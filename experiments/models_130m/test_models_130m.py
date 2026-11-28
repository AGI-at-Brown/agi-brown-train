import math
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from flax.serialization import msgpack_serialize, to_state_dict

from clubfm.data.transforms.packing import pack_batches
from clubfm.models.qwen_flash_next import Config, forward, init_params
from experiments.models_130m import scoring
from experiments.models_130m.bpb import evaluate

CONFIG = """
[architecture]
name = "qwen_flash_next"

[model]
B = 1
L = 4
D = 16
F = 32
N = 2
K = 1
H = 8
V = 12
T = 16
rope_theta = 10000.0
eos_token_id = 0
ngram_buckets_per_head = 127
linear_num_key_heads = 2
linear_num_value_heads = 4
linear_key_head_dim = 8
linear_value_head_dim = 8
gated_delta_chunk_size = 4
indexer_n_heads = 2
indexer_head_dim = 8
indexer_budget = 4
rotary_dim = 8
num_experts = 4
num_experts_per_tok = 2
moe_intermediate_size = 8
shared_expert_intermediate_size = 8

[training]
epochs = 1
lr = 0.001
num_train_steps = 1
dataset_path = ""

[vocab]
a = 1
"""


class FakeTokenizer:
  """Maps characters a..k to ids 1..11, one token per character."""

  def encode(self, text, add_special_tokens=False):
    return SimpleNamespace(ids=[ord(c) - ord("a") + 1 for c in text if c != " "])

  def encode_batch(self, texts, add_special_tokens=False):
    return [self.encode(text) for text in texts]


@pytest.fixture
def model_files(tmp_path):
  config = tmp_path / "model.toml"
  config.write_text(CONFIG)
  cfg = Config(**__import__("tomllib").loads(CONFIG)["model"])
  params = init_params(cfg, jax.random.key(7))
  weights = tmp_path / "weights.msgpack"
  weights.write_bytes(msgpack_serialize(to_state_dict(params)))
  return config, weights, cfg, params


def make_scorer(model_files, **kwargs):
  config, weights, _, _ = model_files
  scorer = scoring.ClubfmScorer(config, weights, max_length=16, length_bucket=8, **kwargs)
  scorer.tokenizer = FakeTokenizer()
  return scorer


def reference_logprobs(cfg, params, sequence):
  tokens = np.asarray([sequence[:-1]], dtype=np.int32)
  logprobs = jax.nn.log_softmax(forward(params, tokens, cfg))[0]
  return np.asarray(logprobs[np.arange(len(sequence) - 1), sequence[1:]])


def test_pack_rows_matches_training_packing():
  documents = [[1, 2, 3], [4, 5, 6, 7, 8], [9], [10, 11, 2, 3, 4, 5]]
  rows, tail = scoring.pack_rows(documents, eos_token_id=0, sequence_length=4)
  tokenizer = SimpleNamespace(
    encode_batch=lambda group, add_special_tokens: [SimpleNamespace(ids=list(d)) for d in group]
  )
  batches = list(pack_batches(documents, tokenizer, batch_size=1, sequence_length=4,
                              eos_token_id=0))
  trained = np.concatenate([np.concatenate([b["observed_ids"], b["target_ids"][:, -1:]], 1)
                            for b in batches])
  np.testing.assert_array_equal(rows, trained)
  # Every token after the first is a target exactly once, including the leftover tail.
  stream = sum((d + [0] for d in documents), [])
  assert rows.shape[0] * 4 + len(tail) - 1 == len(stream) - 1


def test_load_params_accepts_exported_weights_and_full_checkpoints(model_files, tmp_path):
  _, weights, cfg, params = model_files
  loaded = scoring.load_params(cfg, weights)
  checkpoint = tmp_path / "full.msgpack"
  checkpoint.write_bytes(msgpack_serialize({"params": to_state_dict(params), "opt": {"step": 3}}))
  from_checkpoint = scoring.load_params(cfg, checkpoint)
  for a, b, c in zip(jax.tree.leaves(params), jax.tree.leaves(loaded),
                     jax.tree.leaves(from_checkpoint), strict=True):
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)


def test_load_params_rejects_a_different_model(model_files):
  _, weights, cfg, _ = model_files
  with pytest.raises(ValueError):
    scoring.load_params(cfg.replace(D=32), weights)


def test_scores_do_not_depend_on_padding_or_batch_neighbours(model_files):
  _, _, cfg, params = model_files
  rng = np.random.default_rng(0)
  sequences = [list(rng.integers(0, cfg.V, size=n)) for n in (17, 5, 9, 2, 13)]
  results = make_scorer(model_files, batch_size=3).score(sequences)
  for sequence, (logprobs, greedy) in zip(sequences, results, strict=True):
    expected = reference_logprobs(cfg, params, sequence)
    np.testing.assert_allclose(logprobs, expected, atol=1e-5)
    assert greedy.shape == expected.shape


def test_bpb_counts_every_byte_and_token(model_files):
  scorer = make_scorer(model_files, batch_size=4)
  model = SimpleNamespace(
    name="tiny", num_parameters=1, eos_token_id=0, cosmo2_tokenizer=True,
    encode=lambda texts: [scorer.tokenizer.encode(t).ids for t in texts],
    nll=lambda seqs: [float(-lp.sum()) for lp, _ in scorer.score(seqs)],
  )
  texts = ["abcdefghij", "kab", "cdefghijkabcdefghijk"]
  record = evaluate(model, "toy", texts, sequence_length=8, matched_rows=2)
  assert record["bytes"] == sum(len(t) for t in texts)
  assert record["tokens"] == sum(len(t) + 1 for t in texts) - 1
  assert math.isclose(record["bpb"], record["loss_per_token"] * record["tokens"]
                      / (record["bytes"] * math.log(2)))
  assert "first_rows_loss" in record


def test_harness_loglikelihood_matches_direct_scoring(model_files):
  pytest.importorskip("lm_eval")
  from experiments.models_130m.harness import ClubfmLM

  _, _, cfg, params = model_files
  lm = ClubfmLM(None, None, scorer=make_scorer(model_files, batch_size=2))
  requests = [SimpleNamespace(args=("abc ", "de")), SimpleNamespace(args=("", "fg"))]
  (first, _), (second, _) = lm.loglikelihood(requests)
  # "abc " + "de": the trailing space moves to the continuation; tokens are a b c | d e.
  assert math.isclose(first, reference_logprobs(cfg, params, [1, 2, 3, 4, 5])[-2:].sum(),
                      abs_tol=1e-4)
  # Empty context is replaced by EOS.
  assert math.isclose(second, reference_logprobs(cfg, params, [0, 6, 7]).sum(), abs_tol=1e-4)


def test_harness_rolling_scores_every_token_once(model_files):
  pytest.importorskip("lm_eval")
  from experiments.models_130m.harness import ClubfmLM

  _, _, cfg, params = model_files
  lm = ClubfmLM(None, None, scorer=make_scorer(model_files, batch_size=2))
  text = "abcdefghijk" * 3  # 33 tokens: longer than the 16-token context, so 3 windows
  (total,) = lm.loglikelihood_rolling([SimpleNamespace(args=(text,))])
  tokens = [0] + lm.tok_encode(text)
  expected = sum(
    reference_logprobs(cfg, params, tokens[max(0, end - 16) : end + 1])[-(end - start):].sum()
    for start, end in ((0, 16), (16, 32), (32, 33))
  )
  assert math.isclose(total, expected, abs_tol=1e-4)


def test_model_config_ignores_training_keys_and_switched_off_features(tmp_path):
  config = tmp_path / "trained.toml"
  config.write_text(
    CONFIG.replace("[training]", "[training]\nwarmup_steps = 1000\nmin_lr_ratio = 0.1")
    .replace("eos_token_id = 0", "eos_token_id = 0\ndense_attention = false")
  )
  spec = scoring.load_model_config(config)
  assert spec.weight_cfg == Config(**__import__("tomllib").loads(CONFIG)["model"])
  assert spec.tokenizer is None and spec.vocab == {"a": 1}


def test_model_config_rejects_switched_on_unknown_features(tmp_path):
  config = tmp_path / "dense.toml"
  config.write_text(CONFIG.replace("eos_token_id = 0", "eos_token_id = 0\ndense_attention = true"))
  with pytest.raises(ValueError, match="dense_attention"):
    scoring.load_model_config(config)


def test_greedy_generation_follows_the_model(model_files):
  from experiments.models_130m.generate import Generator

  _, _, cfg, params = model_files
  generator = Generator(cfg, params, length_bucket=8)
  prompt = [3, 1, 4]
  tokens = generator.generate(prompt, max_new_tokens=20, temperature=0, stop_at_eos=False)
  assert len(tokens) == 20
  context = list(prompt)
  for token in tokens:
    window = np.asarray([context[-cfg.T :]], dtype=np.int32)
    assert token == int(np.argmax(forward(params, window, cfg)[0, -1]))
    context.append(token)


def test_sampling_is_reproducible_and_respects_top_k(model_files):
  from experiments.models_130m.generate import Generator

  _, _, cfg, params = model_files
  generator = Generator(cfg, params, length_bucket=8)
  first = generator.generate([1, 2], max_new_tokens=10, seed=3, stop_at_eos=False)
  assert first == generator.generate([1, 2], max_new_tokens=10, seed=3, stop_at_eos=False)
  # With top_k=1 sampling is greedy.
  assert generator.generate([1, 2], max_new_tokens=10, top_k=1, seed=5, stop_at_eos=False) == \
    generator.generate([1, 2], max_new_tokens=10, temperature=0, stop_at_eos=False)


def test_autocomplete_streams_the_generated_text(model_files):
  import io

  from experiments.models_130m.autocomplete import complete
  from experiments.models_130m.generate import Generator

  _, _, cfg, params = model_files
  generator = Generator(cfg, params, length_bucket=8)
  tokenizer = FakeTokenizer()
  tokenizer.decode = lambda ids: "".join(chr(ord("a") + i - 1) if i else "_" for i in ids)
  out = io.StringIO()
  text = complete(generator, tokenizer, "abc", tokens=12, temperature=0.8, top_k=5, seed=2,
                  out=out)
  expected = generator.generate([1, 2, 3], max_new_tokens=12, temperature=0.8, top_k=5, seed=2)
  assert text == tokenizer.decode(expected)
  assert out.getvalue() == text + "\n"
