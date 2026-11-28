"""Load a trained clubfm model and score token sequences with it.

`load_model_config` reads only the model and tokenizer settings of an experiment TOML, so a
model can be evaluated with the config it was trained with even when that config carries
training-only keys. `load_params` loads exported weights or a full training checkpoint.

`ClubfmScorer.score` returns, for each sequence, the log-probability of every token after the
first and whether the model's top prediction was that token. Sequences are right-padded into a
few fixed shapes so JAX compiles once per shape. Right padding cannot change the real tokens'
scores because every layer is causal (and top-k MoE routing never drops tokens).
"""

from __future__ import annotations

import dataclasses
import tomllib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_state_dict, msgpack_restore

from clubfm.models import qwen_flash_next, transformer
from clubfm.training.trainer import _model_module

ARCHITECTURES = {"qwen_flash_next": qwen_flash_next.Config, "transformer": transformer.Config}


@dataclasses.dataclass(frozen=True)
class ModelSpec:
  weight_cfg: Any
  tokenizer: Any  # tokenizers.Tokenizer, or None for configs with a [vocab] table
  vocab: dict[str, int] | None


def load_model_config(path: str | Path) -> ModelSpec:
  """Read [architecture], [model] and the tokenizer from an experiment TOML.

  [training] is ignored. A [model] key this code does not know is accepted only when it is
  false or 0: such keys switch off features of experimental branches, and the model with the
  feature switched off is the one defined here. Any other unknown key is an error, because the
  weights would then be run as a different model.
  """
  with open(path, "rb") as f:
    settings = tomllib.load(f)
  architecture = settings.get("architecture", {}).get("name", "transformer")
  if architecture not in ARCHITECTURES:
    raise ValueError(f"Unsupported architecture: {architecture}")
  config_class = ARCHITECTURES[architecture]
  known = {field.name for field in dataclasses.fields(config_class)}
  model = {}
  for key, value in settings["model"].items():
    if key in known:
      model[key] = value
    elif value:
      raise ValueError(f"{path}: [model].{key} = {value!r} is not supported by this code")
  weight_cfg = config_class(**model)

  tokenizer = None
  vocab = settings.get("vocab")
  tokenizer_id = settings.get("data", {}).get("tokenizer")
  if tokenizer_id is not None:
    from clubfm.data.tokenization.cosmo2 import COSMO2_TOKENIZER_ID, load_cosmo2_tokenizer

    if tokenizer_id != COSMO2_TOKENIZER_ID:
      raise ValueError(f"Unsupported tokenizer: {tokenizer_id}")
    tokenizer = load_cosmo2_tokenizer()
  elif vocab is None:
    raise ValueError("Configure [data].tokenizer or provide a [vocab] table")
  return ModelSpec(weight_cfg=weight_cfg, tokenizer=tokenizer, vocab=vocab)


def load_params(weight_cfg, weights_path: str | Path):
  """Load params from an exported weights file or a full training checkpoint, as float32."""
  state = msgpack_restore(Path(weights_path).read_bytes())
  if isinstance(state, dict) and "params" in state and "opt" in state:
    state = state["params"]  # full training-state checkpoint
  template = jax.eval_shape(lambda: _model_module(weight_cfg).init_params(weight_cfg))
  params = from_state_dict(template, state)
  expected = jax.tree.map(lambda leaf: leaf.shape, template)
  actual = jax.tree.map(lambda leaf: np.shape(leaf), params)
  if expected != actual:
    raise ValueError(f"{weights_path} does not match the configured model shapes")
  return jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float32), params)


def pack_rows(token_lists: Iterable[Sequence[int]], eos_token_id: int, sequence_length: int):
  """Pack documents exactly like `clubfm.data.transforms.packing.pack_batches`.

  Each document gets a trailing EOS, and consecutive rows of `sequence_length + 1` tokens overlap
  by one token, so every token after the first is a target exactly once. Returns the full rows
  and the leftover tail (which `pack_batches` would drop) so callers can score the whole text.
  """
  stream: list[int] = []
  for ids in token_lists:
    stream.extend(ids)
    stream.append(eos_token_id)
  rows = []
  start = 0
  while start + sequence_length + 1 <= len(stream):
    rows.append(stream[start : start + sequence_length + 1])
    start += sequence_length
  return np.asarray(rows, dtype=np.int32).reshape(-1, sequence_length + 1), stream[start:]


class ClubfmScorer:
  def __init__(
    self,
    config_path: str | Path,
    weights_path: str | Path,
    *,
    batch_size: int = 16,
    max_length: int | None = None,
    length_bucket: int = 128,
  ):
    spec = load_model_config(config_path)
    weight_cfg = spec.weight_cfg
    self.weight_cfg = weight_cfg
    self.tokenizer = spec.tokenizer
    self.eos_token_id = weight_cfg.eos_token_id
    # Longest input the model sees: the training context unless overridden.
    self.max_length = max_length or weight_cfg.T
    self.batch_size = batch_size
    self.length_bucket = length_bucket
    self.params = load_params(weight_cfg, weights_path)
    forward = _model_module(weight_cfg).forward

    @jax.jit
    def _score(params, inputs, targets):
      logits = forward(params, inputs, weight_cfg).astype(jnp.float32)
      logprobs = jax.nn.log_softmax(logits)
      target_logprobs = jnp.take_along_axis(logprobs, targets[..., None], axis=-1)[..., 0]
      return target_logprobs, jnp.argmax(logits, axis=-1) == targets

    self._score = _score

  def _bucket(self, length: int) -> int:
    bucket = -(-length // self.length_bucket) * self.length_bucket
    return min(bucket, self.max_length)

  def score(self, sequences: Sequence[Sequence[int]]):
    """Return (logprobs, greedy) arrays of length len(seq) - 1 for each token sequence.

    Each sequence may be at most `max_length + 1` tokens; its first token is context only.
    """
    results: list = [None] * len(sequences)
    order = sorted(range(len(sequences)), key=lambda i: -len(sequences[i]))
    for start in range(0, len(order), self.batch_size):
      indices = order[start : start + self.batch_size]
      longest = len(sequences[indices[0]]) - 1
      if longest > self.max_length:
        raise ValueError(f"Sequence of {longest + 1} tokens exceeds max_length + 1")
      width = self._bucket(max(longest, 1))
      inputs = np.full((self.batch_size, width), self.eos_token_id, dtype=np.int32)
      targets = np.full((self.batch_size, width), self.eos_token_id, dtype=np.int32)
      for row, index in enumerate(indices):
        tokens = np.asarray(sequences[index], dtype=np.int32)
        inputs[row, : len(tokens) - 1] = tokens[:-1]
        targets[row, : len(tokens) - 1] = tokens[1:]
      logprobs, greedy = jax.device_get(self._score(self.params, inputs, targets))
      for row, index in enumerate(indices):
        count = len(sequences[index]) - 1
        results[index] = (logprobs[row, :count].astype(np.float64), greedy[row, :count])
    return results
