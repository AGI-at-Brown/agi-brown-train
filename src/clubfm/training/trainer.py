"""
Basic transformer implementation.
"""


from __future__ import annotations

import tomllib
from itertools import batched
from tempfile import TemporaryFile
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.sharding import PartitionSpec as P

from clubfm.models.transformer import Config, Transformer, forward, init_params
from clubfm.modules.loss import cross_entropy
from clubfm.training.optim.adam import adam_init, adam_update

if TYPE_CHECKING:
  from tokenizers import Tokenizer


@struct.dataclass
class TrainConfig:
  epochs: int
  lr: float
  vocab: dict[str, int] | None
  num_train_steps: int
  dataset_path: str
  batch_size: int
  weight_cfg: Config
  activation_sharding: P = P("data", None, None)
  weight_sharding: P = P(None, None)
  tokenizer: Tokenizer | None = struct.field(pytree_node=False, default=None)

  @property
  def vocab_size(self):
    if self.tokenizer is not None:
      return self.tokenizer.get_vocab_size()
    return len(self.vocab or {})


def load_trainconfig(path, *, tiny=False):
  """Load explicit model and training settings from a TOML experiment file."""
  with open(path, "rb") as f:
    settings = tomllib.load(f)
  model = settings["model"]
  if tiny:
    model = model | settings["tiny_model"]
  weight_cfg = Config(**model)
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

  return TrainConfig(
    **settings["training"],
    vocab=vocab,
    tokenizer=tokenizer,
    batch_size=weight_cfg.B,
    weight_cfg=weight_cfg,
  )


@struct.dataclass
class TrainState:
  weight_cfg: Config = struct.field(pytree_node=False)
  params: Transformer
  opt: dict


def init_train_state(weight_cfg):
  params = init_params(weight_cfg)
  opt = adam_init(params)
  return TrainState(weight_cfg=weight_cfg, params=params, opt=opt)


def encode(s, vocab):
  return list(map(lambda c: vocab[c], s))


def model_apply(train_cfg, params, tokens):
  return forward(params, tokens, train_cfg.weight_cfg)


def train_step(train_cfg, train_state, batch):
  def loss_fn(params):
    logits = model_apply(train_cfg, params, batch["observed_ids"])
    return cross_entropy(logits, batch["target_ids"])

  loss, grad = jax.value_and_grad(loss_fn)(train_state.params)
  new_params, new_opt = adam_update(train_state.params, grad, train_state.opt, train_cfg.lr)
  new_train_state = train_state.replace(params=new_params, opt=new_opt)
  metrics = {"train_loss": loss}
  return new_train_state, metrics


def _sample_batches(train_cfg, tokens):
  key = jax.random.PRNGKey(0)
  for _ in range(train_cfg.num_train_steps):
    key, sample_key = jax.random.split(key)
    indices = jax.random.randint(sample_key, (train_cfg.batch_size,), 0, len(tokens))
    batch = jnp.asarray(tokens[np.asarray(indices)])
    yield {"observed_ids": batch[:, :-1], "target_ids": batch[:, 1:]}


def _tokenized_batches(train_cfg, lines):
  tokenizer = train_cfg.tokenizer
  sequence_length = train_cfg.weight_cfg.T + 1
  if train_cfg.weight_cfg.V < train_cfg.vocab_size:
    raise ValueError("Model V must cover the tokenizer vocabulary")

  # Spool token IDs to disk so random sampling only loads the selected rows.
  with TemporaryFile() as token_file:
    n_examples = 0
    for texts in batched(lines, 1024):
      rows = [row.ids for row in tokenizer.encode_batch(list(texts), add_special_tokens=False)]
      if any(len(row) != sequence_length for row in rows):
        raise ValueError("Each example must have T + 1 tokens after tokenization")
      np.asarray(rows, dtype=np.int32).tofile(token_file)
      n_examples += len(rows)
    if not n_examples:
      raise ValueError("Dataset is empty")

    tokens = np.memmap(
      token_file, dtype=np.int32, mode="r", shape=(n_examples, sequence_length),
    )
    yield from _sample_batches(train_cfg, tokens)


def get_dataset(train_cfg):
  with open(train_cfg.dataset_path) as f:
    if train_cfg.tokenizer is not None:
      yield from _tokenized_batches(train_cfg, (line.rstrip("\n") for line in f))
      return
    lines = [line.strip() for line in f]

  if not lines:
    raise ValueError("Dataset is empty")
  if train_cfg.vocab is None:
    raise ValueError("A tokenizer or character vocabulary is required")
  if any(len(line) != train_cfg.weight_cfg.T + 1 for line in lines):
    raise ValueError("Each example must have T + 1 characters")
  tokens = np.asarray([encode(line, train_cfg.vocab) for line in lines], dtype=np.int32)
  yield from _sample_batches(train_cfg, tokens)


def train_loop(train_cfg):
  if train_cfg.num_train_steps <= 0:
    raise ValueError("num_train_steps must be positive")
  train_state = init_train_state(train_cfg.weight_cfg)
  step_fn = jax.jit(lambda state, batch: train_step(train_cfg, state, batch))
  batch = iter(get_dataset(train_cfg))
  for step in range(train_cfg.num_train_steps):
    train_state, metrics = step_fn(train_state, next(batch))
    jax.debug.print("metrics {}", metrics)
  return train_state, metrics
