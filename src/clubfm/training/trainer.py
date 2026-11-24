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
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from clubfm.models import qwen_flash_next, transformer
from clubfm.modules.loss import cross_entropy
from clubfm.training.distributed import (
  DATA_PARALLEL,
  logical_to_sharding,
  replicate_state,
  shard_batch,
)
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
  weight_cfg: transformer.Config | qwen_flash_next.Config
  activation_sharding: P = P("data", None, None)
  weight_sharding: P = P(None, None)
  tokenizer: Tokenizer | None = struct.field(pytree_node=False, default=None)
  router_aux_loss_coef: float = 0.001

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
  architecture = settings.get("architecture", {}).get("name", "transformer")
  if architecture == "qwen_flash_next":
    weight_cfg = qwen_flash_next.Config(**model)
  elif architecture == "transformer":
    weight_cfg = transformer.Config(**model)
  else:
    raise ValueError(f"Unsupported training architecture: {architecture}")
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
  weight_cfg: transformer.Config | qwen_flash_next.Config = struct.field(pytree_node=False)
  params: transformer.Transformer | qwen_flash_next.Transformer
  opt: dict


def _model_module(weight_cfg):
  if isinstance(weight_cfg, qwen_flash_next.Config):
    return qwen_flash_next
  return transformer


def init_train_state(weight_cfg):
  params = _model_module(weight_cfg).init_params(weight_cfg)
  opt = adam_init(params)
  return TrainState(weight_cfg=weight_cfg, params=params, opt=opt)


def encode(s, vocab):
  return list(map(lambda c: vocab[c], s))


def model_apply(train_cfg, params, tokens):
  return _model_module(train_cfg.weight_cfg).forward(params, tokens, train_cfg.weight_cfg)


def train_step(train_cfg, train_state, batch):
  cfg = train_cfg.weight_cfg
  use_moe = isinstance(cfg, qwen_flash_next.Config) and cfg.num_experts > 0

  def loss_fn(params):
    tokens = batch["observed_ids"]
    if use_moe:
      logits, router_loss = qwen_flash_next.forward(params, tokens, cfg, return_router_aux=True)
    else:
      logits = model_apply(train_cfg, params, tokens)
      router_loss = jnp.float32(0)
    token_loss = cross_entropy(logits, batch["target_ids"])
    loss = token_loss + train_cfg.router_aux_loss_coef * router_loss if use_moe else token_loss
    return loss, (token_loss, router_loss)

  (loss, (token_loss, router_loss)), grad = jax.value_and_grad(loss_fn, has_aux=True)(
    train_state.params
  )
  new_params, new_opt = adam_update(train_state.params, grad, train_state.opt, train_cfg.lr)
  new_train_state = train_state.replace(params=new_params, opt=new_opt)
  metrics = {"train_loss": loss}
  if use_moe:
    metrics.update(cross_entropy=token_loss, router_aux_loss=router_loss)
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
      token_file,
      dtype=np.int32,
      mode="r",
      shape=(n_examples, sequence_length),
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


def train_loop(train_cfg, *, mesh=None, batches=None, on_step=None, log_every=1):
  if train_cfg.num_train_steps <= 0:
    raise ValueError("num_train_steps must be positive")
  if mesh is not None and train_cfg.batch_size % mesh.shape["data"]:
    raise ValueError("Global batch size must be divisible by the data mesh size")

  train_state = init_train_state(train_cfg.weight_cfg)

  def step(state, batch):
    return train_step(train_cfg, state, batch)

  if mesh is None:
    step_fn = jax.jit(step)
  else:
    train_state = replicate_state(train_state, mesh)
    replicated = NamedSharding(mesh, P())
    batch_sharding = logical_to_sharding(P("batch"), mesh, DATA_PARALLEL)
    step_fn = jax.jit(
      step,
      in_shardings=(replicated, batch_sharding),
      out_shardings=(replicated, replicated),
    )
  if batches is None:
    batches = get_dataset(train_cfg)
  metrics = None
  for step_number, examples in enumerate(batches, start=1):
    if mesh is not None:
      examples = shard_batch(examples, mesh)
    train_state, metrics = step_fn(train_state, examples)
    if step_number == 1 or step_number % log_every == 0:
      print(f"step {step_number} loss {float(metrics['train_loss']):.6f}", flush=True)
    if on_step is not None:
      on_step(step_number, train_state)
  if metrics is None:
    raise ValueError("Dataset did not produce a complete training batch")
  return train_state, metrics
