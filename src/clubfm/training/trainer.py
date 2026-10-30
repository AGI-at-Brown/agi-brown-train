"""
Basic transformer implementation.
"""


import jax
import jax.numpy as jnp
from flax import struct
from jax.sharding import PartitionSpec as P

from clubfm.models.transformer import Config, Transformer, forward, init_params
from clubfm.modules.loss import cross_entropy
from clubfm.training.optim.adam import adam_init, adam_update


@struct.dataclass
class TrainConfig:
  activation_sharding: P = P("data", None, None)
  weight_sharding: P = P(None, None)
  epochs: int = 2
  lr: float = 1e-4
  vocab = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "+": 10,
    "=": 11,
  }
  vocab_size = len(vocab)
  num_train_steps: int = 4000
  dataset_path: str = "data_128000.txt"
  batch_size: int = 32
  weight_cfg: Config = Config()


def init_trainconfig(weight_cfg):
  train_cfg = TrainConfig(
    weight_cfg=weight_cfg,
    batch_size=weight_cfg.B,
  )
  return train_cfg


def init_trainconfig_with_steps(weight_cfg, steps):
  train_cfg = TrainConfig(weight_cfg=weight_cfg, batch_size=weight_cfg.B, num_train_steps=steps)
  return train_cfg


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


def get_dataset(train_cfg):
  with open(train_cfg.dataset_path) as f:
    lines = [line.strip() for line in f.readlines()]

  if not lines:
    raise ValueError("Dataset is empty")
  if any(len(line) != train_cfg.weight_cfg.T + 1 for line in lines):
    raise ValueError("Each example must have T + 1 characters")
  tokens = jnp.array([encode(line, train_cfg.vocab) for line in lines])
  n_examples = tokens.shape[0]
  keys = jax.random.split(jax.random.PRNGKey(0), train_cfg.num_train_steps)
  for step in range(train_cfg.num_train_steps):
    idx = jax.random.randint(keys[step], (train_cfg.batch_size,), 0, n_examples)
    batch = tokens[idx]
    yield {"observed_ids": batch[:, :-1], "target_ids": batch[:, 1:]}


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
