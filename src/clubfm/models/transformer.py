"""Notebook transformer: RoPE, grouped-query attention, GELU MLP, tied embeddings."""

import jax
from flax import struct

from clubfm.layers.attention import AttnParams, attn_block
from clubfm.modules.embedding import embed_tokens, project_logits
from clubfm.modules.mlp import MLPParams, mlp_block
from clubfm.modules.normalization import rmsnorm


@struct.dataclass
class Config:
  B: int = 32
  L: int = 4
  D: int = 128
  F: int = 512
  N: int = 4  # number of q heads
  K: int = 4  # number of kv heads
  H: int = 32  # attention head dimension
  V: int = 12
  T: int = 11


@struct.dataclass
class Layer:
  attn: AttnParams
  mlp: MLPParams


@struct.dataclass
class Transformer:
  layers: list[Layer]
  embed: jax.Array


def init_params(cfg, key=None):
  if key is None:
    key = jax.random.PRNGKey(0)
  keys = iter(jax.random.split(key, 6 * cfg.L + 1))
  return Transformer(
    layers=[
      Layer(
        attn=AttnParams(
          wq=jax.random.normal(next(keys), (cfg.D, cfg.N, cfg.H)),
          wk=jax.random.normal(next(keys), (cfg.D, cfg.K, cfg.H)),
          wv=jax.random.normal(next(keys), (cfg.D, cfg.K, cfg.H)),
          wo=jax.random.normal(next(keys), (cfg.N, cfg.H, cfg.D)),
        ),
        mlp=MLPParams(
          w1=jax.random.normal(next(keys), (cfg.D, cfg.F)),
          w2=jax.random.normal(next(keys), (cfg.F, cfg.D)),
        ),
      )
      for _ in range(cfg.L)
    ],
    embed=jax.random.normal(next(keys), (cfg.V, cfg.D)),
  )


def forward(params, tokens, cfg):
  x = embed_tokens(params.embed, tokens)

  for layer in params.layers:
    x = x + attn_block(layer.attn, x, cfg)
    x = x + mlp_block(layer.mlp, x, cfg)
    x = rmsnorm(x, 1.0)

  logits = project_logits(params.embed, x)
  return logits  # raw logits
