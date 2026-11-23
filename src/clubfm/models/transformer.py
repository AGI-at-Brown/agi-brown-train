"""Notebook transformer: RoPE, grouped-query attention, GELU MLP, tied embeddings."""

import jax
from flax import struct

from clubfm.layers.attention import AttnParams, attn_block
from clubfm.modules.embedding import NGRAM_HEADS, embed_ngrams, embed_tokens, project_logits
from clubfm.modules.mlp import BasicMLPParams, mlp_block
from clubfm.modules.normalization import rmsnorm


@struct.dataclass
class Config:
  B: int
  L: int
  D: int
  F: int
  N: int  # number of q heads
  K: int  # number of kv heads
  H: int  # attention head dimension
  V: int
  T: int
  rope_theta: float
  ngram_buckets_per_head: int = 0
  eos_token_id: int | None = None


@struct.dataclass
class NgramParams:
  table: jax.Array
  projection: jax.Array


@struct.dataclass
class Layer:
  attn: AttnParams
  mlp: BasicMLPParams


@struct.dataclass
class Transformer:
  layers: list[Layer]
  embed: jax.Array
  ngram: NgramParams | None = None


def init_params(cfg, key=None):
  if key is None:
    key = jax.random.PRNGKey(0)
  ngram = None
  if cfg.ngram_buckets_per_head < 0:
    raise ValueError("ngram_buckets_per_head must be nonnegative")
  if cfg.ngram_buckets_per_head:
    if cfg.L < 2 or cfg.D % NGRAM_HEADS:
      raise ValueError(
        f"n-gram embeddings require at least two layers and D divisible by {NGRAM_HEADS}"
      )
    table_key, projection_key = jax.random.split(jax.random.fold_in(key, 1))
    #create the table
    ngram = NgramParams(
      table=0.02 * jax.random.normal(
        table_key, (NGRAM_HEADS, cfg.ngram_buckets_per_head, cfg.D // NGRAM_HEADS)
      ),
      projection=jax.nn.initializers.lecun_normal()(projection_key, (cfg.D, cfg.D)),
    )

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
        mlp=BasicMLPParams(
          w1=jax.random.normal(next(keys), (cfg.D, cfg.F)),
          w2=jax.random.normal(next(keys), (cfg.F, cfg.D)),
        ),
      )
      for _ in range(cfg.L)
    ],
    embed=jax.random.normal(next(keys), (cfg.V, cfg.D)),
    ngram=ngram,
  )


def forward(params, tokens, cfg):
  x = embed_tokens(params.embed, tokens)

  for layer_index, layer in enumerate(params.layers):
    if layer_index == 1 and params.ngram is not None:
      features = embed_ngrams(params.ngram.table, tokens, cfg.eos_token_id)
      x = x + features @ params.ngram.projection
    x = x + attn_block(layer.attn, x, cfg)
    x = x + mlp_block(layer.mlp, x)
    x = rmsnorm(x, 1.0)

  logits = project_logits(params.embed, x)
  return logits  # raw logits
