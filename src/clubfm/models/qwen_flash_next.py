"""Qwen Flash Next text scaffold with hybrid attention and dense MLPs."""

import jax
import jax.numpy as jnp
from flax import struct

from clubfm.layers.gated_delta import GatedDeltaParams, gated_delta_block
from clubfm.layers.sparse_attention import (
  CompressedIndexerParams,
  SparseAttentionParams,
  SparseCoreAttnParams,
  sparse_attention_block,
)
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
  linear_num_key_heads: int = 16
  linear_num_value_heads: int = 48
  linear_key_head_dim: int = 128
  linear_value_head_dim: int = 128
  linear_conv_kernel_dim: int = 4
  indexer_n_heads: int = 4
  indexer_head_dim: int = 128
  indexer_compress_ratio: int = struct.field(pytree_node=False, default=4)
  indexer_budget: int = struct.field(pytree_node=False, default=2048)
  rotary_dim: int = struct.field(pytree_node=False, default=64)


@struct.dataclass
class NgramParams:
  table: jax.Array
  projection: jax.Array


@struct.dataclass
class Layer:
  attn: GatedDeltaParams | SparseAttentionParams
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
    ngram = NgramParams(
      table=0.02 * jax.random.normal(
        table_key, (NGRAM_HEADS, cfg.ngram_buckets_per_head, cfg.D // NGRAM_HEADS)
      ),
      projection=jax.nn.initializers.lecun_normal()(projection_key, (cfg.D, cfg.D)),
    )

  layer_keys = jax.random.split(key, cfg.L + 1)
  return Transformer(
    layers=[_init_layer(cfg, layer_key, i) for i, layer_key in enumerate(layer_keys[:-1])],
    embed=jax.random.normal(layer_keys[-1], (cfg.V, cfg.D)),
    ngram=ngram,
  )


def _init_layer(cfg, key, layer_index):
  keys = iter(jax.random.split(key, 12))

  def weight(*shape):
    return 0.02 * jax.random.normal(next(keys), shape)

  if layer_index % 4 == 3:
    attn = SparseAttentionParams(
      indexer=CompressedIndexerParams(
        wq=weight(cfg.D, cfg.indexer_n_heads, cfg.indexer_head_dim),
        wk=weight(cfg.D, cfg.indexer_head_dim),
        q_norm_weight=jnp.zeros(cfg.indexer_head_dim),
        k_norm_weight=jnp.zeros(cfg.indexer_head_dim),
      ),
      core=SparseCoreAttnParams(
        wq=weight(cfg.D, cfg.N, cfg.H),
        wk=weight(cfg.D, cfg.K, cfg.H),
        wv=weight(cfg.D, cfg.K, cfg.H),
        wo=weight(cfg.N, cfg.H, cfg.D),
        q_norm_weight=jnp.zeros(cfg.H),
        k_norm_weight=jnp.zeros(cfg.H),
      ),
    )
  else:
    key_heads = cfg.linear_num_key_heads
    value_heads = cfg.linear_num_value_heads
    key_dim = cfg.linear_key_head_dim
    value_dim = cfg.linear_value_head_dim
    kernel = cfg.linear_conv_kernel_dim
    attn = GatedDeltaParams(
      wq=weight(cfg.D, key_heads, key_dim),
      wk=weight(cfg.D, key_heads, key_dim),
      wv=weight(cfg.D, value_heads, value_dim),
      q_conv=weight(kernel, key_heads, key_dim),
      k_conv=weight(kernel, key_heads, key_dim),
      v_conv=weight(kernel, value_heads, value_dim),
      w_write=weight(cfg.D, value_heads),
      w_decay=weight(cfg.D, value_heads),
      decay_bias=jnp.zeros(value_heads),
      log_decay_rate=jnp.zeros(value_heads),
      w_gate=weight(cfg.D, value_heads, value_dim),
      norm_weight=jnp.zeros(value_dim),
      wo=weight(value_heads, value_dim, cfg.D),
    )
  return Layer(attn=attn, mlp=BasicMLPParams(w1=weight(cfg.D, cfg.F), w2=weight(cfg.F, cfg.D)))


def forward(params, tokens, cfg):
  x = embed_tokens(params.embed, tokens)

  for layer_index, layer in enumerate(params.layers):
    if layer_index == 1 and params.ngram is not None:
      features = embed_ngrams(params.ngram.table, tokens, cfg.eos_token_id)
      x = x + features @ params.ngram.projection
    normalized = rmsnorm(x, 1.0)
    if isinstance(layer.attn, GatedDeltaParams):
      x = x + gated_delta_block(layer.attn, normalized)
    else:
      x = x + sparse_attention_block(
        layer.attn, normalized, cfg, rotary_dim=cfg.rotary_dim,
        block_size=cfg.indexer_compress_ratio, token_budget=cfg.indexer_budget,
      )
    x = x + mlp_block(layer.mlp, x)
    x = rmsnorm(x, 1.0)

  logits = project_logits(params.embed, x)
  return logits  # raw logits
