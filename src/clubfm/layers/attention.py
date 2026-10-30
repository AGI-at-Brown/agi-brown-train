"""Learned query/key/value projections, RoPE, and causal grouped-query attention."""

import jax
import jax.numpy as jnp
from flax import struct

from clubfm.modules.rotary import get_r
from clubfm.ops.attention import grouped_query_attention


@struct.dataclass
class AttnParams:
  wq: jax.Array
  wk: jax.Array
  wv: jax.Array
  wo: jax.Array


def attn_block(attn, tokens, cfg):
  q = jnp.einsum("btd,dnh->btnh", tokens, attn.wq)
  k = jnp.einsum("bsd,dkh->bskh", tokens, attn.wk)
  v = jnp.einsum("bsd,dkh->bskh", tokens, attn.wv)

  # apply RoPE to q and k (not v)
  R = get_r(tokens.shape[1], cfg)  # (T, H, H); T == S for self-attention
  q = jnp.einsum("toj,btnj->btno", R, q)
  k = jnp.einsum("soj,bskj->bsko", R, k)

  qkv = grouped_query_attention(q, k, v)
  o = jnp.einsum("btnh,nhd->btd", qkv, attn.wo)
  return o
