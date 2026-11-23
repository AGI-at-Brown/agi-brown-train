"""
Sparse attention in Qwen 3.8 Flash Next
"""

import jax
import jax.numpy as jnp
from flax import struct

from clubfm.modules.normalization import rmsnorm
from clubfm.modules.rotary import partial_rope
from clubfm.ops.attention import grouped_query_attention

@struct.dataclass
class CompressedIndexerParams:
    wq: jax.Array  # (model_dim, num_query_heads, head_dim)
    wk: jax.Array  # (model_dim, head_dim)
    q_norm_weight: jax.Array  # (head_dim,), shared across query heads
    k_norm_weight: jax.Array  # (head_dim,)

@struct.dataclass
class SparseCoreAttnParams:
  wq: jax.Array
  wk: jax.Array
  wv: jax.Array
  wo: jax.Array
  q_norm_weight: jax.Array  # (head_dim,)
  k_norm_weight: jax.Array  # (head_dim,)

def compressed_indexer(
    params, tokens, *, rope_theta, block_size=4, rotary_dim=64, token_budget=2048
):
    """Return selected block IDs per query, with -1 for unused slots."""
    if block_size <= 0 or token_budget < 0:
        raise ValueError("block_size must be positive and token_budget nonnegative")
    q = jnp.einsum("btd,dnh->btnh", tokens, params.wq)
    q = rmsnorm(q, 1.0 + params.q_norm_weight)
    k = jnp.einsum("btd,dh->bth", tokens, params.wk)

    # Average-pool complete key blocks before normalizing.
    batch, sequence, head_dim = k.shape
    num_blocks = sequence // block_size
    k = k[:, :num_blocks * block_size, :]
    k = k.reshape(batch, num_blocks, block_size, head_dim).mean(axis=2)
    k = rmsnorm(k, 1.0 + params.k_norm_weight)

    query_positions = jnp.arange(sequence)
    key_positions = jnp.arange(num_blocks) * block_size
    q = partial_rope(q, query_positions, rope_theta, rotary_dim)
    k = partial_rope(k, key_positions, rope_theta, rotary_dim)
    block_ends = key_positions + block_size - 1
    block_budget = (token_budget + block_size - 1) // block_size
    return _select_key_blocks(q, k, query_positions, block_ends, block_budget)

# only for the indexer for sparse attention
def _select_key_blocks(q, k, query_positions, block_ends, block_budget):
    head_scores = jnp.einsum("btnh,bkh->btnk", q, k)
    block_scores = jax.nn.relu(head_scores).sum(axis=2)

    complete_blocks = query_positions[:, None] >= block_ends
    causal_scores = jnp.where(complete_blocks, block_scores, -jnp.inf)
    num_selected = min(block_budget, k.shape[1])
    selected_scores, block_ids = jax.lax.top_k(causal_scores, num_selected)
    return jnp.where(selected_scores == -jnp.inf, -1, block_ids)


def sparse_core_attention_block(sparse_core_attn, tokens, config, *, rotary_dim=64):
  q = jnp.einsum("btd,dnh->btnh", tokens, sparse_core_attn.wq)
  k = jnp.einsum("bsd,dkh->bskh", tokens, sparse_core_attn.wk)
  v = jnp.einsum("bsd,dkh->bskh", tokens, sparse_core_attn.wv)

  q = rmsnorm(q, 1.0 + sparse_core_attn.q_norm_weight)
  k = rmsnorm(k, 1.0 + sparse_core_attn.k_norm_weight)

  positions = jnp.arange(tokens.shape[1])
  q = partial_rope(q, positions, config.rope_theta, rotary_dim)
  k = partial_rope(k, positions, config.rope_theta, rotary_dim)

  qkv = grouped_query_attention(q, k, v)
  o = jnp.einsum("btnh,nhd->btd", qkv, sparse_core_attn.wo)
  return o
