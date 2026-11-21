"""Qwen3.8-Flash-Next Gated DeltaNet"""

import jax
import jax.numpy as jnp
from flax import struct

from clubfm.modules.normalization import rmsnorm
from clubfm.ops.gated_delta import gated_delta_scan


@struct.dataclass
class GatedDeltaParams:
  # Weights for projecting input tokens into queries, keys, and values.
  wq: jax.Array  # [model_dim, qk_heads, key_dim]
  wk: jax.Array  # [model_dim, qk_heads, key_dim]
  wv: jax.Array  # [model_dim, value_heads, value_dim]

  # Short convolution kernels mix each projection with nearby past tokens.
  q_conv: jax.Array  # [kernel_size, qk_heads, key_dim]
  k_conv: jax.Array  # [kernel_size, qk_heads, key_dim]
  v_conv: jax.Array  # [kernel_size, value_heads, value_dim]

  # Weights for projecting input tokens into write and decay gates.
  w_write: jax.Array  # [model_dim, value_heads]
  w_decay: jax.Array  # [model_dim, value_heads]
  decay_bias: jax.Array  # [value_heads]
  log_decay_rate: jax.Array  # [value_heads], A in the Qwen report

  # Projects input tokens into the gate applied to the normalized scan output.
  w_gate: jax.Array  # [model_dim, value_heads, value_dim]

  norm_weight: jax.Array  # [value_dim], shared across heads and centered at zero
  wo: jax.Array  # [value_heads, value_dim, model_dim]


def _short_conv(x, kernel):
  """Mix each channel with its recent history using a learned causal kernel."""
  batch, time, heads, width = x.shape
  channels = heads * width
  kernel_size = kernel.shape[0]
  convolved = jax.lax.conv_general_dilated(
    x.reshape(batch, time, channels),
    kernel.reshape(kernel_size, 1, channels).astype(x.dtype),
    window_strides=(1,),
    padding=((kernel_size - 1, 0),),
    dimension_numbers=("NWC", "WIO", "NWC"),
    feature_group_count=channels,
  )
  return jax.nn.silu(convolved.reshape(x.shape))


def _l2_normalize(x):
  """L2-normalize the last dimension of x, with a small epsilon to avoid division by zero."""
  squared_norm = jnp.sum(jnp.square(x), axis=-1, keepdims=True)
  return x * jax.lax.rsqrt(jnp.maximum(squared_norm, 1e-12))


def gated_delta_block(params: GatedDeltaParams, tokens: jax.Array) -> jax.Array:
  """Mix normalized hidden states of shape [batch, time, model_dim].

  The state starts at zero for each sequence. Consecutive groups of value
  heads share one query/key head. No device mesh or collectives are required.
  """
  qk_heads = params.wq.shape[1]
  value_heads = params.wv.shape[1]
  if qk_heads == 0 or value_heads == 0 or value_heads % qk_heads:
    raise ValueError("Value heads must be a positive multiple of query/key heads")
  heads_per_group = value_heads // qk_heads

  # Project to [batch, time, heads, width].
  q = _short_conv(jnp.einsum("btd,dhk->bthk", tokens, params.wq), params.q_conv)
  k = _short_conv(jnp.einsum("btd,dhk->bthk", tokens, params.wk), params.k_conv)
  v = _short_conv(jnp.einsum("btd,dhv->bthv", tokens, params.wv), params.v_conv)

  # Keep normalization, gates, and the recurrent state at least FP32.
  state_dtype = jnp.result_type(q, k, v, jnp.float32)

  # L2-normalize q and k, and repeat them for each value head in the group.
  q = jnp.repeat(_l2_normalize(q.astype(state_dtype)), heads_per_group, axis=2)
  k = jnp.repeat(_l2_normalize(k.astype(state_dtype)), heads_per_group, axis=2)

  # The write gate scales the correction and the decay gate fades the old state.
  write_logits = jnp.einsum("btd,dh->bth", tokens, params.w_write).astype(state_dtype)
  decay_logits = jnp.einsum("btd,dh->bth", tokens, params.w_decay).astype(state_dtype)
  write = jax.nn.sigmoid(write_logits)

  decay_rate = jnp.exp(params.log_decay_rate.astype(state_dtype))
  decay = jnp.exp(-decay_rate * jax.nn.softplus(decay_logits + params.decay_bias))

  # Run the gated delta recurrence producing a representation of the input tokens.
  mixed = gated_delta_scan(q, k, v.astype(state_dtype), decay.astype(state_dtype), write)

  # Gate the normalized scan output before projecting back to the model width.
  gate_logits = jnp.einsum("btd,dhv->bthv", tokens, params.w_gate).astype(state_dtype)
  gated = rmsnorm(mixed, 1.0 + params.norm_weight) * jax.nn.sigmoid(gate_logits)
  return jnp.einsum("bthv,hvd->btd", gated.astype(v.dtype), params.wo)
