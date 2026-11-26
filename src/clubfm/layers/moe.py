"""Mixture-of-experts feed-forward layers."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import struct

from clubfm.modules.mlp import SwiGLUMLPParams, swiglu_block


@dataclass(frozen=True)
class MoEConfig:
  chunk_size: int = 128
  expert_parallel_size: int = 1
  axis_name: str = "expert"


@struct.dataclass
class MoEParams:
  router: jax.Array  # [D, global_experts]
  w1: jax.Array  # [num_local_experts, D, F]
  w2: jax.Array  # [num_local_experts, F, D]


# an efficient moe block
# sorts by router probability
def moe_block(moe, tokens, cfg):
  if cfg.chunk_size <= 0 or cfg.expert_parallel_size <= 0:
    raise ValueError("chunk_size and expert_parallel_size must be positive")
  if tokens.ndim < 2:
    raise ValueError("tokens must have shape [..., D]")
  if moe.router.ndim != 2 or moe.w1.ndim != 3 or moe.w2.ndim != 3:
    raise ValueError("Expected router [D, E], w1 [local_E, D, F], w2 [local_E, F, D]")
  devices = cfg.expert_parallel_size
  local_experts, width, hidden = moe.w1.shape
  if local_experts == 0 or tokens.shape[-1] != width:
    raise ValueError("Expected nonempty experts matching the token width")
  if moe.router.shape != (width, devices * local_experts):
    raise ValueError("Router must cover all expert devices' local experts")
  if moe.w2.shape != (local_experts, hidden, width):
    raise ValueError("Expert output projections must return the token width")

  x = tokens.reshape(-1, width)
  count = x.shape[0]
  if count == 0:
    return tokens
  probabilities = jax.nn.softmax(jnp.einsum("nd,de->ne", x, moe.router), axis=-1)
  assignments = jnp.argmax(probabilities, axis=-1)
  gates = jnp.take_along_axis(probabilities, assignments[:, None], axis=-1)[:, 0]

  # sort and then do bincount
  order = jnp.argsort(assignments, stable=True)
  destinations = assignments[order] // local_experts
  sizes = jnp.bincount(destinations, length=devices)
  starts = jnp.cumsum(sizes) - sizes
  slots = jnp.arange(count) - starts[destinations]
  chunks = (count + cfg.chunk_size - 1) // cfg.chunk_size
  capacity = chunks * cfg.chunk_size
  packed = jnp.zeros((devices, capacity, width), dtype=x.dtype)
  packed = packed.at[destinations, slots].set(x[order])
  expert_ids = jnp.zeros((devices, capacity), dtype=assignments.dtype)
  expert_ids = expert_ids.at[destinations, slots].set(assignments[order] % local_experts)

  # scan dimension [chunks, destinations, chunk_size, ...].
  packed = packed.reshape(devices, chunks, cfg.chunk_size, width).swapaxes(0, 1)
  expert_ids = expert_ids.reshape(devices, chunks, cfg.chunk_size).swapaxes(0, 1)

  def exchange(value):
    if devices == 1:
      return value
    return jax.lax.all_to_all(value, cfg.axis_name, split_axis=0, concat_axis=0)

  # w1 -> gelu -> w2
  def apply_chunk(_, inputs):
    chunk, ids = inputs
    chunk, ids = exchange(chunk), exchange(ids)
    hidden_states = jax.nn.gelu(jnp.einsum("pcd,pcdf->pcf", chunk, moe.w1[ids]))
    output = jnp.einsum("pcf,pcfd->pcd", hidden_states, moe.w2[ids])
    return None, exchange(output)

  _, outputs = jax.lax.scan(apply_chunk, None, (packed, expert_ids))
  outputs = outputs.swapaxes(0, 1).reshape(devices, capacity, width)
  sorted_outputs = outputs[destinations, slots]
  restored = jnp.zeros_like(sorted_outputs).at[order].set(sorted_outputs)
  return (restored * gates[:, None]).reshape(tokens.shape)


@struct.dataclass
class TopKMoEParams:
  router: jax.Array  # [D, E]
  experts: SwiGLUMLPParams  # Each projection has a leading expert axis.
  shared: SwiGLUMLPParams


def init_topk_moe(key, width, num_experts, expert_width, shared_width):
  keys = jax.random.split(key, 7)
  init = jax.nn.initializers.lecun_normal()
  expert_init = jax.nn.initializers.lecun_normal(batch_axis=0)
  return TopKMoEParams(
    router=init(keys[0], (width, num_experts)),
    experts=SwiGLUMLPParams(
      w1=expert_init(keys[1], (num_experts, width, expert_width)),
      w2=expert_init(keys[2], (num_experts, width, expert_width)),
      w3=expert_init(keys[3], (num_experts, expert_width, width)),
    ),
    shared=SwiGLUMLPParams(
      w1=init(keys[4], (width, shared_width)),
      w2=init(keys[5], (width, shared_width)),
      w3=init(keys[6], (shared_width, width)),
    ),
  )


def topk_moe_block(params, tokens, top_k, normalize_weights, *, return_aux=False):
  """Apply single-device top-k routing to tokens of shape [batch, time, width]."""
  num_experts = params.router.shape[-1]
  if not 1 <= top_k <= num_experts:
    raise ValueError("top_k must be between 1 and the number of experts")

  flat = tokens.reshape(-1, tokens.shape[-1])
  logits = jnp.matmul(flat, params.router, preferred_element_type=jnp.float32)
  probabilities = jax.nn.softmax(logits, axis=-1)
  selected_logits, expert_ids = jax.lax.top_k(logits, top_k)
  if normalize_weights:
    weights = jax.nn.softmax(selected_logits, axis=-1)
  else:
    weights = jnp.take_along_axis(probabilities, expert_ids, axis=-1)

  # Sort (token, expert) assignments by expert so each expert runs one grouped matmul over
  # its own tokens. Work scales with top_k, not with the number of experts.
  assignments = expert_ids.reshape(-1)
  order = jnp.argsort(assignments, stable=True)
  token_index = order // top_k
  group_sizes = jnp.bincount(assignments, length=num_experts).astype(jnp.int32)
  sorted_tokens = flat[token_index]
  experts = params.experts
  hidden = jax.nn.silu(jax.lax.ragged_dot(sorted_tokens, experts.w1, group_sizes))
  hidden *= jax.lax.ragged_dot(sorted_tokens, experts.w2, group_sizes)
  outputs = jax.lax.ragged_dot(hidden, experts.w3, group_sizes)
  gated = outputs * weights.reshape(-1)[order][:, None]
  routed = jnp.zeros(flat.shape, gated.dtype).at[token_index].add(gated).astype(outputs.dtype)
  output = routed.reshape(tokens.shape) + swiglu_block(params.shared, tokens)
  if not return_aux:
    return output

  # extra stats return path

  # A uniform router has a load-balancing loss of one for any top-k.
  counts = jnp.bincount(expert_ids.reshape(-1), length=num_experts)
  fractions = counts / max(expert_ids.size, 1)
  mean_probabilities = probabilities.sum(axis=0) / max(flat.shape[0], 1)
  aux_loss = num_experts * jnp.sum(fractions * mean_probabilities)
  return output, aux_loss
