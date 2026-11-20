"""Chunked top-1 expert routing with optional expert-parallel all-to-all."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import struct


@dataclass(frozen=True)
class MoEConfig:
  chunk_size: int = 128
  expert_parallel_size: int = 1
  axis_name: str = "expert"


@struct.dataclass
class MoEParams:
  router: jax.Array   # [D, global_experts]
  w1: jax.Array       # [num_local_experts, D, F]
  w2: jax.Array       # [num_local_experts, F, D]

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
