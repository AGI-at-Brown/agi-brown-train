"""Explicit rotary position matrices from the notebook transformer."""

import jax
import jax.numpy as jnp


def partial_rope(x, positions, rope_theta, rotary_dim):
  """Rotate the first rotary_dim features of (batch, sequence, ..., head_dim)."""
  if rotary_dim < 0 or rotary_dim > x.shape[-1] or rotary_dim % 2:
    raise ValueError("rotary_dim must be even and between 0 and head_dim")
  if positions.shape != (x.shape[1],):
    raise ValueError("positions must contain one position per sequence element")
  if rotary_dim == 0:
    return x

  frequencies = rope_theta ** (
    -jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim
  )

  def rotate_at_position(features, position):
    angles = position * frequencies
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    even = features[..., ::2]
    odd = features[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return jnp.stack((rotated_even, rotated_odd), axis=-1).reshape(features.shape)

  rotated = jax.vmap(rotate_at_position, in_axes=(1, 0), out_axes=1)(x[..., :rotary_dim], positions)
  return jnp.concatenate((rotated.astype(x.dtype), x[..., rotary_dim:]), axis=-1)


def rope(m, cfg):
  big_mat = jnp.zeros((cfg.H, cfg.H))
  for i in range(cfg.H // 2):
    theta = jnp.power(cfg.rope_theta, -2.0 * i / cfg.H)
    block = jnp.array(
      [[jnp.cos(m * theta), -jnp.sin(m * theta)], [jnp.sin(m * theta), jnp.cos(m * theta)]]
    )
    idx = i * 2
    big_mat = big_mat.at[idx : idx + 2, idx : idx + 2].set(block)
  return big_mat


# get rotation matrices for every position 0..T-1
# (T, H, H)
def get_r(T, cfg):
  return jax.vmap(lambda m: rope(m, cfg))(jnp.arange(T))
