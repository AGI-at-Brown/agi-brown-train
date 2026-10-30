"""Explicit rotary position matrices from the notebook transformer."""

import jax
import jax.numpy as jnp


def rope(m, cfg):
  big_mat = jnp.zeros((cfg.H, cfg.H))
  for i in range(cfg.H // 2):
    theta = jnp.power(10000.0, -2.0 * i / cfg.H)
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
