"""Root-mean-square normalization."""

import jax
import jax.numpy as jnp


def rmsnorm(x, gamma):
  rms = jax.lax.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-6)
  return gamma * x / rms
