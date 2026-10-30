"""Two-projection GELU MLP from the notebook transformer."""

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class MLPParams:
  w1: jax.Array
  w2: jax.Array


def mlp_block(mlp, tokens, cfg):
  first = jnp.einsum("btd,df->btf", tokens, mlp.w1)
  second = jax.nn.gelu(first)
  return jnp.einsum("btf,fd->btd", second, mlp.w2)
