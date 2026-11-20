"""All different MLP blocks should be in this file."""

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class BasicMLPParams:
  w1: jax.Array
  w2: jax.Array

@struct.dataclass
class DenseMLPParams:
  layers: list[jax.Array]

@struct.dataclass
class SwiGLUMLPParams:
  w1: jax.Array
  w2: jax.Array
  w3: jax.Array


def mlp_block(mlp, tokens):
  first = jnp.einsum("btd,df->btf", tokens, mlp.w1)
  second = jax.nn.gelu(first)
  return jnp.einsum("btf,fd->btd", second, mlp.w2)

def densemlp_block(mlp, tokens):
  output = tokens
  for layer in mlp.layers:
    output = jax.nn.gelu(jnp.einsum("btd,df->btf", output, layer))
  return output

def swiglu_block(mlp, tokens):
  gate = jax.nn.silu(jnp.einsum("btd,df->btf", tokens, mlp.w1))
  up = jnp.einsum("btd,df->btf", tokens, mlp.w2)
  output = jnp.einsum("btf,fd->btd", gate * up, mlp.w3)
  return output