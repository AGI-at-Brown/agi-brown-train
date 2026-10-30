"""Reference causal grouped-query attention using JAX operations."""

import jax
import jax.numpy as jnp

# GQA
def grouped_query_attention(q, k, v):
  N, H = q.shape[-2:]
  K = k.shape[-2]
  q_reshaped = jnp.reshape(q, (q.shape[0], q.shape[1], K, N // K, H))
  qk = jnp.einsum("btkgh,bskh->btskg", q_reshaped, k) / jnp.sqrt(H)

  mask = jnp.tril(jnp.ones((q.shape[1], k.shape[1]), dtype=bool))[None, :, :, None, None]
  qk = jax.nn.softmax(jnp.where(mask, qk, -jnp.inf), axis=2)

  qkv = jnp.reshape(jnp.einsum("btskg,bskh->btkgh", qk, v), (qk.shape[0], qk.shape[1], N, H))
  return qkv
