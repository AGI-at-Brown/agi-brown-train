"""One-hot embedding lookup and tied language-model output projection."""

import jax
import jax.numpy as jnp


def embed_tokens(embed, tokens):
  one_hot = jax.nn.one_hot(tokens, embed.shape[0])
  return jnp.einsum("vd,btv->btd", embed, one_hot)


def project_logits(embed, x):
  return jnp.einsum("btd,vd->btv", x, embed)
