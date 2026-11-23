"""One-hot embedding lookup and tied language-model output projection."""

import jax
import jax.numpy as jnp


def embed_tokens(embed, tokens):
  one_hot = jax.nn.one_hot(tokens, embed.shape[0])
  return jnp.einsum("vd,btv->btd", embed, one_hot)


def project_logits(embed, x):
  return jnp.einsum("btd,vd->btv", x, embed)


#number of has functioned used per bigram and trigram
NGRAM_HEADS_PER_ORDER = 8

#2 since bigram and trigram 
NGRAM_HEADS = 2 * NGRAM_HEADS_PER_ORDER


def embed_ngrams(table, tokens, eos_token_id=None):
  """Gather concatenated causal bigram and trigram embeddings."""
  tokens = tokens.astype(jnp.uint32)
  boundary = jnp.uint32(0xFFFFFFFF if eos_token_id is None else eos_token_id)
  #gather token info
  previous = jnp.pad(tokens, ((0, 0), (1, 0)), constant_values=boundary)[:, :-1]
  older = jnp.pad(tokens, ((0, 0), (2, 0)), constant_values=boundary)[:, :-2]
  older = jnp.where(previous == boundary, boundary, older)

  #create embedding
  bigram = tokens * jnp.uint32(0x9E3779B1) ^ previous * jnp.uint32(0x85EBCA77)
  trigram = bigram ^ older * jnp.uint32(0xC2B2AE3D)

  hashes = jnp.repeat(jnp.stack((bigram, trigram), axis=-1), NGRAM_HEADS_PER_ORDER, axis=-1)
  heads = jnp.arange(NGRAM_HEADS, dtype=jnp.uint32)
  hashes = hashes ^ ((heads + 1) * jnp.uint32(0x9E3779B9))

  # Mix high bits into low bits before taking the bucket remainder.
  hashes = (hashes ^ (hashes >> 16)) * jnp.uint32(0x85EBCA6B)
  hashes = (hashes ^ (hashes >> 13)) * jnp.uint32(0xC2B2AE35)
  hashes = hashes ^ (hashes >> 16)
  buckets = (hashes % table.shape[1]).astype(jnp.int32)

  #looks up the embedding
  features = table[jnp.arange(NGRAM_HEADS), buckets]
  #flattens and concat them
  return features.reshape(*tokens.shape, NGRAM_HEADS * table.shape[-1])
