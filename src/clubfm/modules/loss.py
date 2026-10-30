"""Next-token training losses."""

import jax


# Per-token cross-entropy, averaged over batch and time.
def cross_entropy(logits, target_ids):
  labels = jax.nn.one_hot(target_ids, logits.shape[-1])
  return -(labels * jax.nn.log_softmax(logits)).sum(-1).mean()
