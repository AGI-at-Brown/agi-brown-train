import jax
import jax.numpy as jnp
import numpy as np

from clubfm.layers.gated_delta import GatedDeltaParams, gated_delta_block
from clubfm.ops.gated_delta import gated_delta_scan


def test_gated_delta_scan_matches_loop():
  """Compare the scan with a plain NumPy loop over batches, heads, and time."""
  rng = np.random.default_rng(7)
  q = rng.normal(size=(2, 5, 3, 2)).astype(np.float32)
  k = rng.normal(size=q.shape).astype(np.float32)
  v = rng.normal(size=(2, 5, 3, 4)).astype(np.float32)
  decay = rng.uniform(0.2, 0.9, size=q.shape[:3]).astype(np.float32)
  write = rng.uniform(0.1, 0.7, size=q.shape[:3]).astype(np.float32)
  expected = np.empty_like(v)

  for batch in range(q.shape[0]):
    for head in range(q.shape[2]):
      state = np.zeros((q.shape[-1], v.shape[-1]), dtype=np.float32)
      for t in range(q.shape[1]):
        index = batch, t, head
        state *= decay[index]
        key = k[index]
        error = v[index] - key @ state
        state += write[index] * np.outer(key, error)
        expected[index] = q[index] @ state

  actual = jax.jit(gated_delta_scan)(q, k, v, decay, write)
  np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_gated_delta_block_causality_shape_and_gradients():
  """Check causality, output shape, and finite gradients under JIT."""
  rng = np.random.default_rng(12)
  width, qk_heads, value_heads, key_dim, value_dim = 8, 2, 6, 3, 4
  kernel_size = 3

  def weights(*shape):
    return jnp.asarray(rng.normal(scale=0.2, size=shape), dtype=jnp.float32)

  params = GatedDeltaParams(
    wq=weights(width, qk_heads, key_dim),
    wk=weights(width, qk_heads, key_dim),
    wv=weights(width, value_heads, value_dim),
    q_conv=weights(kernel_size, qk_heads, key_dim),
    k_conv=weights(kernel_size, qk_heads, key_dim),
    v_conv=weights(kernel_size, value_heads, value_dim),
    w_write=weights(width, value_heads),
    w_decay=weights(width, value_heads),
    decay_bias=weights(value_heads),
    log_decay_rate=weights(value_heads),
    w_gate=weights(width, value_heads, value_dim),
    norm_weight=jnp.zeros(value_dim),
    wo=weights(value_heads, value_dim, width),
  )

  tokens = weights(2, 5, width).at[:, 0].set(0)
  apply = jax.jit(gated_delta_block)
  output = apply(params, tokens)
  changed = apply(params, tokens.at[:, -1].add(2.0))

  assert output.shape == tokens.shape
  assert np.isfinite(output).all()
  np.testing.assert_allclose(output[:, :-1], changed[:, :-1], atol=1e-6, rtol=1e-5)
  assert not np.allclose(output[:, -1], changed[:, -1])

  def loss(p, x):
    return jnp.mean(jnp.square(gated_delta_block(p, x)))

  grads = jax.jit(jax.grad(loss, argnums=(0, 1)))(params, tokens)
  assert all(np.isfinite(grad).all() for grad in jax.tree.leaves(grads))
