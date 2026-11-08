import jax
import jax.numpy as jnp
import numpy as np

from clubfm.layers.moe import MoEConfig, MoEParams, moe_block


def test_chunked_moe_matches_reference_and_gradients():
  keys = jax.random.split(jax.random.PRNGKey(4), 4)
  moe = MoEParams(
    router=jax.random.normal(keys[0], (3, 2)),
    w1=jax.random.normal(keys[1], (2, 3, 5)),
    w2=jax.random.normal(keys[2], (2, 5, 3)),
  )
  # Includes a valid zero token and an incomplete final chunk.
  x = jax.random.normal(keys[3], (2, 3, 3)).at[0, 0].set(0)

  def reference(params, tokens):
    probabilities = jax.nn.softmax(jnp.einsum("btd,de->bte", tokens, params.router), axis=-1)
    ids = jnp.argmax(probabilities, axis=-1)
    gates = jnp.max(probabilities, axis=-1)
    hidden = jax.nn.gelu(jnp.einsum("btd,btdf->btf", tokens, params.w1[ids]))
    return jnp.einsum("btf,btfd->btd", hidden, params.w2[ids]) * gates[..., None]

  apply = jax.jit(lambda p, x: moe_block(p, x, MoEConfig(chunk_size=4)))
  np.testing.assert_allclose(apply(moe, x), reference(moe, x), atol=1e-5)
  actual_grad = jax.grad(lambda p: jnp.sum(apply(p, x)))(moe)
  expected_grad = jax.grad(lambda p: jnp.sum(reference(p, x)))(moe)
  for actual, expected in zip(jax.tree.leaves(actual_grad), jax.tree.leaves(expected_grad)):
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_expert_parallel_exchange_matches_single_device():
  if jax.local_device_count() < 2:
    import pytest
    pytest.skip("Requires two devices; use XLA_FLAGS=--xla_force_host_platform_device_count=2")
  keys = jax.random.split(jax.random.PRNGKey(7), 4)
  # All nonzero tokens route to expert 1, including those starting on device 0.
  router = jnp.array([[-1., 1.], [-1., 1.]])
  w1 = jax.random.normal(keys[0], (2, 2, 3))
  w2 = jax.random.normal(keys[1], (2, 3, 2))
  x = jnp.abs(jax.random.normal(keys[2], (2, 1, 5, 2))).at[0, 0, 0].set(0)
  distributed = MoEParams(jnp.stack([router, router]), w1[:, None], w2[:, None])
  cfg = MoEConfig(chunk_size=3, expert_parallel_size=2)
  apply = jax.pmap(lambda p, x: moe_block(p, x, cfg), axis_name="expert",
                   devices=jax.local_devices()[:2])
  actual = apply(distributed, x)
  expected = moe_block(MoEParams(router, w1, w2), x, MoEConfig(chunk_size=3))
  np.testing.assert_allclose(actual, expected, atol=1e-5)
  grads = jax.grad(lambda p: jnp.sum(apply(p, x)))(distributed)
  assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(grads))
