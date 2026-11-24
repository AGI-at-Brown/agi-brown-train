import jax
import jax.numpy as jnp
import numpy as np

from clubfm.layers.moe import MoEConfig, MoEParams, init_topk_moe, moe_block, topk_moe_block
from clubfm.modules.mlp import swiglu_block


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
  router = jnp.array([[-1.0, 1.0], [-1.0, 1.0]])
  w1 = jax.random.normal(keys[0], (2, 2, 3))
  w2 = jax.random.normal(keys[1], (2, 3, 2))
  x = jnp.abs(jax.random.normal(keys[2], (2, 1, 5, 2))).at[0, 0, 0].set(0)
  distributed = MoEParams(jnp.stack([router, router]), w1[:, None], w2[:, None])
  cfg = MoEConfig(chunk_size=3, expert_parallel_size=2)
  apply = jax.pmap(
    lambda p, x: moe_block(p, x, cfg), axis_name="expert", devices=jax.local_devices()[:2]
  )
  actual = apply(distributed, x)
  expected = moe_block(MoEParams(router, w1, w2), x, MoEConfig(chunk_size=3))
  np.testing.assert_allclose(actual, expected, atol=1e-5)
  grads = jax.grad(lambda p: jnp.sum(apply(p, x)))(distributed)
  assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(grads))


def test_topk_swiglu_matches_dense_reference_and_gradients():
  expert_key, token_key = jax.random.split(jax.random.key(9))
  params = init_topk_moe(expert_key, 3, 4, 5, 6)
  tokens = jax.random.normal(token_key, (2, 3, 3))

  def reference(params, tokens, normalize):
    probabilities = jax.nn.softmax(tokens @ params.router, axis=-1)
    ids = jnp.argsort(probabilities, axis=-1)[..., -2:]
    selected = jax.nn.one_hot(ids, 4).sum(axis=-2)
    weights = selected * probabilities
    if normalize:
      weights = weights / weights.sum(axis=-1, keepdims=True)
    experts = jax.vmap(swiglu_block, in_axes=(0, None))(params.experts, tokens)
    output = jnp.einsum("ebtd,bte->btd", experts, weights) + swiglu_block(params.shared, tokens)
    aux = 4 * jnp.sum((selected.mean(axis=(0, 1)) / 2) * probabilities.mean(axis=(0, 1)))
    return output, aux

  for normalize in (False, True):
    apply = jax.jit(lambda p, x: topk_moe_block(p, x, 2, normalize, return_aux=True))
    actual, actual_aux = apply(params, tokens)
    expected, expected_aux = reference(params, tokens, normalize)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(actual_aux, expected_aux, atol=1e-6)

    def objective(fn, params):
      output, aux = fn(params, tokens)
      return jnp.sum(output**2) + aux

    actual_grad = jax.grad(lambda p: objective(apply, p))(params)
    expected_grad = jax.grad(lambda p: objective(lambda p, x: reference(p, x, normalize), p))(
      params
    )
    for actual, expected in zip(jax.tree.leaves(actual_grad), jax.tree.leaves(expected_grad)):
      np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_bfloat16_router_preserves_small_logit_differences():
  params = init_topk_moe(jax.random.key(0), 2, 2, 3, 3)
  params = jax.tree.map(lambda weight: weight.astype(jnp.bfloat16), params)
  params = params.replace(router=jnp.array([[1, 1], [0, 0.001]], dtype=jnp.bfloat16))
  tokens = jnp.ones((1, 1, 2), dtype=jnp.bfloat16)

  # A BF16 matmul rounds both logits to 1, but expert 1 has the larger FP32 logit.
  expert = jax.tree.map(lambda weight: weight[1], params.experts)
  expected = swiglu_block(expert, tokens) + swiglu_block(params.shared, tokens)
  actual = jax.jit(lambda p, x: topk_moe_block(p, x, 1, True))(params, tokens)
  np.testing.assert_array_equal(actual, expected)
