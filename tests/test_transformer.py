import jax
import jax.numpy as jnp
import numpy as np

from clubfm.models.transformer import Config, forward, init_params
from clubfm.training.trainer import init_train_state, init_trainconfig_with_steps, train_step


def test_causal_grouped_query_forward_and_gradients():
  cfg = Config(B=1, L=1, D=8, F=16, N=2, K=1, H=4, V=12, T=4)
  params = init_params(cfg)
  tokens = jnp.array([[0, 1, 2, 3]])
  logits = jax.jit(lambda p, x: forward(p, x, cfg))(params, tokens)
  changed = forward(params, tokens.at[0, 3].set(9), cfg)
  assert logits.shape == (1, 4, 12)
  np.testing.assert_allclose(logits[:, :3], changed[:, :3], atol=1e-5, rtol=1e-5)

  train_cfg = init_trainconfig_with_steps(cfg, 1)
  state = init_train_state(cfg)
  batch = {"observed_ids": tokens, "target_ids": jnp.array([[1, 2, 3, 4]])}
  step = jax.jit(lambda state: train_step(train_cfg, state, batch))
  updated, metrics = step(state)
  assert bool(jnp.isfinite(metrics["train_loss"]))
  assert int(updated.opt["step"]) == 1
  assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(updated.params))
  assert not np.array_equal(state.params.embed, updated.params.embed)
