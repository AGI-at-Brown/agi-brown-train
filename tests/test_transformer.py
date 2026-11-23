from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from clubfm.models.transformer import Config, forward, init_params
from clubfm.training.trainer import TrainConfig, init_train_state, load_trainconfig, train_step


def test_causal_grouped_query_forward_and_gradients():
  cfg = Config(B=1, L=1, D=8, F=16, N=2, K=1, H=4, V=12, T=4, rope_theta=10000.0)
  params = init_params(cfg)
  tokens = jnp.array([[0, 1, 2, 3]])
  logits = jax.jit(lambda p, x: forward(p, x, cfg))(params, tokens)
  changed = forward(params, tokens.at[0, 3].set(9), cfg)
  assert logits.shape == (1, 4, 12)
  np.testing.assert_allclose(logits[:, :3], changed[:, :3], atol=1e-5, rtol=1e-5)

  train_cfg = load_trainconfig("configs/local/transformer_addition.toml").replace(
    weight_cfg=cfg, batch_size=cfg.B, num_train_steps=1,
  )
  state = init_train_state(cfg)
  batch = {"observed_ids": tokens, "target_ids": jnp.array([[1, 2, 3, 4]])}
  step = jax.jit(lambda state: train_step(train_cfg, state, batch))
  updated, metrics = step(state)
  assert bool(jnp.isfinite(metrics["train_loss"]))
  assert int(updated.opt["step"]) == 1
  assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(updated.params))
  assert not np.array_equal(state.params.embed, updated.params.embed)


def test_ngram_embeddings_update_in_jitted_train_step():
  cfg = Config(B=1, L=2, D=16, F=32, N=2, K=1, H=8, V=12, T=4, rope_theta=10000.0)
  baseline = init_params(cfg)
  assert baseline.ngram is None

  cfg = cfg.replace(ngram_buckets_per_head=127)
  state = init_train_state(cfg)
  assert state.params.ngram is not None
  np.testing.assert_array_equal(state.params.embed, baseline.embed)
  for actual, expected in zip(
    jax.tree.leaves(state.params.layers), jax.tree.leaves(baseline.layers), strict=True
  ):
    np.testing.assert_array_equal(actual, expected)

  train_cfg = TrainConfig(
    epochs=1, lr=0.001, vocab={}, num_train_steps=1, dataset_path="", batch_size=1, weight_cfg=cfg
  )
  batch = {
    "observed_ids": jnp.array([[0, 1, 2, 3]]),
    "target_ids": jnp.array([[1, 2, 3, 4]]),
  }
  updated, metrics = jax.jit(lambda state: train_step(train_cfg, state, batch))(state)
  assert bool(jnp.isfinite(metrics["train_loss"]))
  assert int(updated.opt["step"]) == 1
  for name in ("table", "projection"):
    before = getattr(state.params.ngram, name)
    after = getattr(updated.params.ngram, name)
    assert bool(jnp.all(jnp.isfinite(after))), name
    assert not np.array_equal(before, after), name


def test_config_file_and_tiny_overrides(tmp_path):
  path = tmp_path / "experiment.toml"
  source = Path("configs/local/transformer_addition.toml").read_text()
  path.write_text(source.replace("D = 128", "D = 64").replace("lr = 0.0001", "lr = 0.002"))
  cfg = load_trainconfig(path)
  tiny = load_trainconfig(path, tiny=True)
  assert cfg.weight_cfg.D == 64
  assert cfg.lr == 0.002
  assert cfg.batch_size == cfg.weight_cfg.B
  assert tiny.weight_cfg.D == 16
  assert tiny.batch_size == tiny.weight_cfg.B == 2
  assert tiny.weight_cfg.T == cfg.weight_cfg.T
  assert tiny.vocab_size == cfg.weight_cfg.V
