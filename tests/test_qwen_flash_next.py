import jax
import jax.numpy as jnp
import numpy as np

from clubfm.layers.gated_delta import GatedDeltaParams
from clubfm.layers.sparse_attention import SparseAttentionParams
from clubfm.models.qwen_flash_next import Config
from clubfm.training.trainer import TrainConfig, init_train_state, model_apply, train_step


def test_hybrid_qwen_moe_updates_in_jitted_train_step():
  cfg = Config(
    B=1,
    L=4,
    D=16,
    F=32,
    N=2,
    K=1,
    H=8,
    V=12,
    T=8,
    rope_theta=10000.0,
    ngram_buckets_per_head=127,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    indexer_n_heads=2,
    indexer_head_dim=8,
    indexer_budget=4,
    rotary_dim=8,
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=8,
    shared_expert_intermediate_size=8,
  )
  train_cfg = TrainConfig(
    epochs=1, lr=0.001, vocab={}, num_train_steps=1, dataset_path="", batch_size=1, weight_cfg=cfg
  )
  batch = {
    "observed_ids": jnp.arange(8)[None, :],
    "target_ids": jnp.arange(1, 9)[None, :],
  }
  state = init_train_state(cfg)
  assert all(isinstance(layer.attn, GatedDeltaParams) for layer in state.params.layers[:3])
  assert isinstance(state.params.layers[3].attn, SparseAttentionParams)
  logits = model_apply(train_cfg, state.params, batch["observed_ids"])
  assert logits.shape == (cfg.B, cfg.T, cfg.V)

  updated, metrics = jax.jit(lambda state: train_step(train_cfg, state, batch))(state)
  assert int(updated.opt["step"]) == 1
  assert all(bool(jnp.isfinite(value)) for value in metrics.values())
  assert float(metrics["router_aux_loss"]) > 0.0
  np.testing.assert_allclose(
    metrics["train_loss"],
    metrics["cross_entropy"] + train_cfg.router_aux_loss_coef * metrics["router_aux_loss"],
  )
  for before, after in zip(state.params.layers, updated.params.layers, strict=True):
    for old, new in zip(jax.tree.leaves(before.mlp), jax.tree.leaves(after.mlp), strict=True):
      assert bool(jnp.all(jnp.isfinite(new)))
      assert not np.array_equal(old, new)
