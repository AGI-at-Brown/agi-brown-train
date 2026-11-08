"""Handwritten Adam from the transformer notebook."""

import jax
import jax.numpy as jnp


def adam_init(params):
  return {
    "m": jax.tree.map(jnp.zeros_like, params),
    "v": jax.tree.map(jnp.zeros_like, params),
    "step": 0,
  }


def adam_update(params, grad, opt_state, lr, b1=0.9, b2=0.999, eps=1e-8):
  step = opt_state["step"] + 1
  new_m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, opt_state["m"], grad)
  new_v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * jnp.square(g), opt_state["v"], grad)

  m_hat = jax.tree.map(lambda m: m / (1 - b1**step), new_m)
  v_hat = jax.tree.map(lambda v: v / (1 - b2**step), new_v)

  new_params = jax.tree.map(
    lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat
  )
  return new_params, {"m": new_m, "v": new_v, "step": step}

# the implementation in SmolLM2's paper
def adamw_update(params, grad, opt_state, lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.01):
  step = opt_state["step"] + 1
  new_m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, opt_state["m"], grad)
  new_v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * jnp.square(g), opt_state["v"], grad)

  m_hat = jax.tree.map(lambda m: m / (1 - b1**step), new_m)
  v_hat = jax.tree.map(lambda v: v / (1 - b2**step), new_v)

  new_params = jax.tree.map(
    lambda p, mh, vh: p - lr * (mh / (jnp.sqrt(vh) + eps) + weight_decay * p),
    params,
    m_hat,
    v_hat,
  )
  return new_params, {"m": new_m, "v": new_v, "step": step}
