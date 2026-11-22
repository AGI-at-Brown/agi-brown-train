import jax
import jax.numpy as jnp


def gated_delta_scan(q, k, v, decay, write):
  """Run the recurrence on [batch, time, heads, width] inputs."""
  batch, _, heads, key_dim = k.shape
  state = jnp.zeros((batch, heads, key_dim, v.shape[-1]), dtype=v.dtype)

  def step(state, inputs):
    """Run one step of the recurrence."""
    q_t, k_t, v_t, decay_t, write_t = inputs
    state = decay_t[..., None, None] * state

    remembered = jnp.einsum("bhkv,bhk->bhv", state, k_t)
    correction = write_t[..., None] * (v_t - remembered)
    state = state + k_t[..., :, None] * correction[..., None, :]

    output = jnp.einsum("bhkv,bhk->bhv", state, q_t)
    return state, output

  time_major = tuple(jnp.swapaxes(x, 0, 1) for x in (q, k, v, decay, write))
  _, output = jax.lax.scan(step, state, time_major)
  return jnp.swapaxes(output, 0, 1)
