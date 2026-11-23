import jax
import jax.numpy as jnp


def gated_delta_scan(q, k, v, decay, write):
  """Run the recurrence on [batch, time, heads, width] inputs."""
  # width: number of features in one head at one token
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


def gated_delta_chunkwise_parallel_scan(q, k, v, decay, write, *, chunk_size=32):
  """Run the recurrence with batched matrix operations within each chunk.

  Queries and keys are [batch, time, heads, key_dim], values and outputs
  are [batch, time, heads, value_dim], and gates are [batch, time, heads].
  The chunk size must be a positive static integer under JIT.
  """
  if chunk_size < 1:
    raise ValueError("chunk_size must be positive")

  batch, time, heads, key_dim = k.shape
  if time == 0:
    return v

  chunk_size = min(chunk_size, time)
  padding = -time % chunk_size
  num_chunks = (time + padding) // chunk_size
  value_dim = v.shape[-1]
  dtype = jnp.result_type(q, k, v, decay, write, jnp.float32)
  state = jnp.zeros((batch, heads, key_dim, value_dim), dtype=dtype)

  def chunk(x, pad_value=0):
    # Put chunks first for scan, with batch and heads leading each matrix.
    pad_width = ((0, 0), (0, padding)) + ((0, 0),) * (x.ndim - 2)
    padded = jnp.pad(x.astype(dtype), pad_width, constant_values=pad_value)
    return padded.reshape(batch, num_chunks, chunk_size, *x.shape[2:]).transpose(
      1, 0, 3, 2, *range(4, x.ndim + 1)
    )

  positions = jnp.arange(chunk_size)
  strictly_lower = positions[:, None] > positions[None, :]
  causal = positions[:, None] >= positions[None, :]

  def step(state, inputs):
    q_chunk, k_chunk, v_chunk, decay_chunk, write_chunk = inputs
    prefix = jnp.cumprod(decay_chunk, axis=-1)

    # For row t and column j, multiply the decays after write j through t.
    # A reverse cumulative product handles zero decays without division.
    next_decay = jnp.pad(decay_chunk[..., 1:], ((0, 0), (0, 0), (0, 1)), constant_values=1)
    factors = jnp.where(strictly_lower, next_decay[..., None, :], 1)
    pair_decay = jax.lax.cumprod(factors, axis=3, reverse=True)

    keys_transposed = jnp.swapaxes(k_chunk, -1, -2)
    key_scores = k_chunk @ keys_transposed
    write_scale = write_chunk[..., None]
    lower = jnp.where(strictly_lower, write_scale * pair_decay * key_scores, 0)
    system = lower + jnp.eye(chunk_size, dtype=dtype)
    rhs = write_scale * (v_chunk - prefix[..., None] * (k_chunk @ state))
    corrections = jax.lax.linalg.triangular_solve(
      system, rhs, left_side=True, lower=True, unit_diagonal=True
    )

    read_scores = jnp.where(causal, pair_decay * (q_chunk @ keys_transposed), 0)
    output = prefix[..., None] * (q_chunk @ state) + read_scores @ corrections
    state = (
      prefix[..., -1, None, None] * state
      + keys_transposed @ (pair_decay[..., -1, :, None] * corrections)
    )
    return state, output

  inputs = (chunk(q), chunk(k), chunk(v), chunk(decay, 1), chunk(write))
  _, output = jax.lax.scan(step, state, inputs)
  output = output.transpose(1, 0, 3, 2, 4).reshape(batch, time + padding, heads, value_dim)
  return output[:, :time].astype(v.dtype)
