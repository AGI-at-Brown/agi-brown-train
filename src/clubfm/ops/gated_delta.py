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

  q_chunk, k_chunk, v_chunk = chunk(q), chunk(k), chunk(v)
  decay_chunk, write_chunk = chunk(decay, 1), chunk(write)
  prefix = jnp.cumprod(decay_chunk, axis=-1)

  # For row t and column j, multiply the decays after write j through t.
  next_decay = jnp.pad(decay_chunk[..., 1:], ((0, 0), (0, 0), (0, 0), (0, 1)), constant_values=1)
  factors = jnp.where(strictly_lower, next_decay[..., None, :], 1)
  pair_decay = jax.lax.cumprod(factors, axis=factors.ndim - 1, reverse=True)

  keys_transposed = jnp.swapaxes(k_chunk, -1, -2)
  write_scale = write_chunk[..., None]
  lower = jnp.where(strictly_lower, write_scale * pair_decay * (k_chunk @ keys_transposed), 0)
  system = lower + jnp.eye(chunk_size, dtype=dtype)

  # Solve each chunk's corrections as an affine function of its incoming state.
  rhs = write_scale * jnp.concatenate((v_chunk, -prefix[..., None] * k_chunk), axis=-1)
  corrections = jax.lax.linalg.triangular_solve(
    system, rhs, left_side=True, lower=True, unit_diagonal=True,
  )
  correction_offset, correction_matrix = jnp.split(corrections, (value_dim,), axis=-1)
  write_keys = keys_transposed * pair_decay[..., -1, None, :]
  transition = prefix[..., -1, None, None] * jnp.eye(key_dim, dtype=dtype)
  transition = transition + write_keys @ correction_matrix
  offset = write_keys @ correction_offset

  def compose(earlier, later):
    left, left_offset = earlier
    right, right_offset = later
    return right @ left, right @ left_offset + right_offset

  _, chunk_states = jax.lax.associative_scan(compose, (transition, offset), axis=0)
  incoming = jnp.concatenate((jnp.zeros_like(chunk_states[:1]), chunk_states[:-1]), axis=0)
  read_scores = jnp.where(causal, pair_decay * (q_chunk @ keys_transposed), 0)
  read_matrix = prefix[..., None] * q_chunk + read_scores @ correction_matrix
  output = read_matrix @ incoming + read_scores @ correction_offset
  output = output.transpose(1, 0, 3, 2, 4).reshape(batch, time + padding, heads, value_dim)
  return output[:, :time].astype(v.dtype)
