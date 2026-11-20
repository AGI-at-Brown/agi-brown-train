"""
To use:
uv run python profile_clubfm.py --num-gpus 2
uv run tensorboard --logdir artifacts/profiles
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from clubfm.ops.attention import grouped_query_attention


def build_workload(target, /, *positional, **keyword):
  return target, positional, keyword


def attention_inputs(sharding=None):
  batch_size = 2 if sharding is None else 2 * sharding.mesh.size
  q_key, k_key, v_key = jax.random.split(jax.random.key(0), 3)
  q = jax.random.normal(q_key, (batch_size, 128, 8, 32), dtype=jnp.float32)
  k = jax.random.normal(k_key, (batch_size, 128, 2, 32), dtype=jnp.float32)
  v = jax.random.normal(v_key, (batch_size, 128, 2, 32), dtype=jnp.float32)
  return jax.device_put((q, k, v), sharding)


def main():
  parser = argparse.ArgumentParser(description="Capture a JAX trace for TensorBoard.")
  parser.add_argument("--log-dir", type=Path, default=Path("artifacts/profiles"))
  parser.add_argument("--warmup", type=int, default=3, help="Untraced calls before capture.")
  parser.add_argument("--steps", type=int, default=10, help="Calls to capture.")
  parser.add_argument("--num-gpus", type=int, help="GPUs to use, with two examples per GPU.")
  args = parser.parse_args()
  if args.warmup < 0 or args.steps < 1:
    parser.error("--warmup must be nonnegative and --steps must be positive")

  sharding = None
  if args.num_gpus is not None:
    if args.num_gpus < 1:
      parser.error("--num-gpus must be positive")
    try:
      devices = jax.local_devices(backend="gpu")
    except RuntimeError as error:
      parser.error(f"Could not initialize the JAX GPU backend: {error}")
    if args.num_gpus > len(devices):
      parser.error(f"Requested {args.num_gpus} GPUs, but JAX can see {len(devices)}")
    mesh = Mesh(devices[:args.num_gpus], ("batch",))
    sharding = NamedSharding(mesh, PartitionSpec("batch"))

  target, positional, keyword = build_workload(
    jax.jit(grouped_query_attention), *attention_inputs(sharding)
  )
  jax.block_until_ready((positional, keyword))
  for _ in range(args.warmup):
    jax.block_until_ready(target(*positional, **keyword))

  with jax.profiler.trace(args.log_dir):
    for step in range(args.steps):
      with jax.profiler.StepTraceAnnotation("workload", step_num=step):
        jax.block_until_ready(target(*positional, **keyword))

  print(f"Saved JAX trace to {args.log_dir.resolve()}")


if __name__ == "__main__":
  main()
