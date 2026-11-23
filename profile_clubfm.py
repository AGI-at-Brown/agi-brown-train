"""
To use:
uv run python profile_clubfm.py --num-gpus 2
uv run python profile_clubfm.py --target gated_delta_block --num-gpus 2
uv run python profile_clubfm.py --target src/clubfm/layers/gated_delta.py:gated_delta_block
uv run tensorboard --logdir artifacts/profiles
"""

import argparse
import sys
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from clubfm.layers.gated_delta import GatedDeltaParams, gated_delta_block
from clubfm.ops.attention import grouped_query_attention


def attention_inputs(sharding=None):
  batch_size = 2 if sharding is None else 2 * sharding.mesh.size
  q_key, k_key, v_key = jax.random.split(jax.random.key(0), 3)
  q = jax.random.normal(q_key, (batch_size, 128, 8, 32), dtype=jnp.float32)
  k = jax.random.normal(k_key, (batch_size, 128, 2, 32), dtype=jnp.float32)
  v = jax.random.normal(v_key, (batch_size, 128, 2, 32), dtype=jnp.float32)
  return jax.device_put((q, k, v), sharding), {}


def gated_delta_inputs(sharding=None):
  batch_size = 2 if sharding is None else 2 * sharding.mesh.size
  model_dim, qk_heads, value_heads, head_dim = 256, 2, 8, 32
  kernel_size = 4
  keys = iter(jax.random.split(jax.random.key(0), 11))

  def normal(*shape):
    return jax.random.normal(next(keys), shape, dtype=jnp.float32)

  scale = model_dim ** -0.5
  params = GatedDeltaParams(
    wq=normal(model_dim, qk_heads, head_dim) * scale,
    wk=normal(model_dim, qk_heads, head_dim) * scale,
    wv=normal(model_dim, value_heads, head_dim) * scale,
    q_conv=normal(kernel_size, qk_heads, head_dim) * kernel_size ** -0.5,
    k_conv=normal(kernel_size, qk_heads, head_dim) * kernel_size ** -0.5,
    v_conv=normal(kernel_size, value_heads, head_dim) * kernel_size ** -0.5,
    w_write=normal(model_dim, value_heads) * scale,
    w_decay=normal(model_dim, value_heads) * scale,
    decay_bias=jnp.zeros(value_heads, dtype=jnp.float32),
    log_decay_rate=jnp.zeros(value_heads, dtype=jnp.float32),
    w_gate=normal(model_dim, value_heads, head_dim) * scale,
    norm_weight=jnp.zeros(head_dim, dtype=jnp.float32),
    wo=normal(value_heads, head_dim, model_dim) * scale,
  )
  tokens = normal(batch_size, 128, model_dim)
  if sharding is not None:
    params = jax.device_put(params, NamedSharding(sharding.mesh, PartitionSpec()))
    tokens = jax.device_put(tokens, sharding)
  return (params, tokens), {}


WORKLOADS = {
  "grouped_query_attention": (grouped_query_attention, attention_inputs),
  "gated_delta_block": (gated_delta_block, gated_delta_inputs),
}


def import_callable(spec):
  module, separator, name = spec.partition(":")
  if not separator or not module or not name:
    raise ValueError("Expected path.py:function or module:function")
  if module.endswith(".py"):
    path = Path(module).resolve()
    if not path.is_file():
      raise ValueError(f"Python file not found: {path}")
    package_root = path.parent
    parts = [path.stem]
    while (package_root / "__init__.py").is_file():
      parts.insert(0, package_root.name)
      package_root = package_root.parent
    sys.path.insert(0, str(package_root))
    if len(parts) > 1:
      imported = import_module(".".join(parts))
      if Path(imported.__file__).resolve() != path:
        raise ValueError(f"Module name resolves to a different file: {imported.__file__}")
    else:
      module_name = f"_profile_target_{path.stem}_{hash(path):x}"
      imported = sys.modules.get(module_name)
      if imported is None:
        file_spec = spec_from_file_location(module_name, path)
        imported = module_from_spec(file_spec)
        sys.modules[module_name] = imported
        file_spec.loader.exec_module(imported)
  else:
    imported = import_module(module)
  function = getattr(imported, name)
  if not callable(function):
    raise ValueError(f"{spec} is not callable")
  return function


def parse_shape(spec):
  try:
    shape = tuple(int(dimension) for dimension in spec.split(","))
  except ValueError as error:
    raise argparse.ArgumentTypeError("Expected comma-separated dimensions") from error
  if any(dimension < 0 for dimension in shape):
    raise argparse.ArgumentTypeError("Dimensions must be nonnegative")
  return shape


def main():
  parser = argparse.ArgumentParser(description="Capture a JAX trace for TensorBoard.")
  parser.add_argument(
    "--target", default="grouped_query_attention",
    help="Built-in target name, path.py:function, or module:function.",
  )
  parser.add_argument(
    "--inputs",
    help="Input factory as path.py:function or module:function. Receives batch sharding "
    "or None and returns (positional_args, keyword_args).",
  )
  parser.add_argument(
    "--shape", type=parse_shape, action="append", metavar="DIM,DIM,...",
    help="Generate a random float32 positional argument. Repeat in argument order. "
    "With --num-gpus, give global shapes and axis 0 is sharded.",
  )
  parser.add_argument("--log-dir", type=Path, default=Path("artifacts/profiles"))
  parser.add_argument("--warmup", type=int, default=3, help="Untraced calls before capture.")
  parser.add_argument("--steps", type=int, default=10, help="Calls to capture.")
  parser.add_argument("--num-gpus", type=int, help="GPUs to use, with two examples per GPU.")
  args = parser.parse_args()
  if args.warmup < 0 or args.steps < 1:
    parser.error("--warmup must be nonnegative and --steps must be positive")
  if args.inputs is not None and args.shape is not None:
    parser.error("Use either --inputs or --shape")

  try:
    if args.target in WORKLOADS:
      function, input_factory = WORKLOADS[args.target]
    else:
      function = import_callable(args.target)
      input_factory = next(
        (factory for target, factory in WORKLOADS.values() if target is function), None
      )
    if args.inputs is not None:
      input_factory = import_callable(args.inputs)
  except (ImportError, AttributeError, ValueError) as error:
    parser.error(str(error))

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

  if args.shape is not None:
    if sharding is not None and any(shape[0] % sharding.mesh.size for shape in args.shape):
      parser.error("Each --shape axis 0 must be divisible by --num-gpus")
    keys = jax.random.split(jax.random.key(0), len(args.shape))
    arrays = tuple(
      jax.random.normal(key, shape, dtype=jnp.float32)
      for key, shape in zip(keys, args.shape, strict=True)
    )
    positional, keyword = jax.device_put(arrays, sharding), {}
  elif input_factory is not None:
    positional, keyword = input_factory(sharding)
  else:
    positional, keyword = (), {}
  target = jax.jit(function)
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
