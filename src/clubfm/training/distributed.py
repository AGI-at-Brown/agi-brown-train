"""
GPU parallelism on any number of nodes and GPUs. 

be sure to set CUDA_VISIBLE_DEVICES before calling initialize_gpu().
"""
from typing import NamedTuple

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


class ShardingRules(NamedTuple):
  batch: str | None = None
  sequence: str | None = None
  d_model: str | None = None
  d_ff: str | None = None
  vocab: str | None = None


DATA_PARALLEL = ShardingRules(batch="data")
FSDP = ShardingRules(batch="x", d_model="x")
MODEL_PARALLEL = ShardingRules(d_ff="x")


def logical_to_physical(logical: P, rules: ShardingRules) -> P:
  """Map logical dimensions to mesh axes."""
  return P(*(getattr(rules, axis) if axis is not None else None for axis in logical))


def logical_to_sharding(logical: P, mesh: Mesh, rules: ShardingRules) -> NamedSharding:
  return NamedSharding(mesh, logical_to_physical(logical, rules))


def initialize_gpu(num_processes=1, process_id=0, coordinator_address=None, local_device_ids=None):
  if num_processes < 1 or not 0 <= process_id < num_processes:
    raise ValueError("Expected a positive process count and rank within that count")
  if num_processes > 1:
    if coordinator_address is None:
      raise ValueError("Multiple processes require coordinator_address='host:port'")
    jax.distributed.initialize(
      coordinator_address=coordinator_address,
      num_processes=num_processes,
      process_id=process_id,
      local_device_ids=local_device_ids,
    )
  elif local_device_ids is not None:
    jax.config.update("jax_cuda_visible_devices", ",".join(map(str, local_device_ids)))
  return create_gpu_mesh()

# use all available GPUs on one mesh
def create_gpu_mesh():
  devices = jax.devices("gpu")
  counts = [sum(d.process_index == rank for d in devices) for rank in range(jax.process_count())]
  if len(set(counts)) != 1:
    raise ValueError("Use equal GPU counts per process, or launch one process per GPU")
  return Mesh(np.array(devices), ("data",))


def shard_batch(local_batch, mesh):
  sharding = logical_to_sharding(P("batch"), mesh, DATA_PARALLEL)
  return jax.tree.map(
    lambda x: jax.make_array_from_process_local_data(sharding, np.asarray(x)),
    local_batch,
  )


def replicate_state(state, mesh):
  sharding = NamedSharding(mesh, P())
  return jax.tree.map(
    lambda x: jax.make_array_from_process_local_data(sharding, np.asarray(x)),
    state,
  )
