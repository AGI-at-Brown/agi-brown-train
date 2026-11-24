"""Train Qwen Flash Next on fixed-length, line-delimited text."""

import argparse
from pathlib import Path

import jax
import numpy as np
from flax.serialization import to_bytes
from jax.sharding import Mesh

from clubfm.training.trainer import load_trainconfig, train_loop


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=Path("configs/cluster/qwen_flash_next.toml"))
  parser.add_argument("--dataset-path", type=Path)
  parser.add_argument("--output", type=Path, required=True, help="Final training-state checkpoint")
  parser.add_argument("--steps", type=int)
  parser.add_argument("--tiny", action="store_true")
  parser.add_argument("--num-devices", type=int, default=1)
  args = parser.parse_args()
  if args.steps is not None and args.steps <= 0:
    parser.error("--steps must be positive")

  devices = jax.local_devices()
  if not 1 <= args.num_devices <= len(devices):
    parser.error(f"--num-devices must be between 1 and {len(devices)}")
  mesh = None
  if args.num_devices > 1:
    mesh = Mesh(np.array(devices[:args.num_devices]), ("data",))

  cfg = load_trainconfig(args.config, tiny=args.tiny)
  dataset_path = args.dataset_path or cfg.dataset_path
  if not dataset_path or not Path(dataset_path).is_file():
    parser.error("provide an existing dataset file with --dataset-path or training.dataset_path")
  cfg = cfg.replace(dataset_path=str(dataset_path))
  if args.steps is not None:
    cfg = cfg.replace(num_train_steps=args.steps)

  cfg = cfg.replace(batch_size=cfg.batch_size * args.num_devices)
  state, _ = train_loop(cfg, mesh=mesh)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("xb") as checkpoint:
    checkpoint.write(to_bytes(state))


if __name__ == "__main__":
  main()
