"""Run the notebook transformer on reproducible synthetic addition data."""

import argparse
from pathlib import Path

from clubfm.data.sources.addition import generate_data
from clubfm.models.transformer import Config
from clubfm.training.trainer import init_trainconfig_with_steps, train_loop


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--steps", type=int, default=100)
  parser.add_argument("--examples", type=int, default=1024)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--dataset-path", type=Path)
  parser.add_argument("--tiny", action="store_true", help="Use a small CPU smoke configuration")
  args = parser.parse_args()
  if args.steps <= 0 or args.examples <= 0:
    parser.error("--steps and --examples must be positive")

  # A supplied dataset is read unchanged. Otherwise regenerate a seeded toy dataset.
  dataset_path = args.dataset_path
  if dataset_path is None:
    dataset_path = Path("artifacts/addition") / f"data_{args.examples}_seed{args.seed}.txt"
    generate_data(args.examples, dataset_path, args.seed)

  cfg = Config(B=2, L=1, D=16, F=32, N=2, K=1, H=8) if args.tiny else Config()
  train_cfg = init_trainconfig_with_steps(cfg, args.steps).replace(dataset_path=str(dataset_path))
  train_loop(train_cfg)


if __name__ == "__main__":
  main()
