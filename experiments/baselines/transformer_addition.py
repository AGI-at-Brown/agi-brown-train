"""Run the notebook transformer on reproducible synthetic addition data."""

import argparse
from pathlib import Path

from clubfm.data.sources.addition import generate_data
from clubfm.training.trainer import load_trainconfig, train_loop


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--config", type=Path, default=Path("configs/local/transformer_addition.toml"),
  )
  parser.add_argument("--steps", type=int)
  parser.add_argument("--examples", type=int, default=1024)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--dataset-path", type=Path)
  parser.add_argument("--tiny", action="store_true", help="Use a small CPU smoke configuration")
  args = parser.parse_args()
  if (args.steps is not None and args.steps <= 0) or args.examples <= 0:
    parser.error("--steps and --examples must be positive")

  train_cfg = load_trainconfig(args.config, tiny=args.tiny)
  if args.steps is not None:
    train_cfg = train_cfg.replace(num_train_steps=args.steps)

  # A supplied dataset is read unchanged. Otherwise regenerate a seeded toy dataset.
  dataset_path = args.dataset_path
  if dataset_path is None:
    dataset_path = Path("artifacts/addition") / f"data_{args.examples}_seed{args.seed}.txt"
    generate_data(args.examples, dataset_path, args.seed)

  train_cfg = train_cfg.replace(dataset_path=str(dataset_path))
  train_loop(train_cfg)


if __name__ == "__main__":
  main()
