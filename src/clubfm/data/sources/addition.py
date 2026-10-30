"""Synthetic three-digit addition examples from the transformer notebook."""

import random
from pathlib import Path


def generate_data(num_examples, dataset_path, seed=0):
  # A local RNG keeps dataset generation reproducible without changing global state.
  rng = random.Random(seed)
  path = Path(dataset_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w") as f:
    for _ in range(num_examples):
      a = rng.randint(0, 999)
      b = rng.randint(0, 999)
      f.write(f"{a:03d}+{b:03d}={a + b:04d}\n")
