# AGI Brown Train

```bash
uv run python -m experiments.baselines.transformer_addition --tiny --steps 3
```

See [the baseline guide](experiments/baselines/README.md) for usage and limitations.

## Setup

Use Python 3.12 or newer. From this directory:

```bash
uv sync --extra dev
```

The base dependency uses JAX's default installation. Install a compatible accelerator if needed.

To test on ARC AGI 3, add your API key as `ARC_API_KEY` to .env in the root folder.

## Trained models

The 130M FineWeb-Edu and vintage (pre-1931) models, and the scripts to sample from and evaluate
them, are in [experiments/models_130m](experiments/models_130m/README.md).
