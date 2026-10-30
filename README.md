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
