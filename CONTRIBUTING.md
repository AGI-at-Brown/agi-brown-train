# Contributing

Please explain the change, and include validation and reproduction commands.

- Put reusable code in `src/clubfm/` and research recipes in `experiments/`.
- Keep imports free of downloads, training, and provisioning.
- Keep JAX state and random keys explicit; document shape and sharding assumptions.

Before opening a PR, add relevant tests and run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Record findings in `docs/reports/`, including negative results. Include the
hypothesis, baseline, hardware, metrics, and steps to reproduce.
