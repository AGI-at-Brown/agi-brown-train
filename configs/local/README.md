# Local configuration

`transformer_addition.toml` contains the addition baseline's model dimensions,
training settings, vocabulary, and optional `tiny_model` overrides. Batch size is
read from the model's `B` field.

Run from the repository root:

```bash
uv run python -m experiments.baselines.transformer_addition --config configs/local/transformer_addition.toml --tiny --steps 3
```

Copy the file to configure another experiment. `--steps` overrides the file's
`num_train_steps`; `--tiny` merges `tiny_model` over `model`. The runner uses
`--dataset-path` when supplied, otherwise it generates a dataset.
