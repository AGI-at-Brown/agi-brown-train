# Dataset recipes

Bind source versions, preparation steps, and mixture choices using reusable data code.

## Train on a local Parquet shard

From the repository root, pass a TinyStories, Cosmopedia v2, or FineWeb-Edu
training shard directly to the Qwen Flash Next experiment.

```bash
uv sync
uv run --locked python -m experiments.pretraining.qwen_flash_next \
  --dataset-path "$HOME/scratch/datasets/tinystories/train-00000.parquet" \
  --tiny --steps 10 --output "$HOME/scratch/tinystories-smoke.msgpack"
```

The loader reads the `text` column in file order, skips null or blank rows,
and inserts an end-of-document token between documents. Training stops at
the requested step limit or when the shard runs out of complete batches.
Remove `--tiny` to use the default cluster model.
