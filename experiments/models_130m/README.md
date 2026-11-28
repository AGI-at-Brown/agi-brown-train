# 130M models

Two ~130M Qwen Flash Next models (D=768, 8 layers, 4 experts top-2; ~126M parameters, ~38M of
them in the tied embedding), trained with the same shape, steps and batch:

| Model | Training text | Result |
|---|---|---|
| FineWeb-Edu | 2.28B tokens of FineWeb-Edu `sample/10BT` | val loss 3.114; 0.977 bpb on FineWeb-Edu, 1.223 on WikiText-103; 0-shot average 40.2 |
| Vintage | pre-1931 text: PG-19 and Shakespeare | val loss ~3.3 on PG-19 |

Both load with [qwen_flash_next_130m.toml](qwen_flash_next_130m.toml). The weights are shared
outside git as exported bf16 `.msgpack` files; full training checkpoints load the same way.
Run every command from the repository root.

```bash
# Interactive autocomplete: load once, then type prompts; each gets N more tokens
uv run python -m experiments.models_130m.autocomplete \
  experiments/models_130m/qwen_flash_next_130m.toml fineweb-edu-130m.msgpack --tokens 50

# Sample text from one prompt
uv run python -m experiments.models_130m.generate \
  experiments/models_130m/qwen_flash_next_130m.toml fineweb-edu-130m.msgpack "The water cycle is"

# Bits per byte on a Parquet file with a text column (add --hf MODEL_ID to compare baselines)
uv run python -m experiments.models_130m.bpb --out bpb.jsonl --dataset NAME=validation.parquet \
  --clubfm web130 experiments/models_130m/qwen_flash_next_130m.toml fineweb-edu-130m.msgpack

# Zero-shot benchmarks with lm-evaluation-harness (use a GPU for the full task set)
uv sync --extra eval
uv run python -m experiments.models_130m.harness --out web130.json \
  --clubfm experiments/models_130m/qwen_flash_next_130m.toml fineweb-edu-130m.msgpack
```

| File | What it does |
|---|---|
| `scoring.py` | Loads the config (model and tokenizer sections only) and weights; scores token sequences |
| `generate.py` | Samples text; `Generator.stream` yields one token at a time |
| `autocomplete.py` | Interactive prompt loop on top of `generate.py` |
| `bpb.py` | Bits per byte against Hugging Face baselines |
| `harness.py` | lm-evaluation-harness connector (log-likelihood tasks) |
| `test_models_130m.py` | Tests on a tiny random model; `uv run pytest` runs them |
