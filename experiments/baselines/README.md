# Transformer addition baseline

Adapted from the later complete implementation in `transformer.ipynb`.
Function names, Flax parameter dataclasses, einsum expressions, two-space indentation,
and the handwritten Adam update are preserved.

From the repository root, after `uv sync --extra dev`:

```bash
# Small CPU smoke run (also exercises grouped-query attention).
uv run python -m experiments.baselines.transformer_addition --tiny --steps 3

# Notebook model dimensions; defaults to 100 updates rather than the long sweep.
uv run python -m experiments.baselines.transformer_addition --steps 100

# Read an existing file instead of generating data.
uv run python -m experiments.baselines.transformer_addition --dataset-path path/to/data.txt
```

Generated data goes under ignored `artifacts/addition/`. Examples look like
`123+456=0579`; the model predicts each next character over all 11 input positions.
Generation is seeded and overwrites only its generated dataset file. A supplied
file is read unchanged. The model initialization and batch sampler retain seed 0;
`--seed` controls generated examples only.

## Code map

- `src/clubfm/models/transformer.py`: model configuration, parameter initialization, and forward assembly.
- `src/clubfm/layers/attention.py`: attention parameters, learned projections, and RoPE integration.
- `src/clubfm/ops/attention.py`: reference causal grouped-query attention on projected queries, keys, and values.
- `src/clubfm/modules/rotary.py`: explicit RoPE rotation matrices.
- `src/clubfm/modules/normalization.py`: RMSNorm.
- `src/clubfm/modules/mlp.py`: MLP parameters and GELU feed-forward computation.
- `src/clubfm/modules/embedding.py`: embedding lookup and tied output projection.
- `src/clubfm/modules/loss.py`: mean next-token cross-entropy.
- `src/clubfm/training/optim/adam.py`: Adam state and updates.
- `src/clubfm/training/trainer.py`: training config/state, encoding, sampling, loss integration, and training loop.
- `src/clubfm/data/sources/addition.py`: synthetic dataset generation.
- `experiments/baselines/transformer_addition.py`: run choices and CLI.

Integration changes: independent random keys per parameter initialization, seeded
streaming data generation, dataset length checks, and a JIT-compiled training step.
The early draft and long scaling sweep were not included. Original normal initialization,
one-hot embedding lookup, explicit RoPE matrices, residual/norm placement, and fixed
normalization scale are retained; this is an educational reference, not an optimized recipe.

The sharding configuration fields are retained but not applied. Checkpointing,
held-out evaluation, and distributed training are not implemented. Training loss
alone does not establish addition accuracy or generalization.

## Reusing components

Import from the defining module, for example:

```python
from clubfm.layers.attention import AttnParams, attn_block
from clubfm.modules.normalization import rmsnorm
from clubfm.ops.attention import grouped_query_attention
```

The model composes layers and modules; layers compose modules and operations.
Components do not import the full model. The existing `Config`, `init_params`, and
`forward` entry points remain in `clubfm.models.transformer`. Attention and MLP
parameter dataclasses now live with their corresponding components.
