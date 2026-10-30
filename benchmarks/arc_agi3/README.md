# ARC-AGI-3

Minimal interactive evaluation using the [official SDK](https://github.com/arcprize/ARC-AGI).
This is a bounded research runner, not a competition submission harness.

```bash
uv sync --extra dev --extra arc
# Offline: supply already-downloaded official environment files.
uv run --extra arc python -m benchmarks.arc_agi3.run --games ls20 \
  --environments-dir /path/to/environment_files --max-actions 20
# Online: uses ARC_API_KEY, or the SDK's anonymous access when available.
uv run --extra arc python -m benchmarks.arc_agi3.run --games ls20 --mode online
```

Use full versioned game IDs for reproducible comparisons. Offline mode does not
fetch missing games. Recordings and JSON results go under `artifacts/arc_agi3/`.
The default random policy only checks the integration.

## Connect an architecture

Provide an importable module with `make_policy(seed=0)` returning a callable:

```python
def make_policy(seed=0):
  # Load your parameters and initialize model memory/RNG here.
  def policy(obs, available_actions):
    # Encode obs.frame, run your architecture, and decode a legal action.
    # Return an arcengine.GameAction from available_actions and its data dict.
    # Pointer actions require {'x': int, 'y': int}.
    ...

  return policy
```

```bash
uv run --extra arc python -m benchmarks.arc_agi3.run \
  --games ls20 --policy experiments.architecture.my_agent:make_policy
```

Each game gets a fresh policy. The callable can retain interaction history and
receive every observation, including updated levels and available actions.
WIN, GAME_OVER, or the action budget ends an attempt; no automatic retries.
Invalid actions or missing observations fail visibly. Importing your policy may
load a checkpoint; the runner does not choose tokenization or a checkpoint format.
The addition transformer needs a frame encoder/action decoder and appropriate
training before this benchmark can measure useful ability.

Compare architectures with the same games, seeds, action budget, observation
encoding, and adaptation rules. Record checkpoint/code revision separately.
`won`, `levels_completed`, and `actions` are diagnostics, not official ARC scores;
the SDK's final scorecard is saved separately. This runner does not enforce the
competition protocol or measure inference compute. Network/API errors propagate.

Local tests use the real SDK's types with a tiny scripted environment; they never
contact the service or download games:

```bash
uv run --extra dev --extra arc pytest tests/test_arc_agi3.py
```
