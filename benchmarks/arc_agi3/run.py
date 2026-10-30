"""Bounded ARC-AGI-3 episodes with pluggable architecture policies.

To run:
uv run --extra arc python -m benchmarks.arc_agi3.run \
  --games ls20 --mode online

"""


import argparse
import importlib
import json
from pathlib import Path

from arc_agi import Arcade, OperationMode
from arcengine import GameState


def run_episode(env, policy, max_actions):
  if max_actions <= 0:
    raise ValueError("max_actions must be positive")
  obs = env.observation_space
  if obs is None:
    obs = env.reset()
  actions_taken = 0
  while actions_taken < max_actions:
    if obs is None:
      raise RuntimeError("Environment returned no observation")
    if obs.state in (GameState.WIN, GameState.GAME_OVER):
      break
    available = tuple(env.action_space)
    if not available:
      raise RuntimeError("Environment has no available actions")
    action, data = policy(obs, available)
    if action not in available:
      raise ValueError(f"Policy selected unavailable action: {action}")
    action.validate_data(data)
    obs = env.step(action, data=data)
    actions_taken += 1
  if obs is None:
    raise RuntimeError("Environment returned no observation")
  return {
    "state": obs.state.name,
    "won": obs.state == GameState.WIN,
    "levels_completed": obs.levels_completed,
    "actions": actions_taken,
    "budget_exhausted": actions_taken == max_actions
    and obs.state not in (GameState.WIN, GameState.GAME_OVER),
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--games", nargs="+", required=True, help="Prefer full versioned game IDs")
  parser.add_argument("--policy", default="benchmarks.arc_agi3.random_policy:make_policy")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--max-actions", type=int, default=100)
  parser.add_argument("--mode", choices=["offline", "online"], default="offline")
  parser.add_argument("--environments-dir", default="artifacts/arc_agi3/environments")
  parser.add_argument("--output", type=Path, default=Path("artifacts/arc_agi3/results.json"))
  args = parser.parse_args()
  if args.max_actions <= 0:
    parser.error("--max-actions must be positive")
  module, name = args.policy.split(":", 1)
  factory = getattr(importlib.import_module(module), name)
  arcade = Arcade(
    operation_mode=OperationMode.OFFLINE if args.mode == "offline" else OperationMode.ONLINE,
    environments_dir=args.environments_dir,
    recordings_dir="artifacts/arc_agi3/recordings",
  )
  results = []
  scorecard = None
  try:
    for game in args.games:
      policy = factory(seed=args.seed)
      env = arcade.make(game, seed=args.seed, save_recording=True)
      if env is None:
        raise RuntimeError(f"Could not create {game}; check game files or API access")
      result = run_episode(env, policy, args.max_actions)
      results.append({"game": game, **result})
      print(json.dumps(results[-1]))
  finally:
    scorecard = arcade.close_scorecard()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
      json.dumps(
        {
          "policy": args.policy,
          "seed": args.seed,
          "mode": args.mode,
          "max_actions": args.max_actions,
          "results": results,
          "scorecard": scorecard.model_dump(mode="json") if scorecard is not None else None,
        },
        indent=2,
      )
      + "\n"
    )


if __name__ == "__main__":
  main()
