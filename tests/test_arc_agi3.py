from types import SimpleNamespace

import pytest

pytest.importorskip("arc_agi")
from arcengine import GameAction, GameState

from benchmarks.arc_agi3.random_policy import make_policy
from benchmarks.arc_agi3.run import run_episode


class Environment:
  def __init__(self, states):
    self.states = iter(states)
    self.observation_space = self.observe(next(self.states))
    self.calls = []

  def observe(self, state):
    return SimpleNamespace(
      state=state,
      levels_completed=int(state == GameState.WIN),
      frame=[[[0] * 64 for _ in range(64)]],
    )

  @property
  def action_space(self):
    return [GameAction.ACTION1]

  def step(self, action, data):
    self.calls.append((action, data))
    return self.observe(next(self.states))


def test_win_and_budget():
  env = Environment([GameState.NOT_FINISHED, GameState.WIN])
  result = run_episode(env, make_policy(), 10)
  assert result["won"] and result["actions"] == 1
  assert not result["budget_exhausted"]

  env = Environment([GameState.NOT_FINISHED] * 3)
  result = run_episode(env, make_policy(), 2)
  assert result["actions"] == 2 and result["budget_exhausted"]


def test_game_over_stops_and_invalid_action_fails():
  env = Environment([GameState.GAME_OVER])
  assert run_episode(env, make_policy(), 10)["actions"] == 0
  env = Environment([GameState.NOT_FINISHED])
  with pytest.raises(ValueError, match="unavailable"):
    run_episode(env, lambda obs, actions: (GameAction.ACTION2, {}), 10)
  assert not env.calls


def test_pointer_policy_uses_frame_bounds():
  obs = SimpleNamespace(frame=[[[0] * 64 for _ in range(64)]])
  action, data = make_policy()(obs, [GameAction.ACTION6])
  assert action == GameAction.ACTION6
  assert 0 <= data["x"] < 64 and 0 <= data["y"] < 64
