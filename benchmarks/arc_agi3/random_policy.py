"""Random integration smoke policy; not an architecture-quality baseline."""

import random


def make_policy(seed=0):
  rng = random.Random(seed)

  def policy(obs, available_actions):
    action = rng.choice(available_actions)
    data = {}
    if action.is_complex():
      height, width = len(obs.frame[-1]), len(obs.frame[-1][0])
      data = {"x": rng.randrange(width), "y": rng.randrange(height)}
    return action, data

  return policy
