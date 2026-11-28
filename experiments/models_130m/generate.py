"""Sample text from a trained clubfm model.

There is no KV cache: every new token reruns the model over the whole context, right-padded to
a multiple of --length-bucket so JAX compiles only a few shapes. Padding cannot change the
next-token logits because every layer is causal. Once the context fills the training length,
the oldest tokens are dropped. That is slow but fine for trying out a ~100M model, even on CPU.

Examples:
  python -m experiments.models_130m.generate experiments/models_130m/qwen_flash_next_130m.toml \
    fineweb-edu-130m.msgpack "The water cycle is"
  python -m experiments.models_130m.generate CONFIG WEIGHTS "It was a dark and" --temperature 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from clubfm.training.trainer import _model_module
from experiments.models_130m.scoring import load_model_config, load_params


class Generator:
  def __init__(self, weight_cfg, params, *, eos_token_id=None, length_bucket=128):
    self.weight_cfg = weight_cfg
    self.params = params
    self.eos_token_id = weight_cfg.eos_token_id if eos_token_id is None else eos_token_id
    self.max_length = weight_cfg.T
    self.length_bucket = length_bucket
    forward = _model_module(weight_cfg).forward

    @jax.jit
    def _next_logits(params, tokens, position):
      logits = forward(params, tokens, weight_cfg)
      return logits[0, position].astype(jnp.float32)

    self._next_logits = _next_logits

  def next_logits(self, context):
    """Logits for the token after `context` (its last `max_length` tokens)."""
    context = list(context)[-self.max_length :]
    width = min(-(-len(context) // self.length_bucket) * self.length_bucket, self.max_length)
    tokens = np.full((1, width), self.eos_token_id, dtype=np.int32)
    tokens[0, : len(context)] = context
    return np.asarray(self._next_logits(self.params, tokens, len(context) - 1))

  def generate(self, prompt_tokens, *, max_new_tokens=100, temperature=0.8, top_k=50, seed=0,
               stop_at_eos=True):
    """Return the new tokens. temperature 0 picks the most likely token every time."""
    return list(self.stream(prompt_tokens, max_new_tokens=max_new_tokens,
                            temperature=temperature, top_k=top_k, seed=seed,
                            stop_at_eos=stop_at_eos))

  def stream(self, prompt_tokens, *, max_new_tokens=100, temperature=0.8, top_k=50, seed=0,
             stop_at_eos=True):
    """Yield new tokens one at a time, as `generate` would return them."""
    rng = np.random.default_rng(seed)
    # An empty prompt starts like a new training document: right after an EOS.
    context = list(prompt_tokens) or [self.eos_token_id]
    for _ in range(max_new_tokens):
      logits = self.next_logits(context).astype(np.float64)
      if temperature <= 0:
        token = int(np.argmax(logits))
      else:
        logits = logits / temperature
        if top_k and top_k < logits.size:
          cutoff = np.partition(logits, -top_k)[-top_k]
          logits = np.where(logits >= cutoff, logits, -np.inf)
        probs = np.exp(logits - logits.max())
        token = int(rng.choice(logits.size, p=probs / probs.sum()))
      if stop_at_eos and token == self.eos_token_id:
        return
      context.append(token)
      yield token


def main(argv=None):
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawTextHelpFormatter
  )
  parser.add_argument("config", type=Path, help="Experiment TOML the model was trained with")
  parser.add_argument("weights", type=Path, help="Exported weights or a full checkpoint")
  parser.add_argument("prompt", nargs="?", default="", help="Text to continue")
  parser.add_argument("--max-new-tokens", type=int, default=100)
  parser.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
  parser.add_argument("--top-k", type=int, default=50, help="0 = no limit")
  parser.add_argument("--samples", type=int, default=1)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args(argv)

  spec = load_model_config(args.config)
  if spec.tokenizer is None:
    parser.error("the config has no [data].tokenizer, so text cannot be encoded")
  generator = Generator(spec.weight_cfg, load_params(spec.weight_cfg, args.weights))
  prompt_tokens = spec.tokenizer.encode(args.prompt, add_special_tokens=False).ids
  for sample in range(args.samples):
    tokens = generator.generate(
      prompt_tokens, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
      top_k=args.top_k, seed=args.seed + sample,
    )
    if args.samples > 1:
      print(f"--- sample {sample + 1}")
    print(args.prompt + spec.tokenizer.decode(tokens), flush=True)


if __name__ == "__main__":
  main()
