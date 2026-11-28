"""Interactive autocomplete: load a trained model once, then continue each prompt you type.

Type a prompt and press Enter; the model's next N tokens are printed as they are generated.
Commands:
  /tokens N        generate N tokens per prompt
  /temperature T   sampling temperature (0 = always the most likely token)
  /quit            exit (so do an empty line, Ctrl-D or Ctrl-C)

Examples:
  python -m experiments.models_130m.autocomplete experiments/models_130m/qwen_flash_next_130m.toml \
    fineweb-edu-130m.msgpack --tokens 50
  python -m experiments.models_130m.autocomplete CONFIG WEIGHTS --tokens 30 --prompt "Once upon a"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.models_130m.generate import Generator
from experiments.models_130m.scoring import load_model_config, load_params


def complete(generator, tokenizer, prompt, *, tokens, temperature, top_k, seed, out=sys.stdout):
  """Print the continuation of `prompt` piece by piece and return it as text."""
  prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False).ids
  new_tokens, shown = [], ""
  stream = generator.stream(prompt_tokens, max_new_tokens=tokens, temperature=temperature,
                            top_k=top_k, seed=seed)
  for token in stream:
    new_tokens.append(token)
    # Decode everything so far: one character can span several byte-level tokens.
    text = tokenizer.decode(new_tokens)
    if not text.endswith("�"):
      out.write(text[len(shown):])
      out.flush()
      shown = text
  text = tokenizer.decode(new_tokens)
  out.write(text[len(shown):] + "\n")
  out.flush()
  return text


def main(argv=None):
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawTextHelpFormatter
  )
  parser.add_argument("config", type=Path, help="Experiment TOML the model was trained with")
  parser.add_argument("weights", type=Path, help="Exported weights or a full checkpoint")
  parser.add_argument("--tokens", "-n", type=int, default=50, help="Tokens to generate")
  parser.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
  parser.add_argument("--top-k", type=int, default=50, help="0 = no limit")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--prompt", help="Complete this one prompt and exit")
  args = parser.parse_args(argv)

  spec = load_model_config(args.config)
  if spec.tokenizer is None:
    parser.error("the config has no [data].tokenizer, so text cannot be encoded")
  print(f"Loading {args.weights} ...", file=sys.stderr, flush=True)
  generator = Generator(spec.weight_cfg, load_params(spec.weight_cfg, args.weights))
  settings = {"tokens": args.tokens, "temperature": args.temperature}

  def run(prompt, seed):
    sys.stdout.write(prompt)
    complete(generator, spec.tokenizer, prompt, tokens=settings["tokens"],
             temperature=settings["temperature"], top_k=args.top_k, seed=seed)

  if args.prompt is not None:
    run(args.prompt, args.seed)
    return

  print("Type a prompt and press Enter (/tokens N, /temperature T, /quit).", file=sys.stderr)
  seed = args.seed
  while True:
    try:
      line = input("> ")
    except (EOFError, KeyboardInterrupt):
      print(file=sys.stderr)
      return
    if not line.strip() or line.strip() == "/quit":
      return
    command, _, value = line.strip().partition(" ")
    if command in ("/tokens", "/temperature"):
      try:
        key = command[1:]
        settings[key] = int(value) if key == "tokens" else float(value)
        print(f"{key} = {settings[key]}", file=sys.stderr)
      except ValueError:
        print(f"usage: {command} NUMBER", file=sys.stderr)
      continue
    try:
      run(line, seed)
    except KeyboardInterrupt:
      print("\n(stopped)", file=sys.stderr)
    seed += 1  # a different sample each time, reproducible from --seed


if __name__ == "__main__":
  main()
