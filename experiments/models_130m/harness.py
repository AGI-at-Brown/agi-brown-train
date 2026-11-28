"""Run EleutherAI lm-evaluation-harness tasks on clubfm and Hugging Face models.

`ClubfmLM` connects a trained JAX model to the harness. It implements the two log-likelihood
requests that multiple-choice and perplexity tasks use; text generation is not supported.
Contexts and continuations are tokenized like the harness's own `hf` model, so a clubfm model
and the public baselines see identical inputs.

Examples:
  python -m experiments.models_130m.harness --out web130.json \
    --clubfm experiments/models_130m/qwen_flash_next_130m.toml fineweb-edu-130m.msgpack
  python -m experiments.models_130m.harness --out smollm2.json --hf HuggingFaceTB/SmolLM2-135M
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lm_eval.api.model import LM
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window

from experiments.models_130m.scoring import ClubfmScorer

# Tasks that still separate models below ~1B parameters (the FineWeb "early signal" set),
# all zero-shot. mmlu_continuation is MMLU scored as cloze, not as A/B/C/D letters.
DEFAULT_TASKS = [
  "hellaswag",
  "arc_easy",
  "arc_challenge",
  "piqa",
  "sciq",
  "openbookqa",
  "winogrande",
  "lambada_openai",
  "mmlu_continuation",
]


class ClubfmLM(LM):
  def __init__(self, config, weights, *, batch_size=16, max_length=1024, scorer=None):
    super().__init__()
    self.scorer = scorer or ClubfmScorer(
      config, weights, batch_size=batch_size, max_length=max_length
    )
    self.max_length = self.scorer.max_length
    self.eos_token_id = self.scorer.eos_token_id

  def tok_encode(self, text):
    return self.scorer.tokenizer.encode(text, add_special_tokens=False).ids

  def _encode_pair(self, context, continuation):
    # Same split as the harness's hf model: trailing context spaces move to the continuation.
    spaces = len(context) - len(context.rstrip())
    if spaces:
      continuation = context[-spaces:] + continuation
      context = context[:-spaces]
    whole = self.tok_encode(context + continuation)
    context_tokens = self.tok_encode(context)
    return context_tokens, whole[len(context_tokens) :]

  def _loglikelihood_tokens(self, pairs):
    """Sum of continuation log-probabilities and whether every token was the greedy choice."""
    sequences, lengths = [], []
    for context_tokens, continuation_tokens in pairs:
      if not continuation_tokens:
        raise ValueError("Continuation produced no tokens")
      if len(continuation_tokens) > self.max_length:
        raise ValueError("Continuation is longer than the model's context")
      # Keep the end; the model scores at most max_length targets after one context token.
      sequences.append((list(context_tokens) + list(continuation_tokens))[-(self.max_length + 1) :])
      lengths.append(len(continuation_tokens))
    results = []
    for (logprobs, greedy), length in zip(self.scorer.score(sequences), lengths, strict=True):
      results.append((float(logprobs[-length:].sum()), bool(greedy[-length:].all())))
    return results

  def loglikelihood(self, requests, disable_tqdm=False):
    pairs = []
    for request in requests:
      context, continuation = request.args
      if context == "":
        pairs.append(([self.eos_token_id], self.tok_encode(continuation)))
      else:
        pairs.append(self._encode_pair(context, continuation))
    return self._loglikelihood_tokens(pairs)

  def loglikelihood_rolling(self, requests, disable_tqdm=False):
    results = []
    for request in requests:
      (text,) = request.args
      windows = [
        make_disjoint_window(window)
        for window in get_rolling_token_windows(
          token_list=self.tok_encode(text),
          prefix_token=self.eos_token_id,
          max_seq_len=self.max_length,
          context_len=1,
        )
      ]
      results.append(sum(logprob for logprob, _ in self._loglikelihood_tokens(windows)))
    return results

  def generate_until(self, requests, disable_tqdm=False):
    raise NotImplementedError("ClubfmLM only supports log-likelihood tasks")


def main(argv=None):
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawTextHelpFormatter
  )
  parser.add_argument("--out", type=Path, required=True, help="JSON file for the results")
  model = parser.add_mutually_exclusive_group(required=True)
  model.add_argument("--clubfm", nargs=2, metavar=("CONFIG", "WEIGHTS"))
  model.add_argument("--hf", help="Hugging Face model ID")
  parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
  parser.add_argument("--max-length", type=int, default=1024)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--limit", type=float, help="Examples per task, for quick checks")
  parser.add_argument("--device", default="cuda", help="Device for --hf models")
  args = parser.parse_args(argv)

  from lm_eval import simple_evaluate
  from lm_eval.utils import make_table

  if args.clubfm:
    lm = ClubfmLM(*args.clubfm, batch_size=args.batch_size, max_length=args.max_length)
    name = str(Path(args.clubfm[1]).resolve())
  else:
    lm = "hf"
    name = args.hf
  results = simple_evaluate(
    model=lm,
    model_args=None if args.clubfm else (
      f"pretrained={args.hf},dtype=float32,max_length={args.max_length},device={args.device}"
    ),
    tasks=args.tasks.split(","),
    num_fewshot=0,
    batch_size=args.batch_size,
    limit=args.limit,
    # Bootstrapped error bars (only used for perplexity) fork worker processes that each copy
    # the whole JAX process, which ran a full run out of 48 GB of host memory. Accuracy error
    # bars are computed directly and are unaffected.
    bootstrap_iters=0,
  )
  summary = {
    "model": name,
    "max_length": args.max_length,
    "limit": args.limit,
    "results": results["results"],
    "versions": results.get("versions"),
    "n-samples": results.get("n-samples"),
  }
  # Save before printing, so a console that cannot encode the table cannot lose the results.
  args.out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
  print(make_table(results).encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
  main()
