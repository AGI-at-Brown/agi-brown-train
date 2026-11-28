"""Bits per byte of clubfm and Hugging Face models on held-out Parquet text.

Bits per UTF-8 byte does not depend on the tokenizer, so models with different vocabularies
(GPT-2, Pythia, SmolLM, ours) are directly comparable. Every model sees the same text, packed
like training (documents joined by EOS, rows of --sequence-length + 1 overlapping by one token),
and every token is scored once, including the final partial row.

`first_rows_loss` is the mean loss per token over the first --matched-rows rows. For models that
use the cosmo2 tokenizer it covers exactly the tokens behind train.log's `validation_loss`
(--validation-batches 16 x global batch 16 = 256 rows), so the two should agree.

Examples:
  python -m experiments.models_130m.bpb --out results.jsonl \
    --dataset fineweb_edu=validation.parquet \
    --clubfm web130 experiments/models_130m/qwen_flash_next_130m.toml fineweb-edu-130m.msgpack
  python -m experiments.models_130m.bpb --out results.jsonl --dataset ... \
    --hf HuggingFaceTB/SmolLM2-135M --hf openai-community/gpt2
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from experiments.models_130m.scoring import pack_rows

COSMO2_TOKENIZER_ID = "HuggingFaceTB/cosmo2-tokenizer"


def read_texts(path: str | Path) -> list[str]:
  """Nonblank documents from the text column, in file order (as the trainer reads them)."""
  from clubfm.data.sources.parquet import read_documents

  return list(read_documents(Path(path).expanduser()))


class ClubfmModel:
  def __init__(self, name, config, weights, *, sequence_length, batch_size):
    from experiments.models_130m.scoring import ClubfmScorer

    self.name = name
    self.scorer = ClubfmScorer(
      config, weights, batch_size=batch_size, max_length=sequence_length
    )
    self.eos_token_id = self.scorer.eos_token_id
    self.cosmo2_tokenizer = True
    self.num_parameters = sum(int(np.size(leaf)) for leaf in _leaves(self.scorer.params))

  def encode(self, texts):
    encodings = self.scorer.tokenizer.encode_batch(texts, add_special_tokens=False)
    return [encoding.ids for encoding in encodings]

  def nll(self, sequences):
    return [float(-logprobs.sum()) for logprobs, _ in self.scorer.score(sequences)]


class HuggingFaceModel:
  def __init__(self, name, *, sequence_length, batch_size, dtype="float32"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    self.torch = torch
    self.name = name
    self.batch_size = batch_size
    self.tokenizer = AutoTokenizer.from_pretrained(name)
    self.model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype))
    self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    self.device = next(self.model.parameters()).device
    self.eos_token_id = self.tokenizer.eos_token_id
    reference = AutoTokenizer.from_pretrained(COSMO2_TOKENIZER_ID)
    self.cosmo2_tokenizer = self.tokenizer.get_vocab() == reference.get_vocab()
    self.num_parameters = sum(p.numel() for p in self.model.parameters())
    del sequence_length  # rows are already at most sequence_length + 1 tokens

  def encode(self, texts):
    return self.tokenizer(texts, add_special_tokens=False)["input_ids"]

  def nll(self, sequences):
    torch = self.torch
    out = []
    with torch.no_grad():
      for start in range(0, len(sequences), self.batch_size):
        chunk = sequences[start : start + self.batch_size]
        width = max(len(s) for s in chunk)
        tokens = torch.full((len(chunk), width), self.eos_token_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width - 1), dtype=torch.bool)
        for row, sequence in enumerate(chunk):
          tokens[row, : len(sequence)] = torch.as_tensor(sequence)
          mask[row, : len(sequence) - 1] = True
        tokens = tokens.to(self.device)
        logits = self.model(tokens[:, :-1]).logits.float()
        losses = torch.nn.functional.cross_entropy(
          logits.transpose(1, 2), tokens[:, 1:], reduction="none"
        )
        out.extend((losses.cpu() * mask).sum(1).double().tolist())
    return out


def _leaves(tree):
  import jax

  return jax.tree.leaves(tree)


def evaluate(model, dataset_name, texts, *, sequence_length, matched_rows):
  start = time.time()
  n_bytes = sum(len(text.encode("utf-8")) for text in texts)
  rows, tail = pack_rows(model.encode(texts), model.eos_token_id, sequence_length)
  sequences = [list(row) for row in rows]
  if len(tail) > 1:
    sequences.append(list(tail))
  row_nll = model.nll(sequences)
  total = sum(row_nll)
  n_targets = sum(len(sequence) - 1 for sequence in sequences)
  record = {
    "model": model.name,
    "dataset": dataset_name,
    "parameters": model.num_parameters,
    "documents": len(texts),
    "bytes": n_bytes,
    "tokens": n_targets,
    "loss_per_token": total / n_targets,
    "bpb": total / (n_bytes * math.log(2)),
    "cosmo2_tokenizer": model.cosmo2_tokenizer,
    "sequence_length": sequence_length,
    "seconds": round(time.time() - start, 1),
  }
  if matched_rows and len(rows) >= matched_rows:
    record["first_rows_loss"] = sum(row_nll[:matched_rows]) / (matched_rows * sequence_length)
  return record


def _dataset(value):
  name, separator, path = value.partition("=")
  if not separator or not name or not path:
    raise argparse.ArgumentTypeError("use NAME=PATH")
  return name, path


def main(argv=None):
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawTextHelpFormatter
  )
  parser.add_argument("--out", type=Path, required=True, help="JSON Lines file to append to")
  parser.add_argument("--dataset", type=_dataset, action="append", required=True,
                      help="NAME=PATH to a Parquet file with a text column (repeatable)")
  parser.add_argument("--clubfm", nargs=3, action="append", default=[],
                      metavar=("NAME", "CONFIG", "WEIGHTS"),
                      help="A trained clubfm model: exported weights or a full checkpoint")
  parser.add_argument("--hf", action="append", default=[], help="Hugging Face model ID")
  parser.add_argument("--sequence-length", type=int, default=1024)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--matched-rows", type=int, default=256)
  args = parser.parse_args(argv)
  if not args.clubfm and not args.hf:
    parser.error("give at least one --clubfm or --hf model")

  datasets = [(name, read_texts(path)) for name, path in args.dataset]
  builders = [
    (lambda spec=spec: ClubfmModel(*spec, sequence_length=args.sequence_length,
                                   batch_size=args.batch_size))
    for spec in args.clubfm
  ] + [
    (lambda name=name: HuggingFaceModel(name, sequence_length=args.sequence_length,
                                        batch_size=args.batch_size))
    for name in args.hf
  ]
  for build in builders:
    model = build()
    for dataset_name, texts in datasets:
      record = evaluate(model, dataset_name, texts, sequence_length=args.sequence_length,
                        matched_rows=args.matched_rows)
      print(json.dumps(record), flush=True)
      with args.out.open("a") as f:
        f.write(json.dumps(record) + "\n")
    del model


if __name__ == "__main__":
  main()
