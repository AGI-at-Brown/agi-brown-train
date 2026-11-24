"""Train Qwen Flash Next on fixed-length, line-delimited text."""

import argparse
from contextlib import closing
from itertools import islice
from pathlib import Path
from tempfile import NamedTemporaryFile

import jax
import numpy as np
from flax.serialization import to_bytes
from jax.sharding import Mesh

from clubfm.data.sources.common_crawl import (
  DEFAULT_CRAWL,
  DEFAULT_MAX_BYTES,
  pack_batches,
  stream_documents,
)
from clubfm.training.trainer import load_trainconfig, train_loop


def save_checkpoint(path, state):
  path.parent.mkdir(parents=True, exist_ok=True)
  with NamedTemporaryFile(dir=path.parent, delete=False) as checkpoint:
    temporary = Path(checkpoint.name)
    try:
      checkpoint.write(to_bytes(state))
      checkpoint.close()
      temporary.replace(path)
    finally:
      temporary.unlink(missing_ok=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=Path("configs/cluster/qwen_flash_next.toml"))
  source = parser.add_mutually_exclusive_group()
  source.add_argument("--dataset-path", type=Path)
  source.add_argument("--common-crawl", action="store_true")
  parser.add_argument("--crawl", default=DEFAULT_CRAWL)
  parser.add_argument("--crawl-bytes", type=int, default=DEFAULT_MAX_BYTES)
  parser.add_argument("--output", type=Path, required=True, help="Final training-state checkpoint")
  parser.add_argument("--steps", type=int)
  parser.add_argument("--checkpoint-every", type=int, default=1000)
  parser.add_argument("--tiny", action="store_true")
  parser.add_argument("--num-devices", type=int, default=1)
  args = parser.parse_args()
  if args.steps is not None and args.steps <= 0:
    parser.error("--steps must be positive")

  if args.checkpoint_every <= 0:
    parser.error("--checkpoint-every must be positive")
  if args.output.exists():
    parser.error(f"Checkpoint already exists: {args.output}")

  devices = jax.local_devices()
  if not 1 <= args.num_devices <= len(devices):
    parser.error(f"--num-devices must be between 1 and {len(devices)}")
  mesh = None
  if args.num_devices > 1:
    mesh = Mesh(np.array(devices[:args.num_devices]), ("data",))

  cfg = load_trainconfig(args.config, tiny=args.tiny)
  dataset_path = args.dataset_path or cfg.dataset_path
  if not args.common_crawl and (not dataset_path or not Path(dataset_path).is_file()):
    parser.error("provide --common-crawl or an existing dataset file")
  cfg = cfg.replace(dataset_path=str(dataset_path))
  if args.steps is not None:
    cfg = cfg.replace(num_train_steps=args.steps)

  cfg = cfg.replace(batch_size=cfg.batch_size * args.num_devices)
  latest = args.output.with_suffix(".latest" + args.output.suffix)

  def checkpoint_periodically(step, state):
    if step % args.checkpoint_every == 0:
      save_checkpoint(latest, state)

  if args.common_crawl:
    eos = cfg.weight_cfg.eos_token_id
    if args.crawl_bytes <= 0:
      parser.error("--crawl-bytes must be positive")
    if cfg.tokenizer is None or eos is None or cfg.vocab_size > cfg.weight_cfg.V:
      parser.error("Common Crawl requires a tokenizer, an EOS token, and a matching model V")
    with closing(stream_documents(args.crawl, args.crawl_bytes)) as documents:
      packed = pack_batches(
        documents, cfg.tokenizer, batch_size=cfg.batch_size,
        sequence_length=cfg.weight_cfg.T, eos_token_id=eos,
      )
      batches = packed if args.steps is None else islice(packed, args.steps)
      state, _ = train_loop(
        cfg, mesh=mesh, batches=batches, on_step=checkpoint_periodically, log_every=100,
      )
  else:
    state, _ = train_loop(cfg, mesh=mesh)
  save_checkpoint(args.output, state)


if __name__ == "__main__":
  main()
