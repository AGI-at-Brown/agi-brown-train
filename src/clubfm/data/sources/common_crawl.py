"""Stream Common Crawl WET text and pack it into next-token batches."""

import gzip
from collections.abc import Iterable, Iterator
from typing import BinaryIO
from urllib.request import urlopen

import numpy as np
from tokenizers import Tokenizer

BASE_URL = "https://data.commoncrawl.org/"
DEFAULT_CRAWL = "CC-MAIN-2026-47"
DEFAULT_MAX_BYTES = 40_000_000_000


def _wet_payloads(stream: BinaryIO) -> Iterator[bytes]:
  for line in stream:
    if not line.strip():
      continue
    if not line.startswith(b"WARC/"):
      raise ValueError("Expected a WARC record")
    headers = {}
    for line in stream:
      if not line.strip():
        break
      name, _, value = line.partition(b":")
      headers[name.lower()] = value.strip()
    length = int(headers[b"content-length"])
    if length < 0:
      raise ValueError("Negative WARC content length")
    payload = stream.read(length)
    if len(payload) != length:
      raise EOFError("Truncated WARC payload")
    if headers.get(b"warc-type") == b"conversion" and payload.strip():
      yield payload


def stream_documents(
  crawl: str = DEFAULT_CRAWL, max_bytes: int = DEFAULT_MAX_BYTES,
) -> Iterator[str]:
  """Yield WET text up to the uncompressed payload byte budget."""
  if max_bytes <= 0:
    raise ValueError("Common Crawl byte budget must be positive")
  manifest = f"{BASE_URL}crawl-data/{crawl}/wet.paths.gz"
  remaining = max_bytes
  with urlopen(manifest, timeout=60) as response, gzip.GzipFile(fileobj=response) as paths:
    for path in paths:
      url = BASE_URL + path.decode("ascii").strip()
      with urlopen(url, timeout=60) as response, gzip.GzipFile(fileobj=response) as wet:
        for payload in _wet_payloads(wet):
          chunk = payload[:remaining]
          remaining -= len(chunk)
          yield chunk.decode("utf-8", errors="replace")
          if remaining == 0:
            return
  raise EOFError(f"{crawl} ended with {remaining:,} text bytes left in the budget")


def _document_groups(documents: Iterable[str]) -> Iterator[list[str]]:
  group = []
  characters = 0
  for document in documents:
    group.append(document)
    characters += len(document)
    if len(group) >= 64 or characters >= 262_144:
      yield group
      group = []
      characters = 0
  if group:
    yield group


def pack_batches(
  documents: Iterable[str], tokenizer: Tokenizer, *, batch_size: int,
  sequence_length: int, eos_token_id: int,
) -> Iterator[dict[str, np.ndarray]]:
  """Pack documents into batches with a one-token overlap between sequences."""
  batch = np.empty((batch_size, sequence_length + 1), dtype=np.int32)
  row = column = 0
  for group in _document_groups(documents):
    for encoding in tokenizer.encode_batch(group, add_special_tokens=False):
      ids = encoding.ids
      ids.append(eos_token_id)
      start = 0
      while start < len(ids):
        count = min(len(ids) - start, batch.shape[1] - column)
        batch[row, column:column + count] = ids[start:start + count]
        column += count
        start += count
        if column < batch.shape[1]:
          continue
        last_token = batch[row, -1]
        row += 1
        if row == batch_size:
          yield {"observed_ids": batch[:, :-1], "target_ids": batch[:, 1:]}
          batch = np.empty_like(batch)
          row = 0
        batch[row, 0] = last_token
        column = 1
