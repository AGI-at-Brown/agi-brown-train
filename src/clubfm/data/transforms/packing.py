"""Pack documents into fixed-length next-token batches."""

from collections.abc import Iterable, Iterator

import numpy as np
from tokenizers import Tokenizer


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
  documents: Iterable[str],
  tokenizer: Tokenizer,
  *,
  batch_size: int,
  sequence_length: int,
  eos_token_id: int,
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
        batch[row, column : column + count] = ids[start : start + count]
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
