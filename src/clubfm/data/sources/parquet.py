"""Read documents from a local Parquet shard."""

from collections.abc import Iterator
from pathlib import Path

from pyarrow.parquet import ParquetFile


def read_documents(path: str | Path) -> Iterator[str]:
  """Yield nonblank documents from the text column in file order."""
  with ParquetFile(path) as shard:
    if shard.schema_arrow.names.count("text") != 1:
      raise ValueError(f"{path} must contain exactly one 'text' column")
    for batch in shard.iter_batches(batch_size=1024, columns=["text"]):
      for text in batch.column(0).to_pylist():
        if text is None:
          continue
        if not isinstance(text, str):
          raise ValueError(f"The 'text' column in {path} must contain strings")
        if text.strip():
          yield text
