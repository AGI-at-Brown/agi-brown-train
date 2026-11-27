"""Stream Common Crawl WET text."""

import gzip
from collections.abc import Iterator
from typing import BinaryIO
from urllib.request import urlopen

from clubfm.data.transforms.packing import pack_batches as pack_batches

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
