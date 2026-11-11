
"""Streaming reader for Common Crawl WET archives."""

import gzip
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO
from urllib.request import Request, urlopen

from warcio.archiveiterator import ArchiveIterator

BASE_URL = "https://data.commoncrawl.org"
USER_AGENT = "clubfm-data-common-crawl/1.0"

# get WET archive paths
def iter_wet_paths(crawl_id: str, timeout: int = 30) -> Iterator[str]:
    if not crawl_id:
        raise ValueError("crawl_id cannot be empty")

    url = f"{BASE_URL}/crawl-data/{crawl_id}/wet.paths.gz"
    request = Request(url, headers={"User-Agent": USER_AGENT})

    with urlopen(request, timeout=timeout) as response:
        with gzip.GzipFile(fileobj=response) as paths_file:
            for line in paths_file:
                path = line.decode("utf-8").strip()
                if path:
                    yield path

# supports both local WET files and remote WET URLs
def _open_wet_file(wet_path: str | Path, timeout: int):
    path = str(wet_path)

    if path.startswith(("http://", "https://")):
        url = path
    elif path.startswith("crawl-data/"):
        url = f"{BASE_URL}/{path}"
    else:
        return Path(path).open("rb")

    request = Request(url, headers={"User-Agent": USER_AGENT})
    return urlopen(request, timeout=timeout)

# parses and extracts text from open WET stream
def _parse_wet_stream(stream: BinaryIO) -> Iterator[dict[str, str]]:
    for record in ArchiveIterator(stream):
        if record.rec_type != "conversion":
            continue

        target_url = record.rec_headers.get_header("WARC-Target-URI")
        if not target_url:
            continue

        text = record.content_stream().read().decode("utf-8", errors="replace")

        if text.strip():
            yield {"url": target_url, "text": text}

# text records from WET archives
def iter_wet_records(wet_path: str | Path, 
                     timeout: int = 60) -> Iterator[dict[str, str]]:
    with _open_wet_file(wet_path, timeout) as stream:
        yield from _parse_wet_stream(stream)

# stream documents
def iter_documents(crawl_id: str, max_files: int | None = None, 
                   max_documents: int | None = None) -> Iterator[dict[str, str]]:
    if max_files is not None and max_files < 1:
        raise ValueError("max_files must be at least 1")

    if max_documents is not None and max_documents < 1:
        raise ValueError("max_documents must be at least 1")

    document_count = 0

    for file_count, wet_path in enumerate(iter_wet_paths(crawl_id), start=1):
        if max_files is not None and file_count > max_files:
            break

        for document in iter_wet_records(wet_path):
            yield document
            document_count += 1

            if max_documents is not None and document_count >= max_documents:
                return