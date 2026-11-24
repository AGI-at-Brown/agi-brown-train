import io

import pytest
from warcio.warcwriter import WARCWriter

from clubfm.data.sources import common_crawl
from clubfm.data.sources.common_crawl import iter_wet_records


def write_wet_file(path, records):
    stream = path.open("wb")
    writer = WARCWriter(stream, gzip=False)

    for url, text, record_type in records:
        record = writer.create_warc_record(
            url, record_type,
            payload=io.BytesIO(text.encode("utf-8")),
            warc_headers_dict={"Content-Type": "text/plain; charset=utf-8"})
        writer.write_record(record)

    stream.close()

def test_iter_wet_records_extracts_url_and_text(tmp_path):
    path = tmp_path / "sample.warc.wet.gz"
    write_wet_file(
       path, [("https://example-common-crawl.com", "Testing!", "conversion")])

    documents = list(iter_wet_records(path))
    assert len(documents) == 1
    assert documents[0]["url"] == "https://example-common-crawl.com"
    assert documents[0]["text"] == "Testing!"


def test_iter_wet_records_skips_irrelevant_records(tmp_path):
    path = tmp_path / "sample.warc.wet"
    write_wet_file(
       path,
       [("https://example-common-crawl.com/resource", "Some resource", "resource"),
        ("https://example-common-crawl.com/empty", " ", "conversion"),
        ("https://example-common-crawl.com/article", "Some article", "conversion")])

    documents = list(iter_wet_records(path))
    assert len(documents) == 1
    assert documents[0]["url"] == "https://example-common-crawl.com/article"


def test_iter_documents_respects_max_documents(tmp_path, monkeypatch):
    path = tmp_path / "sample.warc.wet"
    write_wet_file(
      path,
      [("https://example-common-crawl.com/1", "Page 1", "conversion"),
       ("https://example-common-crawl.com/2", "Page 2", "conversion"),
       ("https://example-common-crawl.com/3", "Page 3", "conversion")])

    monkeypatch.setattr(common_crawl, "iter_wet_paths", lambda crawl_id: iter([str(path)]))
    documents = list(common_crawl.iter_documents("CC-MAIN-test", max_documents=2))
    assert len(documents) == 2

def test_iter_documents_respects_max_files(tmp_path, monkeypatch):
    paths = []
    for i in range(2):
        path = tmp_path / f"sample_{i}.warc.wet"
        write_wet_file(path, [(f"https://example-common-crawl.com/{i}", 
                               f"Document {i}", "conversion")])
        paths.append(str(path))

    monkeypatch.setattr(common_crawl, "iter_wet_paths", lambda crawl_id: iter(paths))
    documents = list(common_crawl.iter_documents("CC-MAIN-test", max_files=1))
    assert len(documents) == 1


@pytest.mark.parametrize("kwargs", [{"max_files": 0}, {"max_documents": 0}])
def test_iter_documents_rejects_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        list(common_crawl.iter_documents("CC-MAIN-test", **kwargs))