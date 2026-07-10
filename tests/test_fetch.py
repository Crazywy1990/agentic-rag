# tests/test_fetch.py
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.ingest.fetch import fetch_html


def test_fetch_html_skips_existing_file(tmp_path: Path) -> None:
    """Idempotency: if the file is already on disk, no network call at all."""
    existing = tmp_path / "2401.00001.html"
    existing.write_text("<html>cached</html>")

    with patch("rag.ingest.fetch.httpx.get") as mock_get:
        result = fetch_html("2401.00001", tmp_path)

    assert result == existing
    mock_get.assert_not_called()  # the actual guarantee we care about


def test_fetch_html_returns_none_when_html_unavailable(tmp_path: Path) -> None:
    """PDF-only papers: non-200 → skip, and nothing written to disk."""
    mock_resp = MagicMock(status_code=404, headers={"content-type": "text/plain"})

    with (
        patch("rag.ingest.fetch.httpx.get", return_value=mock_resp),
        patch("rag.ingest.fetch.time.sleep"),
    ):  # don't actually wait 3s in tests
        result = fetch_html("1234.99999", tmp_path)

    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_build_corpus_writes_manifest(tmp_path: Path) -> None:
    fake_entry = {
        "arxiv_id": "2401.00001",
        "title": "T",
        "authors": ["A"],
        "published": "2024-01-01",
        "abstract": "abs",
        "categories": ["cs.CL"],
    }
    mock_resp = MagicMock(
        status_code=200, headers={"content-type": "text/html"}, text="<html></html>"
    )

    with (
        patch("rag.ingest.fetch.search_arxiv", return_value=[fake_entry]),
        patch("rag.ingest.fetch.httpx.get", return_value=mock_resp),
        patch("rag.ingest.fetch.time.sleep"),
    ):
        from rag.ingest.fetch import build_corpus

        manifest = build_corpus("q", tmp_path / "raw", target_count=1)

    assert len(manifest) == 1
    assert manifest[0]["html_path"].endswith("2401.00001.html")

    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk == manifest




@pytest.mark.network
def test_search_and_fetch_real(tmp_path: Path) -> None:
    from rag.ingest.fetch import search_arxiv
    results = search_arxiv("cat:cs.CL", max_results=2)
    assert len(results) > 0 and "arxiv_id" in results[0]

    html = fetch_html("2005.11401", tmp_path)
    assert html is not None