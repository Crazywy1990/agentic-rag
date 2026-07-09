# src/rag/ingest/fetch.py
import json
import time
from pathlib import Path

import feedparser
import httpx

ARXIV_API = "https://export.arxiv.org/api/query"
HEADERS = {"User-Agent": "agentic-rag-corpus-builder (github.com/Crazywy1990)"}


def search_arxiv(query: str, max_results: int = 50) -> list[dict]:
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    resp = httpx.get(ARXIV_API, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)
    return [
        {
            "arxiv_id": e.id.split("/abs/")[-1],
            "title": e.title.replace("\n", " ").strip(),
            "authors": [a.name for a in e.authors],
            "published": e.published,
            "abstract": e.summary.replace("\n", " ").strip(),
            "categories": [t.term for t in e.tags],
        }
        for e in feed.entries
    ]


def fetch_html(arxiv_id: str, out_dir: Path) -> Path | None:
    """Download the HTML version; return None if unavailable (PDF-only paper)."""
    out_path = out_dir / f"{arxiv_id.replace('/', '_')}.html"
    if out_path.exists():
        return out_path  # idempotent re-runs
    url = f"https://arxiv.org/html/{arxiv_id}"
    resp = httpx.get(url, headers=HEADERS, timeout=60, follow_redirects=True)
    time.sleep(3)  # arXiv politeness
    if resp.status_code != 200 or "html" not in resp.headers.get("content-type", ""):
        return None
    out_path.write_text(resp.text, encoding="utf-8")
    return out_path


def build_corpus(query: str, out_dir: Path, target_count: int = 50) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for meta in search_arxiv(query, max_results=target_count * 2):  # overfetch; some lack HTML
        if len(manifest) >= target_count:
            break
        path = fetch_html(meta["arxiv_id"], out_dir)
        if path is None:
            continue
        meta["html_path"] = str(path)
        manifest.append(meta)
    (out_dir.parent / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
