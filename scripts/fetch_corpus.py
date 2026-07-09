from rag.config import settings
from rag.ingest.fetch import build_corpus

if __name__ == "__main__":
    manifest = build_corpus(
        query=settings.arxiv_query,
        out_dir=settings.data_dir / "raw",
        target_count=settings.corpus_size,
    )
    print(f"Fetched {len(manifest)} papers -> {settings.data_dir / 'manifest.json'}")
