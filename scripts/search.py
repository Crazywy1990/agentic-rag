import sys

from rag.config import settings
from rag.embed.openai_embedder import OpenAIEmbedder
from rag.store.lancedb_store import LanceDBStore


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: uv run python scripts/search.py "your query here"')
        raise SystemExit(1)
    query = sys.argv[1]

    embedder = OpenAIEmbedder()
    store = LanceDBStore(settings.data_dir / "lancedb", dimension=embedder.dimension)

    results = store.search(embedder.embed([query])[0], k=5)

    print(f'query: "{query}"\n')
    for r in results:
        print(f"[{r.score:.4f}] {r.chunk.chunk_id}  ({r.chunk.section_title})")
        print(f"         {r.chunk.paper_title}")
        print(f"         {r.chunk.text[:200].replace(chr(10), ' ')}...\n")


if __name__ == "__main__":
    main()
