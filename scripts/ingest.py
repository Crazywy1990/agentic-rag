import json
import time

from rag.config import settings
from rag.embed.openai_embedder import OpenAIEmbedder
from rag.ingest.models import Chunk
from rag.store.lancedb_store import LanceDBStore


def main() -> None:
    chunks_path = settings.data_dir / "chunks.jsonl"
    chunks = [Chunk(**json.loads(line)) for line in chunks_path.read_text().splitlines()]
    print(f"loaded {len(chunks)} chunks from {chunks_path}")

    embedder = OpenAIEmbedder()
    store = LanceDBStore(settings.data_dir / "lancedb", dimension=embedder.dimension)

    t0 = time.time()
    vectors = embedder.embed([c.text for c in chunks])
    t_embed = time.time() - t0

    store.add(chunks, vectors)

    print(f"embedded with {embedder.model_id} in {t_embed:.1f}s")
    print(f"store count: {store.count()}")


if __name__ == "__main__":
    main()
