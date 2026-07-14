from pathlib import Path

import lancedb

from rag.ingest.models import Chunk
from rag.store.base import SearchResult

TABLE = "chunks"


class LanceDBStore:
    def __init__(self, db_path: Path, dimension: int) -> None:
        self._db = lancedb.connect(str(db_path))
        self._dim = dimension

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        assert len(chunks) == len(vectors)
        rows = [{"vector": v, **c.model_dump()} for c, v in zip(chunks, vectors, strict=True)]
        if TABLE in self._db.table_names():
            self._db.open_table(TABLE).add(rows)
        else:
            self._db.create_table(TABLE, data=rows)

    def search(self, query_vector: list[float], k: int = 5) -> list[SearchResult]:
        hits = self._db.open_table(TABLE).search(query_vector).metric("cosine").limit(k).to_list()
        return [
            SearchResult(
                chunk=Chunk(**{f: h[f] for f in Chunk.model_fields}),
                score=1.0 - h["_distance"],
            )
            for h in hits
        ]

    def count(self) -> int:
        return self._db.open_table(TABLE).count_rows()
