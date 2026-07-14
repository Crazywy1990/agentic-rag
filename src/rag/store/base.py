from typing import Protocol

from pydantic import BaseModel

from rag.ingest.models import Chunk


class SearchResult(BaseModel):
    chunk: Chunk
    score: float  # higher = more similar (implementations normalize to this)


class VectorStore(Protocol):
    """Swappable store. Implementations: LanceDB (W1). Documented scale path: pgvector/Qdrant."""

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(self, query_vector: list[float], k: int = 5) -> list[SearchResult]: ...

    def count(self) -> int: ...
