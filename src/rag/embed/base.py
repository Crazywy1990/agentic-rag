from typing import Protocol


class EmbeddingProvider(Protocol):
    """Anything that turns texts into vectors. Implementations: OpenAI (prod), Fake (tests)."""

    @property
    def dimension(self) -> int: ...

    @property
    def model_id(self) -> str:
        """Identifies model for provenance — stored alongside vectors."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed. Order-preserving: result[i] corresponds to texts[i]."""
        ...
