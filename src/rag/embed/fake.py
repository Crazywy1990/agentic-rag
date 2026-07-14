import hashlib
import math


class FakeEmbedder:
    """Deterministic, text-sensitive embeddings for hermetic tests. Not semantic."""

    def __init__(self, dimension: int = 8) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return f"fake-{self._dim}d"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            v = [b / 255.0 for b in h[: self._dim]]
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vecs.append([x / norm for x in v])
        return vecs
