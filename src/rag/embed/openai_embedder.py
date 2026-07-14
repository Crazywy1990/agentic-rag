from openai import OpenAI

from rag.config import settings

_DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
BATCH_SIZE = 256  # 1414 chunks -> 6 requests


class OpenAIEmbedder:
    def __init__(self, model: str | None = None, client: OpenAI | None = None) -> None:
        self._model = model or settings.embedding_model
        self._client = client or OpenAI(api_key=settings.openai_api_key)

    @property
    def dimension(self) -> int:
        return _DIMENSIONS[self._model]

    @property
    def model_id(self) -> str:
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            # API guarantees index-aligned data, but sort defensively
            out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return out
