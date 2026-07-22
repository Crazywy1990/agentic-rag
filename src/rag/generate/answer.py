from pydantic import BaseModel

from rag.embed.base import EmbeddingProvider
from rag.generate.gateway_client import GatewayClient
from rag.store.base import SearchResult, VectorStore

SYSTEM_PROMPT = """You answer questions about ML/AI research papers using \
ONLY the provided context chunks.

Rules:
- Every factual claim must cite its source chunk inline as [chunk_id].
- Use only the provided chunks. If they don't contain the answer, say so
  plainly — do not fill gaps from general knowledge.
- Multiple chunks may be cited for one claim: [id1][id2].
- Quote sparingly; synthesize in your own words."""


class Answer(BaseModel):
    question: str
    answer: str
    sources: list[SearchResult]  # what was retrieved, scores included — provenance travels


def build_context(results: list[SearchResult]) -> str:
    return "\n\n".join(
        f'<chunk id="{r.chunk.chunk_id}">\n{r.chunk.text}\n</chunk>' for r in results
    )


def answer_question(
    question: str,
    embedder: EmbeddingProvider,
    store: VectorStore,
    gateway: GatewayClient,
    k: int = 5,
) -> Answer:
    results = store.search(embedder.embed([question])[0], k=k)
    user_msg = f"Context chunks:\n\n{build_context(results)}\n\nQuestion: {question}"
    text = gateway.complete(system=SYSTEM_PROMPT, user=user_msg)
    return Answer(question=question, answer=text, sources=results)
