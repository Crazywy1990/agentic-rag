from unittest.mock import MagicMock

from rag.embed.fake import FakeEmbedder
from rag.embed.openai_embedder import BATCH_SIZE, OpenAIEmbedder


def _fake_openai_client(dim: int = 1536) -> MagicMock:
    """Mock OpenAI client whose embeddings.create echoes index-aligned vectors."""
    client = MagicMock()

    def create(model, input):  # noqa: A002 — matches OpenAI's param name
        data = [MagicMock(index=i, embedding=[float(i)] * dim) for i in range(len(input))]
        resp = MagicMock()
        resp.data = data
        return resp

    client.embeddings.create.side_effect = create
    return client


def test_embed_returns_one_vector_per_input() -> None:
    client = _fake_openai_client()
    embedder = OpenAIEmbedder(model="text-embedding-3-small", client=client)

    vectors = embedder.embed(["a", "b", "c"])

    assert len(vectors) == 3
    assert all(len(v) == embedder.dimension for v in vectors)


def test_embed_batches_large_inputs() -> None:
    client = _fake_openai_client()
    embedder = OpenAIEmbedder(model="text-embedding-3-small", client=client)

    n = BATCH_SIZE * 2 + 5  # forces 3 API calls
    vectors = embedder.embed(["x"] * n)

    assert len(vectors) == n
    assert client.embeddings.create.call_count == 3


def test_embed_preserves_order_within_batch() -> None:
    """API may return out-of-order; embedder sorts by index. Vector[i] must map to input[i]."""
    client = MagicMock()

    def create(model, input):
        # deliberately reverse the returned order to test the sort
        data = [MagicMock(index=i, embedding=[float(i)]) for i in range(len(input))]
        resp = MagicMock()
        resp.data = list(reversed(data))
        return resp

    client.embeddings.create.side_effect = create
    embedder = OpenAIEmbedder(model="text-embedding-3-small", client=client)

    vectors = embedder.embed(["first", "second", "third"])

    assert [v[0] for v in vectors] == [
        0.0,
        1.0,
        2.0,
    ]  # order restored despite reversed API response


def test_dimension_and_model_id() -> None:
    embedder = OpenAIEmbedder(model="text-embedding-3-small", client=MagicMock())
    assert embedder.dimension == 1536
    assert embedder.model_id == "text-embedding-3-small"


def test_fake_embedder_is_deterministic_and_text_sensitive() -> None:
    fake = FakeEmbedder()
    assert fake.embed(["same"]) == fake.embed(["same"])  # deterministic
    assert fake.embed(["a"]) != fake.embed(["b"])  # text-sensitive
    assert len(fake.embed(["x"])[0]) == fake.dimension
