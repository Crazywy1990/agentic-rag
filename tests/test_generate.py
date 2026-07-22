from unittest.mock import MagicMock, patch

import httpx
import pytest
from conftest import make_chunk

from rag.generate.answer import answer_question, build_context
from rag.generate.gateway_client import GatewayClient
from rag.store.base import SearchResult

# ---------------------------------------------------------------- fakes


class FakeEmbedder:
    """Minimal — answer_question only calls embed() on the question."""

    dimension = 8
    model_id = "fake"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dimension for _ in texts]


class FakeStore:
    """Returns preloaded results; records the query vector it was given."""

    def __init__(self, results):
        self._results = results
        self.last_query = None

    def search(self, query_vector, k: int = 5):
        self.last_query = query_vector
        return self._results[:k]

    def add(self, chunks, vectors) -> None:  # Protocol completeness
        raise NotImplementedError

    def count(self) -> int:
        return len(self._results)


class FakeGateway:
    """Records the prompt it was given; returns a canned answer."""

    def __init__(self, reply: str = "The answer is X [0000.00001:0:0]."):
        self.reply = reply
        self.last_system: str | None = None
        self.last_user: str | None = None

    def complete(self, system: str, user: str, model: str = "fake-model") -> str:
        self.last_system = system
        self.last_user = user
        return self.reply


def make_results(*texts: str) -> list[SearchResult]:
    return [
        SearchResult(chunk=make_chunk(t, chunk_index=i), score=0.9 - i * 0.1)
        for i, t in enumerate(texts)
    ]


# ---------------------------------------------------------------- answer_question


def test_retrieved_chunks_enter_the_prompt() -> None:
    results = make_results("alpha finding", "beta finding")
    gateway = FakeGateway()

    answer_question("q?", FakeEmbedder(), FakeStore(results), gateway)

    assert "alpha finding" in gateway.last_user
    assert "beta finding" in gateway.last_user
    assert "0000.00001:0:0" in gateway.last_user  # ids visible for citation
    assert "q?" in gateway.last_user


def test_answer_carries_sources_with_scores() -> None:
    results = make_results("alpha", "beta")

    ans = answer_question("q?", FakeEmbedder(), FakeStore(results), FakeGateway())

    assert ans.question == "q?"
    assert [s.chunk.chunk_id for s in ans.sources] == [r.chunk.chunk_id for r in results]
    assert ans.sources[0].score == pytest.approx(0.9)


def test_k_limits_retrieval() -> None:
    results = make_results("a", "b", "c", "d", "e")

    ans = answer_question("q?", FakeEmbedder(), FakeStore(results), FakeGateway(), k=2)

    assert len(ans.sources) == 2


def test_build_context_wraps_chunks_with_ids() -> None:
    ctx = build_context(make_results("hello world"))
    assert '<chunk id="0000.00001:0:0">' in ctx
    assert "hello world" in ctx
    assert "</chunk>" in ctx


# ---------------------------------------------------------------- GatewayClient


def _mock_response(content: str = "canned", status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"x-cache": "miss"}
    resp.json.return_value = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def test_gateway_client_request_shape() -> None:
    client = GatewayClient(base_url="https://gw.example", api_key="k123")

    with patch("rag.generate.gateway_client.httpx.post", return_value=_mock_response()) as post:
        out = client.complete(system="sys prompt", user="user prompt")

    assert out == "canned"
    assert post.call_args[0][0] == "https://gw.example/v1/chat/completions"
    kwargs = post.call_args[1]
    body = kwargs["json"]
    assert body["model"] == "claude-haiku-4-5"
    assert body["temperature"] == 0.0
    assert body["messages"][0] == {"role": "system", "content": "sys prompt"}
    assert body["messages"][1] == {"role": "user", "content": "user prompt"}
    assert kwargs["headers"]["Authorization"] == "Bearer k123"


def test_gateway_client_raises_on_error_status() -> None:
    client = GatewayClient(base_url="https://gw.example", api_key="k123")

    with (
        patch(
            "rag.generate.gateway_client.httpx.post",
            return_value=_mock_response(status=502),
        ),
        pytest.raises(httpx.HTTPStatusError),
    ):
        client.complete(system="s", user="u")


def test_gateway_client_omits_auth_header_when_no_key() -> None:
    client = GatewayClient(base_url="https://gw.example", api_key="")

    with patch("rag.generate.gateway_client.httpx.post", return_value=_mock_response()) as post:
        client.complete(system="s", user="u")

    assert "Authorization" not in post.call_args[1]["headers"]
