from pathlib import Path

from conftest import make_chunk

from rag.embed.fake import FakeEmbedder
from rag.store.lancedb_store import LanceDBStore


def _loaded_store(tmp_path: Path, texts: list[str]) -> tuple[LanceDBStore, FakeEmbedder, list]:
    embedder = FakeEmbedder()
    store = LanceDBStore(tmp_path / "db", dimension=embedder.dimension)
    chunks = [make_chunk(t, chunk_index=i) for i, t in enumerate(texts)]
    store.add(chunks, embedder.embed([c.text for c in chunks]))
    return store, embedder, chunks


def test_add_then_count(tmp_path: Path) -> None:
    store, _, _ = _loaded_store(tmp_path, ["one", "two", "three"])
    assert store.count() == 3


def test_search_returns_k_results_scores_descending(tmp_path: Path) -> None:
    store, embedder, _ = _loaded_store(tmp_path, ["a", "b", "c", "d", "e"])

    results = store.search(embedder.embed(["a"])[0], k=3)

    assert len(results) == 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_exact_match_scores_near_one(tmp_path: Path) -> None:
    """Pins the score contract: cosine similarity, higher-better, identity ≈ 1.0.

    Regression for the S3 L2-vs-cosine discovery: the default metric produced
    negative 'scores'; this fails if the metric ever silently changes."""
    store, embedder, chunks = _loaded_store(tmp_path, ["alpha content", "beta content"])

    results = store.search(embedder.embed([chunks[0].text])[0], k=2)

    assert results[0].chunk.chunk_id == chunks[0].chunk_id
    assert results[0].score > 0.99
    assert results[0].score > results[1].score


def test_chunk_metadata_survives_round_trip(tmp_path: Path) -> None:
    store, embedder, chunks = _loaded_store(tmp_path, ["survives intact"])

    r = store.search(embedder.embed(["survives intact"])[0], k=1)[0]

    assert r.chunk == chunks[0]  # every field: id, titles, text, token_count


def test_add_appends_not_replaces(tmp_path: Path) -> None:
    """Documents the append behavior that motivates the rebuild-always policy."""
    store, embedder, _ = _loaded_store(tmp_path, ["first"])
    more = [make_chunk("second", chunk_index=1)]
    store.add(more, embedder.embed(["second"]))

    assert store.count() == 2
