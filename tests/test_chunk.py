from rag.ingest.chunk import CHUNK_SIZE, chunk_paper
from rag.ingest.models import Paper, Section


def make_paper(section_texts: list[str]) -> Paper:
    return Paper(
        arxiv_id="0000.00001",
        title="Test Paper",
        abstract="An abstract.",
        sections=[
            Section(title=f"Section {i}", text=t, index=i) for i, t in enumerate(section_texts)
        ],
    )


def test_chunks_respect_token_limit() -> None:
    long_text = "Retrieval systems require careful evaluation. " * 400  # forces multiple chunks
    chunks = chunk_paper(make_paper([long_text]))
    assert len(chunks) > 1
    # Prefix is added after chunking, so allow headroom for it rather than exact CHUNK_SIZE
    assert all(c.token_count <= CHUNK_SIZE for c in chunks)


def test_chunk_ids_unique_and_deterministic() -> None:
    paper = make_paper(["Some text here.", "Other text there."])
    chunks_a = chunk_paper(paper)
    chunks_b = chunk_paper(paper)
    ids = [c.chunk_id for c in chunks_a]
    assert len(ids) == len(set(ids))
    assert ids == [c.chunk_id for c in chunks_b]  # same input → same ids, always
    assert ids[0] == "0000.00001:-1:0"  # abstract pseudo-section leads
    assert ids[1] == "0000.00001:0:0"  # first real section follows


def test_chunks_carry_citation_metadata_and_prefix() -> None:
    chunks = chunk_paper(make_paper(["Some section content."]))
    abstract, body = chunks[0], chunks[1]
    assert abstract.section_title == "Abstract"
    assert body.paper_title == "Test Paper"
    assert body.section_title == "Section 0"
    assert body.text.startswith("[Test Paper — Section 0]")


def test_chunking_never_crosses_sections() -> None:
    chunks = chunk_paper(make_paper(["Alpha content only.", "Beta content only."]))
    for c in chunks:
        assert not ("Alpha" in c.text and "Beta" in c.text)


def test_chunks_respect_token_limit_with_long_titles() -> None:
    long_title = (
        "A Comprehensive and Extremely Verbose Investigation into Retrieval-Augmented "
        "Generation Systems with Multi-Hop Reasoning over Dense Scientific Corpora"
    )
    paper = Paper(
        arxiv_id="0000.00001",
        title=long_title,
        abstract="x",
        sections=[
            Section(
                title="7 Experimental Results and Extended Ablation Studies",
                text="Retrieval systems require careful evaluation. " * 400,
                index=0,
            )
        ],
    )
    chunks = chunk_paper(paper)
    assert all(c.token_count <= CHUNK_SIZE for c in chunks), max(c.token_count for c in chunks)
