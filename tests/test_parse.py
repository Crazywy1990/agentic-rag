from pathlib import Path

from rag.ingest.parse import parse_paper

FIXTURE = Path(__file__).parent / "fixtures" / "minimal_paper.html"


def test_parse_extracts_title_and_abstract() -> None:
    paper = parse_paper(FIXTURE, arxiv_id="0000.00001")
    assert paper.title == "A Minimal Test Paper on Retrieval"
    assert "tests parsing" in paper.abstract
    assert paper.arxiv_id == "0000.00001"


def test_parse_extracts_sections_in_order() -> None:
    paper = parse_paper(FIXTURE, arxiv_id="0000.00001")
    titles = [s.title for s in paper.sections]
    assert titles == ["1 Introduction", "2 Method"]  # empty S3, References, Acks all dropped


def test_parse_joins_paragraphs() -> None:
    paper = parse_paper(FIXTURE, arxiv_id="0000.00001")
    intro = paper.sections[0]
    assert "combines a retriever" in intro.text
    assert "second paragraph" in intro.text
    assert "\n\n" in intro.text  # paragraphs joined, not concatenated


def test_parse_survives_real_paper() -> None:
    """Smoke test against a real downloaded paper — skipped where data/raw absent (CI)."""
    import pytest

    raw = Path("data/raw")
    if not raw.exists() or not any(raw.glob("*.html")):
        pytest.skip("no downloaded corpus available")
    html = next(raw.glob("*.html"))
    paper = parse_paper(html, arxiv_id=html.stem)
    assert paper.title
    assert len(paper.sections) >= 1


def test_skipped_sections_never_reach_chunks() -> None:
    paper = parse_paper(FIXTURE, arxiv_id="0000.00001")
    all_text = " ".join(s.text for s in paper.sections)
    assert "Some Citation" not in all_text
    assert "thank the test suite" not in all_text
