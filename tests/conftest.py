from rag.ingest.models import Chunk


def make_chunk(text: str, chunk_index: int = 0, section_index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"0000.00001:{section_index}:{chunk_index}",
        arxiv_id="0000.00001",
        paper_title="Test Paper",
        section_title=f"Section {section_index}",
        text=text,
        token_count=10,
    )
