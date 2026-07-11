from chonkie import RecursiveChunker

from rag.ingest.models import Chunk, Paper

CHUNK_SIZE = 512  # tokens, cl100k_base — documented W1 baseline


def chunk_paper(paper: Paper, chunker: RecursiveChunker | None = None) -> list[Chunk]:
    chunker = chunker or RecursiveChunker(
        tokenizer_or_token_counter="cl100k_base", chunk_size=CHUNK_SIZE
    )
    chunks = []
    for section in paper.sections:
        # Contextual prefix: cheap grounding, big retrieval win for "which paper says X" queries
        prefix = f"[{paper.title} — {section.title}]\n"
        for j, c in enumerate(chunker.chunk(section.text)):
            chunks.append(
                Chunk(
                    chunk_id=f"{paper.arxiv_id}:{section.index}:{j}",
                    arxiv_id=paper.arxiv_id,
                    paper_title=paper.title,
                    section_title=section.title,
                    text=prefix + c.text,
                    token_count=c.token_count,
                )
            )
    return chunks