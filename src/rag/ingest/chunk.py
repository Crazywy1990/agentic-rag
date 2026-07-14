import tiktoken
from chonkie import RecursiveChunker

from rag.ingest.models import Chunk, Paper, Section

CHUNK_SIZE = 512  # tokens (cl100k_base), total embedded size incl. citation prefix

_enc = tiktoken.get_encoding("cl100k_base")


def chunk_paper(paper: Paper) -> list[Chunk]:
    chunks = []
    sections = list(paper.sections)
    if paper.abstract:
        sections.insert(0, Section(title="Abstract", text=paper.abstract, index=-1))
    for section in sections:
        # Citation prefix: embedded chunk "knows" its paper + section
        prefix = f"[{paper.title} — {section.title}]\n"
        prefix_tokens = len(_enc.encode(prefix))
        chunker = RecursiveChunker(tokenizer="cl100k_base", chunk_size=CHUNK_SIZE - prefix_tokens)
        for j, c in enumerate(chunker.chunk(section.text)):
            text = prefix + c.text
            chunks.append(
                Chunk(
                    chunk_id=f"{paper.arxiv_id}:{section.index}:{j}",
                    arxiv_id=paper.arxiv_id,
                    paper_title=paper.title,
                    section_title=section.title,
                    text=text,
                    token_count=len(_enc.encode(text)),
                )
            )
    return chunks
