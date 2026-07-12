from pydantic import BaseModel


class Section(BaseModel):
    title: str
    text: str  # paragraphs joined by \n\n
    index: int  # position in paper, for ordering


class Paper(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    sections: list[Section]


class Chunk(BaseModel):
    chunk_id: str  # f"{arxiv_id}:{section_index}:{chunk_index}"
    arxiv_id: str
    paper_title: str
    section_title: str
    text: str  # what gets embedded
    token_count: int
