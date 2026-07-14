import re
from pathlib import Path

from selectolax.parser import HTMLParser

from rag.ingest.models import Paper, Section

# Sections that add retrieval noise, not answer value. Matched against the
# lowercased title with leading numbering stripped ("7 References" -> "references").
SKIP_SECTIONS = {
    "references",
    "bibliography",
    "acknowledgement",
    "acknowledgements",
    "acknowledgments",
}

_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    """Collapse whitespace runs to single spaces."""
    return _WS.sub(" ", text).strip()


def _node_text(node) -> str:
    """Extract text with separators so adjacent tags don't concatenate
    ("2Related Works" -> "2 Related Works")."""
    return _clean(node.text(separator=" ", strip=True))


def _is_skipped(title: str) -> bool:
    normalized = title.lower().lstrip("0123456789.ivxlc ")
    return normalized in SKIP_SECTIONS


def parse_paper(html_path: Path, arxiv_id: str) -> Paper:
    tree = HTMLParser(html_path.read_text(encoding="utf-8"))

    title_node = tree.css_first("h1.ltx_title_document")
    title = _node_text(title_node) if title_node else arxiv_id

    abstract_node = tree.css_first("div.ltx_abstract")
    abstract = _node_text(abstract_node) if abstract_node else ""

    sections = []
    for i, sec in enumerate(tree.css("section.ltx_section")):
        heading = sec.css_first("h2.ltx_title")
        sec_title = _node_text(heading) if heading else f"Section {i}"
        if _is_skipped(sec_title):
            continue
        paras = [_node_text(p) for p in sec.css("div.ltx_para")]
        text = "\n\n".join(p for p in paras if p)
        if text:
            sections.append(Section(title=sec_title, text=text, index=i))

    return Paper(arxiv_id=arxiv_id, title=title, abstract=abstract, sections=sections)
