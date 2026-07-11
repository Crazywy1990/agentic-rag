from pathlib import Path

from selectolax.parser import HTMLParser

from rag.ingest.models import Paper, Section


def parse_paper(html_path: Path, arxiv_id: str) -> Paper:
    tree = HTMLParser(html_path.read_text(encoding="utf-8"))

    title_node = tree.css_first("h1.ltx_title_document")
    title = title_node.text(strip=True) if title_node else arxiv_id

    abstract_node = tree.css_first("div.ltx_abstract")
    abstract = abstract_node.text(strip=True) if abstract_node else ""

    sections = []
    for i, sec in enumerate(tree.css("section.ltx_section")):
        title_node = sec.css_first("h2.ltx_title")
        sec_title = title_node.text(strip=True) if title_node else f"Section {i}"
        paras = [p.text(strip=True) for p in sec.css("div.ltx_para")]
        text = "\n\n".join(p for p in paras if p)
        if text:
            sections.append(Section(title=sec_title, text=text, index=i))

    return Paper(arxiv_id=arxiv_id, title=title, abstract=abstract, sections=sections)