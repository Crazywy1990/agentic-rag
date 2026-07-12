import json
import statistics
from pathlib import Path

from rag.config import settings
from rag.ingest.chunk import chunk_paper
from rag.ingest.parse import parse_paper


def main() -> None:
    manifest = json.loads((settings.data_dir / "manifest.json").read_text())
    out_path = settings.data_dir / "chunks.jsonl"

    all_chunks, failures = [], []
    for entry in manifest:
        try:
            paper = parse_paper(Path(entry["html_path"]), arxiv_id=entry["arxiv_id"])
            if not paper.sections:
                raise ValueError("no sections extracted")
            all_chunks.extend(chunk_paper(paper))
        except Exception as e:  # noqa: BLE001 — survey run: collect failures, don't crash
            failures.append((entry["arxiv_id"], str(e)))

    with out_path.open("w") as f:
        for c in all_chunks:
            f.write(c.model_dump_json() + "\n")

    if failures:
        print(f"\nfailures ({len(failures)}):")
        for aid, err in failures[:10]:
            print(f"  {aid}: {err}")

    if not all_chunks:
        print(f"papers: 0/{len(manifest)} parsed — nothing to chunk")
        return

    counts = [c.token_count for c in all_chunks]
    print(f"papers: {len(manifest) - len(failures)}/{len(manifest)} parsed")
    print(f"chunks: {len(all_chunks)}")
    print(f"tokens/chunk: min={min(counts)} median={statistics.median(counts)} max={max(counts)}")
    print(
        f"total tokens: {sum(counts):,} (embedding cost @$0.02/M: ${sum(counts) * 0.02 / 1e6:.3f})"
    )


if __name__ == "__main__":
    main()
