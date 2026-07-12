# agentic-rag

Agentic RAG over an arXiv ML/AI papers corpus, built eval-first: a rigorous
measurement harness is the centerpiece, and every design decision below has a
stated revisit condition tied to it.

**Architecture:** this service consumes
[llm-gateway](https://github.com/Crazywy1990/llm-gateway) (Project 1) for all
LLM calls — generation and the agentic loop — reusing its semantic caching and
observability. Embeddings are called directly from this service (see decisions).

**Status:** Week 1 — corpus ingestion and chunking complete (50 papers, 1,414
chunks). Next: embeddings + LanceDB vector store behind swappable interfaces.

## Quickstart

```bash
uv sync
uv run pytest                              # hermetic suite
uv run python scripts/fetch_corpus.py      # ~3 min: 50 arXiv papers -> data/raw/ + manifest
uv run python scripts/build_chunks.py      # parse + chunk -> data/chunks.jsonl
./scripts/check.sh                         # all CI gates, locally
```

## Pinned decisions

| Date | Decision | Choice | Rationale | Revisit when |
|---|---|---|---|---|
| 2026-07-08 | Vector store | LanceDB (embedded) | In-process, zero infra, disk-based indexing; ships inside the Cloud Run image | Corpus >1M chunks or multi-writer → pgvector/Qdrant |
| 2026-07-08 | Embeddings | OpenAI text-embedding-3-small, called directly | Cheap defensible default for English technical text; gateway caching is completion-oriented | W2 harness exists → A/B vs voyage-3-large, measured |
| 2026-07-08 | Corpus | 50 arXiv cs.CL papers (HTML-only), manifest-pinned | LaTeXML HTML ≫ PDF parsing; manifest pins exact paper versions | W2 eval authoring → date-bounded, cross-referential query |
| 2026-07-11 | Chunk size | 512 tokens total (cl100k_base), citation prefix measured per-section | Precision-first for dense technical text; hard ≤512 invariant, test-enforced | W2 harness → A/B vs 1024 |
| 2026-07-11 | Chunk context | `[paper title — section title]` prefix injected pre-embedding | Lightweight contextual retrieval; chunk self-identifies for ~30 tokens | W2 → measure vs no-prefix |
| 2026-07-11 | Chunk boundaries | Never cross sections; bibliography skipped | Section breaks are real semantic boundaries | Only if eval shows cross-section answers failing |