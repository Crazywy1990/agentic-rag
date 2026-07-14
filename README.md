# agentic-rag

Agentic RAG over an arXiv ML/AI papers corpus, built eval-first: a rigorous
measurement harness is the centerpiece, and every design decision below has a
stated revisit condition tied to it.

**Architecture:** this service consumes
[llm-gateway](https://github.com/Crazywy1990/llm-gateway) (Project 1) for all
LLM calls — generation and the agentic loop — reusing its semantic caching and
observability. Embeddings are called directly from this service (see decisions).

**Status:** Week 1 — retrieval working end-to-end. 50 papers → 1,559 chunks,
embedded and searchable in LanceDB with citation metadata on every chunk.
Next: citation-grounded generation via the gateway.

## Quickstart

```bash
uv sync
cp .env.example .env                       # then fill in keys
uv run pytest                              # hermetic suite
uv run python scripts/fetch_corpus.py      # ~3 min: 50 arXiv papers -> data/raw/ + manifest
uv run python scripts/build_chunks.py      # parse + chunk -> data/chunks.jsonl
uv run python scripts/ingest.py            # embed + store -> data/lancedb (~$0.01)
uv run python scripts/search.py "what is agentic retrieval?"
./scripts/check.sh                         # all CI gates, locally
```

Rebuilding after chunking changes: the store is a derived artifact — always
`rm -rf data/lancedb` (or rebuild) before re-ingesting; `add()` appends.

## Pinned decisions

| Date | Decision | Choice | Rationale | Revisit when |
|---|---|---|---|---|
| 2026-07-08 | Vector store | LanceDB (embedded) | In-process, zero infra, disk-based indexing; ships inside the Cloud Run image | Corpus >1M chunks or multi-writer → pgvector/Qdrant |
| 2026-07-08 | Embeddings | OpenAI text-embedding-3-small, called directly | Cheap defensible default for English technical text; gateway caching is completion-oriented | W2 harness exists → A/B vs voyage-3-large, measured |
| 2026-07-08 | Corpus | 50 arXiv cs.CL papers (HTML-only), manifest-pinned | LaTeXML HTML ≫ PDF parsing; manifest pins exact paper versions | W2 eval authoring → date-bounded, cross-referential query |
| 2026-07-11 | Chunk size | 512 tokens total (cl100k_base), citation prefix measured per-section | Precision-first for dense technical text; hard ≤512 invariant, test-enforced | W2 harness → A/B vs 1024 |
| 2026-07-11 | Chunk context | `[paper title — section title]` prefix injected pre-embedding | Lightweight contextual retrieval; chunk self-identifies for ~30 tokens | W2 → measure vs no-prefix |
| 2026-07-11 | Chunk boundaries | Never cross sections; references/acknowledgements skipped | Section breaks are real semantic boundaries; bibliography is retrieval noise (found by inspection) | Only if eval shows cross-section answers failing |
| 2026-07-12 | Store/embedder interfaces | Python Protocols; score contract: cosine similarity, higher-better | Structural typing, zero coupling; backend distance conventions never leak past the store | — |
| 2026-07-12 | Distance metric | cosine, explicit | LanceDB default (L2) gave uninterpretable scores — found by inspection, pinned by regression test | — |
| 2026-07-12 | Abstract handling | pseudo-section, chunk_id `{id}:-1:0` | Densest retrieval target in any paper | W2: measure abstract hit-rate |
| 2026-07-12 | Re-ingest policy | rebuild-always; store is a derived artifact | Append-by-default silently mixes stale and fresh vectors | Corpus grows enough to need incremental ingest |