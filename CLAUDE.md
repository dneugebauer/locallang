# CLAUDE.md — LocalLang

Project instructions for Claude Code. Read this before making any changes.

## Project State

- **Current version:** v1.5.0
- **Phase 1 (terminal MVP):** complete
- **Phase 2 (React + FastAPI frontend):** next
- **PRD:** `docs/prd.md` — source of truth for phases, architecture intent, and open questions

## Commit Convention

Conventional commits: `type: short description in imperative mood`

```
feat: add metadata filtering to retriever
fix: handle empty sidecar yaml gracefully
docs: update PRD phase 2 requirements
chore: pin langchain to 0.3.26
refactor: extract sidecar loader into adapters/meta.py
```

Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`

Version bump commits include the version: `feat: improve ingestion pipeline UX (v1.4.2)`

## Version Bumping

Bump the version in `docs/prd.md` on every sufficient product change — same commit as the change:

- **MAJOR** — new deployment phase complete
- **MINOR** — meaningful new feature or capability
- **PATCH** — fixes, config changes, small improvements
- **No bump** — PRD-only edits (rewording, adding sections, clarifying content)

After bumping, update `project_state.md` in memory before ending the session.

## PRD Quality

Keep `docs/prd.md` clean business/technical documentation. Do not embed worked examples, sample Q&A, or step-by-step calculations to make the RAG system answer them. If retrieval quality is poor, fix the system (retrieval weights, system prompt, chunk size) — not the source documents.

## Technical Constraints

These are load-bearing decisions — do not change without discussion:

- **LangChain pinned to `0.3.x`** — `1.x` introduces `langchain-protocol` incompatible with Python 3.13. Do not upgrade.
- **`ChatOllama`, not `OllamaLLM`** — required for token-by-token streaming. `OllamaLLM` buffers the full response.
- **EnsembleRetriever output must be sliced** — both sub-retrievers return k results each; without the slice in `retriever.py` you get up to 2k results passed to the LLM.
- **Chroma max upsert batch: 5461** — pipeline batches at 5000. Do not increase.
- **Ollama must be installed via official script on WSL2** — snap sandbox blocks CUDA libraries, falls back to CPU.
- **`PDF_SOURCE_DIR` drives both PDF and CSV adapters** — both `ingest.py` and `chat.py` must call both adapters when this var is set. Do not add a separate `CSV_SOURCE_DIR`.

## Key Distinctions

- **`reindex.py`** — wipes the Chroma store and rebuilds from scratch. Use when switching document sets.
- **`python -m core.ingest`** — incremental only. Skips unchanged files via MD5 registry. Use to add new documents to an existing index.

## Metadata & Scope Gate

The metadata filter (`client`, `matter`, `doc_type`, `date`) is applied as a **hard pre-filter** via Chroma's `where` clause — documents outside the scope are excluded before similarity scoring. This is architecturally intentional: in professional contexts a missed filter is an ethical wall violation, not a relevance miss.

Phase 2 FastAPI layer must enforce that a scope is present before passing a query to the retriever. The `filter: dict | None = None` signature in `build_retriever()` is permissive for developer use — the API layer is where enforcement lives.

## Architecture Overview

```
PDFs → pdf_adapter (reads .meta.yaml sidecar)
     → ingest.py (MD5 diff → extractor.py → splitter → embedder → Chroma)

Query → scope gate (client · matter · doc_type)
      → build_retriever() with Chroma where clause + BM25 pre-filter
      → EnsembleRetriever (60% semantic, 40% keyword) → top-k slice
      → ChatOllama → streamed answer + source citations
```
