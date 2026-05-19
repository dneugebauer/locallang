# LocalLang — Product Requirements Document

**Version:** v1.5.0
**Updated:** 2026-05-19

---

## Overview

LocalLang is a fully private, locally-hosted RAG (Retrieval-Augmented Generation) platform for querying document libraries in plain English. All inference runs on-premise — no cloud APIs, no data egress, no per-query cost.

---

## Version History

| Version | Date | Summary |
|---|---|---|
| v1.5.0 | 2026-05-19 | Sidecar metadata (.meta.yaml), LLM auto-extraction at ingest, Chroma metadata filtering, PDF-first positioning |
| v1.4.2 | 2026-05-11 | Ingestion pipeline UX — GPU warmup, live progress bar, benchmark before embed |
| v1.3.3 | 2026-05-11 | CSV adapter, GPU name in startup banner, incremental ingest improvements |

---

## Architecture

### Ingestion Pipeline

1. MD5-based registry diff — skip unchanged files
2. Load documents via PDF adapter; read `.meta.yaml` sidecar if present
3. LLM-based metadata extraction — for new PDFs without a sidecar, extract `doc_type`, `client`, `matter`, `date`, `summary` and write `.meta.yaml`
4. Chunk with `RecursiveCharacterTextSplitter` (1000 tokens, 150 overlap)
5. Embed with `nomic-embed-text` via Ollama
6. Upsert into Chroma vector store (batches of 5000)

### Retrieval

Every query passes through a scope gate before reaching the retriever. The scope (client, matter, doc_type, date range) is resolved to a Chroma `where` clause, which is applied as a **hard pre-filter** — documents not matching the scope are excluded before any similarity scoring occurs. BM25 receives the same pre-filtered document set.

After filtering, retrieval is a hybrid ensemble — 60% semantic (Chroma dense vector) + 40% keyword (BM25), fused and sliced to `NUM_RETRIEVED_CHUNKS`.

The pre-filter model matters for professional contexts: a missed filter is not a relevance miss — it is a potential ethical wall violation. Chroma's `where` clause enforces the boundary at the query layer, not in post-processing.

### LLM

`ChatOllama` (not `OllamaLLM`) — required for token-by-token streaming. System prompt instructs the model to answer only from retrieved context and reason step-by-step for any calculations.

### Key Technical Constraints

- LangChain pinned to `0.3.x` — `1.x` introduces `langchain-protocol` incompatible with Python 3.13
- Chroma max upsert batch: 5461 — pipeline batches at 5000
- Ollama must be installed via official script on WSL2 — snap sandbox blocks CUDA libraries
- `PDF_SOURCE_DIR` scans both PDFs and CSVs (supplementary) — both adapters run when this dir is set

---

## Metadata & Filtering

Every PDF ingested for the first time receives a `.meta.yaml` sidecar automatically:

```yaml
client: Acme Corp
date: '2024-03-15'
doc_type: contract
matter: M-2024-01
summary: Service agreement between Acme Corp and Smith & Associates.
```

Sidecars are written once and never overwritten — editable by hand. The extraction model is configurable separately from the chat model (`EXTRACT_METADATA_MODEL`) so a smaller, faster model can be used for tagging without affecting response quality.

Metadata fields are indexed in Chroma and available for equality filtering at query time.

---

## Phases

### Phase 1 — Terminal MVP ✅ (complete, v1.5.0)

- Terminal chat with interactive model picker, conversation history, LaTeX sanitizer
- Incremental ingestion with MD5 registry; `reindex.py` for full rebuilds
- Hybrid retriever (Chroma + BM25 ensemble)
- PDF adapter with per-file `.meta.yaml` sidecar support
- LLM-based metadata auto-extraction at ingest time
- Chroma metadata filtering in retriever
- Supplementary CSV and code repo adapters
- GPU warmup and live progress bar during embedding

### Phase 2 — Web Frontend (next, est. 16–24 hours)

**Goal:** Browser-based UI accessible on LAN without technical knowledge.

**Stack:** React frontend + FastAPI backend

**Requirements:**
- Browser-based chat UI (replaces `chat.py`)
- Model selector dropdown
- **Matter selection is required before a query can be submitted.** The scope gate (client, matter, doc_type, date range) is the primary entry point — not an optional filter panel. Unscoped queries are only permitted via an explicit "Search across all matters" mode with a confirmation step.
- Scope selection is enforced at the API layer (FastAPI), not just the UI — a request without a valid scope is rejected, not passed through unfiltered.
- Source citation display with page reference
- Session management (per-user conversation history)
- FastAPI serves the retrieval chain and Ollama proxying

### Phase 3 — Automated Ingestion (future)

**Goal:** Keep the document index current without manual re-indexing.

- Prefect pipeline for scheduled ingestion runs
- Connector framework for external data sources
- New documents auto-generate metadata sidecars on first ingest

### Phase 4 — Multi-User & Access Control (future)

- Authentication and session management
- Per-user or per-group document scope (metadata filter at login)
- Audit logging — query log with user, timestamp, retrieved sources
- Client profile system — per-context system prompts and document scope

---

## Open Questions

- Phase 2 auth layer — build in-house (FastAPI + JWT) or integrate an existing provider?
- Multi-location document sync — rsync over Tailscale vs. shared object storage?
- Phase 3 connector priority — which external sources to support first?
