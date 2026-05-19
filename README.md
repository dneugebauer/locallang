# LocalLang

A fully local, private RAG (Retrieval-Augmented Generation) system for querying documents in plain English. No data leaves your machine — no cloud APIs, no rate limits, no cost per query.

## Overview

Index a directory of PDFs, then ask questions about them from the terminal. Answers are grounded in your documents and include source citations.

Documents are automatically tagged with structured metadata (client, matter, doc type, date) at ingest time using a local LLM. Metadata is stored in `.meta.yaml` sidecar files alongside your sources — editable by hand, never overwritten once created. The retriever can scope any query to a subset of documents using these fields.

Built on [LangChain](https://python.langchain.com/), [Ollama](https://ollama.com/), and [Chroma](https://www.tropic.io/chroma).

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) running locally — install via the **official script**, not snap:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

> **WSL2 + NVIDIA GPU:** the snap package does not have access to WSL2's CUDA libraries and will fall back to CPU. The official installer auto-detects the GPU and runs ~20x faster for embeddings.

Pull the required models:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:14b   # best for code queries
ollama pull llama3.2:3b         # faster, good for testing
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — set TARGET_REPO_PATH and/or PDF_SOURCE_DIR

# 3. Index your documents
python reindex.py

# 4. Start chatting
python chat.py
```

## Configuration

All settings live in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `TARGET_REPO_PATH` | _(empty)_ | Path to a local code repository |
| `PDF_SOURCE_DIR` | _(empty)_ | Path to a directory of PDF documents |
| `CHROMA_PERSIST_DIR` | `./chroma_store` | Where the vector store is saved |
| `LLM_MODEL` | `qwen2.5-coder:14b` | Ollama model for answering queries |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `CHUNK_SIZE` | `1000` | Token size for text splitting |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `NUM_RETRIEVED_CHUNKS` | `3` | Chunks retrieved per query |
| `SEMANTIC_WEIGHT` | `0.6` | Weight for semantic (vector) search |
| `KEYWORD_WEIGHT` | `0.4` | Weight for BM25 keyword search |
| `EXTRACT_METADATA` | `true` | Auto-extract metadata for new PDFs at ingest |
| `EXTRACT_METADATA_MODEL` | _(LLM_MODEL)_ | Model used for extraction — can be lighter than your chat model |
| `EXTRACT_METADATA_CHARS` | `3000` | Characters of document text fed to the extractor |

## Ingestion

```bash
# Start a new RAG context (wipes existing index, then ingests from scratch)
python reindex.py

# Cap the number of documents ingested — useful for fast test runs
python reindex.py --limit 10000

# Add to or update the current context (only processes new/changed files)
python -m core.ingest

# Force re-process all files without wiping the store
python -m core.ingest --force
```

Use `reindex.py` when switching projects or sources. Use `core.ingest` to augment an existing index with new documents. The pipeline tracks processed files in `registry.json` using MD5 hashes — incremental runs skip unchanged files.

## Metadata & Filtering

Every PDF is automatically tagged at ingest time. The extractor reads the first `EXTRACT_METADATA_CHARS` characters, calls the configured Ollama model, and writes a `.meta.yaml` sidecar beside the source file:

```yaml
# sample_docs/contract_acme.meta.yaml
client: Acme Corp
date: '2024-03-15'
doc_type: contract
matter: M-2024-01
summary: Service agreement between Acme Corp and Smith & Associates covering Q1 2024.
```

**Sidecar files are written once and never overwritten.** Edit them by hand at any time — the next ingest will pick up your changes. Set `EXTRACT_METADATA=false` to skip auto-extraction entirely.

For a code repository, place a `.meta.yaml` at the repo root to tag all files from that source:

```yaml
# /path/to/your/repo/.meta.yaml
client: Internal
doc_type: codebase
matter: backend-v2
```

### Using filters in retrieval

Pass a `filter` dict to `build_retriever()` to scope both Chroma and BM25 to matching documents:

```python
retriever = build_retriever(vectorstore, all_docs, filter={"client": "Acme Corp"})
```

Simple equality matching on any metadata field is supported. Multiple keys are ANDed together.

## Terminal Chat

```bash
python chat.py
```

On startup, an interactive model picker fetches all available Ollama models and lets you select with arrow keys. The cursor defaults to `LLM_MODEL` from `.env`.

The startup banner shows the active model, detected GPU, and chunk count for quick confirmation that the system is configured correctly.

Conversation history is maintained within a session — follow-up questions have full context of prior exchanges.

**Commands:**
- `exit` / `quit` — close the session
- `clear` — clear the terminal and reset conversation history

Sources are displayed after each answer. LaTeX math formatting is automatically stripped for clean terminal output.

## Project Structure

```
locallang/
├── chat.py                    # Terminal chat entry point (model picker, history, sanitizer)
├── reindex.py                 # Wipe and rebuild index from scratch
├── config.py                  # All configuration (reads from .env)
├── generate_sample_data.py    # Generates mock sales CSV for testing
├── requirements.txt
├── .env.example
├── core/
│   ├── ingest.py              # Ingestion pipeline orchestrator
│   ├── extractor.py           # LLM-based metadata extraction → writes .meta.yaml sidecars
│   ├── retriever.py           # Hybrid semantic + BM25 retriever with optional metadata filter
│   ├── chain.py               # LLM prompt, conversation history, streaming
│   └── registry.py            # MD5-based file change tracking
├── adapters/
│   ├── meta.py                # Sidecar loader (load_sidecar, load_dir_sidecar)
│   ├── github_adapter.py      # Supplementary code repo loader
│   ├── pdf_adapter.py         # PyMuPDF PDF loader
│   └── csv_adapter.py         # Supplementary CSV loader (one document per row)
└── sample_docs/               # Gitignored — drop test PDFs here
```

## Performance Tips

- Use `llama3.2:3b` during development for fast responses; switch to `qwen2.5-coder:14b` for production quality
- Set `EXTRACT_METADATA_MODEL` to a smaller model (e.g. `llama3.2:3b`) to speed up ingestion without affecting chat quality
- Lower `NUM_RETRIEVED_CHUNKS` to reduce time-to-first-token
- Set `OLLAMA_MAX_LOADED_MODELS=2` in your environment before starting Ollama to keep both the embedding model and LLM loaded simultaneously
- Install Ollama via the official script (not snap) for GPU support on WSL2 — snap runs sandboxed and cannot access CUDA libraries
- On an RTX 5080, nomic-embed-text embeds at ~130 chunks/sec on GPU vs ~6 chunks/sec on CPU
- Chroma enforces a max upsert batch size of 5461 — the pipeline batches in groups of 5000 automatically

## Architecture

### Chat loop

```mermaid
flowchart LR
    A([User question]) --> B{Hybrid Retriever}
    B -->|Semantic 60%| C[Chroma]
    B -->|Keyword 40%| D[BM25]
    C & D --> E[Top-k fusion]
    E --> F[ChatOllama]
    F --> G([Streamed answer\n+ source citations])
    G --> A
```

### System architecture

```mermaid
flowchart TD
    subgraph src ["Local files"]
        S2[PDFs]
        M[(".meta.yaml\nsidecars")]
    end

    subgraph adapters ["Adapter layer"]
        A2[pdf_adapter\n+ file sidecar]
    end

    subgraph ingest ["Ingestion pipeline · core/ingest.py"]
        I1[MD5 registry diff\nskip unchanged files]
        I2[extractor.py\nLLM metadata extraction\nwrites .meta.yaml]
        I3[RecursiveCharacter\nTextSplitter]
        I4[OllamaEmbeddings\nnomic-embed-text]
    end

    subgraph retrieval ["Retrieval · core/retriever.py"]
        R1[Chroma\nsemantic search]
        R2[BM25\nkeyword search]
        R3[EnsembleRetriever\nfusion + top-k slice]
    end

    S2 --> A2
    M -.->|merged into metadata| A2

    A2 --> I1 --> I2 --> I3 --> I4 --> DB[(Chroma\nvector store)]
    I2 -.->|writes| M

    F{filter?} -->|optional| R1 & R2
    DB --> R1
    Q([User question]) --> R1 & R2
    R1 & R2 --> R3
    R3 --> LLM[ChatOllama\nqwen2.5-coder:14b · llama3.2:3b]
    LLM --> ANS([Streamed answer + citations])
```
