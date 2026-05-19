"""
LLM-based metadata extraction for document sidecars.

Runs once per source file (PDF/CSV) that lacks a .meta.yaml.
Writes the sidecar so subsequent ingests read it directly without another LLM call.
"""
import json
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

import config

_EXTRACT_PROMPT = """Extract metadata from the document text below. Output ONLY valid JSON with these exact keys:
- doc_type: type of document (e.g. contract, invoice, memo, brief, policy, recipe, report, agreement, other)
- client: primary client or organization name, or empty string if unknown
- matter: case number, project code, batch reference, or empty string if none
- date: primary document date in YYYY-MM-DD format, or empty string if unknown
- summary: one sentence describing the document's purpose

Document text:
{content}"""

_SUPPORTED_EXTENSIONS = {".pdf", ".csv"}


def enrich_documents(docs: list[Document]) -> list[Document]:
    """Extract and apply metadata for any PDF/CSV source that lacks a sidecar.

    Writes a .meta.yaml beside each source file on first run; subsequent ingests
    pick up the sidecar directly without calling the LLM again.
    """
    if not config.EXTRACT_METADATA:
        return docs

    llm = ChatOllama(
        model=config.EXTRACT_METADATA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        format="json",
        temperature=0,
    )

    # Group docs by source so we extract once per file
    by_source: dict[str, list[Document]] = {}
    for doc in docs:
        src = doc.metadata.get("source", "")
        by_source.setdefault(src, []).append(doc)

    for src, src_docs in by_source.items():
        src_path = Path(src)
        if src_path.suffix not in _SUPPORTED_EXTENSIONS:
            continue

        sidecar_path = src_path.parent / (src_path.stem + ".meta.yaml")
        if sidecar_path.exists():
            continue

        extracted = _extract(llm, src_docs[0].page_content)
        if not extracted:
            continue

        _write_sidecar(sidecar_path, extracted)

        for doc in src_docs:
            doc.metadata.update(extracted)

        print(f"  [meta] {src_path.name} → {extracted.get('doc_type', '?')} / {extracted.get('client', '?')}")

    return docs


def _extract(llm: ChatOllama, content: str) -> dict:
    prompt = _EXTRACT_PROMPT.format(content=content[:config.EXTRACT_METADATA_CHARS])
    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        data = json.loads(raw)
        return {k: str(v).strip() for k, v in data.items() if isinstance(v, (str, int, float, bool))}
    except Exception as e:
        print(f"  [warn] Metadata extraction failed: {e}")
        return {}


def _write_sidecar(path: Path, metadata: dict) -> None:
    try:
        with path.open("w") as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True, sort_keys=True)
    except Exception as e:
        print(f"  [warn] Could not write sidecar {path.name}: {e}")
