"""
Ingestion pipeline. Run directly:
    python -m core.ingest [--force]
"""
import argparse
import sys
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import config
from core.registry import load_registry, save_registry, get_changed_files, update_registry
from adapters.github_adapter import load_repo_documents, collect_repo_file_paths
from adapters.pdf_adapter import load_pdf_documents, collect_pdf_file_paths


def run_ingest(force: bool = False) -> None:
    print("=== LocalLang Ingestion Pipeline ===\n")

    registry = {} if force else load_registry(config.INDEX_REGISTRY_PATH)

    # Collect all source file paths for diffing
    all_paths = []
    if config.TARGET_REPO_PATH:
        repo_paths = collect_repo_file_paths(config.TARGET_REPO_PATH)
        all_paths.extend(repo_paths)
        print(f"Repo files found:  {len(repo_paths)}")
    if config.PDF_SOURCE_DIR:
        pdf_paths = collect_pdf_file_paths(config.PDF_SOURCE_DIR)
        all_paths.extend(pdf_paths)
        print(f"PDF files found:   {len(pdf_paths)}")

    if not all_paths:
        print("\n[error] No source paths configured. Set TARGET_REPO_PATH and/or PDF_SOURCE_DIR in .env")
        sys.exit(1)

    new_files, modified_files, unchanged_files = get_changed_files(all_paths, registry)
    print(f"\nNew: {len(new_files)}  |  Modified: {len(modified_files)}  |  Unchanged: {len(unchanged_files)}")

    files_to_process = new_files + modified_files
    if not files_to_process:
        print("\nNothing to index. Use --force to re-index everything.")
        return

    # Load documents only for changed/new files
    print(f"\nLoading {len(files_to_process)} file(s)...")
    docs_to_index = _load_documents_for_paths(
        {path for path, _ in files_to_process}
    )

    if not docs_to_index:
        print("[warn] No content extracted from changed files.")
        return

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs_to_index)
    print(f"Split into {len(chunks)} chunk(s).")

    # Embed and upsert into Chroma
    print(f"Embedding with {config.EMBEDDING_MODEL} via Ollama...")
    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )

    Path(config.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )

    # Use source+chunk_index as stable IDs to enable upsert
    ids = [f"{c.metadata.get('source', 'unknown')}::chunk{i}" for i, c in enumerate(chunks)]
    vectorstore.add_documents(chunks, ids=ids)
    print(f"Upserted {len(chunks)} chunk(s) into Chroma.")

    # Update registry
    for path, hash_val in files_to_process:
        update_registry(registry, path, hash_val)
    save_registry(registry, config.INDEX_REGISTRY_PATH)

    print(f"\nDone. Registry saved to {config.INDEX_REGISTRY_PATH}")


def _load_documents_for_paths(target_paths: set[str]):
    from langchain_core.documents import Document
    docs = []

    if config.TARGET_REPO_PATH:
        repo_docs = load_repo_documents(config.TARGET_REPO_PATH)
        docs.extend(d for d in repo_docs if d.metadata.get("source") in target_paths)

    if config.PDF_SOURCE_DIR:
        pdf_docs = load_pdf_documents(config.PDF_SOURCE_DIR)
        docs.extend(d for d in pdf_docs if d.metadata.get("source") in target_paths)

    return docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the local RAG vector store.")
    parser.add_argument("--force", action="store_true", help="Re-index all files, ignoring registry.")
    args = parser.parse_args()
    run_ingest(force=args.force or config.FORCE_REINDEX)
