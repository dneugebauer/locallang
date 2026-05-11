"""
Terminal chat loop.
    python chat.py
"""
import os
import sys
from pathlib import Path

import requests
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import config
from core.retriever import build_retriever
from core.chain import build_llm, make_messages, format_docs
from adapters.github_adapter import load_repo_documents
from adapters.pdf_adapter import load_pdf_documents


def check_ollama() -> bool:
    try:
        r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def load_all_docs():
    docs = []
    if config.TARGET_REPO_PATH:
        docs.extend(load_repo_documents(config.TARGET_REPO_PATH))
    if config.PDF_SOURCE_DIR:
        docs.extend(load_pdf_documents(config.PDF_SOURCE_DIR))
    return docs


def print_sources(source_docs: list) -> None:
    if not source_docs:
        return
    seen = set()
    lines = []
    for doc in source_docs:
        src = doc.metadata.get("source", "unknown")
        src = os.path.relpath(src)
        page = doc.metadata.get("page")
        label = f"{src}, page {page}" if page else src
        if label not in seen:
            seen.add(label)
            lines.append(f"  • {label}")
    print("\nSources:")
    print("\n".join(lines))


def main() -> None:
    # Health checks
    if not check_ollama():
        print(f"[error] Ollama is not running at {config.OLLAMA_BASE_URL}")
        print("Start it with: ollama serve")
        sys.exit(1)

    chroma_path = Path(config.CHROMA_PERSIST_DIR)
    if not chroma_path.exists():
        print("[error] No vector store found. Run the ingestion pipeline first:")
        print("  python -m core.ingest")
        sys.exit(1)

    print("Loading index...")
    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    vectorstore = Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )

    count = vectorstore._collection.count()
    if count == 0:
        print("[error] Vector store is empty. Run: python -m core.ingest")
        sys.exit(1)

    print("Loading documents for keyword search...")
    all_docs = load_all_docs()

    retriever = build_retriever(vectorstore, all_docs)
    llm = build_llm()

    print(f"\n{'='*50}")
    print(f"  LocalLang Terminal RAG")
    print(f"  Model:  {config.LLM_MODEL}")
    print(f"  Chunks: {count}")
    print(f"  Type 'exit' or 'quit' to leave, 'clear' to clear screen")
    print(f"{'='*50}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if user_input.lower() == "clear":
            os.system("clear")
            continue

        docs = retriever.invoke(user_input)
        messages = make_messages(format_docs(docs), user_input)
        print("\nAssistant: ", end="", flush=True)
        for chunk in llm.stream(messages):
            print(chunk.content, end="", flush=True)
        print("\n")
        print_sources(docs)
        print()


if __name__ == "__main__":
    main()
