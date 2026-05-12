"""
Terminal chat loop.
    python chat.py
"""
import os
import sys
import tty
import termios
from pathlib import Path

import requests
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import config
from core.retriever import build_retriever
from core.chain import build_llm, make_messages, format_docs
from adapters.github_adapter import load_repo_documents
from adapters.pdf_adapter import load_pdf_documents

EMBEDDING_MODELS = {"nomic-embed-text", "mxbai-embed-large", "all-minilm", "nomic-embed-text:latest"}


def check_ollama() -> bool:
    try:
        r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def get_ollama_models() -> list[str]:
    r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
    models = [m["name"] for m in r.json().get("models", [])]
    return [m for m in models if not any(e in m for e in EMBEDDING_MODELS)]


def pick_model(models: list[str], default: str) -> str:
    if not models:
        return default

    # Start cursor on the configured default if present
    selected = next((i for i, m in enumerate(models) if m == default), 0)

    def read_key() -> str:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.buffer.read(1)
            if ch == b"\x1b":
                ch += sys.stdin.buffer.read(2)
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def render(idx: int) -> None:
        # Move cursor up to redraw list
        sys.stdout.write(f"\x1b[{len(models)}A")
        for i, model in enumerate(models):
            cursor = ">" if i == idx else " "
            line = f"  {cursor} {model}"
            sys.stdout.write(f"\r\x1b[K{line}\n")
        sys.stdout.flush()

    print("Select a model (↑↓ to move, Enter to confirm):\n")
    for model in models:
        print(f"    {model}")

    render(selected)

    while True:
        key = read_key()
        if key == b"\x1b[A":  # up
            selected = (selected - 1) % len(models)
            render(selected)
        elif key == b"\x1b[B":  # down
            selected = (selected + 1) % len(models)
            render(selected)
        elif key in (b"\r", b"\n"):  # enter
            print()
            return models[selected]
        elif key in (b"q", b"\x03"):  # q or ctrl-c
            print("\nAborted.")
            sys.exit(0)


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
    if not check_ollama():
        print(f"[error] Ollama is not running at {config.OLLAMA_BASE_URL}")
        print("Start it with: ollama serve")
        sys.exit(1)

    chroma_path = Path(config.CHROMA_PERSIST_DIR)
    if not chroma_path.exists():
        print("[error] No vector store found. Run the ingestion pipeline first:")
        print("  python reindex.py")
        sys.exit(1)

    models = get_ollama_models()
    chosen_model = pick_model(models, default=config.LLM_MODEL)

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
        print("[error] Vector store is empty. Run: python reindex.py")
        sys.exit(1)

    print("Loading documents for keyword search...")
    all_docs = load_all_docs()

    retriever = build_retriever(vectorstore, all_docs)
    llm = build_llm(model=chosen_model)

    print(f"\n{'='*50}")
    print(f"  LocalLang Terminal RAG")
    print(f"  Model:  {chosen_model}")
    print(f"  Chunks: {count}")
    print(f"  Type 'exit' or 'quit' to leave, 'clear' to clear screen and reset conversation")
    print(f"{'='*50}\n")

    history = []

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
            history = []
            continue

        docs = retriever.invoke(user_input)
        messages = make_messages(format_docs(docs), user_input, history)
        print("\nAssistant: ", end="", flush=True)
        answer = ""
        for chunk in llm.stream(messages):
            print(chunk.content, end="", flush=True)
            answer += chunk.content
        print("\n")
        print_sources(docs)
        print()

        history.append((user_input, answer))


if __name__ == "__main__":
    main()
