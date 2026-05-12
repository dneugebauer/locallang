"""
Wipe the current RAG context and build a fresh one from configured sources.
Use this when switching to a new project or starting from scratch.

    python reindex.py
    python reindex.py --limit 10000   # cap rows for a quick test run

For incremental updates to an existing context, use:
    python -m core.ingest
"""
import argparse
import shutil
import sys
from pathlib import Path

import config
from core.ingest import run_ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="Wipe and rebuild the RAG index.")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of documents ingested.")
    args = parser.parse_args()

    chroma = Path(config.CHROMA_PERSIST_DIR)
    registry = Path(config.INDEX_REGISTRY_PATH)

    existing = [p for p in [chroma, registry] if p.exists()]
    if existing:
        print("This will permanently delete the current RAG context:")
        for p in existing:
            print(f"  {p}")
        confirm = input("\nContinue? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)
        print()

    if chroma.exists():
        shutil.rmtree(chroma)
    if registry.exists():
        registry.unlink()

    run_ingest(limit=args.limit)


if __name__ == "__main__":
    main()
