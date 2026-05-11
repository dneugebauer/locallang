import os
from pathlib import Path
from langchain_core.documents import Document
import config


def load_repo_documents(repo_path: str, glob_patterns: list[str] | None = None) -> list[Document]:
    if not repo_path:
        return []

    patterns = glob_patterns or config.FILE_GLOB_PATTERNS
    root = Path(repo_path)
    if not root.exists():
        raise FileNotFoundError(f"Repo path does not exist: {repo_path}")

    docs = []
    seen = set()

    for pattern in patterns:
        for file_path in root.glob(pattern):
            if not file_path.is_file():
                continue
            if file_path.suffix in config.SKIP_EXTENSIONS:
                continue
            if any(skip in file_path.parts for skip in config.SKIP_DIRS):
                continue

            abs_path = str(file_path.resolve())
            if abs_path in seen:
                continue
            seen.add(abs_path)

            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            docs.append(Document(
                page_content=text,
                metadata={
                    "source": abs_path,
                    "file_type": file_path.suffix.lstrip("."),
                    "modified": os.path.getmtime(abs_path),
                    "filename": file_path.name,
                },
            ))

    return docs


def collect_repo_file_paths(repo_path: str, glob_patterns: list[str] | None = None) -> list[str]:
    if not repo_path:
        return []

    patterns = glob_patterns or config.FILE_GLOB_PATTERNS
    root = Path(repo_path)
    paths = []
    seen = set()

    for pattern in patterns:
        for file_path in root.glob(pattern):
            if not file_path.is_file():
                continue
            if file_path.suffix in config.SKIP_EXTENSIONS:
                continue
            if any(skip in file_path.parts for skip in config.SKIP_DIRS):
                continue
            abs_path = str(file_path.resolve())
            if abs_path not in seen:
                seen.add(abs_path)
                paths.append(abs_path)

    return paths
