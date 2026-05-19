import csv
from pathlib import Path
from langchain_core.documents import Document

from adapters.meta import load_sidecar


def load_csv_documents(csv_dir: str) -> list[Document]:
    if not csv_dir:
        return []

    root = Path(csv_dir)
    if not root.exists():
        raise FileNotFoundError(f"CSV directory does not exist: {csv_dir}")

    docs = []
    for csv_path in root.rglob("*.csv"):
        if not csv_path.is_file():
            continue
        try:
            docs.extend(_load_csv(csv_path))
        except Exception as e:
            print(f"  [warn] Could not parse {csv_path.name}: {e}")

    return docs


def _load_csv(csv_path: Path) -> list[Document]:
    docs = []
    sidecar = load_sidecar(csv_path)
    with csv_path.open(encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return []

    columns = list(rows[0].keys())
    abs_path = str(csv_path.resolve())

    for row_num, row in enumerate(rows, start=1):
        content = " | ".join(f"{k}: {v}" for k, v in row.items() if v and v.strip())
        if not content.strip():
            continue
        docs.append(Document(
            page_content=content,
            metadata={
                "source": abs_path,
                "filename": csv_path.name,
                "row": row_num,
                "total_rows": len(rows),
                "columns": ", ".join(columns),
                **sidecar,
            },
        ))

    return docs


def collect_csv_file_paths(csv_dir: str) -> list[str]:
    if not csv_dir:
        return []
    root = Path(csv_dir)
    return [str(p.resolve()) for p in root.rglob("*.csv") if p.is_file()]
