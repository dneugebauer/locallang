from pathlib import Path
from langchain_core.documents import Document
import fitz  # PyMuPDF

from adapters.meta import load_sidecar


def load_pdf_documents(pdf_dir: str) -> list[Document]:
    if not pdf_dir:
        return []

    root = Path(pdf_dir)
    if not root.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")

    docs = []
    for pdf_path in root.rglob("*.pdf"):
        if not pdf_path.is_file():
            continue
        try:
            docs.extend(_load_pdf(pdf_path))
        except Exception as e:
            print(f"  [warn] Could not parse {pdf_path.name}: {e}")

    return docs


def _load_pdf(pdf_path: Path) -> list[Document]:
    docs = []
    sidecar = load_sidecar(pdf_path)
    with fitz.open(str(pdf_path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if not text.strip():
                continue
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": str(pdf_path.resolve()),
                    "filename": pdf_path.name,
                    "page": page_num,
                    "total_pages": len(doc),
                    "directory": str(pdf_path.parent.resolve()),
                    **sidecar,
                },
            ))
    return docs


def collect_pdf_file_paths(pdf_dir: str) -> list[str]:
    if not pdf_dir:
        return []
    root = Path(pdf_dir)
    return [str(p.resolve()) for p in root.rglob("*.pdf") if p.is_file()]
