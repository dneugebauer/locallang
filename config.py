import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
TARGET_REPO_PATH = os.getenv("TARGET_REPO_PATH", "")
PDF_SOURCE_DIR = os.getenv("PDF_SOURCE_DIR", "")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
INDEX_REGISTRY_PATH = os.getenv("INDEX_REGISTRY_PATH", "./registry.json")

# Ingestion
FILE_GLOB_PATTERNS = ["**/*.py", "**/*.md", "**/*.yaml", "**/*.yml", "**/*.json", "**/*.txt", "**/*.toml"]
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
FORCE_REINDEX = os.getenv("FORCE_REINDEX", "false").lower() == "true"

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".pytest_cache", "dist", "build"}
SKIP_EXTENSIONS = {".pyc", ".pyo", ".so", ".egg", ".whl", ".zip", ".tar", ".gz"}

# Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:14b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Metadata extraction (runs once per new PDF/CSV at ingest time)
EXTRACT_METADATA = os.getenv("EXTRACT_METADATA", "true").lower() == "true"
EXTRACT_METADATA_MODEL = os.getenv("EXTRACT_METADATA_MODEL", LLM_MODEL)
EXTRACT_METADATA_CHARS = int(os.getenv("EXTRACT_METADATA_CHARS", "3000"))

# Retrieval
NUM_RETRIEVED_CHUNKS = int(os.getenv("NUM_RETRIEVED_CHUNKS", "5"))
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.6"))
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "0.4"))

# Collection name in Chroma
CHROMA_COLLECTION = "locallang"
