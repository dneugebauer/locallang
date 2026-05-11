import hashlib
import json
from pathlib import Path


def load_registry(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def save_registry(registry: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_changed_files(file_list: list[str], registry: dict) -> tuple[list, list, list]:
    new, modified, unchanged = [], [], []
    for path in file_list:
        h = file_hash(path)
        if path not in registry:
            new.append((path, h))
        elif registry[path]["hash"] != h:
            modified.append((path, h))
        else:
            unchanged.append(path)
    return new, modified, unchanged


def update_registry(registry: dict, path: str, hash_val: str) -> None:
    from datetime import datetime
    registry[path] = {"hash": hash_val, "indexed_at": datetime.utcnow().isoformat()}
