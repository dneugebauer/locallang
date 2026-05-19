from pathlib import Path

import yaml


def load_sidecar(file_path: Path) -> dict:
    """Read an optional {stem}.meta.yaml beside file_path; return {} if absent or unreadable.

    Only scalar values (str, int, float, bool) are kept so Chroma can index them.
    """
    sidecar = file_path.parent / (file_path.stem + ".meta.yaml")
    if not sidecar.exists():
        return {}
    try:
        with sidecar.open() as f:
            data = yaml.safe_load(f) or {}
        return {k: v for k, v in data.items() if isinstance(v, (str, int, float, bool))}
    except Exception:
        return {}


def load_dir_sidecar(dir_path: Path) -> dict:
    """Read an optional .meta.yaml at the root of a directory (e.g. a repo)."""
    sidecar = dir_path / ".meta.yaml"
    if not sidecar.exists():
        return {}
    try:
        with sidecar.open() as f:
            data = yaml.safe_load(f) or {}
        return {k: v for k, v in data.items() if isinstance(v, (str, int, float, bool))}
    except Exception:
        return {}
