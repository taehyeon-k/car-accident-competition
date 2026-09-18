from __future__ import annotations

import os
import tempfile
import gzip
from pathlib import Path
from typing import Any

import torch


FORMAT_VERSION = 1


def atomic_save(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
        path.chmod(0o664)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_gzip_save(value: Any, path: str | Path, compresslevel: int = 3) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
    try:
        with gzip.open(temporary, "wb", compresslevel=compresslevel) as stream:
            torch.save(value, stream)
        os.replace(temporary, path)
        path.chmod(0o664)
    finally:
        temporary.unlink(missing_ok=True)


def load_artifact(path: str | Path, weights_only: bool = False):
    path = Path(path)
    with path.open("rb") as stream:
        compressed = stream.read(2) == b"\x1f\x8b"
    if compressed:
        with gzip.open(path, "rb") as stream:
            return torch.load(stream, map_location="cpu", weights_only=weights_only)
    return torch.load(path, map_location="cpu", weights_only=weights_only)


def save_checkpoint(path: str | Path, **state: Any) -> None:
    from stage3.data.cache import FEATURE_CODE_VERSION, cache_key

    atomic_save({**state, "format_version": FORMAT_VERSION,
                 "feature_version": FEATURE_CODE_VERSION,
                 "feature_cache_key": cache_key(state["config"])}, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    value = torch.load(path, map_location=map_location, weights_only=False)
    if value.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported Stage 3 checkpoint format: {value.get('format_version')}")
    from stage3.data.cache import FEATURE_CODE_VERSION, cache_key

    if value.get("feature_version") != FEATURE_CODE_VERSION or value.get("feature_cache_key") != cache_key(value["config"]):
        raise ValueError("Incompatible Stage 3 feature version; rebuild caches/statistics and retrain. Legacy checkpoints cannot be resumed or used for inference.")
    return value
