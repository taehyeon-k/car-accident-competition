"""Frame-order invariants for native filename IDs."""

from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable


def frame_id(path: str | Path) -> int:
    """Parse the final numeric component without assuming contiguous filenames."""
    numbers = re.findall(
        r"\d+",
        Path(path).stem,
    )
    if not numbers:
        raise ValueError(f"No integer frame id in {path!s}")
    return int(numbers[-1])


def sort_frame_paths(paths: Iterable[str | Path]) -> tuple[list[Path], list[int]]:
    ordered = sorted(
        (Path(p) for p in paths),
        key=frame_id,
    )
    ids = [frame_id(p) for p in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("Frame IDs must be unique within a sample")
    return ordered, ids
