from __future__ import annotations

from pathlib import Path

from .base import ClipRecord, DatasetAdapter


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class DaconAdapter(DatasetAdapter):
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def discover(self) -> list[ClipRecord]:
        base = self.root / "videos" if (self.root / "videos").is_dir() else self.root
        records = [
            ClipRecord(path.stem, path.resolve(), None, metadata={"dataset": "DACON"})
            for path in sorted(base.rglob("*"))
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ]
        if not records:
            raise FileNotFoundError(f"No Stage 3 videos found below {base}")
        return records
