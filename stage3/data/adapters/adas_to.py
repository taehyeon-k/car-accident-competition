from __future__ import annotations

from pathlib import Path

from .base import DatasetAdapter


class AdasToAdapter(DatasetAdapter):
    """Explicit extension point; column mapping requires an actual ADAS-TO release."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def discover(self):
        raise NotImplementedError(
            "ADAS-TO mapping is dataset-release-specific. Provide timestamp, speed, "
            "direct longitudinal acceleration, steering unit, and sign mappings."
        )
