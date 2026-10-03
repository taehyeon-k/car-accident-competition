from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Signals:
    t: np.ndarray
    v: np.ndarray
    a_long: np.ndarray | None
    steering_angle: np.ndarray | None
    yaw_rate: np.ndarray | None = None
    curvature: np.ndarray | None = None
    valid: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.t)
        if n < 2 or np.any(~np.isfinite(self.t)) or np.any(np.diff(self.t) <= 0):
            raise ValueError("Signal timestamps must be finite, strictly increasing, and nontrivial")
        for name in ("v", "a_long", "steering_angle", "yaw_rate", "curvature", "valid"):
            value = getattr(self, name)
            if value is not None and len(value) != n:
                raise ValueError(f"Signal {name} length does not match timestamps")


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    video_path: Path
    signals: Signals | None
    group_keys: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(ABC):
    @abstractmethod
    def discover(self) -> list[ClipRecord]:
        raise NotImplementedError
