from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def load_openpilot_signals(csv_path: str | Path) -> Signals:
    """Load the canonical speed, direct acceleration and steering channels.

    This is the original BATON cleaning rule, shared with DriveDNA so both
    sources reach target interpolation through the same Signals contract.
    """
    required = ["time_s", "vEgo", "aEgo", "steeringAngleDeg"]
    frame = pd.read_csv(csv_path, usecols=required)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_s"])
    frame = frame.drop_duplicates("time_s", keep="first").sort_values("time_s")
    return Signals(
        t=frame["time_s"].to_numpy(np.float64),
        v=frame["vEgo"].to_numpy(np.float64),
        a_long=frame["aEgo"].to_numpy(np.float64),
        steering_angle=frame["steeringAngleDeg"].to_numpy(np.float64),
        valid=np.ones(len(frame), dtype=bool),
    )
