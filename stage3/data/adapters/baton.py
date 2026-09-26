from __future__ import annotations

from pathlib import Path

from .base import ClipRecord, DatasetAdapter, load_openpilot_signals


class BatonAdapter(DatasetAdapter):
    """BATON adapter: m/s, m/s², and steering-wheel degrees.

    BATON/openpilot uses positive ``steeringAngleDeg`` for left steering. This is
    the Stage 3 canonical convention: positive is LEFT and negative is RIGHT.
    """

    REQUIRED = ("time_s", "vEgo", "aEgo", "steeringAngleDeg")

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def discover(self) -> list[ClipRecord]:
        records: list[ClipRecord] = []
        for route in sorted(p for p in self.root.iterdir() if p.is_dir()):
            video, csv_path = route / "qcamera.mp4", route / "vehicle_dynamics.csv"
            if not video.is_file() or not csv_path.is_file():
                continue
            signals = load_openpilot_signals(csv_path)
            records.append(
                ClipRecord(
                    clip_id=route.name,
                    video_path=video,
                    signals=signals,
                    group_keys={"route": route.name.split("_")[1]},
                    metadata={
                        "dataset": "BATON",
                        "signals_path": str(csv_path),
                        "speed_units": "m/s",
                        "acceleration_units": "m/s^2",
                        "steering_units": "degrees",
                        "steering_type": "steering-wheel angle",
                        "steering_sign": "positive LEFT, negative RIGHT",
                    },
                )
            )
        if not records:
            raise FileNotFoundError(f"No BATON routes found below {self.root}")
        return records
