"""DriveDNA-Sample adapter for selected, synchronized video/CAN drives."""

from __future__ import annotations

import json
from pathlib import Path

from .base import ClipRecord, DatasetAdapter, load_openpilot_signals


class DriveDnaAdapter(DatasetAdapter):
    def __init__(self, raw_dir: str | Path, filter_report: str | Path,
                 selection_report: str | Path | None = None):
        self.raw_dir = Path(raw_dir).expanduser().resolve()
        self.filter_report = Path(filter_report).expanduser().resolve()
        self.selection_report = Path(selection_report).expanduser().resolve() if selection_report else None

    def discover(self) -> list[ClipRecord]:
        report = json.loads(self.filter_report.read_text())
        selection = json.loads(self.selection_report.read_text()) if self.selection_report else None
        selected = set(selection["accepted_video_files"]) if selection else None
        records = []
        for row in report["drives"]:
            if row["reasons"]:
                continue
            if selected is not None and row["video_path"] not in selected:
                continue
            csv_path = self.raw_dir / row["csv_path"]
            video_path = self.raw_dir / row["video_path"]
            if not csv_path.is_file() or not video_path.is_file():
                raise FileNotFoundError(f"Selected DriveDNA pair missing: {csv_path}, {video_path}")
            signals = load_openpilot_signals(csv_path)
            if abs(float(signals.t[0])) > 0.2:
                raise ValueError(f"DriveDNA clock must start at video time zero: {csv_path}")
            source_id = f"drivedna__{row['car_model']}__{row['driver']}__{row['drive']}"
            records.append(ClipRecord(
                clip_id=source_id, video_path=video_path, signals=signals,
                group_keys={"original_drive": source_id, "driver": row["driver"]},
                metadata={
                    "dataset": "DriveDNA", "source": "drivedna",
                    "signals_path": str(csv_path), "original_drive": source_id,
                    "car_model": row["car_model"], "driver": row["driver"], "drive": row["drive"],
                    "speed_units": "m/s", "acceleration_units": "m/s^2",
                    "steering_units": "degrees", "steering_type": "steering-wheel angle",
                    "steering_sign": "positive LEFT, negative RIGHT",
                    "telemetry_hz": row["csv_hz"],
                },
            ))
        expected = selection["accepted_drives"] if selection else report["accepted_drives"]
        if len(records) != expected:
            raise ValueError("DriveDNA record count differs from filter report")
        return records
