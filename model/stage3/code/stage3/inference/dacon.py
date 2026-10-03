from __future__ import annotations

from pathlib import Path

import pandas as pd

from stage3.data.adapters.dacon import DaconAdapter

from .predictor import Stage3Predictor


OUTPUT_COLUMNS = ["ID", "sample_index", "accel_label", "steer_label"]


def _checkpoint(model_dir: str | Path) -> Path:
    root = Path(model_dir).expanduser().resolve()
    candidates = (root / "stage3" / "best.pt", root / "best.pt", root / "stage3.pt")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No Stage 3 checkpoint found under {root}")


def predict_stage3(data_dir: str | Path, model_dir: str | Path) -> pd.DataFrame:
    predictor = Stage3Predictor(_checkpoint(model_dir))
    rows = []
    data_root = Path(data_dir).expanduser().resolve()
    if (data_root / "stage3").is_dir():
        data_root = data_root / "stage3"
    for record in DaconAdapter(data_root).discover():
        acceleration, steering, _ = predictor.predict_video(record.video_path)
        rows.extend(
            {"ID": record.clip_id, "sample_index": i, "accel_label": str(a), "steer_label": str(s)}
            for i, (a, s) in enumerate(zip(acceleration, steering))
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
