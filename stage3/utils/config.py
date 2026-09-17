from __future__ import annotations

import importlib
import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and resolve all declared paths against ``root_dir``."""
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    root = Path(cfg.get("root_dir", path.parents[2])).expanduser().resolve()
    cfg["root_dir"] = str(root)
    path_fields = {
        "data": ("manifest", "val_manifest", "cache_dir", "statistics"),
        "flow": ("source_path", "checkpoint"),
    }
    for section, fields in path_fields.items():
        for field in fields:
            value = cfg.get(section, {}).get(field)
            if value and not Path(value).is_absolute():
                cfg[section][field] = str(root / value)
    output = Path(cfg.get("output_dir", "runs/stage3"))
    cfg["output_dir"] = str(output if output.is_absolute() else root / output)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "stage3":
        raise ValueError("stage must be 'stage3'")
    if cfg["calibration"]["focal_mode"] not in {"prior", "geocalib", "known"}:
        raise ValueError("calibration.focal_mode must be prior, geocalib, or known")
    if cfg["flow"]["backend"] not in {"sea_raft", "opencv"}:
        raise ValueError("flow.backend must be sea_raft or opencv")
    if cfg["flow"]["backend"] == "sea_raft" and not cfg["flow"].get("factory"):
        raise ValueError("SEA-RAFT requires flow.factory and a local implementation")
    if len(cfg["model"]["motion_cnn"]["widths"]) != 4:
        raise ValueError("motion_cnn.widths must contain four stages")
    if cfg["data"].get("crop_frames", 0) < 1:
        raise ValueError("data.crop_frames must be positive")


def import_callable(spec: str) -> Callable[..., Any]:
    module, separator, name = spec.partition(":")
    if not separator:
        raise ValueError(f"Expected module:callable, got {spec!r}")
    value = getattr(importlib.import_module(module), name)
    if not callable(value):
        raise TypeError(f"{spec!r} is not callable")
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    for row in rows:
        for name in ("video_path", "signals_path", "cache_path"):
            if row.get(name) and not Path(row[name]).is_absolute():
                row[name] = str((path.parent / row[name]).resolve())
    return rows
