"""Configuration, manifest paths and reproducible local artifact handling."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
import yaml


def load_config(path: str | Path) -> dict:
    """Resolve configured data/weight paths relative to the Stage 2 directory."""
    with open(
        path,
        encoding="utf-8",
    ) as stream:
        config = yaml.safe_load(stream)
    root = Path(
        config.get(
            "root_dir",
            Path(__file__).resolve().parents[1],
        )
    ).resolve()
    config["root_dir"] = str(root)
    for section, names in {
        "data": ("manifest", "val_manifest", "geometry_stats", "feature_dir"),
        "model": (
            "vjepa_checkpoint",
            "dino_checkpoint",
            "rfdetr_checkpoint",
            "depth_checkpoint",
        ),
    }.items():
        for name in names:
            value = config.get(
                section,
                {},
            ).get(name)
            if value and not Path(value).is_absolute():
                config[section][name] = str(root / value)
    config["output_dir"] = str(root / config["output_dir"])
    return config


def read_manifest(path: str | Path) -> list[dict]:
    """Paths inside a JSONL manifest are relative to that manifest's directory."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    for row in rows:
        for name in ("frames_dir", "geometry_dir", "feature_path"):
            if row.get(name):
                row[name] = str(path.parent / row[name])
    return rows


def atomic_save(
    value: object,
    path: str | Path,
) -> None:
    """Publish an artifact only after its entire serialization succeeds."""
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(
            value,
            temporary,
        )
        os.replace(
            temporary,
            path,
        )
    finally:
        temporary.unlink(missing_ok=True)


def file_digest(path: str | Path) -> str:
    """Stream SHA256 so large pretrained checkpoints are never copied into RAM."""
    digest = hashlib.sha256()
    with open(
        path,
        "rb",
    ) as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config: dict) -> None:
    """Validate live-backbone training and configurable adapter settings."""
    if config["stage"] not in {"coarse", "fine", "joint"}:
        raise ValueError("stage must be coarse, fine, or joint")
    model = config["model"]
    if config["stage"] == "joint":
        if model.get("training_mode", "cached_features") != "cached_features":
            raise ValueError("The reviewed joint model trains from frozen feature caches")
        if not config["data"].get("feature_dir"):
            raise ValueError("Joint training requires data.feature_dir")
        return
    t_max = model.get("T_max", 32 if config["stage"] == "coarse" else 64)
    if not isinstance(t_max, int) or isinstance(t_max, bool) or t_max < 2 or t_max % 2:
        raise ValueError("T_max must be a positive even integer")
    if config["stage"] == "fine" and t_max != 64:
        raise ValueError("Fine native windows currently require T_max=64")
    rank = model["lora_rank"]
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError("lora_rank must be a positive integer")
    if model["lora_alpha"] <= 0 or not 0 <= model["lora_dropout"] < 1:
        raise ValueError("LoRA alpha must be positive and dropout must be in [0, 1)")
    count = model.get("unfreeze_last_blocks", 0)
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("unfreeze_last_blocks must be a nonnegative integer")
    if (
        model.get(
            "training_mode",
            "lora",
        )
        == "lora"
    ):
        backbone = "vjepa" if config["stage"] == "coarse" else "dino"
        if not model.get(f"{backbone}_factory"):
            raise ValueError(
                f"Configure a local {backbone}_factory; cached_features is an explicit ablation only"
            )
    elif model["training_mode"] != "cached_features":
        raise ValueError("training_mode must be lora or cached_features")
