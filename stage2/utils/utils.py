"""Configuration, manifest paths and reproducible local artifact handling."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
import yaml

from stage2.model.joint import head_config
from stage2.model.joint_tracking import GEOMETRY_DIM


def source_name(source_id: str) -> str:
    """Dataset name written as the ``source_id`` prefix by ``prepare_workspace``.

    Identifiers without that prefix return unchanged, so per-source reporting
    degrades to one bucket per group rather than failing.
    """
    return str(source_id).split(":", 1)[0]


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
        "data": ("manifest", "val_manifest"),
        "model": (
            "vjepa_checkpoint",
            "dino_checkpoint",
            "rfdetr_checkpoint",
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


def _probability(value, label: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be a probability in [0, 1]")
    return value


def _range(section: dict, low: str, high: str, label: str, floor=None, ceiling=None):
    minimum, maximum = float(section[low]), float(section[high])
    if minimum > maximum:
        raise ValueError(f"{label}: {low} must not exceed {high}")
    if floor is not None and minimum < floor:
        raise ValueError(f"{label}: {low} must be at least {floor}")
    if ceiling is not None and maximum > ceiling:
        raise ValueError(f"{label}: {high} must not exceed {ceiling}")


def validate_augmentation(config: dict) -> None:
    """Check every augmentation probability and range the YAML can set."""
    from stage2.data.augment import augmentation_config

    resolved = augmentation_config(config.get("augmentation"))
    _probability(resolved["horizontal_flip"]["probability"], "horizontal_flip")
    photometric = resolved["photometric"]
    for name, section in photometric.items():
        _probability(section["probability"], f"augmentation.photometric.{name}")
    for name, low, high in (
        ("brightness", "factor_min", "factor_max"),
        ("contrast", "factor_min", "factor_max"),
        ("gamma", "gamma_min", "gamma_max"),
        ("saturation", "factor_min", "factor_max"),
    ):
        _range(
            photometric[name], low, high, f"augmentation.photometric.{name}", floor=0
        )
    _range(
        photometric["jpeg"],
        "quality_min",
        "quality_max",
        "augmentation.photometric.jpeg",
        floor=1,
        ceiling=100,
    )
    _range(
        photometric["gaussian_noise"],
        "std_min",
        "std_max",
        "augmentation.photometric.gaussian_noise",
        floor=0,
    )
    blur = photometric["gaussian_blur"]
    _range(
        blur,
        "sigma_min",
        "sigma_max",
        "augmentation.photometric.gaussian_blur",
        floor=0,
    )
    kernel = blur["kernel_size"]
    if (
        not isinstance(kernel, int)
        or isinstance(kernel, bool)
        or kernel < 1
        or not kernel % 2
    ):
        raise ValueError("gaussian_blur.kernel_size must be an odd positive integer")


def validate_memory_cap(config: dict) -> None:
    """``training_memory.max_frames`` must be null or a positive integer."""
    from stage2.data.joint_sampling import max_train_frames

    max_train_frames(config.get("training_memory"))


def validate_temporal(config: dict) -> None:
    """The three temporal modes must be probabilities summing to one."""
    from stage2.data.joint_sampling import temporal_probabilities

    resolved = temporal_probabilities(config.get("temporal_augmentation"))
    for name, value in resolved.items():
        _probability(value, f"temporal_augmentation.{name}")
    total = sum(resolved.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"temporal_augmentation probabilities must sum to 1, got {total:.6f}"
        )


def validate_config(config: dict) -> None:
    """Validate joint training settings, LoRA wiring and augmentation ranges."""
    if config["stage"] != "joint":
        raise ValueError("stage must be joint")
    model = config["model"]
    if model.get("training_mode") != "online_lora":
        raise ValueError(
            "Joint training runs DINOv3 and V-JEPA online with LoRA; set "
            "model.training_mode: online_lora"
        )
    for backbone in ("vjepa", "dino"):
        if not model.get(f"{backbone}_factory") or not model.get(
            f"{backbone}_checkpoint"
        ):
            raise ValueError(
                f"Joint training requires a local {backbone}_factory and {backbone}_checkpoint"
            )
        blocks = model.get(f"{backbone}_lora_blocks", 4)
        if not isinstance(blocks, int) or isinstance(blocks, bool) or blocks < 1:
            raise ValueError(f"{backbone}_lora_blocks must be a positive integer")
    rank = model.get("lora_rank", 8)
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError("lora_rank must be a positive integer")
    if float(model.get("lora_alpha", 16)) <= 0:
        raise ValueError("lora_alpha must be positive")
    if not 0 <= float(model.get("lora_dropout", 0.05)) < 1:
        raise ValueError("lora_dropout must be in [0, 1)")
    if model.get("geometry_dim", GEOMETRY_DIM) != GEOMETRY_DIM:
        raise ValueError(f"Joint object geometry uses {GEOMETRY_DIM} channels")
    # Raises on a bad hidden_dim/heads divisor or a roi+geometry mismatch.
    head_config(model)
    optimization = config["optimization"]
    for name in ("dino_lora_lr", "vjepa_lora_lr", "new_lr"):
        if name not in optimization:
            raise ValueError(f"optimization.{name} is required")
        if float(optimization[name]) < 0:
            raise ValueError(f"optimization.{name} must be nonnegative")
    if config["logging"].get("checkpoint_metric", "loss") not in {
        "loss",
        "competition_score",
    }:
        raise ValueError("checkpoint_metric must be loss or competition_score")
    validate_temporal(config)
    validate_memory_cap(config)
    validate_augmentation(config)
