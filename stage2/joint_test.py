"""Run the single-stage model on precomputed joint feature caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stage2.data.joint import joint_collate
from stage2.model.joint import JointStage2Model
from stage2.test import evasion_label, side_label
from stage2.utils.joint_losses import constrained_decode
from stage2.utils.utils import read_manifest


def load_model(path: str, device: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("format_version") != 2
        or checkpoint["config"]["stage"] != "joint"
    ):
        raise ValueError("Expected a Stage 2 joint-format checkpoint")
    model = JointStage2Model(checkpoint["config"]["model"].get("geometry_dim", 13))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval().to(device)


def cache_item(row: dict, feature_dir: Path):
    cache = torch.load(
        feature_dir / f"{row['sample_id']}.pt",
        map_location="cpu",
        weights_only=True,
    )
    if cache.get("schema") != 1 or cache.get("sample_id") != row["sample_id"]:
        raise ValueError(f"Invalid feature cache for {row['sample_id']}")
    length = len(cache["frame_ids"])
    global_length = len(cache["global_anchor"])
    return {
        "scene_features": cache["scene_features"],
        "roi_features": cache["roi_features"],
        "geometry": cache["geometry"].float(),
        "object_valid": cache["object_valid"].bool(),
        "time_valid": torch.ones(length, dtype=torch.bool),
        "local_time": torch.linspace(0, 1, length),
        "frame_seconds": torch.arange(length, dtype=torch.float32)
        / float(row.get("native_fps") or 1.0),
        "global_features": cache["global_features"],
        "global_time": (cache["global_anchor"] / max(1, length - 1)).float(),
        "global_valid": torch.ones(global_length, dtype=torch.bool),
        # Collation keeps a single contract; these values are unused at inference.
        "entry_index": 0,
        "entry_supervised": False,
        "collision_index": 0,
        "entry_side": 0,
        "evasion": 0.0,
        "frame_ids": cache["frame_ids"].long(),
        "sample_id": row["sample_id"],
        "source_id": row.get("source_id", row["sample_id"]),
    }


@torch.inference_mode()
def predict(model, item, device):
    batch = joint_collate([item])
    tensor_batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    output = model(tensor_batch)
    entry, collision = constrained_decode(
        output["entry_logits"], output["collision_logits"]
    )
    frame_ids = item["frame_ids"]
    return {
        "entry_frame": int(frame_ids[int(entry)].item()),
        "collision_frame": int(frame_ids[int(collision)].item()),
        "entry_side": side_label(output["side_logits"][0].float().cpu().numpy()),
        "evasion_space": evasion_label(float(output["evasion_logits"][0].item())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model = load_model(args.checkpoint, args.device)
    feature_dir = Path(args.feature_dir)
    with open(args.output, "x", encoding="utf-8") as stream:
        for row in read_manifest(args.manifest):
            result = predict(model, cache_item(row, feature_dir), args.device)
            stream.write(json.dumps({"sample_id": row["sample_id"], **result}) + "\n")


if __name__ == "__main__":
    main()
