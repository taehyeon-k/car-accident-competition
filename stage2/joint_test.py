"""Run the joint model using cached local features and online frozen V-JEPA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stage2.data.joint import joint_collate, joint_item, load_joint_cache
from stage2.model.joint_system import JointSystem
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
    model = JointSystem(checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval().to(device)


def cache_item(row: dict, feature_dir: Path):
    cache, paths = load_joint_cache(row, feature_dir)
    return joint_item(row, cache, paths)


@torch.inference_mode()
def predict(model, item, device):
    batch = joint_collate([item])
    tensor_batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(device).type == "cuda",
    ):
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
