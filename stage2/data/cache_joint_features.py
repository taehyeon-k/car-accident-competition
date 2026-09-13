"""Cache frozen DINOv3 local features with video-global persistent track slots.

The output is one ``.pt`` file per complete video. Detector/depth observations
must already exist in each manifest row's ``geometry_dir``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.io import ImageReadMode, read_image
from torchvision.ops import roi_align

from stage2.data.cache_geometry import frame_paths
from stage2.data.transforms import box_to_grid, letterbox
from stage2.model.backbones import load_local
from stage2.model.tracking import Detection, HungarianTracker
from stage2.model.joint_tracking import object_tensors
from stage2.utils.utils import atomic_save, file_digest, load_config, read_manifest


def _observations(row: dict, frame_ids: list[int], tracking: dict):
    frames, sizes = [], []
    for frame_id in frame_ids:
        saved = torch.load(
            Path(row["geometry_dir"]) / f"{frame_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        sizes.append(tuple(saved["size"]))
        frames.append(
            [
                Detection(box.numpy(), float(score), label, float(proximity))
                for box, score, label, proximity in zip(
                    saved["boxes"],
                    saved["scores"],
                    saved["labels"],
                    saved["proximity"],
                )
            ]
        )
    tracker = HungarianTracker(
        max_gap=tracking.get("max_gap", 2),
        max_center_distance=tracking.get("max_center_distance", 0.2),
        max_match_cost=tracking.get("max_match_cost", 0.65),
        detection_threshold=tracking.get("detection_threshold", 0.2),
    )
    return tracker.track(frames, sizes), sizes


@torch.inference_mode()
def extract_local(model, paths, boxes, valid, device, batch_size):
    scenes, rois = [], []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        transformed, metadata = zip(
            *(
                letterbox(read_image(str(path), mode=ImageReadMode.RGB), 384)
                for path in chunk
            )
        )
        images = torch.stack(transformed).to(device)
        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ):
            output = model.forward_features(images)
        cls = output["x_norm_clstoken"]
        dense = output["x_norm_patchtokens"].reshape(len(chunk), 24, 24, 768)
        cells = F.adaptive_avg_pool2d(dense.permute(0, 3, 1, 2), (4, 4))
        scene = torch.cat((cls[:, None], cells.flatten(2).transpose(1, 2)), dim=1)
        batch_rois = dense.new_zeros(len(chunk), 12, 768)
        roi_boxes = []
        for offset, transform in enumerate(metadata):
            selected = boxes[start + offset][valid[start + offset]]
            roi_boxes.append(box_to_grid(selected, transform, 16).to(device))
        if sum(len(x) for x in roi_boxes):
            pooled = roi_align(
                dense.permute(0, 3, 1, 2),
                roi_boxes,
                output_size=1,
                spatial_scale=1.0,
                aligned=True,
            ).flatten(1)
            cursor = 0
            for offset, selected in enumerate(roi_boxes):
                batch_rois[offset, valid[start + offset].to(device)] = pooled[
                    cursor : cursor + len(selected)
                ]
                cursor += len(selected)
        scenes.append(scene.cpu().half())
        rois.append(batch_rois.cpu().half())
    return torch.cat(scenes), torch.cat(rois)


def cache_manifest(config: dict, manifest: str, device: str, limit: int | None = None):
    model_config = config["model"]
    dino_path = Path(model_config["dino_checkpoint"])
    if not dino_path.is_file():
        raise FileNotFoundError(
            f"Official DINOv3 weights are missing: {dino_path}. "
            "Accept the facebook/dinov3-vitb16-pretrain-lvd1689m license and download the checkpoint."
        )
    rows = read_manifest(manifest)[:limit]
    output_dir = Path(config["data"]["feature_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    dino = load_local(model_config["dino_factory"], dino_path).eval().to(device)
    if torch.device(device).type == "cuda":
        dino.to(dtype=torch.bfloat16)
    provenance = {
        "dino_sha256": file_digest(dino_path),
        "scene_grid": [4, 4],
        "tracking": config["tracking"],
        "geometry_units": "per_frame",
        "track_selection": "video_global_percentile",
    }
    # Stream one video at a time rather than retaining the whole dataset in RAM.
    for row in rows:
        paths, frame_ids = frame_paths(row["frames_dir"])
        tracks, sizes = _observations(row, frame_ids, config["tracking"])
        boxes, geometry, valid, track_ids = object_tensors(
            tracks,
            sizes,
            track_percentile=config["tracking"].get("track_percentile", 90),
            return_track_ids=True,
        )
        scene, roi = extract_local(
            dino, paths, boxes, valid, device, model_config.get("dino_batch_size", 8)
        )
        atomic_save(
            {
                "schema": 2,
                "sample_id": row["sample_id"],
                "frame_ids": torch.tensor(frame_ids),
                "track_ids": track_ids,
                "scene_features": scene,
                "roi_features": roi,
                "geometry": geometry.half(),
                "object_valid": valid,
                "provenance": provenance,
            },
            output_dir / f"{row['sample_id']}.pt",
        )
        print(json.dumps({"sample_id": row["sample_id"], "branch": "local_complete"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cache_manifest(load_config(args.config), args.manifest, args.device, args.limit)


if __name__ == "__main__":
    main()
