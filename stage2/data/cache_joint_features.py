"""Cache frozen DINOv3 local features and sparse V-JEPA 2.1 context.

The output is one ``.pt`` file per complete video. Detector/depth observations
must already exist in each manifest row's ``geometry_dir``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import ImageReadMode, read_image
from torchvision.ops import roi_align

from stage2.data.cache_geometry import frame_paths
from stage2.data.transforms import box_to_grid, letterbox
from stage2.model.backbones import load_local
from stage2.model.tracking import Detection, HungarianTracker
from stage2.utils.utils import atomic_save, file_digest, load_config, read_manifest


def clip_positions(length: int, span: int = 61, stride: int = 48) -> list[torch.Tensor]:
    """Return 16-frame native-stride-4 clips with deterministic tail coverage."""
    if length < 1:
        raise ValueError("A video must contain at least one frame")
    if length < span:
        return [torch.linspace(0, length - 1, 16).round().long()]
    starts = list(range(0, length - span + 1, stride))
    starts.append(length - span)
    return [start + torch.arange(16) * 4 for start in sorted(set(starts))]


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


def object_tensors(tracks, sizes, fps: float, max_objects: int = 12):
    """Build frame-local top objects with history-only motion geometry."""
    length = len(sizes)
    boxes = torch.zeros(length, max_objects, 4)
    geometry = torch.zeros(length, max_objects, 13)
    valid = torch.zeros(length, max_objects, dtype=torch.bool)
    by_frame: list[list[tuple]] = [[] for _ in range(length)]
    for track in tracks:
        history = []
        for t, detection in sorted(track.observations.items()):
            width, height = sizes[t]
            box = np.asarray(detection.box, dtype=np.float32)
            bw, bh = box[2] - box[0], box[3] - box[1]
            area = float(bw * bh / (width * height))
            previous = history[-1] if history else None
            if previous is None:
                dx = dy = dlog = slope = 0.0
            else:
                pt, pbox, parea = previous
                dt = max((t - pt) / fps, 1e-6)
                center = (box[:2] + box[2:]) / 2
                pcenter = (pbox[:2] + pbox[2:]) / 2
                dx = float((center[0] - pcenter[0]) / width / dt)
                dy = float((center[1] - pcenter[1]) / height / dt)
                dlog = float(np.log(area + 1e-8) - np.log(parea + 1e-8))
                slope = dlog / dt
            continuity = min(1.0, (len(history) + 1) / 5)
            priority = (
                0.35 * area
                + 0.25 * float(box[3] / height)
                + 0.20 * float(detection.score)
                + 0.20 * continuity
            )
            values = [
                float((box[0] + box[2]) / (2 * width)),
                float(box[3] / height),
                float(bw / width),
                float(bh / height),
                area,
                float(np.clip((detection.proximity or 0.0) / 5, -1, 1)),
                0.0,
                float(np.clip(dlog, -2, 2)),
                float(np.clip(slope, -5, 5)),
                float(np.clip(dx, -5, 5)),
                float(np.clip(dy, -5, 5)),
                float(detection.score),
                continuity,
            ]
            by_frame[t].append((priority, detection.proximity, box, values))
            history.append((t, box, area))
    for t, candidates in enumerate(by_frame):
        selected = sorted(candidates, key=lambda x: x[0], reverse=True)[:max_objects]
        proximity_order = {
            id(item): rank / max(1, len(selected) - 1)
            for rank, item in enumerate(
                sorted(selected, key=lambda x: float(x[1] or 0.0))
            )
        }
        for slot, item in enumerate(selected):
            _, _, box, values = item
            values[6] = proximity_order[id(item)]
            boxes[t, slot] = torch.from_numpy(box)
            geometry[t, slot] = torch.tensor(values)
            valid[t, slot] = True
    return boxes, geometry, valid


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
            enabled=device != "cpu",
        ):
            output = model.forward_features(images)
        cls = output["x_norm_clstoken"]
        dense = output["x_norm_patchtokens"].reshape(len(chunk), 24, 24, 768)
        cells = F.adaptive_avg_pool2d(dense.permute(0, 3, 1, 2), (2, 3))
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
                batch_rois[offset, : len(selected)] = pooled[
                    cursor : cursor + len(selected)
                ]
                cursor += len(selected)
        scenes.append(scene.cpu().half())
        rois.append(batch_rois.cpu().half())
    return torch.cat(scenes), torch.cat(rois)


@torch.inference_mode()
def extract_global(model, paths, device):
    features, anchors, support = [], [], []
    for positions in clip_positions(len(paths)):
        frames = [
            letterbox(read_image(str(paths[int(i)]), mode=ImageReadMode.RGB), 384)[0]
            for i in positions
        ]
        video = torch.stack(frames, dim=1)[None].to(device)
        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=device != "cpu",
        ):
            output = model(video)
        if isinstance(output, dict):
            output = output.get("dense", output.get("x"))
        tokens = output.reshape(1, 8, 24, 24, 1024).mean(dim=(2, 3))[0]
        pair_anchors = positions.reshape(8, 2).float().mean(1)
        features.append(tokens.cpu().half())
        anchors.append(pair_anchors)
        support.append(
            torch.tensor([[int(positions.min()), int(positions.max())]]).repeat(8, 1)
        )
    return torch.cat(features), torch.cat(anchors), torch.cat(support)


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
    prepared = []
    for row in rows:
        paths, frame_ids = frame_paths(row["frames_dir"])
        tracks, sizes = _observations(row, frame_ids, config["tracking"])
        boxes, geometry, valid = object_tensors(tracks, sizes, float(row["native_fps"]))
        prepared.append((row, paths, frame_ids, boxes, geometry, valid))

    dino = load_local(model_config["dino_factory"], dino_path).eval().to(device)
    if device != "cpu":
        dino.to(dtype=torch.bfloat16)
    local_values = []
    for row, paths, _, boxes, _, valid in prepared:
        local_values.append(
            extract_local(
                dino,
                paths,
                boxes,
                valid,
                device,
                model_config.get("dino_batch_size", 8),
            )
        )
        print(json.dumps({"sample_id": row["sample_id"], "branch": "dino_v3"}))
    del dino
    if device != "cpu":
        torch.cuda.empty_cache()

    vjepa = (
        load_local(
            model_config["vjepa_factory"],
            model_config["vjepa_checkpoint"],
            checkpoint_key=model_config.get("vjepa_checkpoint_key", "ema_encoder"),
        )
        .eval()
        .to(device)
    )
    if device != "cpu":
        vjepa.to(dtype=torch.bfloat16)
    provenance = {
        "dino_sha256": file_digest(dino_path),
        "vjepa_sha256": file_digest(model_config["vjepa_checkpoint"]),
        "sampling": "16 frames, native stride 4, start stride 48, deterministic tail",
    }
    for prepared_item, local in zip(prepared, local_values):
        row, paths, frame_ids, _, geometry, valid = prepared_item
        global_features, global_anchor, global_support = extract_global(
            vjepa, paths, device
        )
        scene, roi = local
        atomic_save(
            {
                "schema": 1,
                "sample_id": row["sample_id"],
                "frame_ids": torch.tensor(frame_ids),
                "scene_features": scene,
                "roi_features": roi,
                "geometry": geometry.half(),
                "object_valid": valid,
                "global_features": global_features,
                "global_anchor": global_anchor,
                "global_support": global_support,
                "provenance": provenance,
            },
            output_dir / f"{row['sample_id']}.pt",
        )
        print(json.dumps({"sample_id": row["sample_id"], "branch": "complete"}))


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
