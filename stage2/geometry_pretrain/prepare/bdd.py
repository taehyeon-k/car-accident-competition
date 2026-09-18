"""Prepare BDD100K 100K-image geometry samples straight from the official zips.

Real annotations only (no teachers here):
  * drivable map (0 background, 1 direct/ego lane, 2 alternative)  <- drivable_maps
  * lane-marking mask (parallel lane markings)                       <- lane poly2d
  * road-curb mask                                                    <- lane/road curb
  * object boxes in the compact taxonomy (masks come later from SAM)  <- box2d
  * vanishing point derived from parallel lane lines                  <- lane poly2d
Images are stored once at the 448x800 model resolution to bound disk and CPU.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from stage2.geometry_pretrain.common import (
    BDD_OBJECT_MAP,
    IGNORE,
    INPUT_HW,
    LABEL_HW,
    DiskGuard,
    atomic_json,
    atomic_savez,
    bdd_poly_points,
    fit_lower_line,
    robust_vanishing_point,
    to_input,
    write_jpeg,
    write_jsonl,
)

SRC_W, SRC_H = 1280, 720
_Z = {}


def _zip(path):
    if path not in _Z:
        _Z[path] = zipfile.ZipFile(path)
    return _Z[path]


def render_lanes(objects):
    """Return lane, curb masks at label resolution plus lines for the VP fit."""
    sx, sy = LABEL_HW[1] / SRC_W, LABEL_HW[0] / SRC_H
    lane = np.zeros(LABEL_HW, np.uint8)
    curb = np.zeros(LABEL_HW, np.uint8)
    lane_ignore = np.zeros(LABEL_HW, np.uint8)
    lines = []
    for obj in objects:
        cat = obj.get("category", "")
        if not cat.startswith("lane/") or not obj.get("poly2d"):
            continue
        pts = bdd_poly_points(obj["poly2d"])
        scaled = np.round(pts * [sx, sy]).astype(np.int32)[None]
        direction = (obj.get("attributes") or {}).get("direction", "parallel")
        if cat == "lane/road curb":
            cv2.polylines(curb, scaled, False, 1, 1)
        elif cat == "lane/crosswalk" or direction == "vertical":
            # Painted but not a lane boundary: neither positive nor negative.
            cv2.polylines(lane_ignore, scaled, False, 1, 2)
        else:
            cv2.polylines(lane, scaled, False, 1, 1)
        if direction == "parallel" and cat != "lane/crosswalk":
            line = fit_lower_line(pts, SRC_H)
            if line is not None:
                lines.append(line)
    lane[(lane_ignore > 0) & (lane == 0)] = IGNORE
    vp = robust_vanishing_point(lines, SRC_W, SRC_H)
    return lane, curb, vp


def process(task):
    name, split, args = task
    out_img = Path(args["image_root"]) / split / f"{name}.jpg"
    out_road = Path(args["label_root"]) / f"{name}.road.npz"
    out_meta = Path(args["label_root"]) / f"{name}.meta.json"
    if out_img.is_file() and out_road.is_file() and out_meta.is_file():
        meta = json.loads(out_meta.read_text())
    else:
        labels = json.loads(_zip(args["labels_zip"]).read(f"100k/{split}/{name}.json"))
        frame = labels["frames"][0]
        objects = frame.get("objects", [])
        raw = np.frombuffer(_zip(args["images_zip"]).read(f"100k/{split}/{name}.jpg"), np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (SRC_H, SRC_W):
            return None
        write_jpeg(out_img, to_input(bgr))
        drv_name = f"labels/{split}/{name}_drivable_id.png"
        try:
            drv = cv2.imdecode(
                np.frombuffer(_zip(args["drivable_zip"]).read(drv_name), np.uint8), cv2.IMREAD_UNCHANGED
            )
            # Majority vote at stride 4 via area-weighted one-hot.
            onehot = np.stack([(drv == c).astype(np.float32) for c in range(3)], -1)
            frac = cv2.resize(onehot, (LABEL_HW[1], LABEL_HW[0]), interpolation=cv2.INTER_AREA)
            drivable = frac.argmax(-1).astype(np.uint8)
            # BDD ids are 0 background / 1 direct / 2 alternative, identical to ours.
        except KeyError:
            drivable = np.full(LABEL_HW, IGNORE, np.uint8)
        lane, curb, vp = render_lanes(objects)
        atomic_savez(out_road, drivable=drivable, lane=lane, curb=curb)
        sx, sy = INPUT_HW[1] / SRC_W, INPUT_HW[0] / SRC_H
        boxes = []
        for obj in objects:
            cls = BDD_OBJECT_MAP.get(obj.get("category"))
            if cls is None or not obj.get("box2d"):
                continue
            b = obj["box2d"]
            attrs = obj.get("attributes") or {}
            boxes.append(
                {
                    "cls": cls,
                    "box": [b["x1"] * sx, b["y1"] * sy, b["x2"] * sx, b["y2"] * sy],
                    "occluded": bool(attrs.get("occluded", False)),
                    "truncated": bool(attrs.get("truncated", False)),
                    "source": "bdd_human",
                }
            )
        meta = {
            "attributes": labels.get("attributes", {}),
            "boxes": boxes,
            "vp": None if vp is None else {"x": vp[0], "y": vp[1], "support": vp[2], "source": "derived_bdd_lanes"},
        }
        atomic_json(out_meta, meta)
    return {
        "id": f"bdd100k/{name}",
        "source": "bdd100k",
        "split": "train" if split == "train" else "val",
        "group": f"bdd100k/{name}",  # each 100K image is a keyframe of a distinct video
        "image": str(out_img),
        "road": str(out_road),
        "road_label": "human",
        "meta": str(out_meta),
        "fps": None,
        "attributes": meta["attributes"],
        "has_vp": meta["vp"] is not None,
        "num_boxes": len(meta["boxes"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zips", default="/workspace/data/geometry/bdd100k/zips")
    parser.add_argument("--image-root", default="/workspace/data/geometry/processed/bdd100k")
    parser.add_argument("--label-root", default="/workspace/cache/geometry_pretrain/labels/bdd100k")
    parser.add_argument("--manifest-dir", default="/workspace/cache/geometry_pretrain/manifests")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--min-free-gb", type=float, default=12.0)
    args = parser.parse_args()
    guard = DiskGuard(args.image_root, args.min_free_gb)
    guard.check("before BDD preparation")
    zips = Path(args.zips)
    shared = {
        "labels_zip": str(zips / "bdd100k_labels.zip"),
        "images_zip": str(zips / "bdd100k_images_100k.zip"),
        "drivable_zip": str(zips / "bdd100k_drivable_maps.zip"),
        "image_root": args.image_root,
        "label_root": args.label_root,
    }
    names = zipfile.ZipFile(shared["labels_zip"]).namelist()
    for split, limit in (("val", args.max_val), ("train", args.max_train)):
        items = sorted(n.split("/")[-1][:-5] for n in names if n.startswith(f"100k/{split}/") and n.endswith(".json"))
        if limit:
            items = items[:limit]
        rows = []
        with Pool(args.workers) as pool:
            for i, row in enumerate(
                tqdm(pool.imap(process, [(n, split, shared) for n in items], chunksize=32), total=len(items), desc=split)
            ):
                if row is not None:
                    rows.append(row)
                if i % 5000 == 0:
                    guard.check("during BDD preparation")
        write_jsonl(Path(args.manifest_dir) / f"bdd100k_{split}.jsonl", rows)
        print(split, len(rows), "vp:", sum(r["has_vp"] for r in rows))


if __name__ == "__main__":
    main()
