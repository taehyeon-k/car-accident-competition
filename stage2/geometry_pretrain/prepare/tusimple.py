"""Prepare TuSimple clips: real lane labels on frame 20 plus temporal pairs.

Only frames 12, 17, 19 and 20 of each 20-frame (20 fps) clip are fetched,
giving flow pairs with dt = 50 ms (19->20), 150 ms (17->20) and 400 ms (12->20).
Split is at clip level: official train_set -> train, a fixed random subset of
600 official test_set clips (with test_label.json) -> held-out validation.

Ego-lane labels are inferred conservatively from the real lane annotations:
the nearest lane left and right of the bottom-center camera position must both
reach the lower image region with a plausible width; otherwise ego-lane
supervision is masked (all IGNORE), never guessed.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from stage2.geometry_pretrain.common import (
    IGNORE,
    LABEL_HW,
    DiskGuard,
    atomic_json,
    atomic_savez,
    fit_lower_line,
    robust_vanishing_point,
    to_input,
    write_jpeg,
    write_jsonl,
)

SRC_W, SRC_H = 1280, 720
FPS = 20.0
PAIRS = ((19, 20), (17, 20), (12, 20))


def lane_arrays(label):
    lanes = []
    hs = np.asarray(label["h_samples"], np.float32)
    for xs in label["lanes"]:
        xs = np.asarray(xs, np.float32)
        ok = xs >= 0
        if ok.sum() >= 2:
            lanes.append(np.stack([xs[ok], hs[ok]], 1))
    return lanes


def ego_lane_mask(lanes):
    """Polygon between nearest left/right lanes at the bottom, or None."""
    cx = SRC_W / 2
    cands = []
    for pts in lanes:
        bottom = pts[pts[:, 1].argmax()]
        if bottom[1] < 0.72 * SRC_H:  # lane must reach the lower image region
            continue
        # Extrapolate linearly to the image bottom from the lowest points.
        line = fit_lower_line(pts, SRC_H)
        if line is None:
            continue
        x_bottom = line[0] * (SRC_H - 1) + line[1]
        cands.append((x_bottom, pts, line))
    left = [c for c in cands if c[0] < cx]
    right = [c for c in cands if c[0] > cx]
    if not left or not right:
        return None
    lx, lpts, lline = max(left, key=lambda c: c[0])
    rx, rpts, rline = min(right, key=lambda c: c[0])
    width = rx - lx
    if not (250 < width < 1300):
        return None
    y_top = max(lpts[:, 1].min(), rpts[:, 1].min())
    ys = np.linspace(y_top, SRC_H - 1, 40)

    def x_at(pts, line, y):
        if y <= pts[:, 1].max():
            order = np.argsort(pts[:, 1])
            return np.interp(y, pts[order, 1], pts[order, 0])
        return line[0] * y + line[1]

    left_poly = [(x_at(lpts, lline, y), y) for y in ys]
    right_poly = [(x_at(rpts, rline, y), y) for y in ys[::-1]]
    poly = np.asarray(left_poly + right_poly, np.float32) * [LABEL_HW[1] / SRC_W, LABEL_HW[0] / SRC_H]
    mask = np.zeros(LABEL_HW, np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    return mask


def process(task):
    label, split, args = task
    raw = label["raw_file"]  # clips/<date>/<clip>/20.jpg
    clip_dir = raw.rsplit("/", 1)[0]
    clip_key = clip_dir.replace("clips/", "").replace("/", "_")
    base = Path(args["root"]) / "TUSimple" / ("train_set" if split == "train" else "test_set")
    images = {}
    for f in sorted({f for p in PAIRS for f in p}):
        out = Path(args["image_root"]) / clip_key / f"{f}.jpg"
        if not out.is_file():
            src = base / clip_dir / f"{f}.jpg"
            bgr = cv2.imread(str(src))
            if bgr is None:
                return None
            write_jpeg(out, to_input(bgr))
        images[f] = str(out)
    road_path = Path(args["label_root"]) / f"{clip_key}.road.npz"
    meta_path = Path(args["label_root"]) / f"{clip_key}.meta.json"
    lanes = lane_arrays(label)
    if not road_path.is_file() or not meta_path.is_file():
        lane = np.zeros(LABEL_HW, np.uint8)
        s = np.asarray([LABEL_HW[1] / SRC_W, LABEL_HW[0] / SRC_H], np.float32)
        for pts in lanes:
            cv2.polylines(lane, [np.round(pts * s).astype(np.int32)[None]], False, 1, 2)
        ego = ego_lane_mask(lanes)
        drivable = np.full(LABEL_HW, IGNORE, np.uint8)
        if ego is not None:
            drivable[ego > 0] = 1
        curb = np.full(LABEL_HW, IGNORE, np.uint8)  # TuSimple has no curb labels
        atomic_savez(road_path, drivable=drivable, lane=lane, curb=curb)
        lines = [l for l in (fit_lower_line(p, SRC_H) for p in lanes) if l is not None]
        vp = robust_vanishing_point(lines, SRC_W, SRC_H)
        atomic_json(
            meta_path,
            {
                "boxes": [],
                "ego_lane": ego is not None,
                "vp": None if vp is None else {"x": vp[0], "y": vp[1], "support": vp[2], "source": "derived_tusimple_lanes"},
            },
        )
    meta = json.loads(meta_path.read_text())
    frame = {
        "id": f"tusimple/{clip_key}/20",
        "source": "tusimple",
        "split": split,
        "group": f"tusimple/{clip_key}",
        "image": images[20],
        "road": str(road_path),
        "road_label": "human_lane+derived_ego",
        "meta": str(meta_path),
        "fps": FPS,
        "has_vp": meta["vp"] is not None,
        "ego_lane": meta["ego_lane"],
    }
    pairs = [
        {
            "id": f"tusimple/{clip_key}/{a}-{b}",
            "source": "tusimple",
            "split": split,
            "group": f"tusimple/{clip_key}",
            "image_t": images[a],
            "image_tplus": images[b],
            "dframes": b - a,
            "dt": (b - a) / FPS,
        }
        for a, b in PAIRS
    ]
    return frame, pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/data/geometry/tusimple")
    parser.add_argument("--image-root", default="/workspace/data/geometry/processed/tusimple")
    parser.add_argument("--label-root", default="/workspace/cache/geometry_pretrain/labels/tusimple")
    parser.add_argument("--manifest-dir", default="/workspace/cache/geometry_pretrain/manifests")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    DiskGuard(args.image_root).check("before TuSimple preparation")
    root = Path(args.root) / "TUSimple"
    train = []
    for f in sorted((root / "train_set").glob("label_data_*.json")):
        train += [json.loads(l) for l in open(f)]
    test = [json.loads(l) for l in open(root / "test_label.json")]
    available = {p.parent.relative_to(root / "test_set").as_posix() for p in (root / "test_set/clips").glob("*/*/20.jpg")}
    test = [t for t in test if t["raw_file"].rsplit("/", 1)[0] in available]
    shared = {"root": args.root, "image_root": args.image_root, "label_root": args.label_root}
    for split, labels in (("train", train), ("val", test)):
        labels = labels[: args.limit] if args.limit else labels
        frames, pairs = [], []
        with Pool(args.workers) as pool:
            for out in tqdm(pool.imap(process, [(l, split, shared) for l in labels], chunksize=16), total=len(labels), desc=split):
                if out is None:
                    continue
                frames.append(out[0])
                pairs.extend(out[1])
        write_jsonl(Path(args.manifest_dir) / f"tusimple_{split}.jsonl", frames)
        write_jsonl(Path(args.manifest_dir) / f"tusimple_{split}_pairs.jsonl", pairs)
        print(split, len(frames), "frames", len(pairs), "pairs", "ego:", sum(f["ego_lane"] for f in frames), "vp:", sum(f["has_vp"] for f in frames))


if __name__ == "__main__":
    main()
