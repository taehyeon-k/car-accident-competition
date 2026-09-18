"""Sample frames and temporal pairs from locally available dashcam footage.

Sources:
  * ``accident``: the Stage-2 CCD / Nexar / AI-Hub videos, from the already
    decoded native-FPS frame directories. Only the video frames are used; the
    four Stage-2 labels are never read. Stage-2 ``train`` videos feed
    pretraining; Stage-2 ``val`` videos are held out for geometry validation,
    which keeps the downstream fixed-split comparison free of any exposure.
  * ``baton``: comma.ai qcamera dashcam routes used by Stage 3 (20 fps),
    split by route.

Static anchors are sampled sparsely in *time* (``--anchor-hz``, capped per
video) so long videos cannot dominate. Each anchor gets temporal partners at
dt ~ 1 frame, ~150 ms and ~400 ms, converted to frames with each video's FPS.
"""

from __future__ import annotations

import argparse
import hashlib
from multiprocessing import Pool
from pathlib import Path

import av
import cv2
import numpy as np
from tqdm import tqdm

from stage2.geometry_pretrain.common import DiskGuard, read_jsonl, to_input, write_jpeg, write_jsonl

TARGET_DT = (None, 0.15, 0.40)  # None = one native frame


def partner_offsets(fps: float) -> list[int]:
    offsets = []
    for dt in TARGET_DT:
        k = 1 if dt is None else max(1, int(round(dt * fps)))
        if k not in offsets:
            offsets.append(k)
    return offsets


def anchor_indices(num_frames: int, fps: float, hz: float, cap: int, max_offset: int) -> list[int]:
    usable = num_frames - max_offset
    if usable <= 0:
        return []
    step = max(1, int(round(fps / hz)))
    idx = list(range(step // 2, usable, step))
    if len(idx) > cap:
        idx = [idx[int(round(i))] for i in np.linspace(0, len(idx) - 1, cap)]
    return idx


def rows_for(source, key, split, group, fps, anchors, offsets, image_of):
    frames, pairs = [], []
    for a in anchors:
        frames.append(
            {
                "id": f"{source}/{key}/{a}",
                "source": source,
                "split": split,
                "group": group,
                "image": image_of(a),
                "road": None,
                "road_label": "pseudo",
                "meta": None,
                "fps": fps,
                "frame_index": a,
                "time": a / fps,
            }
        )
        for k in offsets:
            pairs.append(
                {
                    "id": f"{source}/{key}/{a}-{a + k}",
                    "source": source,
                    "split": split,
                    "group": group,
                    "image_t": image_of(a),
                    "image_tplus": image_of(a + k),
                    "dframes": k,
                    "dt": k / fps,
                }
            )
    return frames, pairs


def process_accident(task):
    rec, split, args = task
    fps = float(rec["native_fps"])
    offsets = partner_offsets(fps)
    anchors = anchor_indices(rec["num_frames"], fps, args["anchor_hz"], args["cap"], max(offsets))
    out_dir = Path(args["image_root"]) / "accident" / rec["sample_id"]
    frames_dir = Path(rec["frames_dir"].replace("/workspace", args["workspace"]))
    needed = sorted({a + k for a in anchors for k in [0] + offsets})
    for i in needed:
        out = out_dir / f"{i:06d}.jpg"
        if out.is_file():
            continue
        bgr = cv2.imread(str(frames_dir / f"{i:06d}.jpg"))
        if bgr is None:
            raise FileNotFoundError(frames_dir / f"{i:06d}.jpg")
        write_jpeg(out, to_input(bgr))
    dataset = rec["source_id"].split(":")[0].lower()
    frames, pairs = rows_for(
        "accident", rec["sample_id"], split, f"accident/{rec['sample_id']}", fps, anchors, offsets,
        lambda i: str(out_dir / f"{i:06d}.jpg"),
    )
    for row in frames + pairs:
        row["dataset"] = dataset
    return frames, pairs


def process_baton(task):
    video, split, args = task
    route = Path(video).parent.name
    container = av.open(video)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    total = stream.frames
    offsets = partner_offsets(fps)
    anchors = anchor_indices(total, fps, args["baton_hz"], args["baton_cap"], max(offsets))
    needed = sorted({a + k for a in anchors for k in [0] + offsets})
    out_dir = Path(args["image_root"]) / "baton" / route
    todo = [i for i in needed if not (out_dir / f"{i:06d}.jpg").is_file()]
    if todo:
        todo_set, last = set(todo), max(todo)
        for i, frame in enumerate(container.decode(stream)):
            if i in todo_set:
                write_jpeg(out_dir / f"{i:06d}.jpg", to_input(frame.to_ndarray(format="bgr24")))
            if i >= last:
                break
    container.close()
    missing = [i for i in needed if not (out_dir / f"{i:06d}.jpg").is_file()]
    if missing:
        drop = set(missing)
        anchors = [a for a in anchors if not ({a + k for k in [0] + offsets} & drop)]
    return rows_for("baton", route, split, f"baton/{route}", fps, anchors, offsets, lambda i: str(out_dir / f"{i:06d}.jpg"))


def baton_split(route: str, val_fraction: float) -> str:
    h = int(hashlib.sha1(route.encode()).hexdigest(), 16) % 1000
    return "val" if h < val_fraction * 1000 else "train"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument("--stage2-manifests", default="/workspace/data/stage2/manifests")
    parser.add_argument("--baton-root", default="/workspace/data/stage3/BATON-Sample")
    parser.add_argument("--image-root", default="/workspace/data/geometry/processed")
    parser.add_argument("--manifest-dir", default="/workspace/cache/geometry_pretrain/manifests")
    parser.add_argument("--anchor-hz", type=float, default=2.0)
    parser.add_argument("--cap", type=int, default=30, help="max static anchors per accident video")
    parser.add_argument("--baton-hz", type=float, default=0.25)
    parser.add_argument("--baton-cap", type=int, default=150)
    parser.add_argument("--baton-val-fraction", type=float, default=0.18)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--only", choices=["accident", "baton"], default=None)
    args = parser.parse_args()
    DiskGuard(args.image_root).check("before video frame sampling")
    shared = vars(args)
    manifest_dir = Path(args.manifest_dir)
    if args.only in (None, "accident"):
        for split in ("train", "val"):
            recs = read_jsonl(Path(args.stage2_manifests) / f"{split}.jsonl")
            frames, pairs = [], []
            with Pool(args.workers) as pool:
                for f, p in tqdm(pool.imap(process_accident, [(r, split, shared) for r in recs]), total=len(recs), desc=f"accident-{split}"):
                    frames += f
                    pairs += p
            write_jsonl(manifest_dir / f"accident_{split}.jsonl", frames)
            write_jsonl(manifest_dir / f"accident_{split}_pairs.jsonl", pairs)
            print("accident", split, len(recs), "videos", len(frames), "frames", len(pairs), "pairs")
    if args.only in (None, "baton"):
        videos = sorted(str(p) for p in Path(args.baton_root).glob("*/qcamera.mp4"))
        tasks = [(v, baton_split(Path(v).parent.name, args.baton_val_fraction), shared) for v in videos]
        out = {"train": ([], []), "val": ([], [])}
        with Pool(args.workers) as pool:
            for (f, p), (_, split, _) in zip(tqdm(pool.imap(process_baton, tasks), total=len(tasks), desc="baton"), tasks):
                out[split][0].extend(f)
                out[split][1].extend(p)
        for split, (f, p) in out.items():
            write_jsonl(manifest_dir / f"baton_{split}.jsonl", f)
            write_jsonl(manifest_dir / f"baton_{split}_pairs.jsonl", p)
            print("baton", split, len({r["group"] for r in f}), "routes", len(f), "frames", len(p), "pairs")


if __name__ == "__main__":
    main()
