"""Training labels for the unlabelled NEXAR positives, one label set per CV fold (and one for the fixed 279/70 split).

For split s (fold k, or 'fixed'):
  COLLISION  = time_of_event (public Nexar metadata) + offset_s, offset_s = median over the labelled NEXAR clips in the
               TRAINING part of s of (GT collision time - time_of_event)  [labels only; the model never sees FPS]
  ENTRY      = argmax of the split's teacher ENTRY probability within the 2 s before that COLLISION; teacher = the E4_NT_both +
               E2_NT_both_bnd2 checkpoints of that split (fold-k CV models were trained without fold k)
  attributes = unknown (masked in training)
Output cache_nexar_u/labels_<split>.json: {sample_id: {collision_frame, entry_frame, nframes, frames}}.
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import cv2
import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from .model import load as load_model
from .nexar_expand import CAND, OUT, VID
from .train import segments

RES = C.REPO / "stage2/aux_signal_experiments/results"
FOLDS = C.REPO / "stage2/long_context_v2_experiments/folds"
TEACHERS = ["E4_NT_both", "E2_NT_both_bnd2"]


def motion28(m):
    return np.concatenate([np.abs(m[:, :3]), m[:, 3:4], np.log1p(100 * m[:, 4:])], 1).astype(np.float32)


def unl_item(sid):
    fr = np.load(OUT / f"{sid}.frames.npy"); x = np.load(OUT / f"{sid}.npy", mmap_mode="r")
    g = segments(motion28(np.load(OUT / f"{sid}.motion.npy")), fr); r = segments(np.load(OUT / f"{sid}.residual.npy"), fr)
    return fr, torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.concatenate([g, r], 1))


def fps_of(vid):
    cap = cv2.VideoCapture(str(VID / f"{vid}.mp4")); f = cap.get(cv2.CAP_PROP_FPS); cap.release(); return float(f)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--window-s", type=float, default=2.0)
    ap.add_argument("--full", nargs="*", default=None, help="only build the 'all' split with these full-data teacher runs (fixed seeds)")
    a = ap.parse_args()
    dev = torch.device("cuda"); meta = {r["video_id"]: r for r in csv.DictReader(open(CAND))}
    unl = sorted(p.name[:-4] for p in OUT.glob("nexaru_*.npy") if p.name.count(".") == 1)
    fps = {sid: fps_of(sid.split("_")[1]) for sid in unl}
    splits = {f"fold{k}": (C.rows(str(FOLDS / f"fold{k}_train.jsonl")), [f"cv/fold{k}_seed"]) for k in range(5)}
    splits["fixed"] = (C.rows("train"), ["seed"])
    if a.full is not None:  # full-data refit: teachers trained on all 349 labelled clips, offset from all labelled NEXAR clips
        global TEACHERS
        TEACHERS = a.full; splits = {"all": (C.rows("all"), ["seed"])}
    for name, (train_rows, prefixes) in splits.items():
        nex = [r for r in train_rows if C.source(r) == "NEXAR"]
        off = float(np.median([int(r["collision_frame"]) / float(r["native_fps"]) - float(meta[r["sample_id"].split("_")[1]]["time_of_event"]) for r in nex]))
        # fixed teacher set for every split: CV seeds 0-2 (fold models) / fixed seeds 0-1
        allowed = {"0", "1", "2"} if name.startswith("fold") else ({"0", "1"} if name == "fixed" else {"0", "1", "2", "3"})
        cks = [p for t in TEACHERS for pre in prefixes for p in sorted((RES / t).glob(f"{pre}*/checkpoint.pt"))
               if p.parent.name.split("seed")[-1] in allowed]
        models = [load_model(p, dev) for p in cks]
        labels = {}
        for sid in unl:
            fr, x, mot = unl_item(sid); f = fps[sid]
            c = int(round((float(meta[sid.split("_")[1]]["time_of_event"]) + off) * f)); c = min(max(c, int(fr[1])), int(fr[-1]))
            v = torch.ones(1, len(fr), dtype=torch.bool, device=dev)
            probs = torch.stack([m(x[None].to(dev), v, motion=mot[None].to(dev))["entry_logits"][0].float().softmax(-1) for m in models]).mean(0).cpu().numpy()
            ok = (fr <= c) & (fr >= c - a.window_s * f)
            e = int(fr[np.argmax(np.where(ok, probs, -1))]) if ok.any() else int(fr[np.argmax(probs)])
            labels[sid] = {"collision_frame": c, "entry_frame": min(e, c), "nframes": int(np.load(OUT / f"{sid}.nframes.npy"))}
        C.dump(OUT / f"labels_{name}.json", labels)
        print(name, "offset %.3fs" % off, "teachers", len(models), "labelled", len(labels), flush=True)


if __name__ == "__main__": main()
