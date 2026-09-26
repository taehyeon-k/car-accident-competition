"""Quality check of the lane-intrusion pseudo-labels (Exp 2) before any training uses them.

Coverage: valid frames / clips per source.
GT alignment (FPS only for the offline 0.3 s check): with e, c = GT ENTRY / COLLISION sampled positions and g = max(c - e, 3):
  rise@ENTRY   = mean score on [e, c] - mean score on [e - g, e)       (should be > 0 if the label marks the intrusion)
  rise@shifted = the same with e moved g positions earlier/later (null: a generic trend toward the collision also rises)
  AUC          = P(score after > score before) pooled over clips
  onset        = first position with state >= CROSSING that persists 2 positions, nearest to e; hit if within 0.3 s of GT ENTRY
Overlays: cache_lane/qa/overlay_*.jpg (RGB, ego-lane corridor, contact pixels, state) at e-2g, e-g, e, (e+c)/2, c.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from stage2.geometry_pretrain.common import INPUT_HW, crop_to_aspect
from stage2.long_context_v2_experiments import common as C
from stage2.spotting_experiments.extract_features import indexed_frames
from .pseudo_lane import OUT, NEAR_FRAC

STATES = ["OUTSIDE", "APPROACH", "CROSSING", "INSIDE"]


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg): return np.nan
    return float(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean())


def window_rise(score, e, c, shift=0):
    g = max(c - e, 3); e2, c2 = e + shift, c + shift
    if e2 - g < 0 or c2 >= len(score): return None
    before, after = score[e2 - g:e2], score[e2:c2 + 1]
    if (before < 0).any() or (after < 0).any(): return None
    return after.mean() - before.mean(), auc(after, before)


def main():
    import sys
    global OUT
    if len(sys.argv) > 1: OUT = Path(sys.argv[1])
    rows = {r["sample_id"]: r for r in C.rows("all")}; fps = C.fps_table()
    stats = defaultdict(list); per_source = defaultdict(lambda: defaultdict(list)); examples = []
    for sid, r in rows.items():
        d = np.load(OUT / f"{sid}.lane.npz"); fr = d["frames"]; state, score = d["state"], d["score"]
        src = C.source(r); e = int(np.abs(fr - int(r["entry_frame"])).argmin()); c = int(np.abs(fr - int(r["collision_frame"])).argmin())
        per_source[src]["valid_frame"].append((state >= 0).mean()); per_source[src]["valid_clip"].append(bool(d["valid_clip"]))
        per_source[src]["states"].append(np.bincount(state[state >= 0], minlength=4) / max((state >= 0).sum(), 1))
        g = max(c - e, 3)
        for key, shift in (("gt", 0), ("shift_early", -g), ("shift_late", g)):
            w = window_rise(score, e, c, shift)
            if w is not None: stats[f"rise_{key}"].append(w[0]); stats[f"auc_{key}"].append(w[1]); per_source[src][f"rise_{key}"].append(w[0])
        cross = np.where((state[:-1] >= 2) & (state[1:] >= 2))[0]
        if len(cross):
            o = int(cross[np.argmin(np.abs(cross - e))])
            hit = abs(int(fr[o]) - int(r["entry_frame"])) / fps[sid] <= .300001
            stats["onset_hit"].append(hit); per_source[src]["onset_hit"].append(hit)
            stats["onset_norm_err"].append(abs(o - e) / max(len(fr) - 1, 1))
        else:
            stats["onset_hit"].append(False); per_source[src]["onset_hit"].append(False)
        if len(examples) < 400: examples.append((sid, e, c))
    out = {"n_clips": len(rows)}
    for k, v in stats.items(): out[k] = float(np.nanmean(v)); out[f"{k}_n"] = len(v)
    out["frac_clips_rise_gt_pos"] = float(np.mean(np.array(stats["rise_gt"]) > 0))
    out["per_source"] = {s: {k: (float(np.mean(v)) if k != "states" else np.mean(v, 0).round(3).tolist()) for k, v in d_.items()} for s, d_ in per_source.items()}
    (OUT / "qa").mkdir(exist_ok=True); C.dump(OUT / "qa/qa_summary.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "per_source"}, indent=1))
    for s, v in out["per_source"].items(): print(s, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})
    overlays(rows, examples)


def overlays(rows, examples, per_source=3):
    rng = np.random.default_rng(0); by_src = defaultdict(list)
    for ex in examples: by_src[C.source(rows[ex[0]])].append(ex)
    picks = [x for s in sorted(by_src) for x in [by_src[s][i] for i in rng.choice(len(by_src[s]), min(per_source, len(by_src[s])), replace=False)]]
    panels = []
    for sid, e, c in picks:
        d = np.load(OUT / f"{sid}.lane.npz"); fr = d["frames"]
        ego = np.unpackbits(d["ego"], axis=-1)[..., :INPUT_HW[1] // 4].astype(bool)
        con = np.unpackbits(d["contact"], axis=-1)[..., :INPUT_HW[1] // 4].astype(bool)
        paths = dict(indexed_frames(rows[sid]["frames_dir"])); g = max(c - e, 3)
        tiles = []
        for name, t in (("e-2g", e - 2 * g), ("e-g", e - g), ("ENTRY", e), ("mid", (e + c) // 2), ("COLL", c)):
            t = int(np.clip(t, 0, len(fr) - 1))
            img = cv2.imread(str(paths[int(fr[t])]), cv2.IMREAD_REDUCED_COLOR_2)
            img = cv2.resize(crop_to_aspect(img)[0], (INPUT_HW[1] // 2, INPUT_HW[0] // 2))
            m = cv2.resize(ego[t].astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            k = cv2.resize(con[t].astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            img[m] = (0.6 * img[m] + 0.4 * np.array([255, 128, 0])).astype(np.uint8); img[k] = (0, 255, 0)
            y0 = int(img.shape[0] * (1 - NEAR_FRAC)); cv2.line(img, (0, y0), (img.shape[1], y0), (255, 255, 255), 1)
            st = int(d["state"][t]); txt = f"{name} {STATES[st] if st >= 0 else 'INVALID'} d={d['dmin'][t]:.2f}"
            cv2.putText(img, txt, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2); tiles.append(img)
        strip = np.hstack(tiles); cv2.putText(strip, sid, (5, strip.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        panels.append(strip)
    for i in range(0, len(panels), 4):
        cv2.imwrite(str(OUT / f"qa/overlay_{i // 4}.jpg"), np.vstack(panels[i:i + 4]), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__": main()
