"""Feasibility diagnostic for object-centric ENTRY (offline only; uses GT COLLISION to pick the opponent, FPS only for the metric).

Opponent = the vehicle track with the largest box area within the last few frames up to the GT COLLISION frame.
For the opponent's trajectory we test simple physical ENTRY rules and report the 0.3 s hit rate against GT ENTRY:
  corridor_k : first frame (going back from COLLISION, contiguous) where the box overlaps the ego corridor |x - 0.5| < k
  area_q     : first frame where the box area reaches q x (area at COLLISION)
  onset      : last frame before COLLISION where the lateral velocity toward the centre becomes positive and stays positive
Also: how often an opponent track exists, how long before ENTRY it is first seen.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CACHE = Path("/workspace/car-accident/stage2/objtrack/cache_tracks")
VEH = (2, 3, 4, 6, 8)


def rows():
    with open("/workspace/data/stage2/usable/usable_only.csv") as f:
        return list(csv.DictReader(f))


def smooth(x, k=5):
    if len(x) < k: return x
    return np.convolve(np.pad(x, (k // 2, k // 2), mode="edge"), np.ones(k) / k, "valid")


def opponent(d, cf, look=5):
    fr, box, cls, tr = d["frame"], d["box"], d["cls"], d["track"]
    m = (fr <= cf) & (fr >= cf - look) & np.isin(cls, VEH) & (tr >= 0)
    if not m.any(): return None
    area = (box[m, 2] - box[m, 0]) * (box[m, 3] - box[m, 1])
    tid = tr[m][area.argmax()]
    sel = (tr == tid) & (fr <= cf)
    order = np.argsort(fr[sel])
    return fr[sel][order], box[sel][order]


def rules(f, b, cf):
    cx = (b[:, 0] + b[:, 2]) / 2; area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    left, right = b[:, 0], b[:, 2]
    out = {}
    for k in (0.1, 0.2, 0.3):
        inside = (right > 0.5 - k) & (left < 0.5 + k)
        t = len(f) - 1
        while t > 0 and inside[t - 1]: t -= 1
        out[f"corridor_{k}"] = f[t] if inside[-1] else f[-1]
    a_end = area[-1]
    for q in (0.1, 0.25, 0.5):
        idx = np.nonzero(area >= q * a_end)[0]; out[f"area_{q}"] = f[idx[0]] if len(idx) else f[-1]
    v = -np.gradient(smooth(np.abs(cx - 0.5))) if len(cx) > 1 else np.zeros(1)  # >0 = moving toward centre
    t = len(f) - 1
    while t > 0 and v[t - 1] > 0: t -= 1
    out["onset"] = f[t]
    out["first_seen"] = f[0]
    return out


def main():
    hits = defaultdict(list); lead = []; srcs = defaultdict(lambda: defaultdict(list)); n_missing = 0; n = 0
    for r in rows():
        p = CACHE / f"{r['sample_id']}.npz"
        if not p.exists(): continue
        n += 1; d = np.load(p); fps = float(r["fps"]); ef, cf = int(r["entry_frame"]), int(r["collision_frame"])
        o = opponent(d, cf)
        if o is None: n_missing += 1; continue
        res = rules(*o, cf)
        lead.append((ef - res["first_seen"]) / fps)
        for k, v in res.items():
            if k == "first_seen": continue
            h = abs(v - ef) / fps <= 0.300001; hits[k].append(h); srcs[r["source"]][k].append(h)
        hits["collision_minus_gap_prior"].append(False)
    print(f"clips {n}, no opponent track {n_missing}")
    print("opponent first seen before ENTRY (s): median %.2f, frac seen >=0.3s before ENTRY %.2f" % (np.median(lead), np.mean(np.array(lead) >= 0.3)))
    for k, v in hits.items():
        if k.startswith("collision_minus"): continue
        print(f"{k:14s} hit {np.mean(v):.3f}  " + "  ".join(f"{s} {np.mean(srcs[s][k]):.2f}" for s in sorted(srcs)))


if __name__ == "__main__": sys.exit(main())
