"""Flow-grid features (3 x 5, as stage2/night/flowgrid.py) for the 670 pseudo-labelled NEXAR extras used by the XN4 family, so a
flow-enabled XN4 can train with its extras. Grey frames as nexar_expand.py builds them for these clips (full-resolution decode ->
grey -> 160 x 90 INTER_AREA). Native frames only (extras are not stride-augmented). Output cache_flowgrid/k1/nexaru_<vid>.npy.
Usage: python -m stage2.night.flowgrid_nexar [--procs 10]
"""
from __future__ import annotations

import argparse, json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from stage2.long_context_v2_experiments import common as C
from stage2.night.flowgrid_rt import flowgrid

VID = Path("/workspace/data/stage2/nexar_unlabeled/videos"); OUT = C.REPO / "stage2/night/cache_flowgrid/k1"


def one(sid):
    tgt = OUT / f"{sid}.npy"
    if tgt.exists(): return
    cv2.setNumThreads(1); cap = cv2.VideoCapture(str(VID / f"{sid.split('_', 1)[1]}.mp4")); g = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        g.append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (160, 90), interpolation=cv2.INTER_AREA))
    cap.release(); np.save(tgt, flowgrid(g).astype(np.float16))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=10); a = ap.parse_args()
    sids = sorted(set().union(*[set(json.loads((C.REPO / f"stage2/aux_signal_experiments/cache_nexar_u/labels_fold{f}.json").read_text())) for f in range(5)]))
    with Pool(a.procs) as p:
        for i, _ in enumerate(p.imap_unordered(one, sids), 1):
            if i % 50 == 0: print(i, len(sids), flush=True)
    print("done", flush=True)


if __name__ == "__main__": main()
