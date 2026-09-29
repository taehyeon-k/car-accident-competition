"""openpilot (v0.11.1 driving_vision, traced fp32 TorchScript: stage2/night/openpilot_vision.ts) features for Stage 2.

Per retained frame t of view k (native frames[::k]; k2 / k3 = the stride caches' kept frames), the model sees the pair (frame t-1,
frame t) of that view (row 0 uses (t, t)), warped into openpilot's model frame by openpilot_probe.to_model_frame (assumed 100 deg
horizontal FOV, vanishing point at the image centre). Per frame (71 values):
  lead: 3 hypotheses x (x, y, v, a) at t = 0, signed-log squashed (12); lead probabilities (sigmoid) (3)
  pose: translation + rotation means (6); lane-line probabilities (sigmoid) (8)
  lane lines: lateral position of the 4 lines at the 1st and 9th points (8); road edges: lateral position at the 1st point (2)
  hidden state: 32-d PCA of the 512-d vector (label-free PCA on native frames of all clips) (32)
Output cache_op/k{k}/<sid>.npy float16. No labels, no FPS.
Usage: python -m stage2.night.op_features [--workers 6]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import cv2
import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.night.openpilot_probe import to_model_frame, yuv6

HERE = Path(__file__).resolve().parent; OUT = HERE / "cache_op"; RAW = HERE / "cache_op_raw"; STRIDE = C.REPO / "stage2/generalization/cache_stride"
SL = json.loads((HERE / "openpilot_vision.json").read_text())
ssq = lambda x: np.sign(x) * np.log1p(np.abs(x))
sig = lambda x: 1 / (1 + np.exp(-x))


class Frames(torch.utils.data.Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return torch.from_numpy(yuv6(to_model_frame(cv2.imread(str(self.paths[i])), 100.0, 0.5)))


def select(o):
    """o [N, 1576] float32 -> [N, 39] compact outputs + [N, 512] hidden"""
    s = lambda k: o[:, SL[k][0]:SL[k][1]]
    lead = s("lead").reshape(len(o), 3, -1)[:, :, :4].reshape(len(o), 12)                      # per hypothesis x, y, v, a at t0 (means)
    lanes = s("lane_lines").reshape(len(o), 4, 33, 2, 2)[:, :, [0, 8], 0, 0].reshape(len(o), 8)  # lateral y (mean) at points 0, 8
    edges = s("road_edges").reshape(len(o), 2, 33, 2, 2)[:, :, 0, 0, 0]
    comp = np.concatenate([ssq(lead), sig(s("lead_prob")), s("pose")[:, :6], sig(s("lane_lines_prob")), ssq(lanes), ssq(edges)], 1)
    return comp.astype(np.float32), s("hidden_state").astype(np.float32)


@torch.inference_mode()
def extract(net, sid, fdir, workers):
    paths = sorted(Path(fdir).glob("*.jpg")); dl = torch.utils.data.DataLoader(Frames(paths), batch_size=64, num_workers=workers)
    yuv = torch.cat([b for b in dl]).cuda()   # [n, 6, 128, 256] uint8
    res = {}
    for k in (1, 2, 3):
        kept = np.arange(len(paths)) if k == 1 else np.load(STRIDE / f"k{k}" / f"{sid}.npz")["kept"]; outs = []
        for t in range(len(kept)):
            a, b = yuv[kept[max(t - 1, 0)]], yuv[kept[t]]; x = torch.cat([a, b])[None]
            o = net(x, x); outs.append((o[0] if isinstance(o, (list, tuple)) else o).float())
        comp, hid = select(torch.cat(outs).cpu().numpy()); res[k] = (comp, hid)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); a = ap.parse_args()
    net = torch.jit.load(str(HERE / "openpilot_vision.ts"), map_location="cuda").eval()
    rows = sorted(C.rows("all"), key=lambda r: -int(r["num_frames"]))
    for k in (1, 2, 3): (RAW / f"k{k}").mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows, 1):
        sid = r["sample_id"]
        if all((RAW / f"k{k}" / f"{sid}.npz").exists() for k in (1, 2, 3)): continue
        for k, (comp, hid) in extract(net, sid, r["frames_dir"], a.workers).items():
            np.savez(RAW / f"k{k}" / f"{sid}.npz", comp=comp.astype(np.float16), hid=hid.astype(np.float16))
        if i % 25 == 0: print(i, len(rows), sid, flush=True)
    # label-free PCA of the hidden state (native frames of all clips, every 4th frame), then write the final caches
    H = np.concatenate([np.load(RAW / "k1" / f"{r['sample_id']}.npz")["hid"][::4].astype(np.float32) for r in rows])
    mu = H.mean(0); _, _, vt = np.linalg.svd(H - mu, full_matrices=False); comp32 = vt[:32]; sc = ((H - mu) @ comp32.T).std(0) + 1e-6
    np.savez(HERE / "op_hidden_pca32.npz", mu=mu, comp=comp32, scale=sc)
    for k in (1, 2, 3):
        (OUT / f"k{k}").mkdir(parents=True, exist_ok=True)
        for r in rows:
            d = np.load(RAW / f"k{k}" / f"{r['sample_id']}.npz"); z = ((d["hid"].astype(np.float32) - mu) @ comp32.T) / sc
            np.save(OUT / f"k{k}" / f"{r['sample_id']}.npy", np.concatenate([d["comp"].astype(np.float32), z], 1).astype(np.float16))
    print("done", flush=True)


if __name__ == "__main__": main()
