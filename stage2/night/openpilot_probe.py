"""Feasibility probe: openpilot v0.11.1 driving_vision.onnx on Stage 2 dashcam frames (no labels used by the model).

openpilot's model frame ("medmodel", 512 x 256, focal 910 px, principal point (256, 47.6)) is a narrow (~31 deg) forward view. Our
dashcams are wide with unknown intrinsics, so the warp is approximated: assume a horizontal FOV (default 100 deg) -> focal
f = W / (2 tan(FOV / 2)); scale s = 910 / f; take the source window of 512/s x 256/s pixels whose vanishing point (assumed at the
image centre column and at VP_ROW x height) maps to (256, 47.6); resize to 512 x 256 RGB -> YUV420 -> 6 channels of 128 x 256
(Y[0::2,0::2], Y[1::2,0::2], Y[0::2,1::2], Y[1::2,1::2], U, V); two consecutive frames stacked = 12 channels; big_img = same.
Prints per-frame lead probability, lead distance (first hypothesis, t=0) and lane-line probabilities around COLLISION.
Usage: python -m stage2.night.openpilot_probe SAMPLE_ID [--fov 100] [--vp-row 0.5]
"""
from __future__ import annotations

import argparse, pickle, base64
from pathlib import Path

import cv2
import numpy as np
import onnx, onnxruntime as ort

from stage2.long_context_v2_experiments import common as C

M = Path("/workspace/pretrained/openpilot_v0.11.1/driving_vision.onnx")


def slices():
    md = {p.key: p.value for p in onnx.load(str(M)).metadata_props}
    return pickle.loads(base64.b64decode(md["output_slices"]))


def to_model_frame(bgr, fov, vp_row):
    H, W = bgr.shape[:2]; f = W / (2 * np.tan(np.radians(fov) / 2)); s = 910.0 / f
    sw, sh = 512 / s, 256 / s; x0 = W / 2 - 256 / s; y0 = vp_row * H - 47.6 / s
    src = np.float32([[x0, y0], [x0 + sw, y0], [x0, y0 + sh]]); dst = np.float32([[0, 0], [512, 0], [0, 256]])
    return cv2.warpAffine(bgr, cv2.getAffineTransform(src, dst), (512, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def yuv6(bgr512):
    yuv = cv2.cvtColor(bgr512, cv2.COLOR_BGR2YUV_I420)  # (384, 512): Y 256 x 512, then U, V (128 x 256 each, planar)
    y = yuv[:256]; u = yuv[256:320].reshape(128, 256); v = yuv[320:384].reshape(128, 256)
    return np.stack([y[0::2, 0::2], y[1::2, 0::2], y[0::2, 1::2], y[1::2, 1::2], u, v])


def run(paths, fov=100.0, vp_row=0.5):
    sess = ort.InferenceSession(str(M), providers=["CPUExecutionProvider"]); prev = None; outs = []
    for p in paths:
        cur = yuv6(to_model_frame(cv2.imread(str(p)), fov, vp_row)); prev = cur if prev is None else prev
        x = np.concatenate([prev, cur])[None].astype(np.uint8)
        outs.append(sess.run(None, {"img": x, "big_img": x})[0][0].astype(np.float32)); prev = cur
    return np.stack(outs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("sid"); ap.add_argument("--fov", type=float, default=100.0); ap.add_argument("--vp-row", type=float, default=0.5)
    a = ap.parse_args(); r = {x["sample_id"]: x for x in C.rows("all")}[a.sid]; sl = slices()
    paths = sorted(Path(r["frames_dir"]).glob("*.jpg")); e, c = int(r["entry_frame"]), int(r["collision_frame"])
    lo, hi = max(0, e - 30), min(len(paths), c + 5); o = run(paths[lo:hi], a.fov, a.vp_row)
    sig = lambda z: 1 / (1 + np.exp(-z))
    lp = sig(o[:, sl["lead_prob"]]); lead = o[:, sl["lead"]]; llp = sig(o[:, sl["lane_lines_prob"]])
    print(f"{a.sid}: ENTRY {e} COLLISION {c} | frame: lead_prob(t0) lead_x0 lead_y0 | lane probs (4 lines)")
    for i, fr in enumerate(range(lo, hi)):
        if (fr - lo) % 3 and fr not in (e, c): continue
        tag = " <ENTRY" if fr == e else (" <COLLISION" if fr == c else "")
        print(f"  {fr:5d} {lp[i, 0]:.2f}  x {lead[i, 0]:6.1f} y {lead[i, 1]:6.1f} | " + " ".join(f"{v:.2f}" for v in llp[i, ::2][:4]) + tag)


if __name__ == "__main__": main()
