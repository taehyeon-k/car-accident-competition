"""Export RF-DETR Small (the project checkpoint used for cache_tracks) as a self-contained TorchScript network for the submission,
so the evaluation server needs neither rfdetr, transformers nor supervision for detection.

  export (detector venv):  /workspace/venvs/det/bin/python -m stage2.actor.export_rfdetr export OUT_DIR
      -> OUT_DIR/rfdetr_small.ts (traced fp32 network: normalised [B, 3, R, R] -> (boxes cxcywh [B, Q, 4], logits [B, Q, C]))
         OUT_DIR/rfdetr_small.json (resolution, means, stds, num_select) and ref_<clip>.npz (predict() detections for checks)
  check (main venv):       /venv/main/bin/python -m stage2.actor.export_rfdetr check OUT_DIR
      -> re-implemented preprocessing + PostProcess on the traced network vs the reference predict() detections.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch

FR = Path("/workspace/data/stage2/frames"); CLIP = "nexar_00136"; KEEP = (1, 2, 3, 4, 6, 8); BATCH = 8


def export(out):
    from PIL import Image
    from rfdetr import RFDETRSmall
    from stage2.objtrack.extract_tracks import WEIGHTS
    out.mkdir(parents=True, exist_ok=True)
    m = RFDETRSmall(pretrain_weights=WEIGHTS); net = m.model.model.eval().cuda(); res = int(m.model.resolution)
    dev = torch.device("cuda"); print("trace device", next(net.parameters()).device); x = torch.randn(BATCH, 3, res, res, device=dev)  # the trace fixes the batch size: callers pad to BATCH
    with torch.no_grad():
        o = net(x); print("raw output type", type(o), list(o) if isinstance(o, dict) else len(o))

    class Wrap(torch.nn.Module):
        def __init__(self, n): super().__init__(); self.n = n
        def forward(self, x):
            o = self.n(x)
            return (o["pred_boxes"], o["pred_logits"]) if isinstance(o, dict) else (o[0], o[1])
    w = Wrap(net).eval()
    with torch.no_grad():
        ts = torch.jit.trace(w, x, check_trace=False, strict=False)
    ts.save(str(out / "rfdetr_small.ts"))  # traced on CUDA: device constants are baked in, load with map_location="cuda"
    meta = {"resolution": res, "means": list(map(float, m.means)), "stds": list(map(float, m.stds)), "num_select": 300, "threshold": 0.3, "keep_classes": KEEP, "batch": BATCH}
    (out / "rfdetr_small.json").write_text(json.dumps(meta, indent=1))
    paths = sorted((FR / CLIP).glob("*.jpg"))[:48]
    dets = m.predict([Image.open(p).convert("RGB") for p in paths], threshold=0.3)
    np.savez(out / f"ref_{CLIP}.npz", **{f"f{i}_xyxy": d.xyxy for i, d in enumerate(dets)}, **{f"f{i}_conf": d.confidence for i, d in enumerate(dets)},
             **{f"f{i}_cls": d.class_id for i, d in enumerate(dets)})
    print("exported", res, meta)


def detect(ts, meta, imgs, dev):
    """re-implemented RF-DETR predict(): RGB uint8 HxWx3 list -> per image (xyxy pixels, scores, labels) with score > threshold"""
    import torchvision.transforms.functional as TF
    R = meta["resolution"]; mean = torch.tensor(meta["means"], device=dev).view(1, 3, 1, 1); std = torch.tensor(meta["stds"], device=dev).view(1, 3, 1, 1)
    x = torch.stack([TF.resize(torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).to(dev).float() / 255, [R, R], antialias=False) for a in imgs])
    n = len(x); B = meta["batch"]; bs, ls = [], []
    for s in range(0, n, B):
        xb = (x[s:s + B] - mean) / std
        if len(xb) < B: xb = torch.cat([xb, xb[-1:].expand(B - len(xb), -1, -1, -1)])
        b_, l_ = ts(xb); bs.append(b_[:min(B, n - s)]); ls.append(l_[:min(B, n - s)])
    boxes, logits = torch.cat(bs), torch.cat(ls)
    prob = logits.sigmoid().flatten(1); k = min(meta["num_select"], prob.shape[1])
    idx = torch.argsort(prob, dim=1, descending=True, stable=True)[:, :k]; sc = prob.gather(1, idx)
    qb, lab = idx // logits.shape[2], idx % logits.shape[2]
    cx, cy, w, h = boxes.unbind(-1); xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
    xyxy = torch.gather(xyxy, 1, qb[..., None].expand(-1, -1, 4)); out = []
    for i, a in enumerate(imgs):
        H, W = a.shape[:2]; s = torch.tensor([W, H, W, H], device=dev, dtype=xyxy.dtype)
        b = (xyxy[i] * s).clamp_min(0).clamp(max=s); keep = sc[i] > meta["threshold"]
        out.append((b[keep].cpu().numpy(), sc[i][keep].cpu().numpy(), lab[i][keep].cpu().numpy()))
    return out


@torch.inference_mode()
def check(out):
    from PIL import Image
    dev = torch.device("cuda"); ts = torch.jit.load(str(out / "rfdetr_small.ts"), map_location=dev).eval(); meta = json.loads((out / "rfdetr_small.json").read_text())
    ref = np.load(out / f"ref_{CLIP}.npz"); paths = sorted((FR / CLIP).glob("*.jpg"))[:48]
    got = []
    for s in range(0, len(paths), 16): got += detect(ts, meta, [np.asarray(Image.open(p).convert("RGB")) for p in paths[s:s + 16]], dev)
    nd = [(len(ref[f"f{i}_conf"]), len(g[1])) for i, g in enumerate(got)]
    diffs = []
    for i, (b, s, l) in enumerate(got):
        rb, rs = ref[f"f{i}_xyxy"], ref[f"f{i}_conf"]
        if len(rb) == len(b) and len(b): diffs.append(max(np.abs(np.sort(rs) - np.sort(s)).max(), np.abs(np.sort(rb, 0) - np.sort(b, 0)).max()))
    print(f"frames {len(nd)}; same detection count {np.mean([a == b for a, b in nd]):.3f}; total ref {sum(a for a, _ in nd)} vs traced {sum(b for _, b in nd)}; "
          f"max |score / box px| diff {max(diffs) if diffs else 'n/a'}")


if __name__ == "__main__":
    {"export": export, "check": check}[sys.argv[1]](Path(sys.argv[2]))
