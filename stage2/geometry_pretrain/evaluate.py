"""Held-out geometry evaluation and representation-drift analysis.

    # geometry metrics of a trained run (heads + backbone from its best.pt)
    python -m stage2.geometry_pretrain.evaluate geometry --run-dir RUN [--split val]
    # original vs adapted DINOv3 feature preservation (+ PCA visualizations)
    python -m stage2.geometry_pretrain.evaluate drift --adapted RUN/backbone_best.pth --out DIR
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stage2.geometry_pretrain.datasets import PairFlowDataset, StaticGeometryDataset, load_manifest
from stage2.geometry_pretrain.models.geometry_dino import DinoBackbone
from stage2.geometry_pretrain.train import MEAN, STD, build_model, evaluate, load_config

ORIGINAL = "/workspace/pretrained/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-timm-converted.pth"
EVAL_CAPS = {"bdd100k": 3000, "tusimple": 600, "accident": 948, "baton": 1301}
PAIR_CAPS = {"tusimple": 1800, "accident": 1500, "baton": 1500}


def eval_loaders(manifest_dir, sources, caps, pair_caps, batch=16, workers=5, seed=0):
    rng = random.Random(seed)
    rows, pairs = [], []
    for s in sources:
        r = load_manifest(manifest_dir, [s], "val")
        if caps.get(s) and len(r) > caps[s]:
            r = rng.sample(r, caps[s])
        rows += r
        p = load_manifest(manifest_dir, [s], "val", pairs=True)
        if p and pair_caps.get(s) and len(p) > pair_caps[s]:
            p = rng.sample(p, pair_caps[s])
        pairs += p
    sl = DataLoader(StaticGeometryDataset(rows, False), batch_size=batch, num_workers=workers)
    pl = DataLoader(PairFlowDataset(pairs, False), batch_size=batch // 2, num_workers=workers) if pairs else None
    return sl, pl, len(rows), len(pairs)


def cmd_geometry(args):
    run = Path(args.run_dir)
    ck = torch.load(run / args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    cfg["model"]["checkpoint"] = None  # all weights come from the run checkpoint
    model = build_model(cfg, "cuda")
    model.load_state_dict(ck["model"])
    sl, pl, n, npairs = eval_loaders(cfg["data"]["manifest_dir"], args.sources, EVAL_CAPS, PAIR_CAPS)
    m = evaluate(model, sl, pl, "cuda", cfg["tasks"])
    m["_n_images"], m["_n_pairs"], m["_step"] = n, npairs, ck.get("step")
    out = run / f"eval_{args.checkpoint.replace('.pt', '')}.json"
    out.write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))


# ----------------------------------------------------------------------------
# Drift
# ----------------------------------------------------------------------------


def linear_cka(x, y):
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    hsic = (x.T @ y).norm() ** 2
    return float(hsic / ((x.T @ x).norm() * (y.T @ y).norm()))


def pca_rgb(feat, basis=None):
    """feat: C,h,w -> h,w,3 uint8 using a PCA basis (fit on this map if None)."""
    C, h, w = feat.shape
    x = feat.reshape(C, -1).T.float()
    mu = x.mean(0, keepdim=True) if basis is None else basis[0]
    if basis is None:
        _, _, v = torch.pca_lowrank(x - mu, q=3, center=False)
        basis = (mu, v)
    y = (x - basis[0]) @ basis[1]
    lo, hi = y.quantile(0.01, 0), y.quantile(0.99, 0)
    y = ((y - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    return (y.reshape(h, w, 3).cpu().numpy() * 255).astype(np.uint8), basis


@torch.no_grad()
def cmd_drift(args):
    dev = "cuda"
    orig = DinoBackbone("vits16", ORIGINAL).to(dev).eval()
    adapted = DinoBackbone("vits16", args.adapted).to(dev).eval()
    rng = random.Random(0)
    rows = []
    for s in args.sources:
        r = load_manifest(args.manifest_dir, [s], "val")
        rows += rng.sample(r, min(len(r), args.per_source))
    loader = DataLoader(StaticGeometryDataset(rows, False), batch_size=16, num_workers=4)
    stats = {k: [] for k in ("cos", "cls_cos", "norm_o", "norm_a", "gram_absdiff", "gram_corr")}
    per_src = {}
    samples_o, samples_a = [], []
    g = torch.Generator().manual_seed(0)
    for batch in loader:
        x = (batch["image"].to(dev).float() - MEAN.to(dev)) / STD.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fo, fa = orig(x), adapted(x)
        po, pa = fo["patch"].float(), fa["patch"].float()
        B, C = po.shape[:2]
        cos = F.cosine_similarity(po, pa, dim=1).flatten(1).mean(1)
        cls = F.cosine_similarity(fo["cls"].float(), fa["cls"].float(), dim=1)
        no, na = po.norm(dim=1).flatten(1).mean(1), pa.norm(dim=1).flatten(1).mean(1)
        a = F.normalize(po.flatten(2), dim=1)
        b = F.normalize(pa.flatten(2), dim=1)
        ga, gb = a.transpose(1, 2) @ a, b.transpose(1, 2) @ b
        absdiff = (ga - gb).abs().flatten(1).mean(1)
        gac = ga.flatten(1) - ga.flatten(1).mean(1, keepdim=True)
        gbc = gb.flatten(1) - gb.flatten(1).mean(1, keepdim=True)
        corr = (gac * gbc).sum(1) / (gac.norm(dim=1) * gbc.norm(dim=1))
        for k, v in zip(stats, (cos, cls, no, na, absdiff, corr)):
            stats[k] += v.cpu().tolist()
        for i, sid in enumerate(batch["source"].tolist()):
            per_src.setdefault(sid, []).append(float(cos[i]))
        idx = torch.randint(po.shape[2] * po.shape[3], (B, 64), generator=g).to(dev)
        samples_o.append(torch.gather(po.flatten(2), 2, idx[:, None].expand(-1, C, -1)).transpose(1, 2).reshape(-1, C))
        samples_a.append(torch.gather(pa.flatten(2), 2, idx[:, None].expand(-1, C, -1)).transpose(1, 2).reshape(-1, C))
    so, sa = torch.cat(samples_o), torch.cat(samples_a)
    from stage2.geometry_pretrain.datasets import SOURCE_IDS

    inv = {v: k for k, v in SOURCE_IDS.items()}
    result = {
        "n_images": len(stats["cos"]),
        "patch_cos_mean": float(np.mean(stats["cos"])),
        "patch_cos_min_image": float(np.min(stats["cos"])),
        "cls_cos_mean": float(np.mean(stats["cls_cos"])),
        "patch_norm_original": [float(np.mean(stats["norm_o"])), float(np.std(stats["norm_o"]))],
        "patch_norm_adapted": [float(np.mean(stats["norm_a"])), float(np.std(stats["norm_a"]))],
        "gram_mean_abs_diff": float(np.mean(stats["gram_absdiff"])),
        "gram_correlation": float(np.mean(stats["gram_corr"])),
        "linear_cka_patches": linear_cka(so, sa),
        "patch_cos_by_source": {inv[k]: float(np.mean(v)) for k, v in per_src.items()},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "drift.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    # PCA feature maps: one shared basis per image (fit on original) for honest comparison.
    tiles = []
    for s in args.sources:
        r = load_manifest(args.manifest_dir, [s], "val")
        for row in random.Random(1).sample(r, 2):
            img = cv2.imread(row["image"])
            x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)[None].to(dev).float()
            x = (x - MEAN.to(dev)) / STD.to(dev)
            po, pa = orig(x)["patch"][0], adapted(x)["patch"][0]
            ro, basis = pca_rgb(po)
            ra, _ = pca_rgb(pa, basis)
            own, _ = pca_rgb(pa)
            sz = (400, 224)
            tiles.append(np.hstack([cv2.resize(img, sz)] + [cv2.resize(t[:, :, ::-1], sz, interpolation=cv2.INTER_NEAREST) for t in (ro, ra, own)]))
    grid = np.vstack(tiles)
    for i, label in enumerate(["RGB", "original PCA", "adapted (orig basis)", "adapted (own PCA)"]):
        cv2.putText(grid, label, (i * 400 + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(out / "pca_features.jpg"), grid)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("geometry")
    g.add_argument("--run-dir", required=True)
    g.add_argument("--checkpoint", default="best.pt")
    g.add_argument("--sources", nargs="+", default=["bdd100k", "tusimple", "accident", "baton"])
    d = sub.add_parser("drift")
    d.add_argument("--adapted", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--manifest-dir", default="/workspace/cache/geometry_pretrain/manifests")
    d.add_argument("--sources", nargs="+", default=["bdd100k", "tusimple", "accident", "baton"])
    d.add_argument("--per-source", type=int, default=150)
    args = parser.parse_args()
    cmd_geometry(args) if args.cmd == "geometry" else cmd_drift(args)


if __name__ == "__main__":
    main()
