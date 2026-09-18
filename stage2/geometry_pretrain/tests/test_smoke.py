"""Smoke tests for the geometry-pretraining pipeline (pytest or ``python -m``).

GPU tests are skipped without CUDA; data tests are skipped without the cache.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from stage2.geometry_pretrain.common import IGNORE, LABEL_HW
from stage2.geometry_pretrain.datasets import (
    PairFlowDataset,
    SourceWeightedSampler,
    StaticGeometryDataset,
    geo_flow,
    geo_image,
    geo_label,
    geo_point,
    load_manifest,
)

MANIFESTS = Path("/workspace/cache/geometry_pretrain/manifests")
ORIGINAL = "/workspace/pretrained/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-timm-converted.pth"
needs_data = pytest.mark.skipif(not (MANIFESTS / "bdd100k_val.jsonl").is_file(), reason="no manifests")
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


def _warp_label_res(img_lr, flow):
    """Backward-warp a stride-4 image with flow given at stride 4."""
    h, w = flow.shape[1:]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img_lr, xs + flow[0], ys + flow[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def test_flow_transform_consistency():
    """A synthetic translation stays exactly consistent through flip + zoom-crop."""
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.integers(0, 255, (LABEL_HW[0] + 20, LABEL_HW[1] + 20, 3)).astype(np.uint8), (0, 0), 2)
    dx, dy = 3, 2
    a = base[10:-10, 10:-10]
    b = base[10 - dy : base.shape[0] - 10 - dy, 10 - dx : base.shape[1] - 10 - dx]
    # b(y, x) = a(y - dy, x - dx)  =>  a(p) = b(p + f) with f = (+dx, +dy)
    flow = np.stack([np.full(LABEL_HW, dx, np.float32), np.full(LABEL_HW, dy, np.float32)])
    assert np.abs(_warp_label_res(b, flow)[5:-5, 5:-5].astype(float) - a[5:-5, 5:-5]).mean() < 1.0
    g = {"flip": True, "crop": (8, 12, 90, 160)}
    a2, b2 = geo_label(a, g, cv2.INTER_LINEAR), geo_label(b, g, cv2.INTER_LINEAR)
    f2 = geo_flow(flow, g)
    err = np.abs(_warp_label_res(b2, f2)[8:-8, 8:-8].astype(float) - a2[8:-8, 8:-8]).mean()
    naive = np.abs(_warp_label_res(b2, flow)[8:-8, 8:-8].astype(float) - a2[8:-8, 8:-8]).mean()
    assert err < 3.0 and err < 0.5 * naive, (err, naive)


def test_point_and_label_transform_agree():
    lab = np.zeros(LABEL_HW, np.uint8)
    lab[30, 50] = 1
    g = {"flip": True, "crop": (10, 20, 90, 160)}
    t = geo_label(lab, g)
    y, x = np.argwhere(t == 1).mean(0)
    px, py = geo_point(((50 + 0.5) / LABEL_HW[1], (30 + 0.5) / LABEL_HW[0]), g)
    assert abs(px * LABEL_HW[1] - (x + 0.5)) < 1.5 and abs(py * LABEL_HW[0] - (y + 0.5)) < 1.5


def test_sampler_source_ratio():
    rows = [{"source": "bdd100k"}] * 10000 + [{"source": "tusimple"}] * 100 + [{"source": "accident"}] * 50
    s = SourceWeightedSampler(rows, {"bdd100k": 0.6, "tusimple": 0.2, "accident": 0.2, "baton": 0.3}, 20000, 0)
    c = Counter(rows[i]["source"] for i in s)
    frac = {k: v / 20000 for k, v in c.items()}
    # baton absent -> renormalized over present sources
    assert abs(frac["bdd100k"] - 0.6) < 0.02 and abs(frac["tusimple"] - 0.2) < 0.02 and abs(frac["accident"] - 0.2) < 0.02


@needs_data
def test_labels_load_and_ranges():
    for src in ("bdd100k", "tusimple", "accident", "baton"):
        rows = load_manifest(MANIFESTS, [src], "val")
        if not rows:
            continue
        ds = StaticGeometryDataset(random.Random(0).sample(rows, min(8, len(rows))), augment=True)
        for i in range(len(ds)):
            b = ds[i]
            assert b["image"].shape == (3, 448, 800) and b["image"].dtype == torch.uint8
            for k in ("drivable", "lane", "curb", "objects", "contact"):
                v = b[k]
                assert v.shape == LABEL_HW
                ok = (v == IGNORE) | (v < 8)
                assert bool(ok.all()), (src, k)
            assert torch.isfinite(b["depth"]).all()


@needs_data
def test_pairs_load():
    pairs = load_manifest(MANIFESTS, ["tusimple"], "val", pairs=True)[:4]
    ds = PairFlowDataset(pairs, augment=True)
    b = ds[0]
    assert b["flow"].shape == (2, *LABEL_HW) and torch.isfinite(b["flow"]).all()


@needs_gpu
def test_gradients_only_in_trainable_blocks():
    from stage2.geometry_pretrain.models.geometry_dino import DinoBackbone, GeometryDINO, OriginalTail
    from stage2.geometry_pretrain.models.losses import anchor_losses

    bb = DinoBackbone("vits16", ORIGINAL, trainable_blocks=4)
    model = GeometryDINO(bb, ["road", "depth", "objects", "contact", "camera", "flow"]).cuda().train()
    tail = OriginalTail(bb).cuda()
    x = torch.randn(2, 3, 448, 800, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = model.forward_static(x)
        orig = tail(p["_backbone_out"])
        # Untrained: the original tail must reproduce the adapted features exactly.
        assert torch.allclose(orig.float(), p["_patch"].float(), atol=1e-2)
        loss = sum(v.float().mean() for k, v in p.items() if not k.startswith("_"))
        loss = loss + model.forward_pair(x, x.flip(-1))["flow"].float().mean()
        f, g = anchor_losses(p["_patch"] * 1.01, orig)
        loss = loss + f + g
    assert torch.isfinite(loss)
    loss.backward()
    for i, blk in enumerate(bb.model.blocks):
        has = any(q.grad is not None and q.grad.abs().sum() > 0 for q in blk.parameters())
        assert has == (i >= 8), (i, has)
    assert bb.model.patch_embed.proj.weight.grad is None
    assert all(q.grad is None for q in tail.parameters())
    assert all(torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
