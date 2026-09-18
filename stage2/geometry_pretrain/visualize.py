"""Qualitative panels and training curves.

    python -m stage2.geometry_pretrain.visualize predictions --runs RUN_A RUN_B --out DIR
    python -m stage2.geometry_pretrain.visualize curves --runs RUN_A RUN_B --out DIR
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from stage2.geometry_pretrain.common import IGNORE, INPUT_HW
from stage2.geometry_pretrain.datasets import PairFlowDataset, StaticGeometryDataset, load_manifest
from stage2.geometry_pretrain.train import MEAN, STD, build_model

W, H = 400, 224
OBJ_PAL = np.array([[0, 0, 0], [255, 128, 0], [0, 200, 255], [255, 0, 255], [0, 255, 0], [255, 255, 0]], np.uint8)
DRV_PAL = np.array([[0, 0, 0], [0, 0, 220], [220, 0, 0]], np.uint8)


def up(x, interp=cv2.INTER_NEAREST):
    return cv2.resize(x, (W, H), interpolation=interp)


def colorize(lab, pal):
    out = pal[np.clip(lab, 0, len(pal) - 1)]
    out[lab == IGNORE] = (70, 70, 70)
    return up(out)


def overlay_binary(img, mask, color, ignore=None):
    o = img.copy()
    m = up(mask.astype(np.uint8)) > 0
    o[m] = color
    if ignore is not None:
        ig = up(ignore.astype(np.uint8)) > 0
        o[ig & ~m] = (o[ig & ~m] * 0.4).astype(np.uint8)
    return o


def depth_color(d, w=None):
    d = d.astype(np.float32)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    c = cv2.applyColorMap((np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    if w is not None:
        c[w <= 0] = (60, 60, 60)
    return up(c, cv2.INTER_LINEAR)


def flow_color(f, scale=None):
    mag, ang = cv2.cartToPolar(f[0].astype(np.float32), f[1].astype(np.float32))
    hsv = np.zeros(f.shape[1:] + (3,), np.uint8)
    hsv[..., 0] = ang * 90 / np.pi
    hsv[..., 1] = 255
    s = scale or max(np.percentile(mag, 99), 1e-3)
    hsv[..., 2] = np.clip(mag * 255 / s, 0, 255)
    return up(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)), s


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def load_run(run):
    ck = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    cfg["model"]["checkpoint"] = None
    model = build_model(cfg, "cuda")
    model.load_state_dict(ck["model"])
    return model.eval(), cfg


@torch.no_grad()
def cmd_predictions(args):
    models = [load_run(r) for r in args.runs]
    names = [Path(r).name for r in args.runs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = models[0][1]["data"]["manifest_dir"]
    for src in args.sources:
        rows = load_manifest(md, [src], "val")
        rows = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
        ds = StaticGeometryDataset(rows, False)
        panels = []
        for i in range(len(ds)):
            b = ds[i]
            img = cv2.cvtColor(b["image"].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
            base = up(img, cv2.INTER_AREA)
            x = (b["image"][None].cuda().float() - MEAN.cuda()) / STD.cuda()
            tgt_row = [label(base, f"{src} RGB"),
                       label(colorize(b["drivable"].numpy(), DRV_PAL), "T drivable"),
                       label(overlay_binary(overlay_binary(base, b["lane"].numpy() == 1, (0, 255, 255), b["lane"].numpy() == IGNORE), b["curb"].numpy() == 1, (255, 0, 255)), "T lane/curb"),
                       label(colorize(b["objects"].numpy(), OBJ_PAL), "T objects"),
                       label(overlay_binary(base, b["contact"].numpy() == 1, (0, 255, 0), b["contact"].numpy() == IGNORE), "T contact"),
                       label(depth_color(b["depth"].numpy(), b["depth_w"].numpy()), "T depth")]
            rows_img = [np.hstack(tgt_row)]
            for (model, _), name in zip(models, names):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = model.forward_static(x)
                r = p["road"][0].float().cpu()
                pred_row = [label(base, name[:28]),
                            label(colorize(r[:3].argmax(0).numpy().astype(np.uint8), DRV_PAL), "P drivable"),
                            label(overlay_binary(overlay_binary(base, (r[3] > 0).numpy(), (0, 255, 255)), (r[4] > 0).numpy(), (255, 0, 255)), "P lane/curb"),
                            label(colorize(p["objects"][0].float().argmax(0).cpu().numpy().astype(np.uint8), OBJ_PAL), "P objects"),
                            label(overlay_binary(base, (p["contact"][0, 0] > 0).float().cpu().numpy() > 0, (0, 255, 0)), "P contact"),
                            label(depth_color(p["depth"][0, 0].float().cpu().numpy()), "P depth")]
                if "camera" in p and bool(b["vp_valid"]):
                    for im, xy, col in ((pred_row[0], p["camera"][0].float().cpu().numpy(), (0, 0, 255)), (tgt_row[0], b["vp"].numpy(), (255, 255, 255))):
                        cv2.circle(im, (int(xy[0] * W), int(xy[1] * H)), 5, col, 2)
                rows_img.append(np.hstack(pred_row))
            panels.append(np.vstack(rows_img + [np.zeros((6, rows_img[0].shape[1], 3), np.uint8)]))
        cv2.imwrite(str(out / f"pred_{src}.jpg"), np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 85])
        # Flow panels.
        pairs = load_manifest(md, [src], "val", pairs=True)
        if not pairs:
            continue
        pairs = random.Random(args.seed).sample(pairs, min(args.n, len(pairs)))
        pds = PairFlowDataset(pairs, False)
        fp = []
        for i in range(len(pds)):
            b = pds[i]
            a = up(cv2.cvtColor(b["image_t"].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR), cv2.INTER_AREA)
            c = up(cv2.cvtColor(b["image_tplus"].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR), cv2.INTER_AREA)
            tf, s = flow_color(b["flow"].numpy())
            tf[up((b["flow_w"].numpy() <= 0).astype(np.uint8)) > 0] //= 3
            row = [label(a, f"{src} t  dt={float(b['dt']):.2f}s"), label(c, "t+dt"), label(tf, "T flow (SEA-RAFT)")]
            for (model, cfg), name in zip(models, names):
                if "flow" not in cfg["tasks"]:
                    continue
                xa = (b["image_t"][None].cuda().float() - MEAN.cuda()) / STD.cuda()
                xb = (b["image_tplus"][None].cuda().float() - MEAN.cuda()) / STD.cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pf = model.forward_pair(xa, xb)["flow"][0].float().cpu().numpy()
                row.append(label(flow_color(pf, s)[0], f"P {name[:20]}"))
            fp.append(np.hstack(row))
        cv2.imwrite(str(out / f"flow_{src}.jpg"), np.vstack(fp), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", sorted(p.name for p in out.glob("*.jpg")))


def cmd_curves(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {Path(r).name: [json.loads(l) for l in open(Path(r) / "metrics.jsonl")] for r in args.runs}
    color = {name: f"C{i}" for i, name in enumerate(runs)}  # one color per run in every panel
    loss_keys = sorted({k for recs in runs.values() for r in recs if r["type"] == "train" for k in r if k.startswith("loss/")})
    n = len(loss_keys)
    cols = 4
    fig, axes = plt.subplots(math_ceil(n / cols), cols, figsize=(4 * cols, 3 * math_ceil(n / cols)))
    for ax, k in zip(np.array(axes).flatten(), loss_keys):
        for name, recs in runs.items():
            tr = [r for r in recs if r["type"] == "train" and k in r]
            if tr:
                ax.plot([r["step"] for r in tr], [r[k] for r in tr], label=name, lw=1, color=color[name])
        ax.set_title(k, fontsize=9)
        ax.set_yscale("log")
    np.array(axes).flatten()[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "train_losses.png", dpi=110)
    plt.close(fig)
    val_keys = args.val_keys
    fig, axes = plt.subplots(math_ceil(len(val_keys) / cols), cols, figsize=(4 * cols, 3 * math_ceil(len(val_keys) / cols)))
    for ax, k in zip(np.array(axes).flatten(), val_keys):
        for name, recs in runs.items():
            va = [r for r in recs if r["type"] == "val" and k in r]
            if va:
                ax.plot([r["step"] for r in va], [r[k] for r in va], marker="o", ms=3, label=name, lw=1, color=color[name])
        ax.set_title(k, fontsize=9)
    fig.legend(*np.array(axes).flatten()[0].get_legend_handles_labels(), loc="lower center", ncol=len(runs), fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "val_metrics.png", dpi=110)
    plt.close(fig)
    print("wrote", out / "train_losses.png", out / "val_metrics.png")


def math_ceil(x):
    return int(np.ceil(x))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predictions")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sources", nargs="+", default=["bdd100k", "tusimple", "accident", "baton"])
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--seed", type=int, default=3)
    c = sub.add_parser("curves")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--val-keys", nargs="+", default=[
        "geometry_score", "bdd100k/drivable_miou", "bdd100k/lane_bf1", "bdd100k/curb_bf1", "tusimple/lane_bf1",
        "bdd100k/object_miou", "bdd100k/contact_bf1", "bdd100k/depth_delta1", "accident/depth_delta1",
        "tusimple/flow_epe", "accident/flow_epe", "bdd100k/vp_err_px", "drift/patch_cos_to_original",
        "accident/drivable_miou", "accident/object_miou", "baton/flow_epe"])
    args = parser.parse_args()
    cmd_predictions(args) if args.cmd == "predictions" else cmd_curves(args)


if __name__ == "__main__":
    main()
