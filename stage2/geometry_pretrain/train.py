"""Geometry-aware DINOv3 pretraining (Phase 0 probe, Phase 1/2 adaptation).

    python -m stage2.geometry_pretrain.train --config stage2/geometry_pretrain/configs/phase1_partial.yaml

Step-based training with a static-image stream and a temporal-pair stream,
source-weighted sampling, bf16 autocast, gradient accumulation, periodic
held-out evaluation, best-checkpoint selection and exact resume.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from stage2.geometry_pretrain.common import DiskGuard, setup_logging, LOG
from stage2.geometry_pretrain.datasets import (
    PairFlowDataset,
    SourceWeightedSampler,
    StaticGeometryDataset,
    infinite,
    load_manifest,
)
from stage2.geometry_pretrain.metrics import GeometryMeter, geometry_score
from stage2.geometry_pretrain.models.geometry_dino import (
    DinoBackbone,
    GeometryDINO,
    OriginalTail,
    count_parameters,
)
from stage2.geometry_pretrain.models.losses import (
    LossManager,
    anchor_losses,
    binary_thin,
    depth_gradient,
    flow_loss,
    seg_ce,
    ssi_depth,
)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def deep_update(base, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path, overrides=()):
    cfg = yaml.safe_load(Path(path).read_text())
    if "base" in cfg:
        base = load_config(Path(path).parent / cfg.pop("base"))
        cfg = deep_update(base, cfg)
    for o in overrides:
        k, v = o.split("=", 1)
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(v)
    return cfg


def to_input(x, device):
    x = x.to(device, non_blocking=True).float()
    return (x - MEAN.to(device)) / STD.to(device)


def build_model(cfg, device):
    m = cfg["model"]
    backbone = DinoBackbone(
        m["arch"], m["checkpoint"], m.get("dinov3_source", "/workspace/pretrained/dinov3-source"),
        trainable_blocks=m.get("trainable_blocks", 0), out_blocks=tuple(m.get("out_blocks", [-1])),
        grad_checkpointing=m.get("grad_checkpointing", False),
    )
    model = GeometryDINO(backbone, cfg["tasks"], m.get("head_dim", 128)).to(device)
    return model


# ----------------------------------------------------------------------------
# Losses for one batch
# ----------------------------------------------------------------------------


def static_losses(preds, batch, cfg):
    lc = cfg.get("loss", {})
    out = {}
    if "road" in preds:
        r = preds["road"]
        out["road"], _ = seg_ce(r[:, :3], batch["drivable"], batch["drivable_w"], boundary_boost=lc.get("boundary_boost", 2.0))
        out["lane_boundary"], _ = binary_thin(r[:, 3], batch["lane"], batch["lanecurb_w"], pos_weight=lc.get("lane_pos_weight", 4.0))
        out["road_edge"], _ = binary_thin(r[:, 4], batch["curb"], batch["lanecurb_w"], pos_weight=lc.get("curb_pos_weight", 4.0))
    if "depth" in preds:
        d = preds["depth"][:, 0]
        l_ssi, n = ssi_depth(d, batch["depth"], batch["depth_w"])
        out["depth"] = l_ssi + lc.get("depth_grad_weight", 0.5) * depth_gradient(d, batch["depth"], batch["depth_w"]) if n > 0 else l_ssi
    if "objects" in preds:
        cw = torch.tensor(lc.get("object_class_weights", [0.5, 1, 1, 1, 1.5, 1.5]), device=preds["objects"].device)
        out["object"], _ = seg_ce(preds["objects"], batch["objects"], batch["objects_w"], class_weight=cw)
    if "contact" in preds:
        out["contact"], _ = binary_thin(preds["contact"][:, 0], batch["contact"], batch["contact_w"], pos_weight=lc.get("contact_pos_weight", 5.0))
    if "camera" in preds:
        v = batch["vp_valid"]
        if v.any():
            out["camera"] = F.smooth_l1_loss(preds["camera"][v].float(), batch["vp"][v], beta=0.02)
        else:
            out["camera"] = preds["camera"].sum() * 0
    return out


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, static_loader, pair_loader, device, tasks, max_batches=None, original_tail=None):
    model.eval()
    meter = GeometryMeter()
    drift = {"cos": 0.0, "n": 0}
    for i, batch in enumerate(static_loader):
        if max_batches and i >= max_batches:
            break
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds = model.forward_static(to_input(batch["image"], device))
        if "road" in preds:
            meter.update_road(preds["road"].float(), batch)
        if "objects" in preds:
            meter.update_objects(preds["objects"].float(), batch)
        if "contact" in preds:
            meter.update_contact(preds["contact"].float(), batch)
        if "depth" in preds:
            meter.update_depth(preds["depth"][:, 0].float(), batch)
        if "camera" in preds:
            meter.update_camera(preds["camera"].float(), batch)
        if original_tail is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                orig = original_tail(preds["_backbone_out"])
            cos = F.cosine_similarity(preds["_patch"].float(), orig.float(), dim=1).mean()
            drift["cos"] += float(cos)
            drift["n"] += 1
    if pair_loader is not None and "flow" in tasks:
        for i, batch in enumerate(pair_loader):
            if max_batches and i >= max_batches:
                break
            batch = to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.forward_pair(to_input(batch["image_t"], device), to_input(batch["image_tplus"], device))
            meter.update_flow(out["flow"].float(), batch)
    metrics = meter.compute()
    metrics["geometry_score"] = geometry_score(metrics)
    if drift["n"]:
        metrics["drift/patch_cos_to_original"] = drift["cos"] / drift["n"]
    model.train()
    return metrics


def val_loaders(cfg):
    d = cfg["data"]
    rng = random.Random(0)
    rows = []
    for src in d["val_sources"]:
        r = load_manifest(d["manifest_dir"], [src], "val")
        cap = d.get("val_cap", {}).get(src)
        if cap and len(r) > cap:
            r = rng.sample(r, cap)
        rows += r
    pairs = []
    for src in d["val_pair_sources"]:
        p = load_manifest(d["manifest_dir"], [src], "val", pairs=True)
        cap = d.get("val_pair_cap", {}).get(src)
        if cap and len(p) > cap:
            p = rng.sample(p, cap)
        pairs += p
    nw = d.get("val_workers", 4)
    sl = DataLoader(StaticGeometryDataset(rows, False, label_cfg=d.get("labels")), batch_size=d["val_batch_size"], num_workers=nw)
    pl = DataLoader(PairFlowDataset(pairs, False, min_consistent=d.get("flow_min_consistent", 0.3)), batch_size=d["val_batch_size"] // 2, num_workers=nw) if pairs else None
    return sl, pl, len(rows), len(pairs)


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------


def lr_lambda(step, total, warmup, min_ratio):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. optim.steps=200")
    parser.add_argument("--resume", default="auto", help="'auto' (output_dir/last.pt), a path, or 'none'")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--gpu-mem-fraction", type=float, default=None, help="cap this process on a shared GPU")
    args = parser.parse_args()
    if args.gpu_mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
    cfg = load_config(args.config, args.set)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "train.log")
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    seed = cfg.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    guard = DiskGuard(out_dir, cfg.get("min_free_gb", 12.0))
    guard.check("at start")

    model = build_model(cfg, device)
    if cfg.get("init_from"):
        state = torch.load(cfg["init_from"], map_location="cpu", weights_only=False)["model"]
        keep = {k: v for k, v in state.items() if not k.startswith("backbone.") or cfg.get("init_backbone", False)}
        missing, unexpected = model.load_state_dict(keep, strict=False)
        LOG.info("init_from %s: loaded %d tensors, missing %d (backbone ok), unexpected %d",
                 cfg["init_from"], len(keep), len([m for m in missing if not m.startswith("backbone.")]), len(unexpected))
    ocfg = cfg["optim"]
    anchor_w = cfg.get("anchor", {})
    use_anchor = model.backbone.trainable_blocks > 0 and (anchor_w.get("feature", 0) > 0 or anchor_w.get("gram", 0) > 0)
    original_tail = OriginalTail(model.backbone).to(device) if model.backbone.trainable_blocks > 0 else None
    weights = dict(cfg["loss_weights"])
    weights["anchor_feature"] = anchor_w.get("feature", 0.0)
    weights["anchor_gram"] = anchor_w.get("gram", 0.0)
    loss_manager = LossManager(weights, cfg.get("weighting", "manual")).to(device)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{"params": model.head_parameters(), "lr": ocfg["lr_heads"], "weight_decay": ocfg.get("wd_heads", 1e-4), "name": "heads"}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": ocfg["lr_backbone"], "weight_decay": ocfg.get("wd_backbone", 0.05), "name": "backbone"})
    if loss_manager.log_vars is not None:
        groups.append({"params": list(loss_manager.parameters()), "lr": ocfg.get("lr_loss_weights", 1e-3), "weight_decay": 0.0, "name": "loss_weights"})
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    total = ocfg["steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, total, ocfg.get("warmup", 500), ocfg.get("min_lr_ratio", 0.05))
    )
    n_total, n_train = count_parameters(model), count_parameters(model, True)
    LOG.info("params: total %.2fM trainable %.2fM (backbone trainable %.2fM, heads %.2fM); trainable blocks %d/%d",
             n_total / 1e6, n_train / 1e6, sum(p.numel() for p in backbone_params) / 1e6,
             sum(p.numel() for p in model.head_parameters()) / 1e6, model.backbone.trainable_blocks, model.backbone.depth)

    d = cfg["data"]
    static_rows = load_manifest(d["manifest_dir"], list(d["static_weights"]), "train")
    pair_rows = load_manifest(d["manifest_dir"], list(d.get("pair_weights", {})), "train", pairs=True) if "flow" in cfg["tasks"] else []
    accum = ocfg.get("accumulation", 1)
    bs, pbs = d["batch_size"], d.get("pair_batch_size", 0)
    static_loader = DataLoader(
        StaticGeometryDataset(static_rows, True, d.get("augment"), d.get("labels"), seed),
        batch_size=bs, num_workers=d.get("workers", 5), pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4,
        sampler=SourceWeightedSampler(static_rows, d["static_weights"], total * accum * bs, seed),
    )
    pair_loader = None
    if pair_rows and pbs > 0:
        pair_loader = DataLoader(
            PairFlowDataset(pair_rows, True, d.get("augment"), d.get("flow_min_consistent", 0.3), seed=seed),
            batch_size=pbs, num_workers=d.get("pair_workers", 3), pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4,
            sampler=SourceWeightedSampler(pair_rows, d["pair_weights"], total * accum * pbs, seed + 1),
        )
    LOG.info("train static rows %d (%s), pair rows %d", len(static_rows),
             {s: sum(r["source"] == s for r in static_rows) for s in d["static_weights"]}, len(pair_rows))
    v_static, v_pairs, nv, npv = val_loaders(cfg)
    LOG.info("val static %d, val pairs %d", nv, npv)

    step, best = 0, -1.0
    resume = None
    if args.resume == "auto" and (out_dir / "last.pt").is_file():
        resume = out_dir / "last.pt"
    elif args.resume not in ("auto", "none"):
        resume = Path(args.resume)
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        loss_manager.load_state_dict(ck["loss_manager"])
        step, best = ck["step"], ck["best"]
        torch.set_rng_state(ck["rng"]["torch"])
        np.random.set_state(ck["rng"]["numpy"])
        random.setstate(ck["rng"]["python"])
        LOG.info("resumed from %s at step %d (best %.4f)", resume, step, best)
        # Continue the sampler streams where they stopped (deterministic by epoch index).
        static_loader.sampler.set_epoch(step)
        if pair_loader is not None:
            pair_loader.sampler.set_epoch(step)

    if args.eval_only:
        m = evaluate(model, v_static, v_pairs, device, cfg["tasks"], original_tail=original_tail)
        (out_dir / "eval_only.json").write_text(json.dumps(m, indent=1))
        LOG.info(json.dumps(m, indent=1))
        return

    def save(name, extra=None):
        guard.check(f"before saving {name}")
        ck = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "loss_manager": loss_manager.state_dict(), "step": step, "best": best, "config": cfg,
            "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()},
        }
        if extra:
            ck.update(extra)
        tmp = out_dir / f"{name}.tmp"
        torch.save(ck, tmp)
        os.replace(tmp, out_dir / name)

    def export_backbone(name):
        tmp = out_dir / f"{name}.tmp"
        torch.save(model.backbone_state_dict(), tmp)
        os.replace(tmp, out_dir / name)

    s_iter = infinite(static_loader)
    p_iter = infinite(pair_loader) if pair_loader is not None else None
    metrics_log = open(out_dir / "metrics.jsonl", "a")
    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0, seen = time.time(), 0
    log_every, eval_every, ckpt_every = cfg.get("log_every", 50), cfg.get("eval_every", 1000), cfg.get("ckpt_every", 500)
    agg = {}
    source_counts = {}
    while step < total:
        optimizer.zero_grad(set_to_none=True)
        logs_step = {}
        for _ in range(accum):
            batch = to_device(next(s_iter), device)
            for s in batch["source"].tolist():
                source_counts[f"static_{s}"] = source_counts.get(f"static_{s}", 0) + 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds = model.forward_static(to_input(batch["image"], device))
                losses = static_losses(preds, batch, cfg)
                if use_anchor:
                    orig = original_tail(preds["_backbone_out"])
                    losses["anchor_feature"], losses["anchor_gram"] = anchor_losses(preds["_patch"], orig)
                if p_iter is not None:
                    pb = to_device(next(p_iter), device)
                    for s in pb["source"].tolist():
                        source_counts[f"pair_{s}"] = source_counts.get(f"pair_{s}", 0) + 1
                    fo = model.forward_pair(to_input(pb["image_t"], device), to_input(pb["image_tplus"], device))
                    losses["flow"], _ = flow_loss(fo["flow"], pb["flow"], pb["flow_w"])
                total_loss, logs = loss_manager(losses)
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {logs}")
            (total_loss / accum).backward()
            seen += len(batch["image"]) + (2 * pb["image_t"].shape[0] if p_iter is not None else 0)
            logs["loss/total"] = float(total_loss.detach())
            for k, v in logs.items():
                logs_step[k] = logs_step.get(k, 0.0) + v / accum
        gn = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], ocfg.get("grad_clip", 1.0))
        if backbone_params:
            logs_step["grad_norm/backbone"] = float(torch.nn.utils.clip_grad_norm_(backbone_params, 1e9))
        logs_step["grad_norm/total"] = float(gn)
        optimizer.step()
        scheduler.step()
        step += 1
        for k, v in logs_step.items():
            agg[k] = agg.get(k, 0.0) + v
        if step % log_every == 0:
            rec = {k: v / log_every for k, v in agg.items()}
            rec.update(step=step, lr_heads=optimizer.param_groups[0]["lr"],
                       lr_backbone=optimizer.param_groups[1]["lr"] if backbone_params else 0.0,
                       img_per_s=seen / (time.time() - t0), peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
                       source_counts=source_counts)
            metrics_log.write(json.dumps({"type": "train", **rec}) + "\n")
            metrics_log.flush()
            LOG.info("step %d/%d total %.4f | %s | %.1f img/s | vram %.1fGB", step, total, rec["loss/total"],
                     " ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in rec.items() if k.startswith("loss/") and k != "loss/total"),
                     rec["img_per_s"], rec["peak_vram_gb"])
            agg = {}
        if step % ckpt_every == 0 or step == total:
            save("last.pt")
        if step % eval_every == 0 or step == total:
            m = evaluate(model, v_static, v_pairs, device, cfg["tasks"], original_tail=original_tail)
            metrics_log.write(json.dumps({"type": "val", "step": step, **m}) + "\n")
            metrics_log.flush()
            LOG.info("VAL step %d score %.4f | %s", step, m["geometry_score"],
                     " ".join(f"{k}={v:.3f}" for k, v in m.items() if k.startswith("bdd100k/") or k.startswith("drift")))
            if m["geometry_score"] > best:
                best = m["geometry_score"]
                save("best.pt", {"val_metrics": m})
                export_backbone("backbone_best.pth")
                LOG.info("new best %.4f at step %d", best, step)
            save("last.pt")
    export_backbone("backbone_last.pth")
    summary = {"steps": step, "best_score": best, "train_time_s": time.time() - t0, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
               "samples_seen": seen, "source_counts": source_counts}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=1))
    LOG.info("done: %s", summary)


if __name__ == "__main__":
    main()
