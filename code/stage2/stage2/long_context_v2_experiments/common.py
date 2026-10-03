"""Shared helpers for long-context v2 experiments.

Everything here is FPS/duration/timestamp-blind on the model side: models only see features,
validity masks, frame numbers and normalized positions. FPS is read only inside the offline
metrics (same convention as spotting_experiments.train.official_metrics).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from stage2.spotting_experiments.data import collate, read_rows
from stage2.spotting_experiments.models import build_model
from stage2.spotting_experiments.objective import constrained_anchors
from stage2.spotting_experiments.sampling import normalized_indices, sample_from_cache
from stage2.spotting_experiments.train import macro_f1
from stage2.temporal_pyramid_experiments.data import adaptive_count
from stage2.temporal_pyramid_experiments.models import TemporalPyramid

REPO = Path(__file__).resolve().parents[2]
DENSE = REPO / "stage2/long_context_v2_experiments/cache_dense"
MANIFESTS = Path("/workspace/data/stage2/manifests")
BINS = [(0, 150, "<150"), (150, 300, "150-300"), (300, 500, "300-500"), (500, 750, "500-750"),
        (750, 1000, "750-1000"), (1000, 10 ** 9, ">1000")]
CATASTROPHIC = 0.10  # normalized |error| / (N-1) above this = wrong event selected


def rows(split):
    return read_rows(split if str(split).endswith(".jsonl") else MANIFESTS / f"{split}.jsonl")


def source(row_or_pred):
    return str(row_or_pred["source_id"]).split(":", 1)[0]


def frame_bin(n):
    return next(name for lo, hi, name in BINS if lo <= n < hi)


@lru_cache(maxsize=400)
def dense(sample_id):
    frames = np.load(DENSE / f"{sample_id}.frames.npy")
    fd = Path(os.environ["FEATS_DIR"]) if os.environ.get("FEATS_DIR") else DENSE  # alternative backbone features (same frames)
    feats = np.load(fd / f"{sample_id}.npy", mmap_mode="r")
    return frames, feats


# ---------------------------------------------------------------- sampling rules (indices into frame list)
def select_fixed128(frame_numbers):
    """E3/P1: 192 FPS-blind grid, then 128 evenly spaced grid positions (SpottingFeatures eval)."""
    grid = normalized_indices(frame_numbers, 192)
    return grid[sample_from_cache(np.asarray(frame_numbers)[grid], 128)]


def select_adaptive(frame_numbers):
    """P2 family: identical to PyramidFeatures(P2, eval) and the submission runtime."""
    n = len(frame_numbers)
    count = adaptive_count(n)
    grid = normalized_indices(frame_numbers, 192 if count <= 192 else 383)
    positions = sample_from_cache(np.asarray(frame_numbers)[grid], count)
    if n <= 128:
        positions = np.unique(np.asarray(frame_numbers)[grid], return_index=True)[1]
    return grid[positions]


SELECTORS = {"fixed128": select_fixed128, "adaptive": select_adaptive}
MOTION_DIM = 28


@lru_cache(maxsize=400)
def motion(sample_id):
    m = np.load(DENSE / f"{sample_id}.motion.npy")
    out = np.concatenate([np.abs(m[:, :3]), m[:, 3:4], np.log1p(100 * m[:, 4:])], 1).astype(np.float32)
    return out


def motion_segments(sample_id, abs_idx):
    """max and mean of per-frame motion over native frames (prev sample, this sample]; frame-index only."""
    m = motion(sample_id); feats = np.zeros((len(abs_idx), MOTION_DIM), np.float32)
    prev = abs_idx[0] - 1
    for t, a in enumerate(abs_idx):
        lo = min(prev + 1, a); seg = m[max(lo, 0):a + 1]
        feats[t] = np.concatenate([seg.max(0), seg.mean(0)]); prev = a
    return feats


def make_item(row, sampling, start=0, stop=None, positions=None):
    """Build a dataset item from the dense cache, optionally restricted to frames [start, stop)."""
    frames_all, feats_all = dense(row["sample_id"])
    stop = len(frames_all) if stop is None else stop
    frames_crop = frames_all[start:stop]
    idx = SELECTORS[sampling](frames_crop) if positions is None else np.asarray(positions)
    frames = frames_crop[idx]
    normalized = (frames - frames[0]).astype(np.float32) / max(int(frames[-1] - frames[0]), 1)
    item_motion = torch.from_numpy(motion_segments(row["sample_id"], start + idx)) if (DENSE / f"{row['sample_id']}.motion.npy").exists() else None
    return {"x": torch.from_numpy(np.ascontiguousarray(feats_all[start:stop][idx])), "motion": item_motion,
            "frame_numbers": torch.from_numpy(frames.copy()), "normalized_positions": torch.from_numpy(normalized),
            "entry_index": int(np.abs(frames - int(row["entry_frame"])).argmin()),
            "collision_index": int(np.abs(frames - int(row["collision_frame"])).argmin()),
            "entry_frame": int(row["entry_frame"]), "collision_frame": int(row["collision_frame"]),
            "entry_side": int(row["entry_side"] == "RIGHT"), "evasion": int(row["evasion_space"]),
            "sample_id": row["sample_id"], "source_id": row.get("source_id", "unknown"),
            "num_available_frames": int(stop - start)}


# ---------------------------------------------------------------- models
def load_checkpoint_model(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if cfg.get("head") == "asformer" or cfg.get("architecture") == "asformer":
        model = build_model(cfg)
    else:
        model = TemporalPyramid(global_coarse=cfg.get("global_coarse", False), category_head=cfg.get("category_head", "global"))
    model.load_state_dict(state["model"])
    return model.to(device).eval(), cfg


class ProbEnsemble(torch.nn.Module):
    def __init__(self, models):
        super().__init__(); self.models = torch.nn.ModuleList(models)

    @property
    def uses_motion(self): return any(getattr(m, "uses_motion", False) for m in self.models)

    def forward(self, x, valid, motion=None):
        parts = [m(x, valid, motion=motion) if getattr(m, "uses_motion", False) else m(x, valid) for m in self.models]
        out = {}
        for e in ("entry", "collision"):
            out[f"{e}_logits"] = torch.stack([p[f"{e}_logits"].float().softmax(-1) for p in parts]).mean(0).clamp_min(1e-12).log()
        out["side_logits"] = torch.stack([p["side_logits"].float().softmax(-1) for p in parts]).mean(0).clamp_min(1e-12).log()
        ev = torch.stack([p["evasion_logits"].float().sigmoid() for p in parts]).mean(0)
        out["evasion_logits"] = torch.logit(ev.clamp(1e-6, 1 - 1e-6))
        for k in parts[0]:
            if k not in out and torch.is_tensor(parts[0][k]) and parts[0][k].dtype.is_floating_point:
                out[k] = torch.stack([p[k].float() for p in parts]).mean(0)
        return out


TP = REPO / "stage2/temporal_pyramid_experiments/results"
BASELINES = {
    "E3_fixed128": (["stage2/spotting_experiments/results/E3-ASFormer-349/fixed/seed0/checkpoint.pt"], "fixed128"),
    "P2_seed0": ([f"{TP}/P2/seed0/checkpoint.pt"], "adaptive"),
    "P10_ema_seed0": ([f"{TP}/P10_ema099/seed0/checkpoint.pt"], "adaptive"),
    "P2_ens4": ([f"{TP}/P2/seed{s}/checkpoint.pt" for s in range(4)], "adaptive"),
}


def load_named(name, device):
    paths, sampling = BASELINES[name]
    models = [load_checkpoint_model(REPO / p if not str(p).startswith("/") else p, device)[0] for p in paths]
    return (models[0] if len(models) == 1 else ProbEnsemble(models).eval()), sampling


def collate_m(items):
    batch = collate(items)
    if all(it.get("motion") is not None for it in items):
        m = torch.zeros(len(items), batch["x"].shape[1], MOTION_DIM)
        for j, it in enumerate(items): m[j, :len(it["motion"])] = it["motion"]
        batch["motion"] = m
    else:
        batch["motion"] = torch.zeros(len(items), batch["x"].shape[1], MOTION_DIM)
    return batch


# ---------------------------------------------------------------- inference
@torch.inference_mode()
def run(model, items, device, batch_size=1, keep_logits=False):
    """batch_size=1 matches the submission runtime (GroupNorm in the pyramid is padding-sensitive)."""
    preds = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        batch = collate_m(chunk)
        x, valid = batch["x"].to(device), batch["time_valid"].to(device)
        out = model(x, valid, motion=batch["motion"].to(device)) if getattr(model, "uses_motion", False) else model(x, valid)
        ei, ci = constrained_anchors(out["entry_logits"], out["collision_logits"])
        if getattr(model, "decoder", "direct") == "structured" and "phase_logits" in out:
            from .phase_loss import decode_structured
            pairs = [decode_structured(out["phase_logits"][j, :len(it["x"])]) for j, it in enumerate(chunk)]
            ei = torch.tensor([p_[0] for p_ in pairs]); ci = torch.tensor([p_[1] for p_ in pairs])
        for j, it in enumerate(chunk):
            t = len(it["x"]); frames = it["frame_numbers"].numpy()
            p = {"sample_id": it["sample_id"], "source_id": it["source_id"],
                 "entry_frame": int(frames[int(ei[j])]), "collision_frame": int(frames[int(ci[j])]),
                 "entry_side": int(out["side_logits"][j].argmax()), "evasion_space": int(out["evasion_logits"][j] >= 0),
                 "entry_gt": it["entry_frame"], "collision_gt": it["collision_frame"],
                 "entry_side_gt": it["entry_side"], "evasion_gt": it["evasion"],
                 "num_available_frames": it["num_available_frames"], "num_positions": t}
            if keep_logits:
                p["frames"] = frames.tolist()
                p["entry_logits"] = out["entry_logits"][j, :t].float().cpu().numpy().tolist()
                p["collision_logits"] = out["collision_logits"][j, :t].float().cpu().numpy().tolist()
                for k in ("phase_logits",):
                    if k in out: p[k] = out[k][j, :t].float().cpu().numpy().tolist()
            preds.append(p)
    return preds


# ---------------------------------------------------------------- metrics
@lru_cache(maxsize=1)
def fps_table():
    return {r["sample_id"]: float(r["native_fps"]) for r in rows("all")}


def hit(pred, event):
    return abs(pred[f"{event}_frame"] - pred[f"{event}_gt"]) / fps_table()[pred["sample_id"]] <= .300001


def metrics(preds):
    if not preds: return {"n": 0}
    out = {"n": len(preds)}
    for e in ("entry", "collision"):
        err = np.array([abs(p[f"{e}_frame"] - p[f"{e}_gt"]) for p in preds], float)
        norm = err / np.array([max(p["num_available_frames"] - 1, 1) for p in preds])
        out[f"{e}_acc"] = float(np.mean([hit(p, e) for p in preds]))
        out[f"{e}_mae_frames"] = float(err.mean()); out[f"{e}_median_frames"] = float(np.median(err))
        out[f"{e}_norm_mae"] = float(norm.mean())
        out[f"{e}_catastrophic"] = float((norm > CATASTROPHIC).mean())
    out["side_f1"] = macro_f1([p["entry_side_gt"] for p in preds], [p["entry_side"] for p in preds])
    out["evasion_f1"] = macro_f1([p["evasion_gt"] for p in preds], [p["evasion_space"] for p in preds])
    out["score"] = .35 * out["entry_acc"] + .35 * out["collision_acc"] + .15 * out["side_f1"] + .15 * out["evasion_f1"]
    return out


def breakdown(preds):
    """Overall, by source, by frame-count bin (bins use the full clip length from the manifest)."""
    nframes = {r["sample_id"]: int(r["num_frames"]) for r in rows("all")}
    res = {"overall": metrics(preds)}
    for s in sorted({source(p) for p in preds}):
        res[f"source:{s}"] = metrics([p for p in preds if source(p) == s])
    res["source:non-NEXAR"] = metrics([p for p in preds if source(p) != "NEXAR"])
    for _, _, b in BINS:
        res[f"bin:{b}"] = metrics([p for p in preds if frame_bin(nframes[p["sample_id"]]) == b])
    return res


def short_table(res, keys=("overall", "source:NEXAR", "source:non-NEXAR", "bin:>1000")):
    cells = []
    for k in keys:
        m = res.get(k, {})
        if m.get("n"):
            cells.append(f"{k.split(':')[-1]} n={m['n']} S={m['score']:.3f} E={m['entry_acc']:.2f} C={m['collision_acc']:.2f}")
    return " | ".join(cells)


def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float) + "\n")
