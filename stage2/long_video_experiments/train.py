"""Train a full-coverage tiled spotter without touching other runs.

Uses the existing FPS-blind frozen DINO feature cache. Every cached position is
scored through overlapping local windows; a tiny window-level encoder supplies
whole-video context. This is a low-memory prototype, not native-frame inference.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from stage2.spotting_experiments.data import SpottingFeatures, collate, read_rows
from stage2.spotting_experiments.objective import decode, spotting_loss
from stage2.spotting_experiments.train import normalized_metrics, official_metrics


class TiledSpotter(nn.Module):
    def __init__(self, tile=32, stride=16, hidden=128, dropout=0.3):
        super().__init__()
        self.tile, self.stride = tile, stride
        self.norm = nn.LayerNorm(384)
        self.token = nn.Linear(384, 16)
        self.frame = nn.Sequential(nn.Dropout(dropout), nn.Linear(70 * 16, hidden), nn.GELU())
        self.local = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(2)
        ])
        self.tile_encoder = nn.TransformerEncoderLayer(
            hidden, 4, hidden * 2, dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.context_gate = nn.Linear(hidden * 2, hidden)
        self.event = nn.Linear(hidden, 2)
        self.tile_event = nn.Linear(hidden, 2)
        self.attribute = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.side = nn.Linear(hidden, 2)
        self.evasion = nn.Linear(hidden, 1)

    def forward(self, x, valid):
        batch, length = x.shape[:2]
        h = self.frame(self.token(self.norm(x.float())).flatten(2))
        starts = list(range(0, length - self.tile + 1, self.stride))
        if starts[-1] != length - self.tile:
            starts.append(length - self.tile)
        windows = torch.stack([h[:, s:s + self.tile] for s in starts], dim=1)
        window_valid = torch.stack([valid[:, s:s + self.tile] for s in starts], dim=1)
        nw = len(starts)
        local = windows.reshape(batch * nw, self.tile, -1)
        local_valid = window_valid.reshape(batch * nw, self.tile)
        safe_valid = local_valid.clone()
        safe_valid[~safe_valid.any(-1), 0] = True
        for layer in self.local:
            local = layer(local, src_key_padding_mask=~safe_valid)
            local = local.masked_fill(~local_valid[..., None], 0)
        local = local.reshape(batch, nw, self.tile, -1)
        tile_tokens = (local * window_valid[..., None]).sum(2) / window_valid.sum(2).clamp_min(1)[..., None]
        tile_valid = window_valid.any(2)
        context = self.tile_encoder(tile_tokens, src_key_padding_mask=~tile_valid)
        gate = torch.sigmoid(self.context_gate(torch.cat([local, context[:, :, None].expand_as(local)], -1)))
        local = local + gate * context[:, :, None]
        tile_logits = self.tile_event(context)
        window_logits = self.event(local) + 0.5 * tile_logits[:, :, None]
        merged = torch.zeros(batch, length, 2, device=x.device)
        counts = torch.zeros(batch, length, 1, device=x.device)
        for j, start in enumerate(starts):
            merged[:, start:start + self.tile] += window_logits[:, j] * window_valid[:, j, :, None]
            counts[:, start:start + self.tile] += window_valid[:, j, :, None]
        merged = merged / counts.clamp_min(1)
        neg = torch.finfo(merged.dtype).min / 4
        pooled = (context * tile_valid[..., None]).sum(1) / tile_valid.sum(1).clamp_min(1)[:, None]
        pooled = self.attribute(pooled)
        return {
            "entry_logits": merged[..., 0].masked_fill(~valid, neg),
            "collision_logits": merged[..., 1].masked_fill(~valid, neg),
            "side_logits": self.side(pooled),
            "evasion_logits": self.evasion(pooled).squeeze(-1),
            "tile_logits": tile_logits,
            "tile_starts": starts,
        }


def tile_loss(outputs, batch, tile, negatives=3):
    logits = outputs["tile_logits"]
    starts = torch.tensor(outputs["tile_starts"], device=logits.device)
    target = torch.stack([batch["entry_index"], batch["collision_index"]], -1)
    positive = ((target[:, None] >= starts[None, :, None]) &
                (target[:, None] < (starts + tile)[None, :, None])).float()
    element = F.binary_cross_entropy_with_logits(logits, positive, reduction="none")
    pos = (element * positive).sum(1) / positive.sum(1).clamp_min(1)
    neg = element.masked_fill(positive.bool(), -1).topk(min(negatives, logits.shape[1]), dim=1).values.clamp_min(0).mean(1)
    return (pos + neg).mean()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    predictions = []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch["x"], batch["time_valid"])
        entry, collision = decode(out, batch)
        side = out["side_logits"].argmax(-1)
        evasion = (out["evasion_logits"] >= 0).long()
        for i, sample_id in enumerate(batch["sample_id"]):
            predictions.append({
                "sample_id": sample_id, "source_id": batch["source_id"][i],
                "entry_frame": int(entry[i]), "collision_frame": int(collision[i]),
                "entry_side": int(side[i]), "evasion_space": int(evasion[i]),
                "entry_gt": int(batch["entry_frame"][i]),
                "collision_gt": int(batch["collision_frame"][i]),
                "entry_side_gt": int(batch["entry_side"][i]),
                "evasion_gt": int(batch["evasion"][i]),
                "num_available_frames": int(batch["num_available_frames"][i]),
            })
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tile", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--eval-count", type=int, default=192)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.04)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    device = torch.device("cpu")
    if not args.cpu and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        if free < 2 * 1024 ** 3:
            raise RuntimeError(f"Only {free / 1024**3:.2f} GiB GPU free; refusing to affect other runs")
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        device = torch.device("cuda")
    manifest = Path("/workspace/data/stage2/manifests")
    rows = read_rows(manifest / "all.jsonl")
    train_rows = read_rows(manifest / "train.jsonl")
    val_rows = read_rows(manifest / "val.jsonl")
    cache = Path("/workspace/car-accident/stage2/artifacts/spotting_features/fpsblind_192_7x10_349")
    common = {"cache_dir": cache, "strict_fps_blind": True}
    train_set = SpottingFeatures(train_rows, sample_counts=(96, 128, 160, 192), train=True,
                                 jitter=0.45, drop_probability=0.03, seed=args.seed, **common)
    val_set = SpottingFeatures(val_rows, sample_counts=(args.eval_count,), train=False,
                               seed=args.seed, **common)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collate,
                              num_workers=0, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_set, batch_size=8, collate_fn=collate, num_workers=0)
    model = TiledSpotter(args.tile, args.stride).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    cfg = {"event_target": "exact", "target_width": 1.0}
    args.output.mkdir(parents=True, exist_ok=False)
    history, best_score, best_epoch, best_state, bad = [], -1.0, 0, None, 0
    start_time = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(batch["x"], batch["time_valid"])
            base, _ = spotting_loss(outputs, batch, cfg)
            loss = base + 0.1 * tile_loss(outputs, batch, args.tile)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        predictions = evaluate(model, val_loader, device)
        metrics = normalized_metrics(predictions)
        official = official_metrics(predictions, rows)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "fpsblind_score": metrics["fpsblind_selection_score"],
                  "official_diagnostic_score": official["competition_score"],
                  "entry_hit": official["acc_entry_0.3s"],
                  "collision_hit": official["acc_collision_0.3s"]}
        history.append(record)
        print(json.dumps(record), flush=True)
        if record["fpsblind_score"] > best_score:
            best_score, best_epoch, bad = record["fpsblind_score"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= 5:
            break
    model.load_state_dict(best_state)
    predictions = evaluate(model, val_loader, device)
    fpsblind = normalized_metrics(predictions)
    official = official_metrics(predictions, rows)
    summary = {"best_epoch": best_epoch, "fpsblind": fpsblind, "official_diagnostic": official,
               "wall_seconds": time.monotonic() - start_time,
               "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
               "parameters": sum(p.numel() for p in model.parameters()),
               "config": vars(args) | {"output": str(args.output), "device": str(device)}}
    torch.save({"model": best_state, "summary": summary}, args.output / "checkpoint.pt")
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.output / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
