"""Emit the prompt-v2 experiment matrix (section 16) as explicit config files."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = "/workspace/car-accident"
CACHE = f"{ROOT}/stage2/artifacts/spotting_features/fpsblind_192_7x10"
MANIFESTS = "/workspace/data/stage2/manifests"

BASE = {
    "strict_fps_blind": True,
    "model": "spotter",
    "temporal": "none",
    "multi_rate_diff": False,
    "state_aux": False,
    "category_head": "global",
    "event_target": "exact",
    "target_width": 0.015,
    "order_weight": 0.05,
    "consistency_weight": 0.0,
    "consistency_bins": 64,
    "cache_dir": CACHE,
    "all_manifest": f"{MANIFESTS}/all.jsonl",
    "train_manifest": f"{MANIFESTS}/train.jsonl",
    "val_manifest": f"{MANIFESTS}/val.jsonl",
    "n_tokens": 70, "feature_dim": 384, "token_dim": 32, "hidden": 192, "dropout": 0.3,
    "grid_width": 10,
    # Fixed cadence until E6 introduces sampling augmentation.
    "train_sample_counts": [128], "eval_sample_count": 128,
    "jitter": 0.0, "drop_probability": 0.0, "warp": 0.0, "duplicate_probability": 0.0,
    "epochs": 30, "batch_size": 8, "eval_batch_size": 16, "workers": 0,
    "lr": 0.001, "weight_decay": 0.05, "grad_clip": 1.0,
    "seed": 42,
}

AUGMENTATION = {
    "train_sample_counts": [96, 128, 160],
    "jitter": 0.45,
    "drop_probability": 0.03,
    "warp": 0.15,
    "duplicate_probability": 0.02,
}

DIFF = {"multi_rate_diff": True, "diff_strides": [1, 2, 4, 8], "diff_gated": True}
ATTENTION = {"temporal": "local_attention", "attention_layers": 2,
             "attention_heads": 4, "attention_window": 17}

EXPERIMENTS = {
    # Prompt v2 baseline: the non-dense "simple head".
    "B0_pooled_baseline": {"id": "B0-pooled-baseline", "model": "pooled_regressor"},
    # E1: dense per-frame event head, no temporal mixing.
    "E1_dense": {"id": "E1-dense-event-head"},
    # E2: lightweight temporal head - three candidates from section 7.
    "E2a_dilated": {"id": "E2a-dilated-conv", "temporal": "dilated_conv"},
    "E2b_maxer": {"id": "E2b-temporal-maxer", "temporal": "temporal_maxer"},
    "E2c_attention": {"id": "E2c-local-attention", **ATTENTION},
    # E3: + multi-rate temporal differences.
    "E3_multirate": {"id": "E3-multi-rate-diff", **ATTENTION, **DIFF},
    # E4: + soft event labels (branches from E3).
    "E4_soft": {"id": "E4-soft-targets", **ATTENTION, **DIFF,
                "event_target": "soft_normalized", "target_width": 0.015},
    # E5: + event-state auxiliary loss (branches from E3).
    "E5_state_aux": {"id": "E5-state-aux", **ATTENTION, **DIFF,
                     "state_aux": True, "state_radius": 3, "state_aux_weight": 0.1},
    # E6: + FPS-invariance sampling augmentation (branches from E3).
    "E6_augmentation": {"id": "E6-sampling-augmentation", **ATTENTION, **DIFF, **AUGMENTATION},
    # E7: + temporal consistency between two augmented views.
    "E7_consistency": {"id": "E7-temporal-consistency", **ATTENTION, **DIFF, **AUGMENTATION,
                       "consistency_weight": 0.5},
    # E8: + event-conditioned side / evasion heads.
    "E8_event_conditioned": {"id": "E8-event-conditioned", **ATTENTION, **DIFF, **AUGMENTATION,
                             "consistency_weight": 0.5, "category_head": "event_conditioned"},
}


def main():
    out = Path(__file__).resolve().parent / "configs"
    out.mkdir(exist_ok=True)
    for name, overrides in EXPERIMENTS.items():
        cfg = {**BASE, **overrides}
        (out / f"{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"wrote {name}.json  id={cfg['id']}")


if __name__ == "__main__":
    main()
