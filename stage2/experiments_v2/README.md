# Stage 2 experiments v2

Self-contained implementation of the prompt-v2 experiment matrix. Nothing in
`stage2/spotting_experiments/` or anywhere else in the repository is modified or
imported; the frozen geometry-DINOv3 feature cache is read read-only.

## FPS-blindness contract

No FPS, duration, timestamp, or seconds value reaches feature sampling, target
construction, the model, the loss, checkpoint selection, or decoding. Original
frame numbers are carried through only so predictions can be returned as
original frame indices. `native_fps` is read in exactly one place,
`metrics.py:official_metrics`, after predictions already exist, to reproduce the
DACON ±0.3 s metric offline. `tests/test_v2.py::test_training_path_reads_no_fps_field`
enforces this by scanning executable source with comments and strings stripped.

## What is new relative to `spotting_experiments`

| Component | Prompt v2 | File |
|---|---|---|
| Multi-rate temporal differences `D_s[t]=F[t]-F[t-s]`, gated fusion | §6 | `modules.py:MultiRateDifference` |
| Event-state auxiliary heads (BEFORE/BOUNDARY/AFTER, PRE/CONTACT/POST) | §9 | `objective.py:state_aux_loss` |
| Two-view temporal consistency (JS on binned original-frame coordinates) | §12 | `objective.py:consistency_loss` |
| Local temporal warping and frame duplication | §11 | `sampling.py` |
| P25/P50/P75/P90 frame-error distributions | §17 | `metrics.py:error_distribution` |
| Sampling-density instability report | §18 | `robustness.py` |

## Experiment matrix

Single run per experiment, fixed 201/50 split, **seed 42**. This is a
methodology screen, not a submission run, so there is no cross-validation and no
multi-seed averaging; treat small score gaps as inconclusive.

| Run | Definition |
|---|---|
| `B0_pooled_baseline` | mean-pooled scalar position regression ("simple head") |
| `E1_dense` | dense per-frame event head, no temporal mixing |
| `E2a/E2b/E2c` | + dilated conv / TemporalMaxer / local attention |
| `E3_multirate` | E2c + multi-rate differences `[1,2,4,8]` |
| `E4_soft` | E3 + soft normalized event targets |
| `E5_state_aux` | E3 + event-state auxiliary loss |
| `E6_augmentation` | E3 + drop / stride / warp / duplication |
| `E7_consistency` | E6 + two-view temporal consistency |
| `E8_event_conditioned` | E7 + event-conditioned side / evasion heads |

## Commands

```bash
cd /workspace/car-accident
python -m stage2.experiments_v2.make_configs
python -m stage2.experiments_v2.train \
  --config stage2/experiments_v2/configs/E3_multirate.json \
  --output stage2/experiments_v2/results/E3_multirate
python -m stage2.experiments_v2.summarize
python -m stage2.experiments_v2.robustness \
  --checkpoint stage2/experiments_v2/results/E3_multirate/checkpoint.pt \
  --output stage2/experiments_v2/results/E3_multirate/robustness.json
PYTHONDONTWRITEBYTECODE=1 python -m pytest stage2/experiments_v2/tests -q
```

Each run writes `config.json` (with git commit, seed, and the exact train/val
ids), `history.json`, `predictions.json`, `metrics.json`, and `checkpoint.pt`.
