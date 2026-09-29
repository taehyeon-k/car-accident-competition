# Stage 3 TCP and openpilot transfer screen

Status: complete at the 08:00 KST deadline on 2026-09-29. The best verified validation competition score is **0.8232739392**, from the current V3 control continued through epoch 78. The objective of 0.9000 was not reached; the final gap is 0.0767260608. The machine-readable deadline status is `/workspace/runs/stage3_tcp_transfer/goal_status.json`.

## Research question and controlled setup

Does a CARLA-trained TCP visual representation help BATON Stage 3 relative to the current SEA-RAFT, geometry, and physics V3 architecture? C0 and T1 use 20 epochs; T2 is an 8-epoch warm-start projection fine-tune so that all distinct screens and a winner continuation fit the fixed overnight deadline. All use the existing 1,693/466 BATON route split, seed 42, 96-frame training crops, event sampling and horizontal flip, AdamW 3e-4, cosine schedule, bf16, EMA 0.999, and the existing Stage 3 losses, decoder, and validation code. The decoder and label boundaries are fixed at acceleration ±0.4 m/s² and steering ±7°. Steering Macro-F1 excludes ground-truth STOPPED frames; competition score is 0.7 × acceleration Macro-F1 + 0.3 × steering Macro-F1. Loader workers were reduced from the archived V3 setting of 12 to 2 for C0 because the 12-worker control process exited before completing an epoch under this instance's RAM limit. T1 and T2 use one training loader worker, zero validation workers, and validation batch size 8. Validation remains in manifest order and batches only consecutive equal-duration clips; the seven short validation clips are evaluated separately so bidirectional temporal padding cannot change their predictions. Their RGB-only model ignores the motion tensor, so their dataset skips faulting the large mmap-backed tensor and supplies the same visual features, physics, labels, masks, crops, and flips through the common trainer interface. This changes data transport only, not the experiment inputs consumed by the model or its objective.

The archived full-schedule V3 at the competition decoder scored 0.824 on BATON validation (acceleration 0.785, steering 0.915) and 0.7475 on the Stage 3 leaderboard, according to `stage3/STAGE3_V1_TO_V3_REPORT.md`. The 20-epoch C0 run below is the fair screening control; the archived 100-epoch score is context only.

## Checkpoints and source licenses

| Asset | Source | Local file | SHA-256 | License/provenance |
|---|---|---|---|---|
| TCP reproduction | [Hugging Face model card](https://huggingface.co/maxim-igenbergs/tcp-carla-repro) | `/workspace/pretrained/tcp_repro/best_model.ckpt` | `1792d62a4c7764f6622d4d20b8585bd785d739d752aef2b6447c081c362f7d5e` | Card says Apache 2.0 and explicitly calls this an **unofficial reproduction**. It is not an official TCP checkpoint. |
| TCP encoder tensor cache | Safely extracted from the above checkpoint with `weights_only=True` | `/workspace/pretrained/tcp_repro/tcp_perception_state.pt` | See cache metadata after completion | 218 encoder tensors; strictly matched torchvision ResNet34. |
| TCP code | [OpenDriveLab/TCP](https://github.com/OpenDriveLab/TCP), commit `73cc1ddfb615439b3f035c0285f5c9ffce350418` | `/workspace/pretrained/tcp-source` | — | Apache 2.0. The official repository does not provide a checkpoint in its tracked tree. |
| openpilot vision | [commaai/openpilot v0.11.1](https://github.com/commaai/openpilot/tree/v0.11.1/selfdrive/modeld/models) | `/workspace/pretrained/openpilot_v0.11.1/driving_vision.onnx` | `ee29ee5bce84d1ce23e9ff381280de9b4e4d96d2934cd751740354884e112c66` | Repository MIT license; no separate model license identified. |
| openpilot policy | Same tag | `/workspace/pretrained/openpilot_v0.11.1/driving_policy.onnx` | `78477124cbf3ffe30fa951ebada8410b43c4242c6054584d656f1d329b067e15` | Same. |

The downloaded TCP Lightning checkpoint contained a serialized callback; `stage3/experiments/tcp_transfer/extract_tcp_encoder.py` allowlists only that callback class name and loads with `weights_only=True`, then saves only the tensor dictionary. The official TCP forward pass also takes vehicle speed, navigation command, and target point. These experiments use only `perception`, the RGB ResNet34, so none of those DACON-unavailable inputs enter the student.

## Architecture and preprocessing

```text
C0: BATON RGB → SEA-RAFT → geometry/physics cache → MotionCNN + physics MLP
                                                     → V3 TCN+BiSSM → existing heads

T1: BATON RGB → frozen TCP ResNet34 → pretrained 1000-D visual projection
                                   → V3 TCN+BiSSM → existing heads

T2: BATON RGB → frozen TCP ResNet34 convolutional backbone
              → trainable pretrained 512→1000 projection (LR 1e-5)
              → V3 TCN+BiSSM → existing heads (LR 3e-4)
```

The T2 ablation fine-tunes TCP's pretrained final **projection**, rather than its final residual convolutional block. This is the small pretrained component that can be trained using a compact frozen-feature cache within the available disk budget. It must not be described as layer4 fine-tuning. T1 has 1,244,182 trainable student parameters. T2 has 1,757,182. The frozen TCP ResNet34 contains 21,797,672 parameters, of which 513,000 belong to the projection updated in T2; the effective total model size for each is 23,041,854 parameters.

The cache uses the same PTS-based 10 Hz frame selection as the existing motion cache and checks exact timestamp agreement per clip. It applies TCP's ImageNet RGB mean and standard deviation, resizes each frame to 256 pixels high while preserving BATON's aspect ratio, and separately encodes its horizontal mirror to preserve the existing flip augmentation. TCP's CARLA camera was 900×256 with a 100° horizontal field of view; BATON's camera differs. Cached features contain pooled layer4 (512 values) and the pretrained projection (1,000 values), float16 per frame. T1 consumes the pretrained projection; T2 recomputes it with trainable weights from the frozen pooled feature. This screen tests transfer under a camera-domain shift.

## Runs and commands

| ID | Config | Run directory | Status |
|---|---|---|---|
| C0 | `stage3/experiments/tcp_transfer/control_base.yaml` | `/workspace/runs/stage3_tcp_transfer/C0_control` | Complete: 0.7907317982 at epoch 20. |
| T1 | `stage3/experiments/tcp_transfer/t1_frozen_tcp.yaml` | `/workspace/runs/stage3_tcp_transfer/T1_frozen_tcp` | Complete: 0.6853233877 at epoch 20. |
| T2 | `stage3/experiments/tcp_transfer/t2_tcp_projection.yaml` | `/workspace/runs/stage3_tcp_transfer/T2_tcp_projection` | Complete: 0.6997250224 at epoch 8; warm-started from T1 best EMA. |
| E1 | validation-only weighted vote | `/workspace/runs/stage3_tcp_transfer/E1_weighted_vote` | Complete: selected C0 alone and scored 0.7907317982. |
| X1 | continued best screen | `/workspace/runs/stage3_tcp_transfer/X1_best_to_100ep` | Stopped at deadline after epoch 78: **0.8232739392**. |
| D1/D2 | — | — | Not run because the openpilot teacher interface failed the moving-footage quality audit; no score claimed. |

Supervisor services are `stage3_tcp_control`, `stage3_tcp_cache`, `stage3_tcp_t1`, and `stage3_tcp_t2`. Logs are in `/workspace/runs/stage3_tcp_transfer/{control,cache,t1,t2}_supervisor.log`. They preserve their result folders and do not overwrite archived V3 checkpoints. The RGB feature cache is `/workspace/cache/stage3/tcp_repro_rgb_v2`.

The cache completed all 2,159 clips. Its two arrays per clip were converted from NPZ members to element-verified NPY sidecars so crop reads can use memory mapping; all 4,318 converted arrays passed exact equality checks before the NPZ sources were removed. C0 was terminated by SIGKILL during validation after epoch 13's training pass, before the epoch-13 checkpoint; the supervisor log does not establish why. It was restarted from the last durable checkpoint. A second SIGKILL interrupted validation after epoch 16's training pass while four other GPU jobs were using most of the instance's RAM. C0 resumed again from epoch 15 with validation batch size 1, zero validation workers, and no pinned validation memory; training settings and the metric remain unchanged. The interrupted, uncommitted training passes are discarded; no score from them is used. `stage3_tcp_watchdog` checks progress every 30 seconds, restarts interrupted runs from checkpoints, and stops this experiment queue at the stated deadline.

After the C0/T1 20-epoch screens and T2 8-epoch warm-start fine-tune, `stage3_tcp_extension` selects the highest scoring completed candidate and continues its checkpoint toward epoch 100 until the 08:00 KST deadline. It initially validates every five epochs; after epoch 70 it validates every two epochs so the final hour produces verified scores before the deadline. A measured epoch-71 loader test was reverted because one prefetch worker increased the 20-batch time from about 31 to 56 seconds under shared-storage contention. This is a separate, exploratory longer-schedule run. T2 versus T1 is an additional-training ablation rather than an equal-budget comparison.

T1, T2, and the longer continuation skip the optional full-training-split diagnostic evaluation after their authoritative validation evaluation. This does not change training, best-checkpoint selection, validation predictions, confusion matrices, or any requested score; it avoids spending the overnight search window recomputing a diagnostic train/validation gap.

Execution checks completed: strict TCP encoder tensor load; one BATON clip cached with exact 301-frame timing and both flip variants; T1 and T2 forward/backward plus Stage 3 loss; and separate one-epoch, one-clip end-to-end runner smokes for T1 and T2, including checkpoint save, warm start, full validation, and metrics output. The smoke scores are not research results and are excluded from the comparison table.

Manual reproduction, from `/workspace/car-accident`:

```bash
python -m stage3.experiments.run --name C0_control --base stage3/experiments/tcp_transfer/control_base.yaml --root /workspace/runs/stage3_tcp_transfer
python -m stage3.experiments.tcp_transfer.cache_tcp --manifest /workspace/data/stage3/manifests/all.jsonl --output /workspace/cache/stage3/tcp_repro_rgb_v2 --batch 16 --decoders 4
python -m stage3.experiments.tcp_transfer.run_tcp --name T1_frozen_tcp --base stage3/experiments/tcp_transfer/t1_frozen_tcp.yaml --root /workspace/runs/stage3_tcp_transfer
python -m stage3.experiments.tcp_transfer.run_tcp --name T2_tcp_projection --base stage3/experiments/tcp_transfer/t2_tcp_projection.yaml --root /workspace/runs/stage3_tcp_transfer
```

## openpilot teacher feasibility

The public v0.11.1 ONNX pair was inspected directly. The vision model takes **two** camera inputs (`img`, `big_img`), each two packed YUV frames of shape `[1,12,128,256]`, and exposes a 512-dimensional hidden state. The policy takes a 25-step feature history, desire pulses, and traffic convention. Its documented plan output contains 33 future samples with position, velocity, acceleration, orientation, and orientation rate; see [`ModelConstants`/`Plan`](https://github.com/commaai/openpilot/blob/v0.11.1/selfdrive/modeld/constants.py) and [`modeld.py`](https://github.com/commaai/openpilot/blob/v0.11.1/selfdrive/modeld/modeld.py). Desired acceleration and curvature are **derived** by openpilot from the plan and vehicle state; they are not direct ONNX output tensors.

BATON supplies one camera with different optics. `probe_openpilot.py` packed BATON RGB in openpilot's YUV layout, duplicated it to both camera inputs, ran the documented 5 Hz/25-step feature history with zero desire pulses and right-hand traffic, and compared its plan at time zero to held-out CAN **only for diagnosis**. After a 5-second warmup:

| Probe route | CAN median speed | Teacher median speed | Teacher speed MAE | Teacher acceleration MAE | Finding |
|---|---:|---:|---:|---:|---|
| `route_1` | 2.23 m/s | 2.06 m/s | 1.19 m/s | 0.48 m/s² | Mixed; speed correlation −0.13, acceleration correlation 0.75. |
| `route_8` | 0 | 0.03 m/s | 0.03 m/s | 0.03 m/s² | Stationary case plausible; does not validate moving behavior. |
| `route_2_1` | 14.23 m/s | 20.80 m/s | 6.66 m/s | 1.44 m/s² | Teacher accelerates in the plan while CAN decelerates. |
| `route_25_1` | 0 | 0.03 m/s | 0.03 m/s | ~0 | Another stationary case. |

An aspect-preserving letterbox on `route_2_1` did not improve the failure: speed MAE 6.64 m/s and acceleration MAE 1.72 m/s². Diagnostic JSON files are under `/workspace/runs/stage3_tcp_transfer/openpilot_probe*.json`. These tests do not establish whether the network weights themselves are poor; the uncalibrated single-camera adaptation is the likely confound. **D1/D2 are not queued or scored.** The next prerequisite is a calibrated two-view mapping or a validated single-camera openpilot teacher interface on BATON. Distilling the current probe outputs would confound teacher quality with representation transfer. No ground-truth BATON CAN value was given to the student or probe policy model.

<!-- MEASURED_RESULTS_START -->
## Measured time-boxed screening results

All rows use the same 466 BATON validation clips, seed 42, and ±0.4 m/s² / 7° decoder.

| ID | Epochs | Competition | Acceleration Macro-F1 | Steering Macro-F1 | Accel MAE (m/s²) | Steering MAE (°) | Speed MAE (m/s) | Best epoch | Train time (h) | Peak train VRAM (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_control | 20 | 0.7907 | 0.7507 | 0.8841 | 0.3087 | 6.5943 | 2.5026 | 20 | 0.62 | 5.69 |
| T1_frozen_tcp | 20 | 0.6853 | 0.7036 | 0.6427 | 0.3113 | 13.9878 | 2.6072 | 20 | 1.45 | 0.10 |
| T2_tcp_projection | 8 | 0.6997 | 0.7063 | 0.6844 | 0.3088 | 12.6499 | 3.0010 | 8 | 0.58 | 0.11 |

| ID | ACCELERATING F1 | DECELERATING F1 | CONSTANT F1 | STOPPED F1 | LEFT F1 | RIGHT F1 | STRAIGHT F1 | Trainable params | Best checkpoint size (MiB) | Cached validation ms/frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_control | 0.6397 | 0.6099 | 0.8022 | 0.9512 | 0.8522 | 0.8284 | 0.9716 | 1880679 | 29.2 | 1.086 |
| T1_frozen_tcp | 0.5625 | 0.5635 | 0.7904 | 0.8980 | 0.5045 | 0.4821 | 0.9414 | 1244182 | 19.4 | 0.505 |
| T2_tcp_projection | 0.5518 | 0.5783 | 0.7931 | 0.9019 | 0.5583 | 0.5485 | 0.9464 | 1757182 | 27.3 | 0.508 |

Cached validation ms/frame excludes SEA-RAFT preprocessing for C0 and TCP RGB encoding for T1/T2. It is not an end-to-end inference latency.

### C0_control confusion matrices

Acceleration, rows = truth, columns = prediction:

```text
  ACCELERATING DECELERATING CONSTANT STOPPED
[14306, 174, 7525, 379]
[522, 11077, 9453, 144]
[7234, 3649, 59312, 451]
[281, 228, 941, 23605]
```

Steering, rows = truth, columns = prediction; ground-truth STOPPED excluded:

```text
  LEFT STRAIGHT RIGHT
[7741, 1661, 51]
[913, 93606, 837]
[61, 2069, 7287]
```

Acceleration-vs-speed-derivative validation loss: 0.1217.

### T1_frozen_tcp confusion matrices

Acceleration, rows = truth, columns = prediction:

```text
  ACCELERATING DECELERATING CONSTANT STOPPED
[10897, 262, 10360, 865]
[362, 9885, 10083, 866]
[4770, 3501, 59984, 2391]
[334, 242, 703, 23776]
```

Steering, rows = truth, columns = prediction; ground-truth STOPPED excluded:

```text
  LEFT STRAIGHT RIGHT
[3503, 5456, 494]
[353, 94609, 394]
[577, 5567, 3273]
```

Acceleration-vs-speed-derivative validation loss: 0.1270.

### T2_tcp_projection confusion matrices

Acceleration, rows = truth, columns = prediction:

```text
  ACCELERATING DECELERATING CONSTANT STOPPED
[10479, 298, 10819, 788]
[364, 10365, 9799, 668]
[4386, 3635, 60527, 2098]
[366, 353, 837, 23499]
```

Steering, rows = truth, columns = prediction; ground-truth STOPPED excluded:

```text
  LEFT STRAIGHT RIGHT
[4117, 4854, 482]
[528, 94364, 464]
[650, 4851, 3916]
```

Acceleration-vs-speed-derivative validation loss: 0.1239.

## Screening interpretation

T1 minus C0: -0.1054 competition score. T2 minus T1: +0.0144.
These are single-seed BATON-validation margins, and the earlier archived V3 leaderboard result shows that BATON validation can overestimate DACON transfer. T2 changes the TCP projection and warm-starts T1 temporal/head weights, so T2 vs T1 is an additional-training comparison, not a from-scratch equal-budget comparison.
Openpilot D1/D2 remain unrun because the uncalibrated teacher probe failed on moving BATON footage. A robust two-view teacher input and output-quality audit are prerequisites. No conclusion about the benefit of valid openpilot distillation can be drawn.
TCP pretraining did not transfer favorably in this setup. T1 trails the equal-budget C0 by 0.1054, and the projection-tuned T2 recovers only 0.0144. The weighted vote selects C0 alone for acceleration and steering, providing no evidence of complementary TCP errors. The current motion/physics V3 remains the preferred representation.

## Deadline continuation result

The highest-scoring screen, C0, was continued from epoch 20 until the watchdog stopped training at exactly 23:00 UTC / 08:00 KST. Epoch 79 was in progress and was discarded; epoch 78 is the last durable and best verified checkpoint.

| Run | Epoch | Competition | Acceleration Macro-F1 | Steering Macro-F1 | Accel MAE (m/s²) | Steering MAE (°) | Speed MAE (m/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 screen | 20 | 0.790732 | 0.750736 | 0.884054 | 0.308667 | 6.594350 | 2.502643 |
| **X1 best** | **78** | **0.823274** | **0.784335** | **0.914131** | **0.294219** | **3.897956** | 4.131500 |

X1 improves the screening control by 0.032542 competition score. Acceleration and steering classification and their continuous MAEs improve, while speed MAE worsens by 1.628856 m/s. This nearly reproduces the archived rounded 0.824 BATON result but remains far below the 0.9000 objective. The checkpoint is `/workspace/runs/stage3_tcp_transfer/X1_best_to_100ep/best.pt` (29.2 MiB, SHA-256 `47d1dce1ed0ecd201beb80fd131ec182c468857471f12a5b0e82411bfd930385`). The checkpoint embeds epoch 78, global step 16,536, metric name `competition_score`, and the exact best score above. A compact final record is stored at `/workspace/runs/stage3_tcp_transfer/X1_best_to_100ep/deadline_metrics.json`.

The next experiment should address acceleration class separation without sacrificing speed calibration, using a route-group-disjoint confirmation split before any leaderboard decision. The TCP variants should not replace V3 based on these results. Openpilot distillation remains blocked on a calibrated, quality-audited teacher interface rather than a larger distillation-weight search.

<!-- MEASURED_RESULTS_END -->

Each completed screen's `metrics.json` and `predictions.npz` are authoritative for per-class F1, confusion matrices, errors, parameter count, VRAM, and runtime. X1's `history.jsonl` and embedded checkpoint validation metrics are authoritative for the deadline continuation because the time-boxed run stopped before the normal final `metrics.json` export. End-to-end RGB inference time for T1/T2 must include TCP feature extraction; the reported cached-feature latency is not full inference cost. The optional corruption robustness suite was not run because neither TCP candidate was competitive and the remaining budget was spent continuing the winning V3 control.

This instance's `/workspace` is **not** a persistent host volume. Checkpoints and feature caches survive stop/start but will be lost on recycle or destroy unless synced elsewhere.
