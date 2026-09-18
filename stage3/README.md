# Stage 3 motion TCN

Stage 3 predicts physical acceleration and steering angle from video motion, then
decodes DACON's four acceleration and three steering categories. Direct CAN
`aEgo` is the primary acceleration target. Speed differentiation is used only by
the low-weight consistency loss. BATON `steeringAngleDeg` is the primary lateral
target and is normalized to positive LEFT / negative RIGHT steering-wheel degrees.

The production flow backend is the frozen official SEA-RAFT-S checkpoint. The
`opencv` backend exists only for fast CPU smoke tests and explicit ablations.
The default focal estimator is a horizontal-FOV prior; GeoCalib is optional and
is never imported in the default path.

All commands below run from `/workspace/car-accident` inside the prepared Docker
container.

## Prepare manifests

```bash
python -m stage3.scripts.inspect_dataset --data-root /workspace/data/stage3/BATON-Sample
python -m stage3.scripts.build_manifest \
  --data-root /workspace/data/stage3/BATON-Sample \
  --output-dir /workspace/data/stage3/manifests \
  --cache-dir /workspace/cache/stage3/motion \
  --segment-seconds 30 --force
```

Splits are route-disjoint. Thirty-second segments bound cache memory while crops
remain 96 frames. `smoke_train.jsonl` and `smoke_val.jsonl` each contain one clip.

## Cache motion

```bash
python -m stage3.scripts.cache_motion \
  --config stage3/configs/baseline_v1_2.workspace.yaml \
  --manifest /workspace/data/stage3/manifests/all.jsonl \
  --data-root /workspace/data/stage3/BATON-Sample --device cuda
python -m stage3.scripts.validate_cache \
  --manifest /workspace/data/stage3/manifests/all.jsonl
```

The schema-2 cache uses versioned, per-channel uint8 motion quantization and gzip.
It stores synchronized raw signals, not fixed labels, so smoothing and target
settings can change without recomputing optical flow. Dense track/rho transport
uses batched CUDA scatter-add with `geometry.tracking_batch_size` (default 32);
the scalar NumPy implementation remains available as the CPU fallback.

## Compute training statistics

```bash
python -m stage3.scripts.compute_statistics \
  --manifest /workspace/data/stage3/manifests/train.jsonl \
  --output /workspace/cache/stage3/physics_stats.pt
```

Only the training split contributes the robust median/MAD statistics.

## Test

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest stage3/tests -q
PYTHONDONTWRITEBYTECODE=1 python -m pytest stage2/tests -q
python -m stage3.run --help
```

## Train and resume

```bash
python -m stage3.run --config stage3/configs/baseline_v1_2.workspace.yaml
python -m stage3.run \
  --config stage3/configs/baseline_v1_2.workspace.yaml \
  --resume /workspace/car-accident/runs/stage3/baseline_v1_2/last.pt
```

For the two-clip execution smoke:

```bash
python -m stage3.run --config stage3/configs/smoke.workspace.yaml
python -m stage3.scripts.smoke_overfit \
  --config stage3/configs/smoke.workspace.yaml --steps 20 --device cuda
```

## Evaluate

```bash
python -m stage3.scripts.evaluate \
  --config stage3/configs/baseline_v1_2.workspace.yaml \
  --checkpoint /workspace/car-accident/runs/stage3/baseline_v1_2/best.pt
```

## Benchmark inference

```bash
python -m stage3.scripts.benchmark_inference \
  --checkpoint /workspace/car-accident/runs/stage3/baseline_v1_2/best.pt \
  --video /workspace/data/stage3/BATON-Sample/route_1/qcamera.mp4 \
  --device cuda
```

## DACON Stage 3 inference

```bash
python -m stage3.scripts.predict_dacon \
  --data-dir /path/to/dacon/data \
  --model-dir /workspace/car-accident/runs/stage3/baseline_v1_2 \
  --output /tmp/stage3_predictions.csv
```

The returned columns are exactly `ID`, `sample_index`, `accel_label`, and
`steer_label`. DACON inference decodes every frame and emits sample indices
`0..N-1`; it does not inspect FPS or resample PTS.

Motion feature revision 3 corrects depth change and source-grid tracking in rho.
Rebuild older motion caches, recompute training statistics, and retrain before
using the corrected features. `cache_motion` automatically rebuilds caches whose
keys differ; training rejects stale caches or statistics. Checkpoints now carry
the feature version and cache identity. Inference and resume reject legacy or
incompatible checkpoints rather than silently mixing feature definitions.

Steering Macro-F1 excludes frames whose ground-truth acceleration label is
`STOPPED`. This scoring mask does not remove rows: inference always returns both
`accel_label` and `steer_label` for every decoded frame.

CUDA geometry now batches rotation and FOE fitting, reuses resident tensors for
both tracking lags, and computes rho before downloading the results. It is
selected automatically with `--device cuda`; `geometry.backend: numpy` retains
the reference fitting implementation (with GPU tracking when CUDA is selected).
Final canonical resizing and robust physics pooling remain on CPU.

Training dequantizes only the selected crop. `inference.cnn_chunk_frames` controls
CNN frame batches (default 32) without splitting temporal context. Inference
section timers synchronize CUDA so asynchronous work is charged to its section.

Targets honor CAN validity and `targets.max_signal_gap_seconds` (default 0.25 s).
Smoothing and speed derivatives operate within contiguous valid runs. Validation
gathers complete clips across processes and removes distributed tail duplicates
before scoring; steering predictions are still required on STOPPED frames.

To reproduce the CUDA checks in the prepared container:

```bash
python -m pytest stage3/tests -q
python -m stage3.scripts.smoke_cuda
python -m stage3.scripts.benchmark_cuda \
  --video /workspace/data/stage3/BATON-Sample/route_1/qcamera.mp4 \
  --frames 301 --repeats 3 --output /tmp/stage3-cuda-benchmark.json
```

The smoke command uses temporary caches and checkpoints, including one bf16
training step and resume. The benchmark compares all-CPU geometry, the former
CPU-fitting/GPU-tracking path, and batched CUDA geometry. CNN equivalence and
memory measurements disable cuDNN TF32 to separate batching from reduced-precision
rounding; normal inference uses the environment's cuDNN precision settings.
