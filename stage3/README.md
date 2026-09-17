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
settings can change without recomputing optical flow.

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
