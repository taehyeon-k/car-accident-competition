# Stage 2 spotting experiments

The strict pipeline never reads FPS or duration while extracting features, sampling,
constructing targets, training, selecting checkpoints, or decoding. `native_fps` is
accessed only by `official_metrics` after predictions exist, to report the offline
DACON-style ±0.3 second metric.

The frozen geometry checkpoint is supplied explicitly with `--backbone-checkpoint`.
Feature caches include their frame-number grid and checkpoint provenance in
`meta.json`. The initial cache uses 192 normalized positions and a 7×10 spatial grid;
each training step resamples it to 96, 128, or 160 positions with jitter and optional
dropping. Every run writes its resolved config, commit, seed/fold, checkpoint,
history, predictions, metrics, parameter count, timing, and peak VRAM.

```bash
python -m stage2.spotting_experiments.extract_features \
  --backbone-checkpoint /workspace/outputs/geometry_pretrain/runs/phase1_partial_noanchor/backbone_best.pth \
  --output /workspace/car-accident/stage2/artifacts/spotting_features/fpsblind_192_7x10
python -m stage2.spotting_experiments.train \
  --config stage2/spotting_experiments/configs/e2_temporal_maxer.json \
  --output stage2/spotting_experiments/results/E2/fixed/seed0
```
