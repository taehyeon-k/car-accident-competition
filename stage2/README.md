# Stage 2 training and inference

The code follows [Stage2_Architecture.md](Stage2_Architecture.md), with the
Diffusion-ISP organization: YAML configuration, entry points, data, model, trainer,
and utilities. See [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md) for findings.

Native training and inference logic is CPU-tested. Local pretrained implementations
and checkpoints still need to be supplied and verified; this is not a ready-to-run
pretrained submission.

## Native manifest

Each JSONL row describes one clip (one object per line):

```json
{"sample_id":"clip_001","source_id":"original_001","frames_dir":"frames/clip_001","geometry_dir":"geometry_cache/clip_001","native_fps":30,"entry_frame":107,"collision_frame":145,"entry_side":"LEFT","evasion_space":1}
```

Image filenames have unique numeric stems, such as 00107.jpg. Labels are original
filename IDs, not array offsets; gaps are allowed. Unknown FPS can be omitted.
source_id groups crops from the same video to prevent split leakage. Manifest paths
are relative to the manifest; configured paths are relative to stage2 unless
root_dir is set.

## Local models

Configure factory/checkpoint fields in both YAML files. Factories use
package.module:callable and construct architectures without downloading weights.
Loading is strict: adapt different checkpoint layouts explicitly. Null factories
fail rather than silently selecting a head-only ablation. See
[third_party/README.md](third_party/README.md) for contracts. The loader does not
download weights, but arbitrary user-provided factories must also be kept offline.

## Workflow

From the repository root, after configuring models and checking the depth adapter's
depth_closer_is_larger orientation:

```bash
python -m stage2.scripts.prepare_manifest --manifest stage2/data/all.jsonl --output-dir stage2/data
python -m stage2.data.cache_geometry --config stage2/configs/coarse.yaml --manifest stage2/data/train.jsonl --device cpu
python -m stage2.data.cache_geometry --config stage2/configs/coarse.yaml --manifest stage2/data/val.jsonl --device cpu
python -m stage2.data.geometry_stats --config stage2/configs/coarse.yaml --output stage2/weights/coarse_geometry_stats.pt
python -m stage2.data.geometry_stats --config stage2/configs/fine.yaml --output stage2/weights/fine_geometry_stats.pt
python -m stage2.run --config stage2/configs/coarse.yaml
python -m stage2.run --config stage2/configs/fine.yaml
```

Cache generation requires actual frozen detector/depth checkpoints even on CPU.
Stages share per-frame caches only when frozen-model/configuration settings match.
Tracks, ranks, temporal geometry and ROIs are rebuilt for each window. Never cache
trainable visual features for LoRA training. The separate training_mode:
cached_features mode uses prepared feature_path records for a head-only ablation;
it is not the native training pipeline.

Inference requires sample_id, frames_dir and optionally geometry_dir, not labels:

```bash
python -m stage2.test --coarse-ckpt /path/coarse.pt --fine-ckpt /path/fine.pt --manifest /path/test.jsonl --output predictions.jsonl --device cpu
```

scripts/export_submission.py merges LoRA and removes auxiliary fine state heads.
scripts/benchmark.py measures Stage 2 latency/memory on the operator's hardware.

## Weights & Biases

W&B is already listed in `requirements.txt`. Install the project dependencies in
your training environment, then set `logging.wandb` in the coarse/fine YAML:

```yaml
logging:
  log_every: 20
  val_every: 1
  wandb:
    enabled: true
    project: stage2-localizer
    entity: your-username-or-team
    name: coarse-experiment-01
    mode: online
    run_id: null
    resume: never
```

Authenticate on your own machine, then use the normal training command:

```bash
wandb login
python -m stage2.run --config stage2/configs/coarse.yaml
```

Never put an API key in YAML or Git; use the login prompt or the `WANDB_API_KEY`
environment variable. W&B logs training loss components/metrics, validation
metrics, epoch, separate learning rates, and selected hyperparameters. Training
metrics are sample-weighted across the logging interval and all workers. Validation
retains duplicate-removing distributed aggregation. No images, datasets, source
files, or checkpoint artifacts are explicitly uploaded. Local checkpoint/manifest
paths are excluded from the logged hyperparameters; W&B may collect normal runtime
metadata and system metrics.

For no account/network, use `enabled: true` with `mode: offline`. Logs are stored
under the stage's output directory in `wandb/`. Upload later, if desired:

```bash
wandb sync /path/to/output/wandb/offline-run-...
```

Set `enabled: false` to disable tracking entirely. Online logging resumption is
separate from model resumption: set the original `run_id` and `resume: must`, and
pass `--resume /path/to/last.pt` to restore training. Keep the same W&B project and
entity. Offline logging starts a new segment and does not support this run-resume
option. Resuming an older checkpoint cannot rewind existing W&B history; use a
new run if its steps have already been logged.

Integration uses [Accelerate trackers](https://huggingface.co/docs/accelerate/en/usage_guides/tracking)
and W&B's documented [initialization options](https://docs.wandb.ai/models/ref/python/functions/init).

## Verification without pretrained weights

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest stage2.tests.test_review -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest stage2.tests.test_tracking -v
python -m black --check stage2
git diff --check
```

Tests exercise real CPU heads/losses and synthetic backbone/cache interfaces.
They do not establish real checkpoint compatibility, GPU/DDP correctness, training
convergence, or production performance.
