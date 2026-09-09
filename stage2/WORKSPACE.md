# Local workspace integration

Use `configs/coarse.workspace.yaml` and `configs/fine.workspace.yaml` on this instance.
They connect the assets under `/workspace/pretrained`, manifests under
`/workspace/data/stage2/manifests`, geometry under `/workspace/cache`, and outputs
under `/workspace/runs`. The original portable example configs are retained.

## Dependencies and model implementations

Python: `/venv/main/bin/python`; tested Torch 2.11.0+cu128 / torchvision 0.26.0+cu128.
Additional versions are pinned in `requirements-workspace.txt`. To provision:

```bash
uv pip install --python /venv/main/bin/python -r requirements-workspace.txt
```

V-JEPA uses the official [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
source at `/workspace/pretrained/vjepa2-source`, revision
`204698b45b3712590f06245fbfba32d3be539812`. To reproduce in a fresh workspace:

```bash
git clone https://github.com/facebookresearch/vjepa2.git /workspace/pretrained/vjepa2-source
git -C /workspace/pretrained/vjepa2-source checkout 204698b45b3712590f06245fbfba32d3be539812
```

Factories in `model/local_assets.py` construct models without downloading weights.
The loader checks all learned tensors strictly. V-JEPA uses `ema_encoder` and the
official constructor settings, with activation checkpointing. Hugging Face DINOv2
uses its local config, dense tokens at 336px, SDPA, and non-reentrant activation
checkpointing. LoRA targets only attention projections in blocks 8–11.

RF-DETR uses the 91-logit COCO checkpoint; vehicle IDs are 3/car, 4/motorcycle,
6/bus, and 8/truck. Its older checkpoint lacks RF-DETR 1.10's empty derived
`_kp_active_mask`; only that empty buffer is supplied. It is never allowed to
substitute random learned weights. Input is resized to 512px and normalized;
postprocessing restores native xyxy coordinates. Depth uses the local processor
and restores native resolution. `depth_closer_is_larger: true` follows the
[Depth Anything V2 paper's inverse-depth convention](https://arxiv.org/html/2406.09414v2#S5.SS2).

## Dataset preparation

Run from `/workspace/car-accident`:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
python -m stage2.scripts.prepare_workspace --extract
```

This creates 201 training and 50 validation records from the 251-row CSV, using
seed 42 and source-group splits. Frames preserve decode order and original
resolution, with zero-based numeric filenames and no FPS resampling. CSV frame
labels are interpreted as zero-based (consistent with the supplied frame/second
columns). Decoded counts must equal CSV counts. Incomplete frame directories
raise an error rather than silently mixing old and new data.

For full training, finish frozen geometry preprocessing and training-only statistics:

```bash
python -m stage2.data.cache_geometry --config stage2/configs/coarse.workspace.yaml --manifest /workspace/data/stage2/manifests/all.jsonl --device cuda
python -m stage2.data.geometry_stats --config stage2/configs/coarse.workspace.yaml --output /workspace/cache/coarse_geometry_stats.pt
python -m stage2.data.geometry_stats --config stage2/configs/fine.workspace.yaml --output /workspace/cache/fine_geometry_stats.pt
python -m stage2.run --config stage2/configs/coarse.workspace.yaml
python -m stage2.run --config stage2/configs/fine.workspace.yaml
```

The smoke run prepares just two clips (one training, one held-out validation), not
all 112,089 frames. Full-data extraction, geometry caching and normalization must
finish before using the full training configs. Smoke statistics are isolated under
`runs/smoke` and cannot accidentally pass the trainer's full-split membership check.

## Repeat the GPU smoke test

```bash
python -m stage2.scripts.prepare_workspace --extract --limit 2
python -m stage2.data.cache_geometry --config stage2/configs/coarse.workspace.yaml --manifest /workspace/data/stage2/manifests/smoke_train.jsonl --device cuda
python -m stage2.data.cache_geometry --config stage2/configs/coarse.workspace.yaml --manifest /workspace/data/stage2/manifests/smoke_val.jsonl --device cuda
python -m stage2.scripts.smoke_workspace --stage coarse
python -m stage2.scripts.smoke_workspace --stage fine
python -m stage2.test --coarse-ckpt /workspace/runs/smoke/coarse/last.pt --fine-ckpt /workspace/runs/smoke/fine/last.pt --manifest /workspace/data/stage2/manifests/smoke_val.jsonl --output /workspace/runs/smoke/predictions.jsonl --device cuda
```

The prediction command requires a new output filename on subsequent runs.
The smoke trainer exercises real Accelerate, BF16, the native dataset, learned
heads, LoRA backward/optimizer updates, validation, checkpoint save and resume.
Reports are `runs/smoke/{coarse,fine}/report.json`. These tiny runs verify execution,
not localization accuracy, convergence, or multi-GPU behavior. W&B is disabled.
