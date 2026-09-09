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
checkpointing. The upgraded coarse backbone is V-JEPA 2.1 ViT-L/16
(1024-dimensional features); the fine backbone is DINOv2 ViT-B/14
(768-dimensional features). The fine head projects global features to 384 dimensions.
LoRA rank is 16, alpha 16, and dropout 0.05. LoRA targets attention projections
in the final four blocks: 20–23 for V-JEPA and 8–11 for DINOv2 (zero-based).
All parameters in those four blocks are also unfrozen, including MLPs, norms,
and base attention weights. Earlier blocks remain frozen. The optimizer applies
`lora_lr` to all trainable visual-backbone parameters and `new_lr` to the heads.
Both stages set `model.T_max: 64`. Coarse sampling creates 64 representatives,
32 tubelets, and 64 event bins; fine windows contain up to 64 native frames.
Both stages use batch size 4 and accumulation 2 (8 samples per optimizer update
on one GPU). Detector and depth weights remain frozen.

Replacement assets are `vjepa2_1_vitl/vjepa2_1_vitl_dist_vitG_384.pt` from
[Meta's official checkpoint](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt)
and `dinov2_base/{config.json,model.safetensors,preprocessor_config.json}` from
[facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base).
The pretrained root is mirrored at `r2:car-accident-dataset/stage2/pretrained/`.
Start new training runs for these architectures; older smaller-backbone training
checkpoints cannot resume into the larger models.

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
python -m stage2.data.geometry_stats --config stage2/configs/coarse.workspace.yaml --output /workspace/cache/coarse_geometry_stats_t64.pt
python -m stage2.data.geometry_stats --config stage2/configs/fine.workspace.yaml --output /workspace/cache/fine_geometry_stats_t64.pt
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
heads, LoRA and unfrozen base-weight backward/optimizer updates, validation,
checkpoint save and resume. It repeats the real training clip across at least two full
batches of four and the held-out clip across one validation batch of four.
Reports are `runs/smoke/{coarse,fine}/report.json`. These tiny runs verify execution,
not localization accuracy, convergence, or multi-GPU behavior. W&B is disabled.

## Backbone upgrade verification (2026-09-09)

Both upgraded stages passed the real-data RTX 4090 BF16 smoke test at batch 4
with accumulation 2 and four fully unfrozen final blocks. Coarse performed one
optimizer update; fine performed two because it samples both event windows.
All adapted LoRA B tensors and all unfrozen base tensors updated, gradients were
finite, validation loss was finite, and checkpoint resume returned epoch 1.
Peak allocated GPU memory was 4.01 GiB coarse and 3.17 GiB fine on these clips.
Those measurements preceded the coarse T_max increase to 64 frames.
Full-run memory can differ with frame validity and object counts. Reports are at
`/workspace/runs/smoke-backbone-upgrade/{coarse,fine}/report.json`.
Replacement source URLs, byte sizes and SHA-256 hashes are recorded in
[configs/pretrained-assets.json](configs/pretrained-assets.json).
