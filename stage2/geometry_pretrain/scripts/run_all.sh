#!/usr/bin/env bash
# End-to-end reproduction of the geometry-aware DINOv3 ViT-S pretraining.
# Run inside the project container (car-accident-dev) from /workspace/car-accident.
# Every step is resumable; re-running skips finished outputs.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GP=stage2.geometry_pretrain
CFG=stage2/geometry_pretrain/configs
RUNS=/workspace/outputs/geometry_pretrain/runs

# ---------------------------------------------------------------- 0. assets
# DINOv3 ViT-S/16 LVD-1689M: Meta's signed URL expired; timm re-hosts the same tensors.
mkdir -p /workspace/pretrained/dinov3_vits16
curl -L -o /workspace/pretrained/dinov3_vits16/timm_model.safetensors \
  https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m/resolve/main/model.safetensors
python -m $GP.scripts.convert_timm_dinov3 --timm /workspace/pretrained/dinov3_vits16/timm_model.safetensors \
  --output /workspace/pretrained/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-timm-converted.pth
# SAM 2.1 Hiera-Small (mask teacher)
for f in config.json model.safetensors preprocessor_config.json processor_config.json; do
  curl -L -o /workspace/pretrained/sam2.1_hiera_small/$f --create-dirs \
    https://huggingface.co/facebook/sam2.1-hiera-small/resolve/main/$f; done

# ---------------------------------------------------------------- 1. datasets (compact subsets only)
# BDD100K official mirror: 100K images + per-image labels (lanes/areas/boxes) + drivable maps (~6.3 GB zipped)
mkdir -p /workspace/data/geometry/bdd100k/zips && cd /workspace/data/geometry/bdd100k/zips
aria2c -x 16 -s 16 -Z -c http://dl.yf.io/bdd100k/bdd100k_labels.zip http://dl.yf.io/bdd100k/bdd100k_drivable_maps.zip \
  http://dl.yf.io/bdd100k/bdd100k_images_100k.zip
cd /workspace/car-accident
# TuSimple: only frames {12,17,19,20} of every train clip + 600 test clips, via HTTP range reads
# from the 23 GB Kaggle bundle (members list built from `remote_zip --list`, ~3 GB fetched).
python -m $GP.scripts.remote_zip --list > /workspace/data/geometry/tusimple/kaggle_members.txt
python -m $GP.scripts.select_tusimple
python -m $GP.scripts.remote_zip --members-file /workspace/data/geometry/tusimple/members_selected.txt \
  --output /workspace/data/geometry/tusimple --workers 24

# ---------------------------------------------------------------- 2. preparation -> manifests
python -m $GP.prepare.bdd --workers 6
python -m $GP.prepare.tusimple --workers 6
python -m $GP.prepare.videos --workers 6          # Stage-2 accident videos + BATON dashcam

# ---------------------------------------------------------------- 3. pseudo labels (cached)
python -m $GP.pseudo_labels.generate --stage static --splits val train          # RF-DETR boxes, SAM2.1 masks, DA-V2 depth
python -m $GP.pseudo_labels.generate --stage flow --splits val train            # SEA-RAFT fw/bw
python -m $GP.train --config $CFG/road_teacher_vitb.yaml                        # BDD-only road teacher
python -m $GP.pseudo_labels.generate --stage road --sources accident baton      # road pseudo labels
python -m $GP.pseudo_labels.generate --stage contact                            # derived contact bands
python -m $GP.pseudo_labels.generate --stage summary                            # cache manifest + stats
python -m pytest -q stage2/geometry_pretrain/tests/test_smoke.py

# ---------------------------------------------------------------- 4. training
python -m $GP.train --config $CFG/phase0_frozen.yaml
python -m $GP.train --config $CFG/phase1_partial.yaml
python -m $GP.train --config $CFG/phase1_partial_noanchor.yaml
# same-protocol probe (fresh heads, frozen backbone) on the adapted backbone
python -m $GP.train --config $CFG/probe_frozen.yaml --set \
  model.checkpoint=$RUNS/phase1_partial_anchor/backbone_best.pth output_dir=$RUNS/probe_adapted

# ---------------------------------------------------------------- 5. evaluation
for r in phase0_frozen phase1_partial_anchor phase1_partial_noanchor probe_adapted; do
  python -m $GP.evaluate geometry --run-dir $RUNS/$r; done
python -m $GP.evaluate drift --adapted $RUNS/phase1_partial_anchor/backbone_best.pth \
  --out /workspace/outputs/geometry_pretrain/drift/phase1_partial_anchor
python -m $GP.visualize predictions --runs $RUNS/phase0_frozen $RUNS/phase1_partial_anchor \
  --out /workspace/outputs/geometry_pretrain/viz
python -m $GP.visualize curves --runs $RUNS/phase0_frozen $RUNS/phase1_partial_anchor $RUNS/phase1_partial_noanchor \
  --out /workspace/outputs/geometry_pretrain/plots

# ---------------------------------------------------------------- 6. Stage-2 original vs adapted
python -m $GP.downstream_probe extract --name original \
  --backbone /workspace/pretrained/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-timm-converted.pth
python -m $GP.downstream_probe extract --name adapted --backbone $RUNS/phase1_partial_anchor/backbone_best.pth
python -m $GP.downstream_probe run --features original adapted --folds 5 --seeds 0 1 2 \
  --output /workspace/outputs/geometry_pretrain/stage2_probe/results.json

# ---------------------------------------------------------------- 7. ablations (4k steps) + static-vs-temporal downstream
bash stage2/geometry_pretrain/scripts/run_ablations.sh
for n in full4k no_flow; do python -m $GP.downstream_probe extract --name abl_$n --backbone $RUNS/abl_$n/backbone_best.pth --frame-cache; done
python -m $GP.downstream_probe run --features abl_full4k abl_no_flow --folds 5 --seeds 0 1 2 \
  --output /workspace/outputs/geometry_pretrain/stage2_probe/results_abl_flow.json
