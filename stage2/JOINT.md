# Joint Stage 2 model

The joint path implements `Stage2_code_modifications.md`, with the user's
2026-09-13 correction: **V-JEPA temporal sampling uses no FPS**. The older
coarse/fine modules are separate from this training path.

## Data and model

- Cache RF-DETR/depth observations once. Associate tracks over the complete video.
- Rank observations by confidence (0.20), approximate ego-lane proximity (0.40),
  absolute lateral-motion percentile (0.30), and positive-growth percentile (0.10).
  Missing motion/history terms are excluded and remaining weights renormalized.
  Motion uses a least-squares slope over the latest five observations; growth uses
  log-area change divided by the observation's frame gap. Both use within-video
  midranks with ties; zero movement/growth receives zero evidence.
- Select the top 12 tracks by their 90th-percentile observation priority
  (`tracking.track_percentile`), breaking ties by track ID. Slots are fixed for the
  full video and its crops, with absent objects zero-filled and masked. The cache
  stores `track_ids`. All 13 geometry channels use index-based motion, never FPS.
- Cache DINOv3 CLS + a 4×4 scene grid, and ROIs scattered into their persistent
  track slots. The spatial Transformer processes 29 tokens per frame.
- Each training crop independently samples `start ∈ [0, entry]` and
  `stop ∈ [collision+1, T]`. Both labels and all local tensors remain aligned.
  Validation and inference use the full supplied video.
- Run frozen V-JEPA online on clean RGB inside that crop. Each clip has 16 frames
  at stride 4 (61-frame span), clip starts are spaced by 48 frames, and a final
  end-aligned clip covers the tail. For fewer than 61 frames, use 16 rounded
  linspace positions, including repeats when necessary. There is no temporal-rate,
  geometric, or photometric augmentation. Preprocessing is 384px letterbox and
  ImageNet normalization, shared by training and inference.
- V-JEPA stays in evaluation/inference mode, with no gradients or activation
  checkpointing. Spatially pooled features feed a trainable 1024→384 projection.
  `model.vjepa_clip_batch_size` limits online encoder workspace (default 1).
- Local/global fusion is followed by two hybrid blocks: radius-16 local attention
  (6 heads), parallel kernel-5 depthwise convolutions at dilations 1/2/4, residual
  fusion, and a 384→1536→384 FFN. A final kernel-5 temporal convolution refines it.
- ENTRY/COLLISION frame-head architectures remain 384→128→1. Separate attribute
  attention modules use detached event probabilities to pool 16 post-Transformer
  scene cells and 12 persistent objects. Each event embedding queries these 28
  tokens. Invalid objects never act as keys/values. Attribute losses train the
  feature paths but do not backpropagate through event probabilities.

## Losses and metrics

Known **training/validation** FPS is required only for seconds-based targets and
metrics. It is never used in V-JEPA sampling or passed to the model. Test manifests
need no FPS and no labels.

ENTRY sigma defaults to 0.15 seconds; COLLISION sigma to 0.10 seconds. Both use
Gaussian temporal cross entropy plus 0.05 × CDF distance. The total weights stay
0.35 ENTRY + 0.35 COLLISION + 0.15 side + 0.15 evasion + 0.05 invalid-order loss.
There is no entry-censoring path.

Every 10 successful optimizer updates, log the six `train/loss*` values. Complete
ENTRY/COLLISION losses include their CDF terms. Competition metrics accumulate
across the full validation interval (`val_every` epochs), separately from loss
logging. Gathered sample counts exclude distributed loader tail duplicates; the
2×2 confusion matrices are combined before macro-F1 calculation. Both classes are
included, with F1=0 for a class with zero denominator.

At each validation interval log train/val ENTRY and COLLISION accuracy within
±0.3 seconds, both attribute macro-F1 values, the weighted competition score,
validation losses, and `train_config/{epoch,lr_lora,lr_new_parameters}`. Decoding is
shared with inference and maximizes the joint event logits subject to ENTRY ≤
COLLISION. With frozen encoders there are no trainable LoRA parameters; its LR is 0.

`logging.checkpoint_metric` accepts `loss` (minimize) or `competition_score`
(maximize). The workspace default remains `loss` until equivalence with the
external official evaluator is confirmed, as required by modification #8. The
score-selection path is implemented and tested. New checkpoints save the best
score and any accumulated competition metrics between validation intervals.

## Local source and commands

Official DINOv3 source is installed at `/workspace/pretrained/dinov3-source`,
revision `6876159a11b4df116f30f667f8c9888617df0751`:

```bash
git clone https://github.com/facebookresearch/dinov3.git /workspace/pretrained/dinov3-source
git -C /workspace/pretrained/dinov3-source checkout 6876159a11b4df116f30f667f8c9888617df0751
```

Factories load local checkpoints strictly; model execution does not download
weights. Keep the source repository's DINOv3 license with it.

The new local cache is **schema 2** under `/workspace/cache/joint_features_v2`.
Schema-1 caches cannot provide the new scene grid and selected ROIs. Regenerate
local features before training; existing compatible detector/depth observations
can be reused. Start a new training run; old joint checkpoints are not migrated.

From `/workspace/car-accident`, in `/venv/main`:

```bash
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m stage2.data.cache_joint_features --config stage2/configs/joint.workspace.yaml --manifest /workspace/data/stage2/manifests/all.jsonl --device cuda
python -m stage2.run --config stage2/configs/joint.workspace.yaml
python -m stage2.joint_test --checkpoint runs/joint/best.pt --manifest /path/to/test.jsonl --feature-dir /workspace/cache/joint_features_v2 --output /path/to/predictions.jsonl --device cuda
```

Inference requires local features for each test video and its original `frames_dir`.
Predicted indices map back to the cache's original frame IDs. The cache builder
also accepts unlabeled manifests without FPS.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest stage2/tests -q -p no:cacheprovider
python -m stage2.scripts.smoke_joint --output-dir /tmp/stage2-joint-smoke --device cuda
```

The smoke script builds local caches for one training and one validation video,
performs one real Accelerate/BF16 optimizer update, validates, saves/resumes, and
runs inference without FPS or labels. W&B is disabled. On the RTX 5090, the
2026-09-13 smoke passed with a frozen V-JEPA and an updated global projection;
peak allocated memory after model construction was approximately 1.48 GiB for
these two 150-frame videos at batch size 1. This verifies execution, not model
accuracy or full-data memory requirements. Full training and real multi-GPU
execution remain unverified.
