# Joint Stage 2 model

The joint path implements `Stage2_code_modifications.md`, with the 2026-09-13
correction (**V-JEPA temporal sampling uses no FPS**), the 2026-09-14 removal of
Depth Anything, and the 2026-09-14 switch to **online DINOv3 + V-JEPA training
with LoRA**. The joint model is the only Stage 2 training path.

## Training summary

Both visual backbones run online with every pretrained weight frozen; only LoRA
adapters on their last few attention blocks and the joint head train. No DINO or
V-JEPA feature is ever cached, because a cached feature would be stale the moment
an adapter updates. RF-DETR stays frozen and fully offline: its per-frame boxes,
classes and scores are the only cache, and even those are re-tracked inside each
sampled crop so a crop's first frame cannot inherit motion from before it.

## Data and model

- Cache RF-DETR observations once on the original frames. There is no depth model
  anywhere in Stage 2, and RF-DETR never runs during training.
- Tracking, ranking and geometry are rebuilt **per crop** from the cached boxes,
  so `dx`, `dy`, log-area growth and continuity only ever see frames inside the
  sampled window.
- Rank observations by confidence (0.20), approximate ego-lane proximity (0.40),
  absolute lateral-motion percentile (0.30), and positive-growth percentile (0.10).
  Missing motion/history terms are excluded and remaining weights renormalized.
  Motion uses a least-squares slope over the latest five observations; growth uses
  log-area change divided by the observation's frame gap. Both use within-video
  midranks with ties; zero movement/growth receives zero evidence.
- Select the top 12 tracks by their 90th-percentile observation priority
  (`tracking.track_percentile`), breaking ties by track ID. Slots are fixed for the
  full video and its crops, with absent objects zero-filled and masked. The cache
  stores `track_ids`. All nine geometry channels use index-based motion, never FPS.
- The geometry token is nine bbox/tracking channels, in this order: `center_x`,
  `bottom_y`, `width`, `height`, `dx_per_frame`, `dy_per_frame`,
  `log_area_growth_per_frame`, `detection_confidence`, `track_continuity`
  (`GEOMETRY_CHANNELS` in `model/joint_tracking.py`). Depth proximity, its rank,
  raw bbox area and the raw per-observation log-area delta were removed; the
  per-frame growth rate already carries the looming signal. Track *ranking* still
  uses ego-lane position, absolute lateral motion, positive growth and confidence,
  but those statistics are computed beside the token rather than inside it.
- The object token is still DINO ROI appearance combined with the geometry
  embedding, which now projects nine channels instead of thirteen.
- DINOv3 runs online and produces the CLS + 4×4 scene grid and the ROIAlign
  object features from the current adapter weights. The spatial Transformer still
  processes 29 tokens per frame.
- Temporal policy: **70%** of training samples use the complete original video,
  **15%** take an ordinary event-preserving crop, and **15%** take the synthetic
  "ENTRY happened before the video started" case. Ordinary crops vary the context
  on each side independently so neither event sits a fixed distance from a
  boundary. The pre-video case starts within 0.3 s *after* the real ENTRY, relabels
  the first visible frame as ENTRY, keeps the real COLLISION at its crop-relative
  position, and preserves `entry_side` and `evasion_space`. A mode whose
  preconditions fail (COLLISION would leave the crop, or the crop would be shorter
  than 16 frames) falls back to the full video instead of degrading silently.
  Validation and inference always use the full supplied video with no augmentation.
- **Two independent temporal mechanisms** act in sequence, and must not be
  confused with one another:

  ```text
  Semantic temporal augmentation        Training memory cap
  ------------------------------        -------------------
  70% full video                        max_frames = 512
  15% ordinary crop
  15% pre-video ENTRY                   purpose: stop long clips building
                                        excessive DINOv3/V-JEPA LoRA
  purpose: generalization and           autograd graphs
  competition-specific augmentation
  ```

  The cap runs **after** the semantic policy, on whatever window it produced. It
  is not a fourth augmentation mode and it never invents, moves or drops a label.
  A window already at or below the limit is returned untouched, so a 50-frame
  clip stays 50 frames. Only longer windows are trimmed, by sampling uniformly
  over every position that still contains both events - never centred on an
  event, never at a fixed offset - using the same seed/epoch/index RNG. For
  `pre_video_entry` the window start is pinned instead, because its whole meaning
  is "ENTRY is the first visible frame"; only the tail is trimmed. If a sample's
  ENTRY->COLLISION span were ever wider than the cap, both paths raise rather
  than relabel. 512 is comfortably above this dataset's longest annotated span
  (**147** frames in train, 115 in val, verified by
  `scripts/check_event_spans.py`), so the cap never has to touch a label today.
  Set `training_memory.max_frames: null` to disable it.
  **Validation and inference are never capped** - with no backward graph they can
  afford the complete video, and they must see it to match submission conditions.
  `train/{original,semantic,final}_temporal_length` and
  `train/memory_crop_applied_fraction` report how often it fires.
- Augmentation is drawn **once per clip** and applied identically to every frame,
  so DINOv3 and V-JEPA always see the same pixels and no photometric flicker is
  introduced. A horizontal flip mirrors the frames and the cached detector boxes
  (`[x1,y1,x2,y2] -> [W-x2,y1,W-x1,y2]`) before tracking, which makes
  `center_x -> 1-center_x` and `dx -> -dx` fall out automatically and keeps track
  identities; it swaps `entry_side` and leaves ENTRY, COLLISION and
  `evasion_space` untouched. Photometric strength (brightness, contrast, gamma,
  saturation, JPEG, blur, noise) is sampled once, including the colour order and
  the noise standard deviation. Every probability and range is set in YAML.
- V-JEPA reads the same augmented frames. Each clip has 16 frames
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

Only the frozen RF-DETR detections are cached. They are **schema 2** (no depth
channels) and carry that version in each `geometry_dir/metadata.pt`; a schema-1
cache is rejected with an explicit message. There is no DINO/V-JEPA feature cache
any more, so no visual feature can go stale against an updated adapter.

LoRA settings live under `model:` (`lora_rank`, `lora_alpha`, `lora_dropout`,
`dino_lora_blocks`, `vjepa_lora_blocks`) and the three learning rates under
`optimization:` (`dino_lora_lr`, `vjepa_lora_lr`, `new_lr`). Inference rebuilds
the adapters on the frozen local pretrained backbones and loads the trained
weights onto them; `model/lora.py:merge_lora` can fold adapters into plain
`nn.Linear` layers for deployment.

From `/workspace/car-accident`, in `/venv/main`:

```bash
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m stage2.run --config stage2/configs/joint.workspace.yaml
python -m stage2.joint_test --checkpoint runs/joint/best.pt --manifest /path/to/test.jsonl --output /path/to/predictions.jsonl --device cuda
```

Inference requires each test video's `frames_dir` and its cached detections.
Predicted indices map back to the original frame IDs. Unlabelled manifests
without FPS are accepted.

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
