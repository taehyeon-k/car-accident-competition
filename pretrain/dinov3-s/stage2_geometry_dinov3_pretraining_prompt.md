# Stage 2 Geometry-Aware DINOv3 Pretraining — Coding Agent Prompt

You are working on my car-accident competition project. Your task is to design, implement, execute, evaluate, and document a **geometry-aware pretraining pipeline for DINOv3 ViT-S** that will later be used as the visual backbone for Stage 2.

Do not only write code. You must actually inspect the repository and available datasets, implement the complete pipeline, run it, debug failures, train the model as far as the available compute permits, evaluate the resulting checkpoint against the original DINOv3 checkpoint, and produce a detailed Markdown report.

---

# 1. Context and goal

The final Stage-2 task has only about **251 fully labeled accident videos**, mixed from CCD, Nexar, and AI-Hub. Previous experiments indicate that the major problem is **overfitting**.

Therefore I do NOT want to directly fine-tune a large foundation model on the 251 Stage-2 labels.

Instead, I want to first transform a generic pretrained **DINOv3 ViT-S** into a driving-geometry-aware visual foundation model by training it on real geometry annotations where available and automatically generated geometry pseudo-labels from dashcam/driving footage.

The idea is:

```text
generic pretrained DINOv3 ViT-S
        ↓
driving datasets with real geometry labels
+
unlabeled dashcam frames / frame pairs
        ↓
real geometry supervision
+
geometry-specific teacher models
        ↓
confidence-filtered pseudo labels
        ↓
multi-task geometric pretraining
        +
representation-preservation loss
        +
temporal correspondence supervision
        ↓
geometry-aware DINOv3 ViT-S
        ↓
later used as Stage-2 backbone
```

The purpose is NOT merely to make the model adapt to the appearance of accident videos.

The purpose is to force DINOv3 patch features to encode the geometry most useful for:

1. vehicle entry into the ego lane;
2. exact entry timing;
3. collision timing;
4. LEFT/RIGHT entry side;
5. traversable evasive space around the ego vehicle.

Do not use the four Stage-2 ground-truth labels during the geometry-pretraining objective.

---

# 2. Current storage constraint

The current server has only approximately:

```text
74 GB free storage
```

Treat this as a hard engineering constraint.

Do NOT attempt to download full CCD, full Nexar, full AI-Hub, full BDD100K video, or other very large raw datasets merely because they are available.

Reserve substantial free space for:

- pseudo-label caches;
- checkpoints;
- temporary files;
- extracted archives;
- visualizations;
- logs;
- training outputs.

Aim to keep at least approximately **10–15 GB free safety margin** throughout the experiment.

Before downloading anything:

1. inspect current free disk space;
2. inspect existing datasets already on disk;
3. estimate download size;
4. estimate extracted size;
5. estimate pseudo-label cache size;
6. log the decision.

Do not silently consume almost all remaining disk.

---

# 3. Dataset strategy

Do NOT make CCD/Nexar/AI-Hub the primary geometry-pretraining corpus.

For this pretraining stage, accident frequency is less important than learning robust:

- lane geometry;
- ego-road structure;
- drivable area;
- road boundaries;
- vehicles and obstacles;
- relative depth;
- temporal correspondence;
- vehicle-to-road contact geometry.

Use datasets with real geometry annotations where possible, then add target-domain accident/dashcam footage through pseudo-label distillation.

## 3.1 Primary dataset: BDD100K 100K Images

BDD100K should be the **main static geometry dataset**.

Download only the compact subsets needed for this task.

Preferred components:

```text
BDD100K 100K Images
BDD100K lane-marking labels
BDD100K drivable-area labels
BDD100K object-detection labels
```

Optionally use instance/semantic segmentation labels only if storage permits and they provide meaningful additional value.

Do NOT download the full BDD100K video corpus.

BDD100K should supervise:

- lane markings / lane boundaries;
- drivable area;
- road/curb-related structure where labels support it;
- cars;
- trucks;
- buses;
- motorcycles;
- bicycles;
- pedestrians;
- other Stage-2-relevant obstacles.

Use real BDD annotations whenever possible instead of replacing them with teacher pseudo-labels.

BDD100K should likely contribute approximately **55–65% of training samples**, but make the ratio configurable.

## 3.2 Secondary dataset: TuSimple Lane

Use TuSimple because it provides:

- front-facing ego-camera imagery;
- lane labels;
- short consecutive driving clips;
- temporal frame sequences useful for optical flow / correspondence.

Use it for:

- lane-boundary supervision;
- temporal lane consistency;
- optical-flow pseudo-labeling;
- dense correspondence learning.

Do not let TuSimple dominate the corpus because its road scenarios are comparatively structured.

Target approximately **15–20% of samples**, configurable.

## 3.3 Optional dataset: selected KITTI Raw sequences

KITTI is optional and should NOT be downloaded in full initially.

Only add a **small selected subset** if the first BDD100K + TuSimple experiment demonstrates that geometry pretraining is promising.

If used, choose a limited set of synchronized KITTI Raw drives totaling approximately:

```text
8–15 GB maximum
```

Use KITTI primarily for:

- calibrated temporal driving sequences;
- LiDAR-based geometry sanity checks;
- depth anchoring;
- camera calibration;
- track geometry where available.

Do NOT download the full KITTI raw/depth collection under the current storage constraint.

## 3.4 Existing CCD / Nexar / AI-Hub footage

Inspect what accident/dashcam footage is already present on the server.

Use all relevant footage that is already locally available, but do not require downloading the entire datasets.

Use existing accident footage primarily for:

- target-domain appearance;
- unusual collision poses;
- dangerous interactions;
- motion blur;
- camera shake;
- intersections;
- side-entry vehicles;
- challenging victim trajectories.

These samples can receive pseudo labels from the geometry teacher models.

Target approximately **10–20% of geometry-pretraining sampling**, depending on how much reliable footage is locally available.

## 3.5 Initial experiment dataset

The first real experiment should use:

```text
BDD100K 100K Images
+
TuSimple train/validation clips
+
whatever relevant CCD/Nexar/AI-Hub footage is already present
```

Do NOT add KITTI before this experiment unless it is already locally available.

The purpose is to test whether geometry-aware adaptation itself is useful before spending more disk space.

## 3.6 Approximate storage budget

Try to stay near this conceptual budget:

```text
~10 GB   BDD100K images + selected labels
~10 GB   TuSimple
~0–10 GB existing accident footage already present
---------------------------------------------
~20–30 GB source data

~10–15 GB compressed pseudo-label cache
~5 GB     checkpoints
~5 GB     visualizations/logs/temp
---------------------------------------------
leave ~10–15 GB free safety margin
```

These are planning targets, not exact file sizes.

Always inspect real usage.

---

# 4. Dataset sampling policy

Do not treat every frame as an independent sample.

Avoid huge redundancy from neighboring video frames.

For BDD100K static-image tasks:

- use available annotated images;
- do not generate unnecessary duplicate crops.

For TuSimple and accident videos:

- use sparse frame sampling for static tasks;
- use denser time-based sampling for temporal correspondence.

Prefer timestamps / FPS-aware sampling.

Do not assume all datasets have identical FPS.

Suggested configurable source weights:

```text
BDD100K              0.55–0.65
TuSimple             0.15–0.20
selected KITTI       0.10–0.15  [only if enabled]
existing accident    0.10–0.20
```

Normalize dynamically when a source is absent.

Create dataset-level balancing so one very large source cannot overwhelm all others.

---

# 5. Train/validation split rules

Create a fixed held-out geometry-validation subset that is NEVER used for backbone pretraining.

Split by:

```text
original video / sequence / drive
```

not individual frames.

A frame from the same source video must never appear in both train and validation.

For image-only BDD samples, respect existing official train/validation split where appropriate.

For TuSimple and accident clips, split at clip/video level.

For KITTI, split at drive/sequence level.

Document every split in a manifest.

---

# 6. First inspect the existing project

Before implementing anything:

1. Inspect the complete repository structure.
2. Locate all existing Stage-2 code, configs, datasets, manifests, utilities, pretrained-model handling, logging, caching, and training infrastructure.
3. Identify all currently available CCD, Nexar, AI-Hub, BDD100K, TuSimple, KITTI, or other dashcam/driving data already on disk.
4. Check free disk space.
5. Identify current GPU, VRAM, CPU, storage, Python environment, installed packages, and available pretrained checkpoints.
6. Reuse existing project infrastructure when appropriate instead of duplicating functionality.

Do not destructively modify existing Stage-2 training code.

Create the geometry-pretraining implementation as a clean module/package that can later be plugged into the Stage-2 architecture.

Before coding, write a short implementation plan to the terminal/log, then proceed immediately.

---

# 7. Geometry representations to learn

The shared DINOv3 representation should learn the following tasks.

## A. Road and lane geometry — highest priority

Teach the model:

- lane markings / lane boundaries;
- ego-lane region when it can be reliably inferred;
- general drivable-area segmentation;
- road boundary;
- curb/barrier/non-traversable boundary if reliable.

Do NOT treat this as generic semantic segmentation only.

Lane boundaries are especially important because Stage 2's entry event is fundamentally related to the victim vehicle crossing the ego-lane boundary.

For ego-lane pseudo labels, if the teacher does not directly provide ego-lane identity, infer it conservatively from:

- lane boundaries;
- the bottom-center ego-camera location;
- lane topology;
- perspective consistency.

If confidence is low, mask the ego-lane supervision instead of generating a fake label.

Use BDD100K and TuSimple real lane/drivable labels whenever possible.

Use pseudo labels only where true annotations are unavailable.

Use boundary-aware losses for thin structures.

Possible components:

```text
L_road =
    λ_drive * segmentation_loss(drivable_area)
  + λ_lane  * segmentation_loss(lane_mask)
  + λ_bound * boundary_loss(lane_boundary)
  + λ_edge  * segmentation_loss(road_edge)
```

Use Dice/Focal/BCE/CE or an appropriate combination.

## B. Object / obstacle geometry

Teach the model to represent objects that matter for Stage 2:

- cars;
- trucks;
- buses;
- motorcycles;
- bicycles;
- pedestrians where available;
- barriers;
- other substantial obstacles.

Prefer real BDD100K object annotations where available.

Prefer instance masks when a reliable segmentation teacher is available.

Bounding boxes alone are less useful because exact vehicle-road interaction depends on object shape.

Do NOT spend capacity learning irrelevant fine-grained COCO categories.

Create a compact Stage-2-relevant class taxonomy.

## C. Depth

Generate dense or semi-dense relative-depth pseudo labels.

Use a strong current monocular depth teacher already available or install a reliable one.

Depth supervision should emphasize relative scene geometry.

Do not pretend pseudo metric depth is exact if the teacher is only scale/shift invariant.

Prefer a robust loss such as scale-and-shift-invariant log-depth loss or an equivalent appropriate depth objective.

Preserve confidence / valid masks.

If selected KITTI is later enabled, use LiDAR geometry only as a limited real-depth anchor / sanity check.

## D. Vehicle-to-road contact geometry

Create a derived pseudo-label representing the approximate lower vehicle / road-contact region.

This can be obtained from:

- vehicle instance mask;
- lower visible contour;
- road/drivable mask;
- depth consistency;
- temporal consistency when possible.

It does NOT need to literally detect tires.

The objective is to make DINO features informative about the spatial point/region that determines when a vehicle crosses a lane boundary.

Use this only when pseudo-label confidence is sufficiently high.

## E. Camera / perspective geometry

If a sufficiently reliable pretrained estimator is available, teach lightweight auxiliary targets such as:

- horizon;
- vanishing point;
- camera pitch;
- possibly focal/perspective quantities.

This task has lower priority than road, objects, depth, and correspondence.

Do not include this task if pseudo-label quality is poor.

## F. Temporal correspondence / optical flow — mandatory

This is critical.

An image-only geometry model can understand:

- lane;
- car;
- depth;
- road;

but Stage 2 also needs:

- vehicle approaching the ego lane;
- vehicle crossing a boundary;
- vehicles converging;
- sudden collision dynamics.

Use adjacent or nearby frames:

```text
I_t
I_{t+Δ}
```

Pass both independently through the shared DINOv3 encoder:

```text
F_t     = DINO(I_t)
F_tplus = DINO(I_t+Δ)
```

Use a lightweight pairwise correspondence / flow head.

Generate pseudo-ground-truth optical flow or dense correspondence using a strong pretrained optical-flow model.

If SEA-RAFT is already available in the project/environment, strongly consider using it. Otherwise select another strong reliable teacher such as RAFT or an equivalent current model.

Train with confidence/occlusion-aware supervision where possible.

Use multiple temporal gaps such as:

```text
Δ = 1 frame
Δ ≈ 100–200 ms
Δ ≈ 300–500 ms
```

depending on source FPS.

Prefer time-based sampling rather than assuming all datasets have identical FPS.

TuSimple and locally available accident videos should be the primary sources for this temporal objective.

---

# 8. Teacher models

Select strong practical pretrained teacher models for:

1. vehicle/obstacle instance segmentation when true masks are unavailable;
2. monocular depth;
3. optical flow;
4. optionally lane/road geometry for unlabeled accident footage;
5. optionally camera perspective.

Before committing to each teacher:

- verify that the model actually runs;
- inspect qualitative outputs on representative BDD100K/TuSimple/accident samples;
- prefer teachers whose training domain includes street/driving imagery;
- record exact model/checkpoint/version in the report.

If one chosen teacher performs badly, replace it instead of blindly continuing.

Teacher models must remain frozen.

Whenever a real dataset annotation exists, prefer it over a teacher prediction unless there is a clear documented reason otherwise.

---

# 9. Pseudo-label generation and caching

Pseudo-label generation should be a separate reproducible stage.

Do NOT run all teachers online during every training iteration.

Implement cached pseudo-label generation.

Recommended structure conceptually:

```text
cache/
  geometry_pretrain/
    <dataset>/
      <sequence_or_image_id>/
        road.*
        objects.*
        depth.*
        contact.*
        metadata.json
      pairs/
        <pair_id>_flow.*
```

You may choose a more storage-efficient format if needed.

Every pseudo-label should have:

- prediction;
- confidence;
- valid mask where applicable;
- teacher identity/version;
- original frame/time reference.

Do not store huge float32 tensors unnecessarily.

Use storage-efficient representations such as:

- uint8 / PNG masks;
- float16 depth;
- float16 flow;
- low-resolution flow when appropriate;
- compressed NPZ;
- sparse structures;
- JSON only for lightweight metadata.

Pseudo-label creation must be resumable.

---

# 10. Pseudo-label cache storage rules

Storage efficiency is mandatory.

Before caching each target type, estimate expected total size.

Especially:

## Optical flow

Do NOT cache full-resolution float32 flow for every neighboring frame pair.

Instead:

- cache only pairs actually sampled by the training manifest;
- use float16;
- optionally store at feature/patch resolution or another justified reduced resolution;
- compress;
- store validity/occlusion masks efficiently.

## Depth

Use float16 or another compact representation.

Do not store redundant depth for frames that will never be sampled.

## Object masks / lane masks

Prefer indexed PNG or compact bit/uint8 representations.

## DINO features

Do NOT pre-cache all DINO feature tensors unless a measured storage calculation shows it is clearly beneficial and affordable.

Default should be to compute DINO features during training.

If cache usage grows beyond the planned budget, stop and revise the cache strategy rather than filling the disk.

---

# 11. Confidence filtering

Do not treat teacher predictions as ground truth.

For each task use confidence weighting:

```text
L_task =
    mean(confidence * valid_mask * task_loss)
```

or threshold unreliable regions entirely.

Whenever practical, improve confidence using:

- teacher output confidence;
- temporal agreement;
- geometric consistency;
- forward/backward flow consistency;
- agreement between multiple related teachers;
- agreement between teacher pseudo outputs and available real BDD annotations.

For example:

```text
real BDD vehicle box
+
segmentation teacher mask
        ↓
accept mask only if geometrically consistent
```

Bad pseudo-labels should be excluded rather than forced into training.

---

# 12. Dataset construction

Build a geometry-pretraining dataset from as much useful driving footage as can fit within the storage budget.

Use:

1. BDD100K real geometry labels;
2. TuSimple lane labels + temporal clips;
3. locally available accident footage with pseudo labels;
4. optionally selected KITTI later.

Do NOT limit pretraining to the 251 fully labeled Stage-2 videos.

However:

- avoid extreme oversampling of long videos;
- avoid treating neighboring nearly-identical frames as independent examples;
- sample across datasets, cameras, roads, weather, lighting, and scene types;
- keep source balancing configurable.

For static image tasks, use sparse frame sampling.

For temporal tasks, sample frame pairs/clips more densely.

---

# 13. Model architecture

Base model:

```text
DINOv3 ViT-S
```

Use the official pretrained checkpoint.

Expose dense patch features from appropriate DINO transformer layers.

Implement configurable feature extraction such as:

```text
last layer
or
multi-layer aggregation from selected final blocks
```

Prefer a simple design initially.

The conceptual architecture should be:

```text
                         DINOv3 ViT-S
                              │
                       dense patch map
                              │
       ┌──────────────┬───────┼─────────┬─────────────┐
       ▼              ▼       ▼         ▼             ▼
   road/lane        depth   objects   contact      camera
      head           head     head      head         head

frame t ── DINO ─┐
                  ├── temporal correspondence / flow head
frame t+Δ ─ DINO ─┘
```

Heads should be lightweight.

The backbone is the asset we care about, not the heads.

Avoid unnecessarily huge decoders.

---

# 14. Training curriculum

Do NOT start by fully fine-tuning DINOv3.

Run controlled phases.

## Phase 0 — frozen-backbone probing

Freeze 100% of DINOv3.

Train only geometry heads.

Purpose:

Determine how much of the required geometry is already linearly / shallowly decodable from original DINOv3 features.

Save metrics.

This is an important baseline and must not be skipped.

## Phase 1 — partial geometry adaptation

Starting from the public DINOv3 checkpoint:

- unfreeze only the final portion of transformer blocks;
- keep early/mid layers frozen;
- use a much smaller learning rate for backbone than heads.

Reasonable starting point:

```text
heads LR:       ~1e-4
backbone LR:    ~1e-6 to 5e-6
weight decay:   appropriate AdamW value
mixed precision: bf16 when supported
```

Do not assume these values are optimal.

Tune conservatively based on actual training behavior.

Start with the last ~1/4 to ~1/3 of blocks trainable.

Only unfreeze more layers if the evidence shows clear underfitting.

## Phase 2 — optional broader adaptation

Only attempt if Phase 1 shows that:

- geometry performance is still significantly limited by frozen features;
- validation metrics improve rather than merely training loss;
- representation-preservation metrics remain healthy.

Do not automatically fully unfreeze the network.

---

# 15. Preserve the original DINO representation

Catastrophic forgetting is a major concern.

Maintain a separate frozen copy of the original DINOv3 model:

```text
                     frozen original DINO
                    /
image ─────────────<
                    \
                     adapted DINO
```

Implement a representation-preservation loss.

At minimum compare corresponding patch features after normalization.

Prefer also implementing a patch-relation / Gram-style preservation term.

For example:

```text
F_orig = frozen_DINO(x)
F_new  = adapted_DINO(x)

G_orig = normalized(F_orig) @ normalized(F_orig).T
G_new  = normalized(F_new)  @ normalized(F_new).T
```

Then:

```text
L_anchor =
    λ_feature * feature_alignment(F_new, F_orig)
  + λ_gram    * gram_alignment(G_new, G_orig)
```

Do not force identical features so strongly that domain adaptation becomes impossible.

Make all anchor weights configurable.

Track anchor loss and feature drift throughout training.

---

# 16. Multi-task optimization

Do NOT simply assign all losses weight 1 and assume there is no gradient conflict.

Implement a clean multi-task loss manager.

At minimum support:

- manually configured weights;
- uncertainty-based learned weighting OR GradNorm.

If practical, also support PCGrad, but do not delay the main experiment for an overly complicated optimizer implementation.

Log each individual loss separately:

```text
loss/road
loss/lane_boundary
loss/depth
loss/object
loss/contact
loss/flow
loss/camera
loss/anchor_feature
loss/anchor_gram
loss/total
```

Also log learned task weights if dynamic weighting is used.

Tasks with unavailable labels for a sample should simply be masked.

---

# 17. Data augmentation

Geometry pretraining must not corrupt target geometry.

Use transformations that can be applied consistently to image + target.

Good candidates:

- brightness/contrast/color jitter;
- mild blur/noise/compression;
- horizontal flip if all geometry labels are correctly transformed;
- mild resize/crop if masks, flow vectors, coordinates, and camera quantities are transformed consistently.

For temporal pairs:

- apply identical geometric transforms to both frames;
- photometric augmentation can differ slightly if appropriate;
- update flow vectors after geometric transforms.

Avoid strong augmentations that destroy geometric correctness.

---

# 18. Resolution

Do not default blindly to 224×224.

Lane boundaries, distant vehicles, and vehicle-road contact are thin/small structures.

Test a practical higher spatial resolution compatible with DINOv3, likely approximately:

```text
448–512 short side
```

if VRAM permits.

Use aspect-ratio-preserving resizing/padding when appropriate.

Record actual training resolution.

If memory is limited, reduce batch size and use gradient accumulation before aggressively reducing spatial resolution.

---

# 19. Evaluation

Evaluation is mandatory.

Do not judge success only by training loss.

On the fixed held-out geometry subset evaluate:

## Road / lane

- mIoU for drivable area;
- lane mask IoU where meaningful;
- boundary F1 / precision / recall;
- ego-lane IoU for valid pseudo-label cases.

## Depth

At least suitable relative-depth metrics such as:

- AbsRel where meaningful;
- SILog or scale-invariant equivalent;
- rank/relative-depth consistency if absolute scale is unavailable.

## Objects

Depending on chosen formulation:

- class mIoU;
- mask AP;
- box AP;
- or other appropriate compact metrics.

## Contact geometry

- localization error;
- IoU;
- distance-to-target lower contour/contact region.

## Flow / correspondence

- EPE;
- angular or endpoint metrics;
- valid/occlusion-aware variants where possible.

## Camera geometry

If included:

- horizon error;
- vanishing-point pixel/angular error;
- pitch error if available.

---

# 20. Representation-preservation evaluation

Compare original DINOv3 and geometry-adapted DINOv3.

At minimum report:

- mean patch cosine similarity;
- feature norm statistics;
- Gram/relation similarity;
- optionally linear CKA over representative images.

Also visually inspect nearest-neighbor / patch correspondence or PCA feature maps if practical.

The goal is to determine whether geometry adaptation improved driving geometry without destroying general dense representations.

---

# 21. Stage-2 relevance evaluation

The final purpose is Stage 2, so geometry metrics alone are insufficient.

After geometry pretraining, run a controlled downstream comparison:

```text
A. original frozen DINOv3 ViT-S
B. geometry-adapted frozen DINOv3 ViT-S
```

Use EXACTLY the same lightweight Stage-2 probe/head and training protocol for both.

Do not give model B extra trainable layers.

If an existing compatible Stage-2 temporal head exists in the repository, reuse it.

Otherwise implement a minimal controlled probe:

```text
DINO per-frame features
        ↓
small temporal model
        ↓
ENTRY heatmap
COLLISION heatmap
entry_side
evasion_space
```

The probe should be intentionally small so the comparison measures backbone quality rather than head capacity.

Use the approximately 251 fully labeled videos only here, not during geometry pretraining.

Because the dataset is tiny:

- use source-aware and video-level splits;
- preferably 5-fold CV if computationally feasible;
- at minimum use a carefully fixed source-stratified validation split;
- never split frames from one original video across folds.

Track:

- entry timing accuracy under the competition tolerance;
- collision timing accuracy under the competition tolerance;
- entry_side Macro-F1 / accuracy as appropriate;
- evasion_space Macro-F1 / accuracy as appropriate;
- combined competition-relevant score if the repository already implements it.

Report mean and standard deviation over folds/seeds when feasible.

If full CV is too expensive, run at least one controlled original-vs-adapted comparison and explain the limitation.

---

# 22. Important ablations

Run as many of these as compute reasonably allows, in this priority order:

1. original frozen DINO + heads;
2. geometry heads trained with frozen DINO;
3. partial DINO adaptation without anchor loss;
4. partial DINO adaptation with anchor loss;
5. remove flow/correspondence task;
6. remove depth task;
7. remove lane/road task;
8. static geometry only vs static + temporal correspondence;
9. BDD100K only vs BDD100K + TuSimple;
10. BDD100K + TuSimple vs + accident pseudo-labeled footage;
11. if later enabled, evaluate whether selected KITTI adds measurable value.

The most important scientific question is:

```text
Does geometry-aware adaptation improve Stage-2-relevant representation
without simply overfitting pseudo labels?
```

---

# 23. Engineering requirements

The implementation must support:

- reproducible configs;
- deterministic seeds where practical;
- AMP / bf16;
- gradient accumulation;
- checkpoint saving;
- resume from checkpoint;
- automatic best-checkpoint selection;
- dataloader workers;
- pseudo-label caching;
- interrupted-run recovery;
- clear progress bars;
- structured logging;
- TensorBoard or W&B if the repository already uses it;
- CLI entrypoints;
- YAML configuration.

Do not hard-code machine-specific paths unnecessarily.

Provide config fields for:

```text
dataset paths
dataset source weights
cache paths
cache size limits
DINO checkpoint
teacher checkpoints
frame sampling
pair sampling
resolution
batch size
gradient accumulation
trainable DINO blocks
task weights
anchor weights
optimizer
scheduler
epochs
num workers
output directory
```

---

# 24. Resource-awareness

Inspect actual GPU VRAM before choosing batch size.

Prefer correctness and stable experiments over unnecessarily large batches.

If memory is insufficient:

1. reduce batch size;
2. use gradient accumulation;
3. use bf16;
4. reduce decoder/head size;
5. only then consider lowering image resolution.

Do not silently switch to a tiny resolution that destroys lane/contact information.

Also monitor disk usage during:

- dataset extraction;
- pseudo-label generation;
- checkpointing.

Fail gracefully if disk space approaches the configured safety threshold.

---

# 25. Smoke tests before full run

Before real training:

1. verify BDD100K labels load correctly;
2. verify TuSimple lane/video data load correctly;
3. run teacher inference on several frames;
4. visualize pseudo labels;
5. run pseudo-label caching on a small subset;
6. inspect cache size;
7. load cached labels;
8. run one forward/backward pass;
9. train for a few hundred steps;
10. check losses decrease;
11. check outputs are not NaN;
12. check DINO gradients only exist in intended blocks;
13. verify temporal augmentations transform flow correctly;
14. verify checkpoint save/resume;
15. verify source-weighted sampling works correctly.

Only after these tests pass should you launch the real pretraining.

---

# 26. Visual diagnostics

Create visualizations for randomly selected geometry-validation images.

For example:

```text
RGB
lane boundary prediction / target
drivable prediction / target
object masks
depth heatmap
contact region
flow vectors
```

Create before-vs-after comparisons for:

```text
original DINO features
geometry-adapted DINO features
```

Save representative examples, including failure cases.

Include examples from multiple sources:

- BDD100K;
- TuSimple;
- existing accident footage;
- KITTI only if enabled.

---

# 27. Actual execution

This task is not complete when the code compiles.

You must:

1. inspect storage and datasets;
2. download only the approved compact dataset components;
3. generate enough pseudo labels to train;
4. run Phase 0;
5. run at least one meaningful partial-backbone geometry adaptation;
6. evaluate on held-out geometry data;
7. run original-vs-adapted downstream Stage-2 probe if possible;
8. save the best adapted DINO checkpoint;
9. write the final report.

For the first experiment, prioritize:

```text
BDD100K
+
TuSimple
+
existing local accident footage
```

Do not add a large new dataset merely to make the experiment bigger.

If a long full training run is impossible because of the current compute environment, do the maximum meaningful run possible and explicitly report:

- how many samples;
- source distribution;
- how many epochs/steps;
- whether convergence was reached;
- storage used;
- estimated limitation.

Do not fabricate completed results.

---

# 28. Expected project structure

Adapt this to the existing repository conventions, but conceptually something like:

```text
stage2/
  geometry_pretrain/
    __init__.py
    datasets.py
    manifests.py
    pseudo_labels/
      generate.py
      teachers/
      confidence.py
    models/
      geometry_dino.py
      heads/
      losses.py
    train.py
    evaluate.py
    visualize.py
    downstream_probe.py
    configs/
      geometry_dino_vits.yaml
      geometry_probe.yaml
```

Do not create this exact structure if the existing project has a better convention.

---

# 29. Required output artifacts

At the end I expect at least:

```text
1. working source code
2. training configs
3. dataset manifests
4. dataset download/preparation commands
5. pseudo-label generation configs
6. cached pseudo-label manifest
7. original-DINO geometry probe results
8. adapted-DINO geometry results
9. Stage-2 original-vs-adapted probe comparison if feasible
10. best DINOv3 adapted checkpoint
11. loss/metric plots
12. qualitative prediction visualizations
13. exact reproduction commands
14. final Markdown report
```

Save the report somewhere obvious, for example:

```text
reports/stage2_geometry_dino_pretraining.md
```

---

# 30. Final report format

The final report should contain:

## Executive summary

Answer directly:

- Did geometry pretraining work?
- Did it improve held-out geometry?
- Did it preserve original DINO representation?
- Did it improve downstream Stage-2 performance?
- Which geometry tasks helped most?
- Which dataset sources helped most?
- Is the checkpoint worth using for the final Stage-2 model?

## Data

Report:

- datasets used;
- exact downloaded subsets;
- disk footprint per dataset;
- number of videos/sequences;
- number of sampled images;
- number of temporal pairs;
- source distribution;
- train/validation split;
- pseudo-label acceptance/rejection rates.

## Teachers

For each teacher:

- model name;
- checkpoint;
- task;
- source;
- confidence strategy;
- qualitative assessment.

## Architecture

Document:

- DINO version;
- input resolution;
- layer(s) used;
- each head;
- trainable/frozen blocks;
- parameter counts.

## Losses

Show the complete final objective and weights.

## Training

Report:

- GPU;
- epochs;
- steps;
- batch size;
- effective batch size;
- learning rates;
- optimizer;
- scheduler;
- total training time;
- peak VRAM;
- peak disk usage.

## Geometry results

Provide tables comparing:

```text
original DINO frozen
vs
adapted DINO
```

for every task.

## Representation drift

Report feature-preservation metrics.

## Stage-2 probe results

Compare:

```text
original DINO
vs
geometry-adapted DINO
```

under an identical downstream setup.

## Dataset contribution

If compute allows, compare at least:

```text
BDD100K only
BDD100K + TuSimple
BDD100K + TuSimple + local accident footage
```

If KITTI is added later, report its incremental contribution separately.

## Ablations

Report completed ablations.

## Failure analysis

Show examples where:

- lane labels/teachers fail;
- depth is unreliable;
- vehicle segmentation fails;
- collision blur causes teacher failure;
- flow fails;
- contact target is unreliable;
- geometry-adapted model regresses.

## Storage analysis

Report:

- source-data footprint;
- pseudo-label cache footprint;
- checkpoint footprint;
- peak temporary storage;
- remaining free space.

## Recommendation

Provide a concrete final recommendation about which checkpoint/config should be used as the Stage-2 visual encoder.

Also state whether downloading selected KITTI data is justified by the results of the initial experiment.

---

# 31. Scientific discipline

Do not optimize only for pseudo-label metrics.

Remember that pseudo labels come from teacher models.

A model perfectly reproducing teacher outputs is not automatically a better Stage-2 representation.

The most important measurements are:

1. held-out geometric generalization;
2. representation preservation;
3. downstream Stage-2 transfer.

Avoid data leakage.

Do not use Stage-2 validation/test labels to choose pseudo-label teacher thresholds.

Do not use competition test data.

Clearly distinguish:

- true human annotations;
- pseudo labels;
- derived geometric targets.

Prefer real BDD100K/TuSimple annotations where available.

---

# 32. Decision rules

Use these rules during implementation:

- Prefer BDD100K real geometry labels over unnecessary pseudo labels.
- Use TuSimple mainly for lane + temporal correspondence.
- Use existing accident footage for target-domain pseudo-label exposure.
- Do not download full KITTI initially.
- Add selected KITTI only if the first experiment suggests geometry adaptation is worthwhile.
- Prefer reliable simple components over unnecessarily complex ones.
- Prefer confidence masking over noisy supervision.
- Prefer partial backbone adaptation over full unfreezing.
- Prefer higher spatial detail when lane/contact geometry requires it.
- Prefer dense temporal correspondence over generic static SSL for motion.
- Preserve original DINO representations unless adaptation demonstrably helps.
- Do not assume every proposed auxiliary task is beneficial.
- Drop tasks whose teachers are unreliable or whose gradients clearly hurt validation.
- Keep every important decision configurable and documented.
- Never fill the disk simply to maximize dataset size.

---

# 33. Final deliverable

Finish by printing:

1. path to the final report;
2. path to the best geometry-adapted DINO checkpoint;
3. command to reproduce dataset preparation;
4. command to reproduce pseudo-label generation;
5. command to reproduce geometry pretraining;
6. command to run geometry evaluation;
7. command to run the Stage-2 original-vs-adapted comparison;
8. current storage usage and remaining free space;
9. a short conclusion stating whether the adapted backbone should replace the original DINOv3 backbone;
10. whether selected KITTI should be added in the next experiment.

Do not stop after planning.

Inspect → prepare compact datasets → smoke-test → generate pseudo labels → train → evaluate → compare → report.
