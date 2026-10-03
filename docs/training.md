# Training on preprocessed data

Run commands from the repository root. Install `pip install -r requirements-training.txt` and restore the inference checkpoints with `python scripts/fetch_checkpoints.py`. Datasets and caches are supplied separately; none are tracked here. CUDA is required by the original Stage 2 trainer. Stage 3's original recipe uses CUDA/bf16; Stage 1's reconstruction also supports CPU for inspection.

## Provenance

| Stage | Training source | Reproduction status |
|---|---|---|
| 1 | Submitted model/preprocessing and recovered checkpoint config; new trainer and tensor adapter | Reconstruction, **not** the missing original trainer or a claim to reproduce 0.953197975 |
| 2 | Original trainer from `handoff-2026-09-24`, all twelve saved best-model refit configs | Exact family settings retained; original preprocessed inputs still required |
| 3 | Training modules bundled with the v8/V3 submission; config recovered from V3 checkpoint | Original V3 model/loss/trainer retained; dataset/statistics paths made portable |

## Stage 1

The checkpoint recovers ConvNeXt-Tiny, 224px local patches, 192×320 global frames, 64-bin burst profiles, 25 epochs, batch 1, accumulation 8, seed 42, backbone/head LR 3e-6/1e-4, weight decay .05, three frozen-backbone epochs, one partially unfrozen block, and patience 3. Its config also records augmentation variants, modality dropout, consistency weight .2 and robust-CVaR validation. The original implementations of those training procedures were not in any inspected branch or submission.

The reconstruction reuses the exact submitted model and deterministic preprocessing. It uses fused BCE plus sparse/burst auxiliary BCE (weight .2) and global auxiliary BCE (.1), a warmup/cosine scheduler, last-block unfreezing and macro-F1 validation at threshold .5. It **does not recreate** the missing variant augmentation, consistency loss, modality-dropout procedure, or CVaR selection. These are explicitly preserved as recovered metadata rather than invented as original behavior. Training with `--init checkpoint` is fine-tuning; `--init imagenet` is a fresh ImageNet-initialized run.

Prepare one JSONL row per video, with `video_path` and `label` (0 ORIGINAL, 1 RERECORDED), then:

```bash
python code/stage1/preprocess.py --manifest data/stage1/train_videos.jsonl --output data/stage1/train
python code/stage1/preprocess.py --manifest data/stage1/val_videos.jsonl --output data/stage1/val
python code/stage1/train.py --train-manifest data/stage1/train/manifest.jsonl --val-manifest data/stage1/val/manifest.jsonl --init imagenet
```

Each tensor file contains `patches [8,4,3,224,224]`, `global_frames [4,3,192,320]`, and `burst_profiles [2,16,9,64]`; RGB tensors are float values in [0,1]. The runtime defines the exact patch/profile ordering. Dataset rows point to tensors through `tensor_path`, relative to their manifest.

## Stage 2 — Best accident-event model

The retained source is `code/stage2/stage2/aux_signal_experiments/train.py`. The wrapper `code/stage2/train.py` supplies saved winning settings and portable cache locations. The original shared modules retain optional branches because the winner's trainer imports their model, batching and direct-loss helpers; the wrapper selects only the winning recipe.

Required manifest fields: `sample_id`, `source_id`, `entry_frame`, `collision_frame`, `entry_side` (LEFT/RIGHT), `evasion_space`, `num_frames`; preprocessing additionally needs `frames_dir`. FPS is used only for offline metrics and weak-label construction, never as a model feature.

| Preprocessed input | Files and schema |
|---|---|
| Dense DINO features | `<sample_id>.npy`, float16 `[N,70,384]`; `<sample_id>.frames.npy`, int64 `[N]` original frame indices |
| Native global motion | Dense-cache `<sample_id>.motion.npy`, float32 `[N,14]` |
| Residual motion | Residual-cache `<sample_id>.residual.npy`, float32 `[N,36]`, squashed as in `extract_residual.py` |
| Temporal-rate views | Stride-cache `k2/<sample_id>.npz`, `k3/<sample_id>.npz`: `kept`, `motion`, `residual`; motion recomputed between retained frames |
| XN4 expansion | NEXAR-cache feature/frame/motion/residual arrays plus `labels_all.json`; 670 weakly labeled extras in the original refit |

For an existing preprocessed dataset:

```bash
python code/stage2/train.py --family E4 --seed 0 --train-manifest data/stage2/all.jsonl --val-manifest data/stage2/val.jsonl --dense-cache cache/stage2/dense --residual-cache cache/stage2/residual --stride-cache cache/stage2/stride --output runs/stage2/E4_sa_full/seed0
```

Repeat seeds 0–3 for E4 and E2. XN4 additionally needs `--nexar-cache cache/stage2/nexar`. Saved family settings: AdamW LR .001, decay .05, batch 4, 30-epoch OneCycle schedule, stride probabilities .5/.25/.25. The full refits stop at epoch **7 (E4), 8 (E2), 12 (XN4)**; the schedule remains parameterized for 30 epochs. Full-data refits use fixed stopping epochs, not the overlapping validation set for selection. Final inference averages four seeds of each family.

All retained configs have phase loss **disabled**. E2 adds `boundary=bnd2`, `w_bnd=1`. XN4 masks unknown attributes and uses weak ENTRY weight .5. Extra MM-AU/CCD pool consistency and later distillation/flow-grid branches are not enabled.

Preprocessing source remains inspectable: `extract_dense.py`, `extract_motion.py`, `extract_residual.py`, `extract_stride_motion.py`, and NEXAR weak-label helpers. Dense extraction can use the retained backbone with `--backbone-checkpoint model/stage2/backbone.pth`; the training backbone loader falls back to bundled DINOv3 source. Historical extraction defaults contain server paths, so pass explicit manifest/output paths or inspect their CLI before use.

## Stage 3 — Best vehicle-behavior model

`code/stage3/config.yaml` preserves the checkpoint's original training values. Set its manifest, cache and statistics paths to your preprocessed dataset. Each manifest row needs a `cache_path` pointing to an original schema-2 motion cache. Cache identity is validated against the flow/geometry/timing config.

The cache stores quantized motion `[T,10,96,168]` (`motion_q`, `motion_offset`, `motion_scale`), physics `[T,20]`, actual/target times, a time-valid mask, and raw CAN signals (`t`, `speed`, `a_long`, `steering_angle`, optional `yaw_rate`). `CachedMotionDataset` regenerates smoothed targets, selects event-aware 96-frame crops and applies horizontal flips. Physics statistics come only from training frames.

```bash
PYTHONPATH=code/stage3/inference/code python -m stage3.scripts.compute_statistics --manifest data/stage3/train.jsonl --output cache/stage3/physics_stats.pt
python code/stage3/train.py --config code/stage3/config.yaml
```

The retained trainer uses AdamW, cosine LR with 5% warmup, bf16, EMA .999, 100 epochs, no early stopping, and competition-score checkpoint selection. The trained best was epoch 91. Training validation uses the original ±.25/5° decoder; the separate inference override supplies the best leaderboard decoder ±.4/7°. Do not substitute the inference thresholds into historical validation and call it the original selection procedure.

The submitted `best.pt` lacks optimizer/scheduler state. It is an inference checkpoint, so original `--resume` needs a full training checkpoint produced by the trainer, not this stripped submission checkpoint. Training/validation route grouping must be supplied consistently with your manifests.
