# Handoff — Stage 3 v2 (first written 2026-09-22; updated 2026-09-24 after all runs finished)

Read `stage3/experiments/STAGE3_V2_REPORT.md` first (results and conclusions). This file is only
"how to pick the work back up".

## 0. Critical: what survives

`/workspace` on this Vast.ai instance is **not** a mounted volume
(`vast-capabilities | jq '.instance.workspace_is_volume'` -> false).

* **stop/start** the instance: everything below survives (storage is still billed).
* **recycle/destroy**: everything below is **lost**, including all checkpoints, the 98 GB uncompressed
  cache mirror, the 26 GB DINO cache and the 21 GB WAFT cache.

**Backed up 2026-09-24** to `r2:car-accident-dataset/stage3/runs/stage3_v2_2026-09-24/`:
`runs/` (all 19 runs: checkpoints, metrics, configs, history, logs; 193 files, 904 MiB, verified with
`rclone check`), `stage3_v2_branch.bundle` (git commits on `stage3-v2-experiments` since `c437a5b`),
`SETUP_REPORT.md` and `stage3_v2_code.patch`. Restore:

```bash
rclone copy r2:car-accident-dataset/stage3/runs/stage3_v2_2026-09-24/runs /workspace/car-accident/runs/stage3_v2 --s3-no-check-bucket
cd /workspace/car-accident && rclone copyto r2:car-accident-dataset/stage3/runs/stage3_v2_2026-09-24/stage3_v2_branch.bundle /tmp/s3v2.bundle --s3-no-check-bucket \
  && git fetch /tmp/s3v2.bundle stage3-v2-experiments:stage3-v2-experiments
```
Single-file uploads need `--s3-no-check-bucket` (the key cannot CreateBucket). Work from other sessions
(`submission/`, `stage2/`, `reports/`) is **not** in this backup or in these commits.
(The large caches are all regenerable — see §3 — so they are not worth uploading.)

## 1. State of the runs (updated 2026-09-24 ~02:00)

**All planned runs are finished.** Results and conclusions are in `STAGE3_V2_REPORT.md`. Final model:
`runs/stage3_v2/V3_tcnssm_100ep/best.pt` (hybrid temporal block, 100 epochs, best epoch 91), val **0.7954**.

| run | epochs | val |
|---|---|---|
| `stage3_v1_baseline` | 20 | 0.7025 |
| `A1_refinement` | 20 | 0.6873 |
| `B1_multiscale` / `C2_ms_cross_attention` / `C1_ms_gated` | 20 | 0.7087 / 0.7104 / 0.7124 |
| `E1_dual_tcn` / `E2_bissm` / `E3_tcn_ssm` | 20 | 0.7430 / 0.7468 / 0.7613 |
| `K1_dualtcn_ms_gated` / `K2_tcnssm_ms_gated` | 20 | 0.7422 / 0.7525 |
| `G3_tcnssm_class_ordinal` / `F3_tcnssm_aux` | 20 | 0.7626 / 0.7533 |
| `D1_tcnssm_dino_pooled` / `D2_tcnssm_ms_cross_dino_tokens` | 20 | 0.7632 / 0.7678 |
| `V1_baseline_40ep` / `H1_waft_a2_40ep` | 40 | 0.7413 / 0.7403 |
| `V1_baseline_80ep` | 80 | 0.7660 |
| `V2_final_tcnssm_80ep` | 71 (early stop) | 0.7850 |
| **`V3_tcnssm_100ep`** | 100 | **0.7954** |

Every run has `config.yaml`, `history.jsonl`, `best.pt`, `last.pt`, `metrics.json`, `predictions.npz`.
`V1_baseline_80ep`, `K1`, `K2` and `D2` were interrupted and resumed. `D2`'s metrics bundle was regenerated with
`--eval-only` after its first final evaluation was stopped (pinned-memory stall, see §2).

## 2. Queue / harness

* `stage3/experiments/exp.sh NAME --set k=v ...` runs one experiment with the standard manifests. Use **`WORKERS=12`**:
  the default 4 leaves the data loader CPU-bound (~5x slower for DINOv3-token runs).
* `stage3/experiments/resume.sh NAME` continues an interrupted run from `runs/stage3_v2/NAME/last.pt` via
  `run.py --resume`. It uses the run's saved `config.yaml`; edit `data.num_workers` there first if needed.
* `stage3/experiments/queue.sh` keeps `MAX` runs alive, popping lines from `runs/stage3_v2/queue.txt`.
  **Use `MAX=2`, not 3.** Three runs OOM the 32 GB GPU (each reserves ~12 GB). Two runs with heavy full-clip
  evaluation can also push the container's 170.8 GB memory cgroup to its limit, because pinned host memory
  (`pin_memory=True`) grows to ~65 GB per process and is never released. The symptom is 100% kernel CPU, 0% GPU,
  and no progress, with no error message. Check `awk '/^shmem /' /sys/fs/cgroup/memory.stat`.
  The queue is currently stopped and `queue.txt` is empty.
* `early_stopping` (patience 4) is on in the base config. It cut `V2` at epoch 71. Pass
  `--set early_stopping.enabled=false` for long final runs.
* `stage3/experiments/summarize.py` rebuilds `stage3/experiments/results/` (tables + per-run plots).
* `stage3/experiments/flow_diagnostics.py` computes SEA-RAFT vs WAFT low-level statistics (already generated).

Next work, in priority order (details in the report, §8):
1. Route-group-disjoint re-validation of `V3_tcnssm_100ep` vs `V1_baseline_80ep`.
2. Multi-seed runs of the final config: `--set seed=<n> --set model.temporal.type=tcn_ssm --set optimization.epochs=100 --set early_stopping.enabled=false`.
3. Fix the hybrid's speed-MAE regression (3.46 vs 1.32 for v1 at 80 epochs).
4. Decision-threshold sweep / hybrid emission for the ACCEL/DECEL -> CONSTANT boundary.

## 3. Assets on disk (and how to rebuild them)

| path | size | rebuild |
|---|---|---|
| `/workspace/cache/stage3/motion` | 20 GB | `rclone copy r2:car-accident-dataset/stage3/cache/motion ...` |
| `/workspace/cache/stage3/motion_raw` | 98 GB | `cd motion && ls *.pt \| xargs -P20 -I{} sh -c "gzip -dc {} > ../motion_raw/{}"` (byte-identical content, uncompressed; used by `manifests_raw` for fast loading) |
| `/workspace/cache/stage3/motion_waft_a2` | 21 GB | `python -m stage3.scripts.cache_motion --config stage3/configs/waft_a2.workspace.yaml --manifest /workspace/data/stage3/manifests_waft_a2/all.jsonl --device cuda` (~4 h) |
| `/workspace/cache/stage3/visual_dinov3s_4x7` | 26 GB | `python -m stage3.scripts.cache_visual --manifest /workspace/data/stage3/manifests_raw/all.jsonl --output ... --decoders 6` (~1 h) |
| `/workspace/cache/stage3/physics_stats*.pt` | tiny | `stage3.scripts.compute_statistics` (SEA-RAFT stats match the R2 reference checkpoint exactly) |
| `/workspace/data/stage2/frames` | 11 GB | `python -m stage2.scripts.prepare_workspace --workspace /workspace --extract --workers 16` |
| `stage2/artifacts/spotting_features/fpsblind_192_7x10` | 2.5 GB | `python -m stage2.spotting_experiments.extract_features --backbone-checkpoint /workspace/outputs/geometry_pretrain/runs/phase1_partial_noanchor/backbone_best.pth --output <same dir>` (3.5 min) |
| `/workspace/pretrained/{sea_raft,waft,dinov3-source,dinov3_vits16}` | ~1.2 GB | R2 / GitHub / gdown (WAFT a1+a2 checkpoints came from the authors' Google Drive) |

Manifest variants: `manifests/` (gz caches), `manifests_raw/` (uncompressed mirror — **use these for training**),
`manifests_waft_a2/` (WAFT caches). Disk is currently 189/250 GB used; a second uncompressed mirror does not fit.

## 4. Stage 2 (untouched, ready to train)

Verified end-to-end on this server (see `/workspace/SETUP_REPORT.md`): frames, manifests, geometry backbone and
the regenerated FPS-blind feature cache are all in place, and 1-epoch smoke runs of both pipelines passed.

```bash
cd /workspace/car-accident && source /venv/main/bin/activate
python -m stage2.spotting_experiments.train \
  --config stage2/spotting_experiments/configs/e3_asformer.json \
  --output stage2/spotting_experiments/results/E3-ASFormer/fixed/seed0_rerun
```
Use a fresh `--output` directory so the committed results are not overwritten.

## 5. Open issues

1. **Split caveat**: `route_24_1`/`route_25_1` are in val while sibling recordings of the same routes are in train.
   Absolute numbers may be optimistic; a route-group-disjoint re-check is pending.
2. **Single seed** for everything. Margins under ~0.01 (B1/C1/C2, WAFT vs SEA-RAFT) are not resolved.
3. **`test_distributed_validation`** fails on this 1-GPU box (both ranks agree; launcher exit code only).
4. **Committed locally only**: branch `stage3-v2-experiments` (`62d3a86` and the commit after it), not pushed to the git
   remote. A git bundle of the branch plus all of `runs/stage3_v2` is backed up to
   `r2:car-accident-dataset/stage3/runs/stage3_v2_2026-09-24/` (see §0). `runs/` is untracked in git.
