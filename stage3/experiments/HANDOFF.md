# Handoff — 2026-09-22, server paused mid-experiment

Read `stage3/experiments/STAGE3_V2_REPORT.md` first (results and conclusions). This file is only
"how to pick the work back up".

## 0. Critical: what survives

`/workspace` on this Vast.ai instance is **not** a mounted volume
(`vast-capabilities | jq '.instance.workspace_is_volume'` -> false).

* **stop/start** the instance: everything below survives (storage is still billed).
* **recycle/destroy**: everything below is **lost**, including all checkpoints, the 98 GB uncompressed
  cache mirror, the 26 GB DINO cache and the 21 GB WAFT cache.

Nothing has been pushed to R2 or committed to git remote. Before any recycle/destroy, back up at minimum:

```bash
# ~500 MB total: code diff + every run's config/history/metrics/checkpoints
cd /workspace/car-accident && git diff > /workspace/stage3_v2_code.patch
rclone copy /workspace/car-accident/runs/stage3_v2 r2:car-accident-dataset/stage3/runs/stage3_v2_2026-09-22 \
  --exclude 'logs/**'
rclone copy /workspace/stage3_v2_code.patch r2:car-accident-dataset/stage3/runs/stage3_v2_2026-09-22/
rclone copy /workspace/SETUP_REPORT.md r2:car-accident-dataset/stage3/runs/stage3_v2_2026-09-22/
```
(The large caches are all regenerable — see §3 — so they are not worth uploading.)

## 1. State of the runs

Finished, with `metrics.json` (val competition score):

| run | epochs | val |
|---|---|---|
| `stage3_v1_baseline` | 20 | 0.7025 |
| `A1_refinement` | 20 | 0.6873 |
| `B1_multiscale` | 20 | 0.7087 |
| `C2_ms_cross_attention` | 20 | 0.7104 |
| `C1_ms_gated` | 20 | 0.7124 |
| `E1_dual_tcn` | 20 | 0.7430 |
| `E2_bissm` | 20 | 0.7468 |
| `E3_tcn_ssm` | 20 | **0.7613** (best) |
| `V1_baseline_40ep` | 40 | 0.7413 |
| `H1_waft_a2_40ep` | 40 | 0.7403 |

Interrupted (killed at the user's request; `best.pt` + `last.pt` + `history.jsonl` present, no `metrics.json`):

| run | stopped at | best val so far |
|---|---|---|
| `V1_baseline_80ep` | epoch 71/80 | 0.7640 |
| `K1_dualtcn_ms_gated` | epoch 19/20 | 0.7404 |
| `K2_tcnssm_ms_gated` | epoch 11/20 | 0.7177 |

Resume one (the trainer restores model/EMA/optimizer/scheduler/epoch and refuses mismatched features):

```bash
cd /workspace/car-accident && source /venv/main/bin/activate
python -m stage3.run --config runs/stage3_v2/V1_baseline_80ep/config.yaml \
                     --resume runs/stage3_v2/V1_baseline_80ep/last.pt
```
`stage3/experiments/run.py` has no `--resume`; after a resumed run finishes, produce the metrics/confusion/
predictions bundle with:
```bash
python -m stage3.experiments.run --name V1_baseline_80ep --eval-only \
  --set data.manifest=/workspace/data/stage3/manifests_raw/train.jsonl \
  --set data.val_manifest=/workspace/data/stage3/manifests_raw/val.jsonl --set data.num_workers=4
```
(`--eval-only` skips training and evaluates the existing `best.pt`. Adding a real `--resume` to the harness is a
5-line change and is the cleaner fix.)

## 2. Queue / harness

* `stage3/experiments/exp.sh NAME --set k=v ...` — one run with the standard manifests (env `WORKERS`, default 4).
* `stage3/experiments/queue.sh` — keeps `MAX` (default 2; 3 fits on this 32 GB GPU) runs alive, popping lines from
  `runs/stage3_v2/queue.txt`. **The queue is currently stopped and `queue.txt` is empty.**
* `stage3/experiments/summarize.py` — rebuilds `stage3/experiments/results/{summary.csv,summary.json,summary.md}`
  and per-run `curves.png` / `confusion.png`. Re-run it after any new result.
* `stage3/experiments/flow_diagnostics.py` — SEA-RAFT vs WAFT low-level statistics
  (`results/flow_diagnostics_val.json`, already generated).

To restart the planned work, write these lines into `runs/stage3_v2/queue.txt` and launch `queue.sh`
(all on top of the selected hybrid temporal block):

```
V2_final_tcnssm_80ep --set model.temporal.type=tcn_ssm --set optimization.epochs=80
K2_tcnssm_ms_gated_rerun --set model.temporal.type=tcn_ssm --set model.motion_encoder.type=multiscale --set model.fusion.type=gated
G3_tcnssm_class_ordinal --set model.temporal.type=tcn_ssm --set model.auxiliary.ordinal_accel=true --set model.auxiliary.ordinal_steer=true --set model.auxiliary.class_loss=ordinal --set loss.class_loss=ordinal --set loss.weights.accel_class=0.3 --set loss.weights.steer_class=0.3
F3_tcnssm_aux --set model.temporal.type=tcn_ssm --set model.auxiliary.jerk=true --set model.auxiliary.steer_rate=true --set loss.weights.jerk=0.05 --set loss.weights.steer_rate=0.02
D1_tcnssm_dino_pooled --set model.temporal.type=tcn_ssm --set model.visual.enabled=true --set model.visual.mode=pooled --set data.visual_cache_dir=/workspace/cache/stage3/visual_dinov3s_4x7
D2_tcnssm_ms_cross_dino_tokens --set model.temporal.type=tcn_ssm --set model.motion_encoder.type=multiscale --set model.fusion.type=cross_attention --set model.visual.enabled=true --set model.visual.mode=tokens --set data.visual_cache_dir=/workspace/cache/stage3/visual_dinov3s_4x7
```
Priority order if time is short: **`V2_final_tcnssm_80ep` first** — it is the actual deliverable and the only run
that separates the architecture gain from the schedule gain. Then `K2`, then the class heads (they target the
ACCEL/DECEL->CONSTANT boundary that dominates the remaining error), then DINOv3.

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
4. **Nothing committed**: all Stage 3 v2 code is uncommitted working-tree changes plus untracked new files.
   `git status` in `/workspace/car-accident` shows them; `git diff` + `git status --porcelain` capture everything.
