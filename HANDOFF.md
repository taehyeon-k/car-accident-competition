# Handoff — updated 2026-09-26 (originally 2026-09-24; server destroyed after each)

**Start command for a new session (unchanged):** "Clone https://github.com/taehyeon-k/car-accident-competition.git into
/workspace/car-accident, check out branch handoff-2026-09-24, and read HANDOFF.md fully. Follow its section 2 to restore the
Stage 2 and Stage 3 training environment from GitHub and R2 (r2:car-accident-dataset). Don't delete anything from R2."
The branch name stays `handoff-2026-09-24`; it now also contains the 2026-09-25 session (§6), the 2026-09-26 leaderboard results +
Stage 3 10 Hz rule (§8 — **read §8 first**, it overrides older Stage 3 notes below), and the 2026-09-26 Stage 2 studies and the
**v7 submission** (§9, §10 — **read §10 first for Stage 2**).

Start here on a fresh Vast.ai server. This file says what exists, where it is stored,
how to restore a training-ready environment, and what to do next. Older, narrower notes:
`stage3/experiments/HANDOFF.md` (Stage 3 v2 runs), `reports/*.md` (Stage 2 experiments),
`/workspace/SETUP_REPORT.md` (original setup; copy in R2 `handoff/2026-09-24/workspace_notes/`).

## 0. Where things live

* **GitHub** `taehyeon-k/car-accident-competition`, branch **`handoff-2026-09-24`** — all code,
  configs, reports, the `submission/` package *code*, and small result JSONs. `main` is older.
  If the branch is not on GitHub, it is in R2 as a git bundle (§2.1).
* **R2** `r2:car-accident-dataset` (rclone remote `r2`) — data, checkpoints, caches, zips.
  Never delete from R2; only `rclone copy` into new prefixes.
* `/workspace` on Vast is **not** persistent. Anything not in GitHub/R2 is lost on destroy.

### R2 layout (what matters now)

| R2 path | What | Local destination |
|---|---|---|
| `stage2/usable/` (`usable_only.csv` + 349 videos) | Stage 2 labeled set (251 original + 98 external MM-AU/CausalCrash) | `/workspace/data/stage2/usable/` |
| `stage2/manifests_349/` | current split: `all` 349 / `train` 279 / `val` 70 (+ `pre_resplit_251/`) | `/workspace/data/stage2/manifests/` |
| `stage2/cache/fpsblind_192_7x10_349/` (1 GB) | FPS-blind DINO feature cache, 192 positions, 349 clips | `car-accident/stage2/artifacts/spotting_features/fpsblind_192_7x10_349/` |
| `stage2/cache/temporal_pyramid_cache_383/` (6.7 GB) | 383-position cache for the pyramid models | `car-accident/stage2/temporal_pyramid_experiments/cache_383/` |
| `stage2/runs/2026-09-24/` | Stage 2 checkpoints: `spotting_experiments/E3-ASFormer-349`, `long_video_experiments`, `temporal_pyramid_experiments` (P1–P10, `REFIT_P2_all349`, ensembles) | matching `car-accident/stage2/.../results/` dirs |
| `geometry_pretrain/outputs/runs/phase1_partial_noanchor/` | frozen geometry-tuned DINOv3-S backbone (`backbone_best.pth`) | `/workspace/outputs/geometry_pretrain/runs/phase1_partial_noanchor/` |
| `stage2/pretrained/dinov3_vits16/` | original DINOv3-S weights (Stage 3 visual cache only) | `/workspace/pretrained/dinov3_vits16/` |
| `stage2/external/` | external-data pipeline, labeling annotations (+ snapshots) | `/workspace/data/stage2_external/` (only to relabel) |
| `stage3/BATON-Sample/` (4 GB) | Stage 3 training videos + CAN signals | `/workspace/data/stage3/BATON-Sample/` |
| `stage3/cache/motion/` (20 GB) | SEA-RAFT motion caches (gz), key `490a9c89f2ddec80bfed` | `/workspace/cache/stage3/motion/` |
| `stage3/cache/physics_stats*.pt` | physics normalization stats | `/workspace/cache/stage3/` |
| `stage3/manifest_variants/{manifests,manifests_raw,manifests_waft_a2}` | Stage 3 manifests (absolute `/workspace/...` paths) | `/workspace/data/stage3/<same>` |
| `stage3/pretrained/sea_raft/`, `stage3/pretrained/waft/` | flow models | `/workspace/pretrained/{sea_raft,waft}/` |
| `stage3/runs/stage3_v2_2026-09-24/runs/` | all Stage 3 v2 runs incl. **V3_tcnssm_100ep** (best) | `car-accident/runs/stage3_v2/` |
| `submissions/2026-09-24/submit_v{1..4}_*.zip` | built DACON submissions (§4) | `/workspace/outputs/` |
| `dacon/Baseline.zip`, `dacon/analysis/` | DACON official baseline/sample data + Stage 3 OPEN analysis | `/workspace/data/dacon_baseline/` |
| `stage2/runs/2026-09-25/long_context_v2_experiments/{results,folds}` | 2026-09-25 LC-v2 + phase-loss checkpoints (all runs, fixed split + 5-fold CV, incl. OOF preds) and the 5-fold split files | `car-accident/stage2/long_context_v2_experiments/{results,folds}` |
| `stage2/cache/long_context_v2_cache_dense/` (6.4 GB) | dense native-frame DINO cache (`<id>.npy`, `.frames.npy`, `.motion.npy`) used by all LC-v2 runs and v5 | `car-accident/stage2/long_context_v2_experiments/cache_dense/` |
| `stage2/runs/2026-09-25/nexar65_experiments/{results,candidate_seed0,candidate_all_seeds}` | NEXAR-specialist heads (checkpoints) and exported candidates | `car-accident/stage2/nexar65_experiments/<same>` |
| `submissions/2026-09-25/` | `submit_v5_*.zip`, `submit_hybrid_*.zip`, `submit_nexar_specialist_*.zip` (+ smoke JSONs) (§4) | `/workspace/outputs/` |
| `stage2/runs/2026-09-26/phase_study/results/` | 2026-09-26 phase-supervision study: all 560 runs (checkpoints, configs, histories, predictions with phase log-probs), summary JSONs | `car-accident/stage2/phase_study/results/` |
| `stage2/runs/2026-09-26/aux_signal_experiments/{results,logs}` | 2026-09-26 complementary-signal study + goal campaign + v7 full-data refits (`E4_full`, `E2_full`, `XN4_full`): checkpoints, configs, histories, predictions | `car-accident/stage2/aux_signal_experiments/{results,logs}` |
| `stage2/runs/2026-09-26/aux_signal_experiments/cache_residual` (19 MB) | **residual motion** per native frame, 349 clips (needed by E4/E2/XN4 and v7) | `car-accident/stage2/aux_signal_experiments/cache_residual` |
| `stage2/runs/2026-09-26/aux_signal_experiments/{cache_geo,cache_lane_v2,cache_lane}` | geometry-head features / lane pseudo-labels (experiments only, not used by v7) | same under `stage2/aux_signal_experiments/` |
| `stage2/cache/nexar_unlabelled_670/` (10.7 GB) | the 670 unlabelled NEXAR positives: DINO 7x10 at sampled frames, global + residual motion, `labels_{fold0..4,fixed,all}.json`, `nexar_candidates.csv` | `car-accident/stage2/aux_signal_experiments/cache_nexar_u` |
| `submissions/2026-09-26/` | **`submit_v7_E4E2XN4full_V3_acc0p5_steer3.zip`** (+ smoke JSON); Codex-built `submit_v6_ES7-auxens_V3_acc0p5_steer5.zip` (+ smoke) | `/workspace/outputs/` |
| `handoff/2026-09-24/` | git bundle (**refreshed 2026-09-26**; older bundles in `archive_*`), HANDOFF.md copy, SETUP_REPORT, prompts, patch, previous agent memory notes; `uncommitted_snapshot_2026-09-26/` = patch + tarball of a parallel Codex session's uncommitted Stage 3 work and `submission_tools/v6_stage2` (not committed by this session) | — |

Obsolete (do not download): `stage2/cache/{geometry,joint_features_v1,v2}`, `stage2/pretrained/{vjepa*,rfdetr*,sam2*,depth*,dinov3_vitb16}`,
`stage2/{frames,videos,aihub,unusable}`, `stage2/outputs/submit*` (old joint model).

## 1. Current best models and scores

| Stage | Model | Offline validation | Leaderboard |
|---|---|---|---|
| 1 | original `global_g1_threshold_0_50` model (unchanged; `submission/model/stage1/`) | — | **0.953** (v1) |
| 2 | **v5: LC-v2 15-head ensemble (C0/X_ema/M_motion/PH) + motion fusion**, trained on 279 clips (not refit) | 0.767 fixed / 0.769 CV (NEXAR 0.619 / 0.670) | v5: **0.5314** (v3 P2 refit: 0.4618; v1 E3: 0.441) |
| 2 (offline best, not yet on LB) | **v7: E4 + E2 + XN4 residual-motion ensemble** (12 heads, full-data refit; §10) | CV of the recipe **0.7995 / NEXAR 0.721** | v7 zip built 2026-09-26, **not submitted yet** — ask the user |
| 3 | **V3_tcnssm_100ep** (SEA-RAFT → geometry/physics → MotionCNN → TCN+SSM, 100 ep, best ep 91) + steering **5°** + accel **±0.5 m/s²** | 0.7954 (BATON val) | specialist zip (accel ±0.5): **0.7388**; ±0.75: 0.7313; ±0.25: 0.7274; ±1.0: 0.6972; v4 (steer 1.5°): 0.6900 |

Stage 2 facts that matter:
* Offline val overestimates the leaderboard. MM-AU clips (short, ~100 frames) are easy; long clips (>1000 frames, NEXAR)
  score ~0.46–0.49 for every model, which matches the leaderboard. The real problem is event selection in long clips.
* Every Stage 2 model is FPS/duration/timestamp-blind by design (user requirement); only frame numbers/count are used.
* Details: `reports/stage2_temporal_pyramid_framecount_experiments.md`, `reports/stage2_iterative_search.md`,
  `stage2/long_video_experiments/REPORT.md`.

Stage 3 facts that matter — **the 20 fps / frame-rate notes below are superseded by §8** (the competition states Stage 3 input and
output are 10 Hz, and the baseline samples are not representative of the test set):
* DACON OPEN sample videos are 1164×874 HEVC, ~1200 frames = 60 s, i.e. **20 fps**; labels sit at
  `frame_index = 2 × sample_index` (sample = 0.1 s). Container timestamps are bogus. Format matches comma2k19 segments.
* The submission (like DACON's own baseline) emits **one row per decoded frame**. On the 50 OPEN labels: per-frame rows 0.342,
  same predictions read at frame 2k 0.613, 10 Hz input (`frames[::2]`) 0.601. The v1 leaderboard (0.654) is far above 0.342,
  so the test set may really be 10 Hz — **unresolved; decide with a leaderboard A/B**.
* Steering threshold 5° never fires on DACON clips (model outputs ~0.6–4.5° at LEFT/RIGHT). 1.25–1.5° fits
  (leave-one-video-out steer F1 0.29→0.46). Acceleration thresholds (±0.25 m/s²) showed no out-of-sample gain from changing.

## 2. Restore the environment (fresh Vast PyTorch image, RTX 5090 / Blackwell needs cu128+ wheels)

### 2.1 Code
```bash
cd /workspace
git clone https://github.com/taehyeon-k/car-accident-competition.git car-accident
cd car-accident && git checkout handoff-2026-09-24
# The R2 bundle is always the newest state. If the branch is missing on GitHub, or GitHub's HANDOFF.md is older than
# 'updated 2026-09-26' (or `git log -1` is not the latest handoff commit), update from the bundle:
#   rclone copy r2:car-accident-dataset/handoff/2026-09-24/handoff-2026-09-24.bundle /workspace/
#   git fetch /workspace/handoff-2026-09-24.bundle +handoff-2026-09-24:handoff-2026-09-24 && git checkout handoff-2026-09-24
# (a checked-out branch can't be fetched into: use `git checkout --detach` first, or `git pull /workspace/handoff-2026-09-24.bundle handoff-2026-09-24`)
```

### 2.2 Python (reuse the image's `/venv/main`; do not replace torch)
```bash
source /venv/main/bin/activate
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
uv pip install pandas scipy scikit-learn opencv-python-headless av accelerate safetensors pytest timm einops matplotlib pyyaml tqdm
```
rclone must have a remote named `r2` (Cloudflare R2, bucket `car-accident-dataset`); check with `rclone lsd r2:car-accident-dataset`
(ListBuckets on the root returns 403 — that is normal; list the bucket path instead). Upload new prefixes with `--s3-no-check-bucket`.

### 2.3 Shared assets
```bash
R=r2:car-accident-dataset; C="rclone copy --transfers 32 --checkers 32"
git clone https://github.com/facebookresearch/dinov3.git /workspace/pretrained/dinov3-source
git -C /workspace/pretrained/dinov3-source checkout 6876159a11b4df116f30f667f8c9888617df0751
$C $R/geometry_pretrain/outputs/runs/phase1_partial_noanchor /workspace/outputs/geometry_pretrain/runs/phase1_partial_noanchor
$C $R/stage3/pretrained/sea_raft /workspace/pretrained/sea_raft
```

### 2.4 Stage 2
```bash
$C $R/stage2/usable /workspace/data/stage2/usable
$C $R/stage2/manifests_349 /workspace/data/stage2/manifests --exclude 'pre_resplit_251/**'
cd /workspace/car-accident
# Frames (12 GB, ~10 min): the script re-derives the same split and refuses to overwrite a different one.
python -m stage2.scripts.prepare_workspace --workspace /workspace --extract --workers 16
# Feature caches (download instead of re-encoding)
$C $R/stage2/cache/fpsblind_192_7x10_349 stage2/artifacts/spotting_features/fpsblind_192_7x10_349
ln -s fpsblind_192_7x10_349 stage2/artifacts/spotting_features/fpsblind_192_7x10   # old configs point here; superset is fine
$C $R/stage2/cache/temporal_pyramid_cache_383 stage2/temporal_pyramid_experiments/cache_383
# Checkpoints/results
$C $R/stage2/runs/2026-09-24/temporal_pyramid_experiments stage2/temporal_pyramid_experiments/results
$C $R/stage2/runs/2026-09-24/spotting_experiments/E3-ASFormer-349 stage2/spotting_experiments/results/E3-ASFormer-349
$C $R/stage2/runs/2026-09-24/long_video_experiments stage2/long_video_experiments/results
# 2026-09-25 session (long-context v2 / phase loss / v5, NEXAR specialist, hybrid):
$C $R/stage2/runs/2026-09-25/long_context_v2_experiments/results stage2/long_context_v2_experiments/results
$C $R/stage2/runs/2026-09-25/long_context_v2_experiments/folds stage2/long_context_v2_experiments/folds   # also in git
$C $R/stage2/cache/long_context_v2_cache_dense stage2/long_context_v2_experiments/cache_dense           # or regenerate, see below
for s in results candidate_seed0 candidate_all_seeds; do $C $R/stage2/runs/2026-09-25/nexar65_experiments/$s stage2/nexar65_experiments/$s; done
# 2026-09-26 (phase study, complementary signals, goal campaign, v7): results + residual motion + NEXAR expansion
$C $R/stage2/runs/2026-09-26/phase_study/results stage2/phase_study/results
for d in results logs cache_residual; do $C $R/stage2/runs/2026-09-26/aux_signal_experiments/$d stage2/aux_signal_experiments/$d; done
$C $R/stage2/cache/nexar_unlabelled_670 stage2/aux_signal_experiments/cache_nexar_u      # 10.7 GB, only for XN* training
# optional (experiments only): cache_geo, cache_lane_v2, cache_lane from the same prefix; cache_hr (14 GB) is NOT backed up —
# regenerate with python -m stage2.aux_signal_experiments.extract_hr if ever needed (the HR experiment failed).
# Residual motion can also be regenerated: python -m stage2.aux_signal_experiments.extract_residual --workers 8 (~5 min).
```
The dense cache is regenerable (~20 min GPU for DINO + motion):
`python -m stage2.long_context_v2_experiments.extract_dense` then `python -m stage2.long_context_v2_experiments.extract_motion`
(defaults: `/workspace/data/stage2/manifests/all.jsonl`, the geometry backbone, output `stage2/long_context_v2_experiments/cache_dense`).
LC-v2 training (fixed split seeds + 5-fold CV seeds, parallel): `stage2/long_context_v2_experiments/run_config.sh <RUN_ID> "0 1 2 3" "0 1" [train.py args]`,
e.g. `--ema 0.99`, `--crop-aug 0.5`, `--motion`, `--loss phase`. Existing run dirs refuse overwrite.
NEXAR specialist: `python -m stage2.nexar65_experiments.batch` (see `stage2/nexar65_experiments/REPORT.md`).
Train (examples; all heads are small, <1 GB VRAM, minutes):
```bash
# Pyramid P2 with early stopping on the 70-clip val split (one seed)
python -m stage2.temporal_pyramid_experiments.train --experiment P2 --seed 0
# Full-data refit for submission (no validation; stop at the seed's validated best epoch: seeds 0..3 -> 7,8,6,13)
python -m stage2.temporal_pyramid_experiments.refit_full --seed 0 --stop-epoch 7 \
  --output stage2/temporal_pyramid_experiments/results/REFIT_P2_all349/seed0
# Ensemble eval on val
python -m stage2.temporal_pyramid_experiments.eval_ensemble --id ENS_P2_seed0123 P2/seed0 P2/seed1 P2/seed2 P2/seed3
# Legacy E3 ASFormer
python -m stage2.spotting_experiments.train --config stage2/spotting_experiments/configs/e3_asformer_349.json --output <dir>
```
New cache for new clips: `python -m stage2.spotting_experiments.extract_features --backbone-checkpoint /workspace/outputs/geometry_pretrain/runs/phase1_partial_noanchor/backbone_best.pth --output <cache dir>`
then `python -m stage2.temporal_pyramid_experiments.expand_cache --old <192 cache> --output <383 cache>`.

### 2.5 Stage 3
```bash
$C $R/stage3/BATON-Sample /workspace/data/stage3/BATON-Sample
$C $R/stage3/cache/motion /workspace/cache/stage3/motion          # 20 GB gz caches
$C $R/stage3/cache /workspace/cache/stage3 --include 'physics_stats*.pt' --max-depth 1
for m in manifests manifests_raw manifests_waft_a2; do $C $R/stage3/manifest_variants/$m /workspace/data/stage3/$m; done
$C $R/stage3/runs/stage3_v2_2026-09-24/runs /workspace/car-accident/runs/stage3_v2
# manifests_raw (fast loading) needs the uncompressed mirror (98 GB); otherwise use manifests/ (gz, slower):
mkdir -p /workspace/cache/stage3/motion_raw && cd /workspace/cache/stage3/motion && \
  ls *.pt | xargs -P20 -I{} sh -c "gzip -dc {} > ../motion_raw/{}"
```
Best run (V3) reproduces with:
```bash
cd /workspace/car-accident
WORKERS=12 stage3/experiments/exp.sh V3_tcnssm_100ep --set model.temporal.type=tcn_ssm \
  --set optimization.epochs=100 --set early_stopping.enabled=false     # ~5.4 h on RTX 5090
```
Optional caches (only for D1/D2 DINO-visual and H1 WAFT runs, which were not better): see `stage3/experiments/HANDOFF.md` §3.

### 2.6 DACON baseline / OPEN sample data
```bash
rclone copy $R/dacon /workspace/data/dacon_baseline_r2 && cd /workspace/data/dacon_baseline && unzip -q ../dacon_baseline_r2/Baseline.zip
```
Contains DACON's train/inference notebooks and 5 OPEN clips per stage with labels.

## 3. Submission package

`submission/` (code in git; weights restored separately):
```
inference.py            predict_stage1/2/3(data_dir, model_dir) -> DataFrame
requirements.txt        DACON preinstalled torch 2.8 stack; nothing extra
model/stage1/           runtime.py + best.pt  (from R2 stage2/outputs/submit/model/stage1/best.pt, sha256 5c1b5214…)
model/stage2/           runtime.py, pyramid_models.py, spotter_models.py, sampling.py, config.json, vendor/dinov3,
                        backbone.pth (= geometry backbone_best.pth), members/p2_seed{0..3}.pt (REFIT_P2_all349, model+config only)
model/stage3/           runtime.py, code/stage3 (copy of repo stage3/), best.pt (V3, optimizer stripped),
                        decoder_overrides.json {"steering":{"threshold_deg":1.5}}, pretrained/sea_raft/{source,model.safetensors}
```
Easiest restore of weights: unzip `submissions/2026-09-24/submit_v4_P2ens-full349_V3_steer1p5.zip` over `submission/`.
Build + test: `python submission_tools/build_zip.py --output /workspace/outputs/submit.zip` then
`python submission_tools/smoke_submission.py --submission <unzipped dir> --checkpoint <unzipped>/model/stage3/best.pt`
(runs all three stages offline with network blocked; the Stage 3 code in `model/stage3/code` must be re-synced from `stage3/`
with `rsync -a --delete --exclude __pycache__ --exclude experiments --exclude configs --exclude '*.md' --exclude cuda_benchmark.json stage3/ submission/model/stage3/code/stage3/` whenever stage3 changes).
Stage 2 runtime was verified bit-exact against the training evaluation (frame selection, features, predictions).

## 4. Submissions built (R2 `submissions/2026-09-24/`, `submissions/2026-09-25/`, `submissions/2026-09-26/`)

| Zip | Stage 2 | Stage 3 | Leaderboard |
|---|---|---|---|
| v1 `submit_v1_E3_V2ep43.zip` | E3 ASFormer (279 clips) | V2 ep43, steer 5° | S1 0.953 / S2 0.441 / S3 0.654 |
| v2 `submit_v2_P2ens-trainsplit_V3.zip` | P2 ensemble (279 clips) | V3, 5° | not submitted when written |
| v3 `submit_v3_P2ens-full349_V3.zip` | P2 ensemble refit (349) | V3, 5° | **S1 0.953197975 / S2 0.461806490 / S3 0.7273634082** (best) |
| v4 `submit_v4_P2ens-full349_V3_steer1p5.zip` | P2 ensemble refit (349) | V3, **1.5°** | S3 0.6899652932 (only change vs v3; worse by 0.037) |
| v5 `2026-09-25/submit_v5_LCv2ens4-motion_V3_acc1_steer5.zip` | LC-v2 15-head ensemble (C0/X_ema/M_motion/PH, 279 clips) + motion fusion | V3, 5°, accel ±1.0 m/s² | S1 0.953 / **S2 0.5314** (best) / S3 0.6972 |
| `2026-09-25/submit_nexar_specialist_steer5_accel0p5.zip` | NEXAR specialist (seed-0 candidate: 3 heads + NEXAR temporal prior) on every clip | V3, 5°, accel ±0.5 | S2 **0.437** (< v3 0.4618); **S3 0.7388** (best) |
| `2026-09-25/submit_hybrid_stage2_steer5_accel0p75.zip` | length-gated: >500 native frames → all-seed NEXAR specialist (9 heads), else P2 ensemble | V3, 5°, accel ±0.75 | S1 0.953 / S2 0.4154 / S3 0.7313 |
| `2026-09-26/submit_v6_ES7-auxens_V3_acc0p5_steer5.zip` (built by a parallel Codex session) | event-specific 18-head ensemble (v5 families + E4/E2/M1), all trained on 279 clips; CV 0.7884 / NEXAR 0.7103 | V3, 5°, accel ±0.5 | not recorded |
| **`2026-09-26/submit_v7_E4E2XN4full_V3_acc0p5_steer3.zip`** | **E4 + E2 + XN4, 4 seeds each, full-data refit** (§10); CV of the recipe 0.7995 / NEXAR 0.721 | V3, **3°**, accel ±0.5 | not submitted yet |

v5 rebuild: `submission_tools/v5_stage2/build_v5.sh`. v7 rebuild: `submission_tools/v7_stage2/build_v7.sh` (needs the `*_full` checkpoints in
`stage2/aux_signal_experiments/results/` and the base zip `submit_nexar_specialist_steer5_accel0p5.zip` in `/workspace/outputs/`);
parity check `submission_tools/v7_stage2/check_v7_parity.py <unzipped dir>`. The specialist/hybrid packages were built from the zips' own code, which is
saved (without weights) in `submission_tools/{nexar_specialist,hybrid_stage2}_package/`; the zips in R2 are the complete versions.

Leaderboard results for v3/v4 recorded 2026-09-25. Steering 1.5° (fit on the 50 OPEN labels) *lost* 0.037 on the
leaderboard vs 5°, so the OPEN-sample threshold fit does not transfer; keep 5° unless new evidence.
Stage 2 leaderboard vs per-source validation: `reports/stage2_leaderboard_vs_nexar.md`.
Ask the user for any newer leaderboard results before deciding next steps.

## 5. Recommended next steps (revised 2026-09-26 evening, see §10)

1. **Ask the user for leaderboard results of v6 / v7.** v7 changes Stage 2 (new ensemble) *and* Stage 3 (steer 3° untested); to read
   Stage 2 alone, build the same zip with steer 5° (edit `submission_tools/v7_stage2/stage3_decoder_overrides.json`, rerun build_v7.sh).
2. **Stage 2**: the plateau is ENTRY (CV 0.69; NEXAR/AIHUB ~0.55). Things that did not help ENTRY are listed in §10 — don't repeat them.
   Untried: labelling ENTRY on more NEXAR positives (labelling app), and 8-seed confirmation of XN2/XN4e0/E3 families.
3. **Stage 3**: accel ±0.5 is the best measured (0.7388 at steer 5°). Steering: 5° 0.7388 / 1.5° 0.690; 3° is in v7 (untested).
4. Stage 1 unchanged at 0.953.

## 6. 2026-09-25 session summary (Stage 2 experiments; no Stage 3 retraining, but the new zips change Stage 3 accel thresholds — see §4)

* Leaderboard of v3/v4 recorded (§4). Hypothesis "the Stage 2 test set looks like NEXAR (long clips)": `reports/stage2_leaderboard_vs_nexar.md`.
* **Long-context v2** (`reports/stage2_long_context_v2_report.md`): NEXAR misses are mostly 0.3–1 s near-misses already present with
  128–256 frames of context, not wrong-event selection. No architecture change reliably improves NEXAR. Best: context-crop 0.5 + EMA 4-seed
  ensemble (fixed 0.757, CV 0.714). Found the GroupNorm padding issue: evaluate at batch 1 (= submission runtime).
* **Phase loss** (`reports/stage2_phase_loss_report.md`): structured phase decoding hurts long clips; the phase model helps as an ensemble member.
  v5 ensemble: fixed val 0.767 / NEXAR 0.619, CV 0.769 / NEXAR 0.670 (leaderboard S2 0.5314, recorded 2026-09-26).
* **NEXAR specialist** (`stage2/nexar65_experiments/REPORT.md`): 3 heads + NEXAR-fitted temporal KDE prior; NEXAR val 0.71–0.74, but it
  was selected on the same 15 val clips and hurts non-NEXAR clips; its submission scored **S2 0.437 on the leaderboard** (< v3 0.4618). **Length-gated hybrid** (`stage2/length_gated_experiments/REPORT.md`): offline 0.80.
* NEXAR specialist and length-gated hybrid were done in a parallel Codex session (prompts in `~/.codex/history.jsonl` on that server, not backed up).
* Takeaway: offline NEXAR-slice gains did not transfer (specialist 0.437, hybrid 0.4154), but the CV-validated v5 ensemble did (0.5314). See §8.

## 7. Constraints from the user (keep following)

* Do not delete anything from R2; do not overwrite valuable checkpoints/runs; keep other jobs' GPU memory safe
  (experiments here used a per-process memory cap and required ≥2–3 GiB free).
* Stage 2 models must stay FPS/duration/timestamp-blind.
* Use `rclone copy` (not sync) to R2; new prefixes need `--s3-no-check-bucket`.
* **Stage 3 is 10 Hz in and 10 Hz out** (competition rule). Do not resample, do not use the baseline samples' frame rate, and do not
  fit thresholds or conventions to the DACON baseline data: it is only an example and the real evaluation data is explicitly unlike it.
* Commit + back up code/reports (GitHub and R2 bundle) before starting new experiments.
* **Keep ≥ 10 GB of RAM free for other jobs (a Codex run shares the machine); never kill another job's running runs.** Launch Stage 2
  training through `stage2/aux_signal_experiments/queue.py` (reads `jobs.txt`, launches only while MemAvailable ≥ 14 GB).
* Judge Stage 2 by 5-fold CV over all 349 clips (80 NEXAR), not the 70-clip fixed split; use ≥ 3 CV seeds (single runs move ±0.03).

## 8. 2026-09-26: leaderboard results and Stage 3 rule

Leaderboard (from the user):

| Submission | S1 | S2 | S3 | Stage 3 settings |
|---|---|---|---|---|
| v3 P2 refit | 0.953 | 0.4618 | 0.7274 | accel ±0.25, steer 5° |
| v4 | — | — | 0.6900 | accel ±0.25, steer 1.5° |
| NEXAR specialist | — | 0.437 | **0.7388** | accel ±0.5, steer 5° |
| **v5 LC-v2 15-head + motion** | 0.953 | **0.5314** | 0.6972 | accel ±1.0, steer 5° |
| length-gated hybrid | 0.953 | 0.4154 | 0.7313 | accel ±0.75, steer 5° |

Verified: the Stage 3 package in v3, v5, hybrid and the NEXAR-specialist zip uses the same `best.pt` (V3) and code; v5/hybrid only add `decoder_overrides.json`
(accel thresholds). So the S3 differences are purely the acceleration threshold.

Reading:
* **Stage 2**: v5 is the first model with a clear LB gain (+0.070 over v3), and it was chosen by 5-fold CV (NEXAR n = 80), not the
  15-clip val slice. Both models tuned on the NEXAR val slice (specialist, hybrid) lost against v3; the hybrid only changes >500-frame
  clips vs the P2 ensemble and dropped 0.046, so the hidden test does contain long clips, and the NEXAR temporal prior hurts on them.
* Stage 3 accel-threshold curve on the LB (same V3 weights, steer 5°): ±0.25 0.7274 → **±0.5 0.7388** → ±0.75 0.7313 → ±1.0 0.6972.
* **Best combined zip does not exist yet**: v5 Stage 2 + Stage 3 accel ±0.5 would combine the two best stage results.
* **Stage 3 rule**: the competition explicitly states Stage 3 input and output are 10 Hz and the baseline data is only an example,
  unlike the real evaluation data. The current runtime emits one row per decoded frame without resampling and V3 trains at
  `target_hz: 10`, so it already follows the rule. The old "per-frame vs 10 Hz A/B" item is dropped.

## 9. 2026-09-26: Stage 2 phase-supervision study (`reports/stage2_phase_supervision_study.md`)

Spec `stage2/stage2_phase_loss_experiments.md`; code `stage2/phase_study/`; 560 runs, 5-fold CV with 3–8 seeds per arm.
* Semantic phase supervision is a small, consistent **auxiliary** gain (+0.005 to +0.007 CV over the matched no-phase control,
  4–5/5 folds, mostly ENTRY). Below the +0.01 bar. Shuffled phase targets hurt, so the effect is semantic.
* Phase at inference (transition, global structured, local re-rank), a phase-conditioned event head, multi-scale heads,
  λ = 2, the monotonic term and soft boundaries are all neutral or harmful.
* **Correction to §6 / the phase-loss report:** the old PH recipe is −0.010 vs control on 8 seeds (old code rerun agrees); its
  0.709 was two lucky seeds. Selecting checkpoints with the structured decoder costs ≈0.008. v5's PH members are a lucky draw
  (0.7688; the PH recipe averages 0.7595 ± 0.003 as v5's 4th family).
* Best recipe: ORD phase λ = 1 + transition 0.75, direct weight 1, direct decoding/selection. As v5's 4th family:
  0.7636 ± 0.003 vs 0.7568 (NT) and 0.7595 (old PH recipe), but not above v5's specific PH checkpoints.
* Train one run: `python -m stage2.phase_study.train --run-id X --phase-rep ord --w-phase 1 --w-tr 0.75`; batches with
  `stage2/phase_study/run.sh RUN "FIXED_SEEDS" "CV_SEEDS" [args]`. Evaluate: `python -m stage2.phase_study.analyze RUNS --control A0_d1`,
  `python -m stage2.phase_study.ensemble_test RUN --seed-sets "0,1 2,3 4,5 6,7"`.
* Runs read features lazily from the memory-mapped dense cache (~1 GB RAM/process). P = 10 parallel jobs saturates 16 CPUs;
  the GPU is not the bottleneck.

## 10. 2026-09-26 (afternoon/evening): Stage 2 complementary signals, goal campaign, v7 submission

Reports: `reports/stage2_complementary_signals_interim.md` (stopped by the user), `reports/stage2_goal_campaign.md`.
Code: `stage2/aux_signal_experiments/` (train.py = harness incl. motion/geo/HR/lane/risk/boundary/extra-NEXAR options, queue.py,
analyze.py, ensemble_test.py, goal_eval.py, event_select.py, extract_residual.py, extract_geo.py, extract_hr.py, pseudo_lane.py,
qa_lane.py, nexar_expand.py, nexar_labels.py, attr_stack.py, m3_late_cue.py).
* **Residual motion input** (camera similarity motion from sparse LK + RANSAC removed, residual summarised; `extract_residual.py`) is the
  big win: NT + global+residual motion (E4) CV 0.740 vs 0.699 without motion (8 seeds). As a late decode cue it does nothing.
* **Boundary heads [h, Δh]** (E2): ≈ E4 alone, better in ensembles. Risk supervision, lane intrusion, geometry features, 14×25 tokens,
  ENTRY loss weight, sharper targets, attribute stacking/selection: no reliable gain (details in the reports).
* **NEXAR expansion (XN4)**: R2 `stage2/manifests/nexar_candidates.csv` has Nexar's public `time_of_event` for all 750 positives
  (matches our COLLISION within 0.3 s for 80–88 % of labelled clips). The 670 unlabelled clips were decoded (`nexar_expand.py`; parity with
  the JPEG caches verified; note `cv2.phaseCorrelate` modifies inputs in place) and labelled per split (`nexar_labels.py`: COLLISION from
  time_of_event + offset, ENTRY pseudo-label from a teacher that never saw the evaluated fold). XN4: long-clip COLLISION catastrophic
  0.071 → 0.013, NEXAR 0.642 → 0.657; ENTRY slightly worse.
* **Goal campaign** (target CV > 0.8 / NEXAR > 0.7): best = equal-weight **E4 + E2 + XN4** (8 seeds each): **CV 0.7995 / NEXAR 0.721**,
  fixed split 0.784 / 0.689. Not met on overall by 0.0005; stopped rather than select combinations on the same CV.
* **v7 submission** (`submission_tools/v7_stage2/`): Stage 2 = E4_full + E2_full + XN4_full (4 seeds each, refit on all 349 clips,
  stop epochs 8/8/9 = CV median; XN4 labels from the full-data E4+E2 teacher), runtime reused from `submission_tools/v6_stage2`
  (Codex session), parity exact (residual and member logits diff 0.0 on 70 clips), offline smoke test passes; Stage 1 unchanged;
  Stage 3 V3 with steer 3.0° / accel ±0.5.
* A parallel Codex session worked on Stage 3 DriveDNA data scaling and built `submission_tools/v6_stage2` + a v6 zip; its uncommitted
  files are snapshotted in R2 `handoff/2026-09-24/uncommitted_snapshot_2026-09-26/` (not committed by this session).
