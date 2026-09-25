# Stage 2 long-context v2 experiments (2026-09-25)

Task: `stage2/Stage2_long_gcontext_experiments.md`. Code: `stage2/long_context_v2_experiments/`. Repo base commit `ae0b464`
(branch `handoff-2026-09-24`). Nothing earlier was overwritten; all new runs are under `stage2/long_context_v2_experiments/results/`.

**Summary.** No change gives a reliable NEXAR / long-clip improvement. The diagnostics show why: most NEXAR misses are
**0.3–1 s near-misses that already occur with 128–256 frames of context**, not distractor-driven event selection. Candidate re-ranking,
denser native frames, pooling, phase supervision and coarse long-range modules therefore have little to act on. The best overall
candidate is **context-crop augmentation + EMA, 4-seed ensemble**: 0.757 on the fixed split (same as the current P2 ensemble;
NEXAR 0.542 vs 0.500) and 0.714 vs 0.702 in 5-fold CV (NEXAR 0.559 vs 0.548). The direction matches on both protocols; the NEXAR part is within noise.

## 0. Setup and two methodology findings

* **Dense native-frame cache** (`extract_dense.py`): geometry-DINOv3-S 7×10 pooled tokens for every frame of all 349 clips
  (125k frames, 6.4 GB, ~20 min). Same preprocessing as the existing caches: max |Δ| vs the 192/383 caches = 0.014 (bf16 noise).
  E3 val metrics reproduce bit-exactly (0.689). It enables arbitrary crops (A), native windows (G) and crop augmentation.
* **Batch-size bug in evaluation.** The pyramid's GroupNorm also normalises over padded positions, so its predictions depend on
  what else is in the batch. The original P2 results were evaluated with batch 4 (P2 ensemble 0.752); batch 1 matches
  the submission runtime and gives **0.757**, the value quoted in HANDOFF. All numbers here use batch 1.
  (Training still uses padded batches of 4, which is part of the P2 recipe.)
* **15 NEXAR val clips are too few**: P2 seeds 0–3 span 0.42–0.50 on NEXAR. So every config is reported twice:
  **fixed 279/70 split** (mean ± std over 3–4 seeds) and **5-fold stratified CV over all 349 clips**, with pooled out-of-fold
  predictions (NEXAR n = 80; 1–2 seeds). CV early-stops on the held-out fold, the same (slightly optimistic) rule as the fixed split.
* Catastrophic miss = |error| / (N−1) > 10 %. All models stay FPS-, duration- and timestamp-blind; FPS is read only by the offline metric.

## 1. Verified prior claims (batch-1 re-evaluation, fixed val)

| Model | all | NEXAR | non-NEXAR | >1000 |
|---|---|---|---|---|
| E3 fixed-128 ASFormer | 0.689 | 0.522 (E .33 / C .67) | 0.734 | 0.492 |
| P2 seed0 | 0.735 | 0.473 (E .47 / C .33) | 0.804 | 0.463 |
| P10 EMA seed0 | 0.738 | 0.481 | 0.807 | 0.467 |
| P2 4-seed ensemble | **0.757** | 0.500 (E .33 / C .47) | 0.826 | 0.492 |
| P5 global coarse / P6 long oversample / P8 hard neg / P9 adaptive target | 0.69 / 0.68 / 0.69 / 0.69 | 0.37 / 0.45 / 0.35 / 0.44 | | |

Confirmed: E3 has the best NEXAR COLLISION (10/15); P2 has a better aggregate but not better long clips; P5/P6/P9 did not help;
the ensemble is the largest aggregate gain.

## 2. Phase 1 — diagnostics

### A. Controlled NEXAR context length (15 val clips; crops keep ENTRY..COLLISION and their relative position)

P2 4-seed ensemble (others in `results/diagnostics/context_length.json`):

| context | score | ENTRY | COLL | ENTRY MAE (fr) | COLL MAE (fr) | E norm | C norm | side F1 | eva F1 | E cat | C cat |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 0.576 | .60 | .40 | 16.0 | 18.2 | .126 | .143 | .87 | .64 | .33 | .53 |
| 256 | 0.553 | .53 | .40 | 28.3 | 30.6 | .111 | .120 | .87 | .64 | .40 | .20 |
| 512 | 0.585 | .53 | .47 | 39.3 | 39.5 | .077 | .077 | .87 | .70 | .13 | .20 |
| 768 | 0.571 | .53 | .40 | 42.4 | 48.1 | .055 | .063 | .87 | .76 | .13 | .13 |
| 1024 | 0.548 | .47 | .40 | 52.5 | 46.3 | .051 | .046 | .87 | .76 | .20 | .13 |
| full (~1200) | 0.500 | .33 | .47 | 67.5 | 59.3 | .056 | .049 | .87 | .60 | .20 | .13 |

E3: 0.416 → 0.522 (collision .33 → .67 as context grows). P2 seed0: 0.471 → 0.546 (512) → 0.473.
Drift: between 1024 and full, 7–20 % of videos switch to another event (>10 % of the clip away), in both directions (hits lost ≈ hits gained).

**Finding:** long context is *not* the main cause. With only 128–256 frames, collision accuracy is already 0.33–0.40 for every
model. Signed errors on those short crops: collision is predicted **0.3–0.4 s early** (median, all models), entry 0.2–0.3 s late,
and 80–87 % of predictions are within 1 s. Extra context raises the absolute error mainly through a few gross switches
(2–3 of 15 clips at full length).

### B. Candidate Recall@K (70 val clips; NMS radius = 1.5 % of positions; hit = official 0.3 s)

| model | NEXAR ENTRY @1/2/4/8/16 | NEXAR COLL @1/2/4/8/16 | overall ENTRY @1/8 | overall COLL @1/8 |
|---|---|---|---|---|
| E3 | .33/.53/.60/.60/.67 | .60/.67/.67/.73/.80 | .54/.83 | .80/.94 |
| P2 seed0 | .40/.47/.73/.73/.73 | .40/.47/.60/.67/.73 | .61/.93 | .80/.91 |
| P10 | .40/.53/.60/.67/.67 | .47/.53/.73/.80/.87 | .64/.91 | .80/.96 |
| P2 ens4 | .27/.67/.80/.80/.80 | .47/.53/.73/.73/.73 | .61/.94 | .83/.94 |

Short clips (<300 frames): top-8 ≈ 0.95–1.00. Every GT frame is reachable at the sampled density (reachable = 1.00).
**Finding:** long-clip top-1 does collapse while top-4/8 is higher (e.g. ENTRY .27 → .80), so re-ranking has headroom, capped at about
0.75–0.80; 20–30 % of NEXAR events are in no top-16 candidate. Per-video candidates: `results/diagnostics/recall_candidates.json`.

## 3. Phase 2/3 — architectures (all use the same frozen dense features, 279/70 split, P2 recipe)

Scores are the official offline score. Fixed = mean ± std over seeds; CV = pooled out-of-fold (349 clips, NEXAR 80).

| run | params | fixed all | fixed NEXAR | fixed non-NEX | CV all | CV NEXAR | CV NEX E / C | CV NEX C-cat |
|---|---|---|---|---|---|---|---|---|
| C0 avg pool (control) | 274k | 0.695±.023 | 0.465±.045 | 0.757±.019 | 0.687±.002 | **0.551±.025** | .44 / .50 | .14 |
| C1 max pool | 274k | 0.710±.018 | 0.491±.048 | 0.769±.016 | 0.689 | 0.497 | .40 / .40 | .15 |
| C2 SGP strided DW | 275k | 0.700±.021 | 0.533±.057 | 0.745±.011 | 0.699 | 0.497 | .34 / .49 | .16 |
| D phase λ=0.1 | 274k | 0.719±.010 | 0.467±.030 | 0.787±.005 | 0.700 | 0.482 | .36 / .45 | .16 |
| D phase λ=0.25 | 274k | 0.712±.027 | 0.515±.040 | 0.765±.023 | 0.689 | 0.528 | .43 / .44 | .14 |
| D phase λ=0.5 | 274k | 0.696±.010 | 0.440±.052 | 0.764±.016 | 0.694 | 0.521 | .39 / .51 | .16 |
| F1 coarse bi-SSM (2 coarsest levels) | 291k | 0.710±.021 | 0.460±.038 | 0.777±.019 | 0.685 | 0.503 | .38 / .46 | .15 |
| F2 coarse local attention | 407k | 0.711±.014 | 0.469±.057 | 0.776±.014 | 0.708 | 0.550 | .48 / .49 | .14 |
| X context-crop aug 0.5 | 274k | 0.735±.018 | 0.517±.043 | 0.794±.018 | 0.693±.004 | 0.527±.010 | .39 / .51 | .14 |
| X EMA 0.99 | 274k | 0.706±.010 | 0.445±.019 | 0.776±.008 | 0.702±.007 | 0.540±.013 | .39 / .52 | .15 |
| X crop 0.5 + EMA | 274k | **0.740±.023** | 0.492±.063 | **0.807±.018** | 0.695±.002 | 0.531±.014 | .43 / .49 | .13 |

4-seed / 2-CV-seed **probability ensembles**:

| ensemble | fixed all | fixed NEXAR | fixed non-NEX | CV all | CV NEXAR | CV >1000 |
|---|---|---|---|---|---|---|
| original P2 ens4 (reference, fixed only) | 0.757 | 0.500 | 0.826 | — | — | — |
| C0 control | 0.744 | 0.511 | 0.807 | 0.702 | 0.548 | 0.543 |
| X crop | **0.764** | 0.555 | 0.820 | 0.714 | 0.542 | 0.527 |
| X EMA | 0.736 | 0.479 | 0.806 | 0.714 | **0.565** | **0.560** |
| X crop + EMA | 0.757 | 0.542 | 0.815 | 0.714 | 0.559 | 0.552 |

**C (pooling).** Max/SGP gain on the fixed split but lose NEXAR in CV (0.497 vs 0.551). Rejected per the brief's rule: aggregate up, long-clip selection not better.
**D (phase auxiliary).** No consistent effect at any λ; phase loss converges (~0.07 CE), so the phase state is learnable but does not suppress distant peaks.
**F (coarse long-range).** Run despite A–E not motivating it, as a negative control. F2 matches the control on CV NEXAR with +49 % params; F1 is worse and 3× slower.
**X (context-crop augmentation).** New: long clips (>256 frames) are seen half the time as a random window of 256..N frames that
still contains ENTRY..COLLISION, then adaptive sampling. Best fixed-split aggregate and short-clip score. In CV it is +0.006 overall and −0.024 on NEXAR
(single models), i.e. a general regulariser, not a long-clip fix. EMA gives the best CV aggregate for single models.

### E. Top-K pair scorer (K = 8 × 8, out-of-fold candidates from an inner 5-fold split; fixed epochs; seed 0)

| variant | fixed all | NEXAR | non-NEX | notes |
|---|---|---|---|---|
| generator (C0 seed0) | 0.696 | 0.456 | 0.761 | |
| full scorer, 20 ep (61k params) | 0.676 | 0.456 | 0.736 | changes 44/70 pairs; OOF-train score 0.84 → memorises clips |
| full scorer, 5 ep | 0.691 | 0.526 | 0.736 | +1 NEXAR clip, −short clips |
| scalars-only scorer | 0.696 | 0.456 | 0.761 | learns to keep the generator argmax |

Pair oracle (some candidate pair hits both events): 0.81 overall, **0.40 on NEXAR**, so the pair scorer's ceiling on NEXAR is low.
No example meets the brief's success pattern (top-1 wrong, GT in top-8, scorer picks it) more often than the reverse, so none are listed.
Not run in CV (fixed-split evidence was negative).

### G. Native-frame refinement (±32 native frames, 3 dilated DW blocks, 54k params; trained on GT+random offsets)

| variant | all | NEXAR | non-NEX |
|---|---|---|---|
| generator | 0.696 | 0.456 | 0.761 |
| refine top-1 / top-3 pairs, all clips | 0.651 | 0.479 | 0.698 |
| refine only clips >128 frames | 0.681 | 0.479 | 0.736 |

Local skill with the anchor *at* GT on val: NEXAR entry 0.40 / collision 0.53, other sources 0.51 / 0.76. That is worse than the global
generator, and skill *rises* when the anchor is offset (it relies on seeing the other event / wider context). **Finding:** precision is not
limited by sampling density; local native frames do not contain a sharper cue than the global model already uses.

### 9. Event-conditioned attributes

Not retrained: the existing P7 run (event-conditioned side/evasion heads, same P2 backbone), re-evaluated at batch 1, gives 0.717 all /
0.489 NEXAR (side F1 0.81 all, 0.87 NEXAR; evasion F1 0.64 all, 0.53 NEXAR) vs P2 seed0 0.735 / 0.473, i.e. no gain from one seed.
Side F1 is not the NEXAR bottleneck (0.87 for the P2 ensemble); evasion F1 (0.49–0.76) is noisy with 15 clips.

## 4. Overfitting, runtime, VRAM

Train-minus-val FPS-blind score at the selected epoch is +0.24…+0.34 for every head (best epochs 4–29, very seed-dependent). The heads are
already heavily over-parameterised for 279 clips; EMA and crop augmentation do not close the gap (+0.28/+0.31).
Peak allocated VRAM ≈ 490 MB for every head (training, batch 4). Temporal head ≈ 3–6 ms/video (F1 13 ms), measured with 6 concurrent jobs.
The encoded frames per video are unchanged (adaptive 128–320); G would add up to 2·R·65 native frames per video.

## 5. Conclusions

**Confirmed**
1. NEXAR failures are mostly **near-misses (0.3–1 s)** that appear even with 128–256 frames of context; systematic collision-early bias ≈ 0.3–0.4 s.
2. Added context causes some gross event switches (≈ 15 % of NEXAR clips catastrophic for collision in CV), but that is the minority failure.
3. Denser/native frames do not fix precision (G, and the old 128→320 result).
4. Batch-size-dependent evaluation of the pyramid; batch 1 is the correct protocol.
5. Across 11 single-model configs, CV NEXAR stays in 0.48–0.55; no architectural change beats the control beyond noise.

**Plausible**
* NEXAR labels have ±0.3–0.5 s ambiguity relative to visual cues (the collision-early bias would also fit a labelling-convention difference).
* Context-crop augmentation + EMA is a small general gain (+0.012 CV aggregate over the control ensemble), mostly on short clips.

**Inconclusive**
* Pair scoring with a much larger candidate-labelled dataset; phase supervision with richer phases; any NEXAR difference below ±0.04 (CV SE on 80 clips).

## 6. Recommendation

* **Next leaderboard submission:** if a Stage 2 slot is spent, use a **crop 0.5 + EMA 0.99, 4-seed ensemble refit on all 349 clips**
  (same runtime and sampling as the v3 package; only the member checkpoints change). Expected change vs v3 is small (CV +0.012 aggregate, NEXAR +0.01,
  both within noise); do not expect a large leaderboard jump.
* A higher-value use of a submission is the diagnostic from `reports/stage2_leaderboard_vs_nexar.md` (a long-clip-only change) to learn whether the test set is long at all.
* **Next experiment:** attack the near-miss mode rather than event selection. Audit NEXAR ENTRY/COLLISION labels against the video for a
  sample of clips (systematic offset? inconsistent convention?). If labels are consistent, add more long labelled clips (NEXAR has
  thousands of public clips), since the long-clip set (65 training clips) is the binding constraint.

## 7. Files

`stage2/long_context_v2_experiments/`: `extract_dense.py`, `common.py` (sampling emulation, metrics), `diagnostics.py` (A, B), `models.py`
(LCPyramid: pool/phase/coarse variants), `train.py`, `pair_scorer.py` (E), `refine.py` (G), `ensemble.py`, `summarize.py`, `run_config.sh`, `folds/`.
Results: `results/<run>/seed*/` and `results/<run>/cv/fold*_seed*/` (config, history, metrics incl. by-source/by-bin/Recall@K, per-video predictions with logits,
checkpoints), `results/diagnostics/`, `results/summary_table.json`, `results/ensemble_*.json`. Dense cache `cache_dense/` (6.4 GB) is not in git.
