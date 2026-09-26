# Stage 2 generalization & robustness research (2026-09-26, 12 h autonomous session) — living report

Rule for this session (from the user): no validation-specific tricks; methods must be general (physical / learning rationale) and judged on
**generalization across sources and robustness under distribution shift**, not only on pooled CV.
Code: `stage2/generalization/` (+ options in `stage2/aux_signal_experiments/train.py` / `model.py`). Runs use the memory-aware queue
(≥ 10 GB RAM kept free for a parallel Codex job).

## Evaluation protocols added
1. **LOSO** (leave-one-source-out): train on 3 of AIHUB / CCD / MMAU / NEXAR, test on the 4th; fixed stop epoch 9 (CV median), so
   the held-out source is never used for selection. Reported per source, mean and **worst source**; 3 seeds (+ seed ensemble).
2. **Frame-rate robustness**: each fold's CV models evaluated on their held-out clips re-sampled to 1/k of the native frame rate
   (frames[::k]; motion recomputed on the retained frames, `extract_stride_motion.py`, verified identical at k = 1). k = 1, 2, 3.
3. **Crop robustness**: random half-length window containing both events (seeded per clip).
4. **ENTRY-gap diagnostics** (`entry_gap.py`): ENTRY hit rate by GT ENTRY→COLLISION gap, predicted vs GT gap, P(E hit | C miss).

## Baseline findings
* **LOSO gap is large**: A0 (no motion) mean 0.628 / worst 0.450 (NEXAR); E4 (residual motion) 0.663 / 0.561. Residual motion is a
  real generalization gain (+0.111 on the worst source, 3/4 sources), mostly COLLISION. **ENTRY barely transfers** (0.28–0.34 on held-out
  AIHUB / NEXAR vs 0.5–0.6 in-domain).
* **Frame rate is a large hidden weakness**: at 1/3 of the native frame rate every model loses 0.10–0.12 (E4 0.749 → 0.633), ENTRY
  collapses (0.63 → 0.39). Not motion-specific (A0 degrades as much).
* **Crop robustness is fine** (E4 0.749 → 0.756 on half-length windows): no reliance on clip-position priors.
* **ENTRY accuracy is set by the ENTRY→COLLISION gap in every source**: <0.5 s ≈ 0.92, >1.5 s ≈ 0.12–0.35; predictions are
  systematically too close to COLLISION (pred gap 0.80–0.86 s vs GT 1.00 s). MMAU (median gap 0.40 s) looks easy for this reason.

## Hypotheses tested
| # | method | rationale | result |
|---|---|---|---|
| H1 | **temporal-rate (stride) augmentation** (each clip at 1/1, 1/2, 1/3 frame rate, p = .5/.25/.25; motion recomputed) | frame-rate invariance | **Robustness win**: 3-seed ens. stride 1/2/3 = 0.768 / 0.753 / 0.701 vs 0.749 / 0.708 / 0.633 (+0.019 / +0.045 / +0.068; stride-3 ENTRY 0.39 → 0.51); single-model CV +0.002; LOSO mean +0.002 (**CCD, the 10 fps source, +0.046**; MMAU −0.027) |
| H2 | per-clip input normalisation (DINO centring / motion median scaling / both) | domain offsets | **rejected**: LOSO mean −0.045 / −0.005 / −0.060, worst −0.10 (NEXAR); absolute appearance and motion magnitudes carry signal |
| H3 | source-balanced sampling | equal source exposure | **neutral**: LOSO mean +0.005, worst −0.020 (2/4 sources) |
| H4 | object-level independent motion (LK tracks not following the camera: grid counts, inward lateral / approaching velocity) | physical ENTRY cue | **rejected** (LOSO mean −0.002, worst −0.019; ENTRY not better) |
| H5 | cross-frame-rate consistency (binned-time KL, native view = teacher, λ .3) + H1 | temporal-corruption robustness (cf. arXiv 2403.20254) | **invariance ↑, accuracy ↓**: stride 1/2/3 = 0.751 / 0.739 / 0.714 (drop −0.037 vs −0.067 for H1) but native −0.017; CV −0.007 |
| H6 | mask MMAU ENTRY labels (convention test) | MMAU ENTRY labelled much closer to COLLISION | **rejected** (LOSO mean −0.014, worst −0.047): MMAU ENTRY labels help |
| H7 | pre-collision truncation augmentation (collision not visible, COLLISION loss masked) | break an "ENTRY = just before COLLISION" shortcut | **neutral** (CV −0.002; LOSO mean +0.009 / worst −0.009); long-gap ENTRY unchanged |
| H8 | causal ENTRY head (ENTRY from positions ≤ t + L only; leak-free, verified) | same shortcut, architecturally | **rejected** (L = 4: CV −0.014, CI excludes 0) and **still predicts ENTRY late on long gaps** |
| H10 | geometry-head features (ego vs alternative drivable area, corridor occupancy) under LOSO | physical evasion cue | **no evasion transfer** (held-out evasion F1 unchanged ≈ 0.55–0.65); LOSO mean +0.009 |
| H11 | per-source class-balanced side / evasion loss weights | label-shift (base-rate) shortcut | **modest consistent gain**: LOSO mean +0.010, 3/4 sources, worst −0.003; side F1 ↑ on all 4 held-out sources; evasion still ≈ chance |
| H12 | feature-space augmentation (10 % token drop + 0.1·σ Gaussian noise on frozen DINO tokens) | appearance robustness | **mixed**: LOSO mean +0.010 (AIHUB +0.036, CCD +0.052, MMAU −0.022, NEXAR −0.027) |
| H13 | EMA weights (0.99) | flat minima generalise | **best cross-source single-model result**: LOSO mean +0.022, worst +0.006, **4/4 sources**; but seed-ensemble mean −0.007 (less diversity) |
| H9 | semi-supervised cross-frame-rate consistency on 1,046 unlabelled AIHUB / CCD / MMAU clips (+ H1); LOSO excludes the held-out source's clips | more domains, no labels | **not a generalization win**: CV −0.009; LOSO mean +0.000 / worst −0.026 (CCD +0.061, AIHUB +0.013, MMAU −0.046, NEXAR −0.026); robustness 0.752 / 0.760 / 0.714 (flatter, like H5) |

**ENTRY conclusion (H6–H8 + H4):** the gap dependence is not a learnable shortcut through the collision nor a missing input cue — even a
model that cannot see the collision places ENTRY late on long-gap clips. Long-gap ENTRY is annotated at the *onset* of a manoeuvre, before
it is visible in these frozen 7×10 features. ENTRY is limited by label definition vs. evidence, so model tricks will not fix it honestly.

## Practical consequences so far
* **The submitted v7 recipe is fragile to frame rate**: its 3-family ensemble scores 0.772 / 0.720 / 0.648 at 1/1, 1/2, 1/3 of the native
  frame rate (ENTRY 0.65 → 0.40). Stride augmentation is the fix that costs nothing at native rate (being applied to all v7 families).
* **Attributes**: side F1 transfers across sources (0.80–0.92 held-out); **evasion F1 does not** (0.51–0.70 held-out, near chance for CCD /
  MMAU / NEXAR; base rates differ: MMAU 72 % positive vs 45–57 %). Motion input slightly hurts evasion transfer. Tested next (H10): physically
  grounded geometry features (ego vs alternative drivable area, corridor occupancy) under LOSO.
* H11 in-domain CV: −0.008 (0/5 folds) — class balancing trades a little in-domain accuracy for cross-source transfer.
* **NEXAR context probe**: NEXAR-held-out models on NEXAR clips cropped to 1/2, 1/4, 1/8 of their length: score flat (0.61–0.63), ENTRY even lower.
  NEXAR's difficulty as an unseen source is appearance/domain + label conventions, **not** long context (so no context-length method).
* Pattern: generic regularisers (H3, H9, H11, H12) help held-out AIHUB / CCD but hurt NEXAR / MMAU; only residual motion (+0.111) and EMA
  (+0.006) improve the worst source.
* **H1 + H13 (stride aug + EMA)** is the best cross-source combination: LOSO mean +0.031, **worst (NEXAR) +0.027**, 3/4 sources (MMAU −0.001);
  complementary (EMA alone +0.022 / +0.006, stride aug alone +0.002 / −0.003). In-domain single-model CV +0.005 (3/5), lowest ENTRY
  catastrophic rate (0.134). **Trade-off**: EMA members are less diverse, so 3-seed ensembles gain less (robustness ensemble 0.752 / 0.735 /
  0.693 vs 0.768 / 0.753 / 0.701 for stride aug alone); EMA alone is not frame-rate robust (0.759 → 0.611 at 1/3 fps).

## Recipe decision (ensemble level)
| recipe (3 families × 3 seeds) | CV all / NEXAR (v5 decoding) | frame rate 1/1, 1/2, 1/3 (plain) | LOSO E4+E2 ensemble: single mean / worst | seed-ens mean / worst |
|---|---|---|---|---|
| v7 (no aug) | 0.784 / 0.717 | 0.772 / 0.720 / 0.648 | (E4 alone) 0.663 / 0.561 | 0.703 / 0.611 |
| **v8 = stride aug** | **0.787** / 0.707 | **0.775 / 0.759 / 0.718** | 0.672 / 0.564 | 0.711 / 0.606 |
| **v9 = stride aug + EMA** | 0.783 / 0.707 | 0.771 / 0.747 / 0.696 | **0.698 / 0.593** (4/4 sources ↑) | **0.720 / 0.613** |

Best single family: E2 + stride aug + EMA (LOSO mean +0.040, worst +0.035, 4/4 sources). Decision rule fixed before the result: cross-source
generalization first (the test set is stated to be unlike the training data). **v9 is the recommended next Stage 2**; v8 is the choice if
frame-rate shift is believed to be the bigger risk than domain shift. Both packages are built and verified (parity exact, smoke passes).

## Confirmation with 6 seeds (final) — recommendation corrected
| LOSO | single-model mean / worst (Δ vs E4) | ensemble mean / worst |
|---|---|---|
| E4 (6 seeds) | 0.659 / 0.543 | 0.699 / 0.610 |
| E4 + stride aug + EMA | +0.034 / +0.037, 4/4 sources | 0.715 / 0.585 |
| E2 + stride aug + EMA | **+0.042 / +0.064, 4/4 sources** | 0.722 / 0.618 |
| **E4+E2, stride aug, 12 members** | 0.673 / 0.570 | **0.721 / 0.617** |
| E4+E2, stride aug + EMA, 12 members | 0.697 / 0.593 | 0.712 / 0.600 |
| mixed EMA / non-EMA, 12 members | 0.685 / 0.578 | 0.714 / 0.607 |

EMA makes **each model** generalize better across sources (confirmed, stronger with 6 seeds), but it **reduces member diversity**: with 3 seeds
per family the EMA ensemble looked better (0.720 vs 0.711), with 6 seeds per family (12 members, the submission size) the non-EMA ensemble
is better (0.721 / 0.617 vs 0.712 / 0.600) and mixing does not recover both. **Recommendation corrected: v8 (stride aug) is the best
ensemble on every ensemble-level measure at submission size** (CV 0.787, frame-rate robustness 0.775 / 0.759 / 0.718, 12-member LOSO
0.721 / 0.617). v9 remains built as an alternative (better single models; use it if the ensemble were much smaller).

## ENTRY visual audit (`stage2_generalization_assets/entry_audit_long_gap.jpg`)
33 long-gap (> 1.5 s) clips where the E4 model predicts ENTRY > 0.3 s late; 6 shown (NEXAR / AIHUB / CCD) at GT−1 s, GT ENTRY, predicted ENTRY,
COLLISION. At **GT ENTRY the other vehicle is still distant and small, or just beginning to deviate (manoeuvre onset)**; at the **predicted
ENTRY (0.9–1.7 s later) it is large and close in the ego lane** — the first moment it is salient in 7×10 pooled global features.
**Future direction (not attempted in this session):** ENTRY needs an **object-centric representation** (per-vehicle boxes / tracks and their
lateral trajectories, e.g. from the detector/tracker that the earlier joint model used), not more global resolution (14×25 tokens overfit)
or global motion statistics. Evasion needs a **label audit** (conventions differ by source).

## Additional robustness / data findings
* **Image quality (4× lower resolution, DINO re-encoded)**: no degradation for any model (within ±0.004; v8-recipe ensemble 0.7754 → 0.7741,
  NEXAR 0.690 → 0.703). 7×10 pooled DINO tokens are low-frequency summaries (token cosine 0.997–0.999 vs clean), so the pipeline is robust
  to camera resolution by construction; no corruption augmentation needed (JPEG pass skipped).
* **Cross-domain data expansion** (the 670 unlabelled NEXAR clips with COLLISION-only supervision from public metadata, LOSO folds where NEXAR
  is a training source): AIHUB +0.013, CCD +0.040, MMAU −0.005 → extra data from one domain transfers modestly to other unseen domains.
* **H14 — label-free test-time attribute thresholds** (`stage2/generalization/attr_threshold.py`). Motivation: on held-out sources evasion is
  *not* at chance (seed-ensemble AUC AIHUB 0.78 / CCD 0.67 / MMAU 0.79 / NEXAR 0.62 for LOSO_E4; side AUC 0.93–0.99), but the mean
  predicted p_eva shifts with the source (CCD 0.70 vs prior 0.51; MMAU 0.54 vs prior 0.72). Tested thresholds computed on the unlabelled
  test set (median, train-prior quantile, logit centring). LOSO evasion macro-F1 mean: fixed 0.623–0.629 → best adaptive 0.634–0.640
  (+0.01 ⇒ +0.0015 score); side: fixed 0.5 is best (adaptive −0.002 … −0.03). **Rejected** — calibration is not the bottleneck; evasion
  is limited by ranking quality (AUC ≤ 0.8), i.e. by what the pooled features encode about free space, not by the decision rule.
* **H16 — frame-rate test-time augmentation** (`stage2/generalization/tta_eval.py`: average the ENTRY/COLLISION distributions of the
  native and 1/2-frame-rate views, projected by frame number). CV (3 seeds, clean input): E4 0.7485 → 0.7541 (+0.006), E4_sa 0.7683 → 0.7659
  (−0.002). **Rejected** — within noise, and once the model is trained with stride augmentation the views already agree; not worth 1.5× runtime.
* **ENTRY error direction on held-out sources** (LOSO_E4_sa, 3 seeds; FPS used offline only). ENTRY hit / late(>0.3 s) / early(<−0.3 s):
  AIHUB 0.31 / **0.61** / 0.09 (median +0.40 s), CCD 0.59 / 0.25 / 0.16, MMAU 0.78 / 0.16 / 0.05, NEXAR 0.34 / 0.22 / **0.45** (median −0.17 s).
  In-domain CV the same sources are nearly unbiased (AIHUB late 0.39, NEXAR early 0.23). COLLISION errors stay symmetric everywhere.
  Opposite biases on AIHUB vs NEXAR ⇒ the held-out ENTRY loss is mainly an **annotation-convention shift** (where each source puts
  "entry": AIHUB earlier, NEXAR later relative to what the other sources teach), not a missing visual cue. Median ENTRY→COLLISION gaps:
  AIHUB 1.27 s, CCD 1.10 s, NEXAR 1.11 s, MMAU 0.40 s. Implication for the leaderboard: ENTRY accuracy on an unseen source is bounded by
  how close its convention is to the training mix; keeping all sources in training (as v8 does) is the robust choice, and an explicit
  ENTRY definition audit across sources (e.g. re-annotating a small shared subset) is the highest-value next step for ENTRY.
* **H15 — event-anchored attribute pooling** (`--anchor-attr`: side/evasion pooled with (attention + detached ENTRY + detached COLLISION
  distributions)/3). LOSO vs E4_sa (same seeds 0–2): AIHUB +0.033, CCD +0.002, MMAU −0.011, NEXAR −0.011 (mean +0.003, worst −0.011).
  Side F1 drops on every source (−0.013 … −0.040), evasion rises on 3/4 (+0.019 … +0.028). **Rejected** — attribute evidence is not
  concentrated at the predicted event positions (side is better read from the whole approach), and the evasion gain is too small/inconsistent.
