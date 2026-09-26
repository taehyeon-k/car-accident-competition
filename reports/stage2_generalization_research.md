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
