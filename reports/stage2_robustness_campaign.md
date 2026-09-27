# Stage 2 robustness campaign — goal: expected leaderboard 0.70 (2026-09-27, ~9 h, living report)

## Target and yardsticks
* LB tracker (4 submissions v5–v8): plain CV at 1/3 of the native frame rate ranks all four correctly (r = 0.99;
  `reports/stage2_leaderboard_correlation.md`); working relation **ΔLB ≈ 0.75 × Δ(1/3-rate CV)**. v8 recipe (clean 1/3-rate CV 0.710) =
  LB 0.5929 ⇒ LB 0.70 needs clean 1/3-rate CV ≈ 0.85 (+0.14). All numbers below are **duplicate-clean** (284 validation clips; 65 clips
  with cross-fold or expansion near-duplicates excluded, `clean_eval.py`), 3-seed probability-averaged ensembles, plain decoding.
* Robustness scorecard: native / 1/2 / 1/3 / 1/4 rate; LOSO (held-out source) at native and 1/3 rate (`robust_eval.py --loso`).
  LOSO at 1/3 rate (E4_sa 0.639) is the closest proxy of the LB *level* found so far (v8 LB 0.593).

## Findings about the evaluation data
* The five DACON Stage 2 sample videos **are CCD clips 000001–000005** (identical frame statistics); 10 fps, 50 frames, no duplicated
  frames. Their `t_collision` equals CCD's official first-accident frame exactly; our manual COLLISION labels are +1 frame on the 3 we
  labelled. Our labels are also later than official metadata for NEXAR (median +0.10 s) and ~equal for MM-AU.
* Training on official-convention COLLISION (`official_labels.py`, arm E4_sbO) does **not** help even when scored against official
  labels (native 0.731 vs 0.744; 1/3 rate 0.706 vs 0.711; COLLISION unchanged, ENTRY −0.05). Keep our labels.
* Low-frame-rate failure mode (E4_sa, 1/3 rate): ENTRY hit 0.56 → 0.47 and errors turn **early** (early > 0.3 s: 11 % → 33 %);
  COLLISION 0.81 → 0.74. CCD at 1/3 rate (3.3 fps, 17 frames) is the worst case (ENTRY 0.75 → 0.32).

## Screen of existing arms at the LB-relevant rate (single family, 3 seeds)
| arm | native | 1/2 | 1/3 | mean | NEXAR @1/3 |
|---|---|---|---|---|---|
| E4_sa (control, v8 family) | 0.756 | 0.736 | 0.689 | 0.727 | 0.643 |
| XC_E4 (extras, COLLISION-only) | 0.730 | 0.749 | **0.712** | 0.730 | 0.630 |
| XU_mc (extras, native only) | 0.759 | 0.733 | 0.708 | 0.733 | 0.648 |
| E4_sa_unl03 (H9 unlabelled rate-consistency) | 0.740 | 0.750 | 0.703 | 0.731 | 0.654 |
| E4_sb (stride mix 1/3 each) | 0.747 | 0.727 | 0.702 | 0.725 | **0.677** |
| E4_sa_c03 (H5 labelled rate-consistency) | 0.736 | 0.728 | 0.700 | 0.721 | 0.650 |
| E4_sc (stride mix .2/.3/.5) | 0.735 | 0.740 | 0.695 | 0.723 | 0.628 |
| E4_sbo (E4_sb + random stride offsets) | 0.739 | 0.735 | 0.692 | 0.722 | 0.626 |
| E4_sa_ema, OT_sa, E2_sa, E4_sa_se2, E2_sa_ema, E4_sa_ab, XN4_sa_ema | | | 0.662–0.691 | | |
Consistency regularisers (H5/H9), rejected on native CV, are among the best at 1/3 rate. Stride 1/4 scores ≈ 1/3 (no further cliff).

## Log
(results appended as arms finish)
* 14:40 — **E4_sbu (stride mix 1/3 each + H9 unlabelled rate-consistency 0.3): 1/3 rate 0.715 / 1/4 0.715 / native 0.740** — best single
  family at low rate; ENTRY @1/3 0.581 (E4_sa 0.504, E4_sb 0.553): consistency directly fixes the low-rate ENTRY collapse.
* 14:40 — E4_sbO (official-convention COLLISION): no gain (see above). E4_sbo (stride offsets): 0.692 @1/3, no gain.
* 14:59 — queue re-prioritised: XSbU (stride mix + MM-AU/CCD extras + unlabelled consistency, 3 families) first, then E4_sbu6
  (consistency 0.6) and E4_sbuc (unlabelled + labelled consistency); dropped XSb/XCb/E4_s4/E4_sbc (superseded).
* 15:10 — ensemble mixes (3 seeds/family, clean): v10 mix XC_E4+XS_E2+XS_XN4 = 0.768 / 0.762 / **0.729** / 0.736 (native/½/⅓/¼);
  E4_sbu+XS_E2+XS_XN4 0.764 / 0.759 / 0.724 / 0.731; E4_sbu+XC_E4+XS_E2+XS_XN4 (12) 0.765 / 0.765 / 0.725 / 0.735. Single-family
  low-rate gains do not add at the ensemble level (errors overlap); ensembles plateau near 0.73 at ⅓ rate.
* 15:12 — more members: E4_sa+E2_sa 6 seeds vs 3: native +0.006, ⅓ rate +0.003 (small).
* 15:14 — denser inference sampling (every retained frame up to 320 positions, no retraining): ⅓ rate +0.004, native −0.007. Dropped.
* LOSO at ⅓ rate (new scorecard item): LOSO_E4_sa pooled 0.639 (worst CCD 0.51), LOSO_OT_sa 0.646 (+0.008).
* Source-clean LOSO labels for the extras (`unl_labels.py --loso`, LOSO_E4_sa teachers, held-out source's extras excluded) built;
  LOSO_XSbU_E4 / LOSO_XSbU_E2 queued.
