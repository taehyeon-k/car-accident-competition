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
* 16:30 — **Backbone audit (P6-lite): frozen original DINOv3 ViT-B** (`extract_dense.py --arch vitb16`, `cache_dense_vitb`, 768-d) vs our
  geometry-adapted ViT-S, same recipe (E4, stride mix ⅓ each): ViT-B 0.733 / 0.735 / 0.709 / 0.704 vs ViT-S 0.747 / 0.727 / 0.702 / 0.709
  (native/½/⅓/¼). Equal on average; ViT-B better ENTRY at low rate (0.585 vs 0.553), ViT-S better NEXAR (0.677 vs 0.633 @⅓).
* 16:40 — **Backbone diversity helps ensembles** (`robust_eval` now supports per-member feature caches): E4_sb + E4_sbB (ViT-S + ViT-B)
  0.758 / 0.755 / 0.716 / 0.726 vs E4_sb + E4_sbu (two ViT-S variants) 0.750 / 0.746 / 0.710 / 0.723 — better at every rate.
  v10 mix + E4_sbB (12): 0.769 / 0.770 / 0.728 / 0.739 (mean 0.751 vs 0.749 for v10 alone).
* 16:45 — ViT-B features for the unlabelled pool (`unl_vitb.py`) being extracted so ViT-B families can use extras + consistency.
* 17:10 — E2_sbB (ViT-B, E2 family): 0.743 / 0.726 / 0.705 / 0.691 — weaker than the E4 ViT-B family at low rate.
* 17:15 — throughput limit: ViT-B + extras + consistency runs take ~11 min each (768-d features exceed the page cache). Queue cut
  to the final candidates: XSbU_XN4, XSbUB_E4 (ViT-B stacked), LOSO_XSbU_E4; dropped XSbUB_E2, E4_sbu6, E4_sbuc, E4_aux, LOSO_XSbU_E2.
* 18:37 — **XSbU ensemble** (E4 / E2 / XN4, each: stride mix ⅓ each + dedup stride-augmented MM-AU/CCD extras + unlabelled consistency 0.3):
  **0.758 / 0.758 / 0.736 / 0.735**, NEXAR @⅓ 0.692 — best at ⅓ rate (v10 mix 0.768 / 0.762 / 0.729 / 0.736, NEXAR @⅓ 0.678;
  v8 recipe ⅓ 0.710). Expected LB ≈ 0.5929 + 0.75 × (0.736 − 0.710) ≈ 0.612.
* 18:38 — v11 = XSbU full-data refits (4 seeds × 3 families; stop epochs 10 / 8 / 15) queued; packaging automated
  (`submission_tools/v11_stage2`, Stage 3 as v10: accel ±0.4, steer 7°).
* 20:34 — **XSbUB_E4** (XSbU recipe on frozen ViT-B features, incl. ViT-B pool features for extras + consistency): 0.735 / 0.766 /
  **0.740** / 0.728 (ENTRY @⅓ 0.602) — best single family at ⅓ rate.
* 20:41 — mixed-backbone ensembles: **XSbU (3 ViT-S families) + XSbUB_E4 (ViT-B) = 0.763 / 0.766 / 0.745 / 0.745**, NEXAR @⅓ 0.691
  (better than XSbU alone at every rate); v10 mix + XSbUB_E4 = 0.770 / 0.766 / 0.731 / 0.745.
* 20:43 — v12 = v11 + XSbUB_E4 refits (dual-backbone runtime: ViT-S + frozen ViT-B, per-member feature routing; functional test
  reproduces training-side predictions exactly; peak GPU 1.3 GB). Build automated.

## Summary (duplicate-clean CV, 3 seeds per family, plain decoding)
| candidate | native | ½ | ⅓ | ¼ | mean | NEXAR @⅓ | expected LB (0.5929 + 0.75·Δ⅓) |
|---|---|---|---|---|---|---|---|
| v8 recipe (LB 0.5929) | 0.764 | 0.750 | 0.710 | — | — | 0.685 | 0.593 |
| v10 (XC_E4 + XS_E2 + XS_XN4; submitted?) | 0.768 | 0.762 | 0.729 | 0.736 | 0.749 | 0.678 | ≈ 0.607 |
| v11 (XSbU, 3 ViT-S families) | 0.758 | 0.758 | 0.736 | 0.735 | 0.747 | 0.692 | ≈ 0.612 |
| **v12 (XSbU + XSbUB_E4, mixed backbones)** | 0.763 | 0.766 | **0.745** | **0.745** | **0.755** | 0.691 | **≈ 0.619** |

## Assessment against the goal (expected LB 0.70)
**Not reached.** The best candidate forecasts ≈ 0.62 (+0.026 over v8), about a quarter of the +0.107 needed; 0.70 would need ≈ 0.85
at ⅓ rate vs 0.745 achieved. What moved the robust metric: equal stride mix (+0.013 @⅓, single family), unlabelled
cross-frame-rate consistency (+0.013 on top; fixes the low-rate ENTRY collapse), metadata-anchored MM-AU/CCD extras (+0.01–0.02),
and backbone diversity (+0.009 in the ensemble). Single-family gains shrink at the ensemble level (errors overlap). What did not:
stride offsets, heavier ⅓ weighting, official-convention COLLISION labels, more seeds (+0.003), denser inference sampling, object
tracks (in-domain), stride-¼ training (dropped for time). The remaining gap is ENTRY at low frame rate (≈ 0.59 @⅓ vs 0.63 native,
and ≈ 0.45 on AIHUB/NEXAR-like footage) and cross-domain shift (LOSO @⅓ ≈ 0.64 — the closest proxy of the LB level). Closing it
needs new information (independently labelled clips in the test's style — CCD-like 10 fps footage — or a representation that sees
small, distant actors), not further regularisation of the same inputs.

## Deliverables
* `submit_v11_XSbU-E4E2XN4_V3_acc0p4_steer7.zip` — built, parity exact, smoke passes (R2 `submissions/2026-09-27/`).
* `submit_v12_XSbU-E4E2XN4+XSbUB-E4_V3_acc0p4_steer7.zip` — dual backbone (+343 MB, second backbone pass at inference); see log.
* Recommendation: submit v12 if the runtime/size budget allows, else v11; both keep v10's Stage 3 (accel ±0.4, steer 7°) so the
  Stage 2 change is isolated.
* 21:14 — **source-clean LOSO** (held-out source's labels, extras and consistency clips all excluded): LOSO_XSbU_E4 vs LOSO_E4_sa,
  native mean/worst 0.694/0.599 vs 0.683/0.582; **⅓ rate 0.663/0.561 vs 0.628/0.514 (+0.035 / +0.048)** — the robust recipe
  also transfers better under combined domain + frame-rate shift (the closest proxy of the LB level).
* 21:13 — v12 built: 16 members + both backbones (zip 556 MB), parity argmax agreement 1.0 (max logit diff 0.004, bf16 ViT-B),
  smoke passes; Stage 2 inference ≈ 1.3× v11.
* 22:15 — source-clean LOSO (mean / worst source), native and ⅓ rate: E4_sa 0.683/0.582, 0.628/0.514; XSbU_E4 0.694/0.599,
  0.663/0.561; XSbUB_E4 (ViT-B) 0.705/0.613, 0.664/0.592; **XSbU_E4 + XSbUB_E4 0.712/0.626, 0.674/0.597**.
* 22:00 — **LB reality check: v10 = 0.5590 (forecast 0.607; −0.034 vs v8).** The ⅓-rate rule failed out of sample; v10's
  extras (metadata-labelled MM-AU/CCD) are the prime suspect and are also in v11/v12, so the v11/v12 forecasts above are withdrawn.
  The LOSO gains above include the same extras and are not a forecast either.

## ENTRY gap / localization study (2026-09-27 late; no labelled extras; `stage2/generalization/entry_scorecard.py`)
Scorecard: ENTRY hit by ENTRY->COLLISION gap and lead-in (pooled native + ⅓ rate), crop-0.25 ENTRY + position slope, long-gap
(≥1.5 s) median ENTRY bias; duplicate-clean, 3-seed ensembles, plain decoding. Base = E4_sbu (stride mix ⅓ each + unlabelled consistency).
| arm | native | ⅓ | ENTRY by gap <0.5 / 0.5-1 / 1-1.5 / 1.5-2.5 / >2.5 s | crop25 ENTRY / slope | long-gap bias |
|---|---|---|---|---|---|---|
| E4_sa (v8 family) | 0.756 | 0.689 | .91 / .62 / .52 / .30 / .20 | .51 / .56 | +0.40 s |
| E4_sb | 0.747 | 0.702 | .96 / .66 / .52 / .29 / .23 | .47 / .50 | +0.40 s |
| E4_sbu | 0.740 | 0.715 | .96 / .65 / .51 / .35 / .20 | .47 / .48 | +0.40 s |
| + gap-balanced ENTRY loss (G) | 0.734 | 0.713 | .96 / .64 / .51 / .33 / .23 | .46 / .49 | +0.40 s |
| **+ window-crop aug 0.3 (C)** | 0.737 | 0.708 | .97 / .61 / .54 / .28 / .27 | **.53 / .60** | +0.40 s |
| + G + C | 0.741 | 0.703 | .98 / .62 / .54 / .24 / .13 | .54 / .49 | +0.50 s |
| + gap-prior head (P), plain / prior decoding | 0.720 / 0.728 | 0.694 / 0.685 | .97/.63/.49/.28/.13 ; .92/.63/.47/.32/.17 | .46/.42 ; .48/.51 | +0.53 / +0.60 s |
| earliest-near-max decoding α 0.5 / 0.3 (on E4_sbu) | 0.721 / 0.674 | 0.616 / 0.567 | .71/.42/.36/.34/.07 ; .49/.28/.23/.25/.03 | .64 / .66 | −0.28 / −0.68 s |
| frozen ViT-B (E4_sbB) / ViT-S+ViT-B | 0.733 / 0.758 | 0.709 / 0.716 | .95/.65/.52/.32/.20 ; .96/.68/.54/.32/.20 | .48/.51 ; .48/.54 | +0.40 s |
| object tracks (OT_sa) | 0.745 | 0.682 | .91 / .60 / .49 / .29 / .23 | .52 / .51 | +0.33 s |
* **Truncation / position prior: fixed by window-crop augmentation** (crop ENTRY .47 → .53, slope .48 → .60) at a small native cost.
* **Long-gap ENTRY is not fixed by any model-side change**: loss reweighting, crops, a learned gap prior, decoding rules, a larger
  backbone and object tracks all leave a median +0.3–0.5 s late bias on ≥1.5 s gaps. The model fires when the other vehicle becomes
  salient; the label marks the manoeuvre onset earlier. Labelled duplicate pairs (7, all short-gap) agree on ENTRY within 0.3 s
  (median 0.07 s), so annotation noise is not demonstrated — but there is no long-gap agreement data.
