# Stage 2 goal campaign: validation > 0.8, NEXAR > 0.7 (2026-09-26)

Code: `stage2/aux_signal_experiments/` (train.py, goal_eval.py, event_select.py, nexar_expand.py, nexar_labels.py, extract_*.py).
Primary protocol (decided up front): **5-fold CV over all 349 labelled clips (80 NEXAR)**, pooled out-of-fold, v5 decoding
(camera-shift COLLISION fusion with β chosen on the other folds, native snap), equal-weight probability ensemble of all CV
seeds of each family. The 70-clip fixed split is reported beside it. Its 15 NEXAR clips make it too noisy to target
(the length-gated hybrid reached 0.80 / 0.71 there and lost on the leaderboard). Models stay FPS/duration/timestamp-blind.

## Result

| ensemble (equal weights, plain average) | CV all | CV NEXAR | CV >1000 | fixed all | fixed NEXAR |
|---|---|---|---|---|---|
| v5 as submitted (C0 + X_ema + M_motion + PH) | 0.769 | 0.670 | 0.660 | 0.764 | 0.619 |
| **E4 + E2 + XN4** (best) | **0.7995** | **0.721** | 0.709 | 0.784 | 0.689 |
| E4 + E2 + XN4 + XN4e0 | 0.7989 | 0.716 | 0.704 | 0.782 | **0.706** |
| E4 + E2 + XN4 + M1 + v5 families | 0.7908 | 0.709 | 0.692 | **0.788** | 0.697 |

**Goal status:** NEXAR > 0.7 is met on CV (0.721) and, for one ensemble, on the fixed split (0.706). Overall is **0.7995 on CV,
0.0005 short of 0.80**, and 0.784–0.788 on the fixed split. Honestly: **not met**, but at the boundary. Every further
change moved CV by about ±0.005, and the remaining options would be selecting ensemble compositions on the same CV clips.
The stretch goal (0.9 / 0.8) was not attempted seriously. It needs ENTRY accuracy of about 0.85 (now 0.69) and evasion F1 of
about 0.9 (now 0.75), and no method here moved ENTRY by more than +0.03.

Best ensemble breakdown (CV): ENTRY 0.688, COLLISION 0.874, side F1 0.940, evasion F1 0.746. Per source (earlier 7-family
ensemble): MMAU 0.91, CCD 0.77, AIHUB 0.74 (ENTRY 0.53), NEXAR 0.70–0.72 (ENTRY 0.55–0.58).

## Families (single-model CV; 8 seeds unless noted; control = matching recipe without the change)

| family | recipe | CV all | NEXAR | E acc | C acc | note |
|---|---|---|---|---|---|---|
| A0 (control) | NT loss, no motion | 0.699 | 0.551 | .571 | .745 | |
| **E4** | NT + global & residual motion input | **0.740** | 0.642 | .608 | .823 | +0.04 over A0; residual motion is the main gain |
| **E2** | E4 + boundary heads [h, Δh] | 0.743 | 0.640 | .608 | .826 | ≈ E4 alone, better in ensembles |
| **XN4** | E4 + 670 unlabelled NEXAR clips (COLLISION from public `time_of_event`, anchored ENTRY pseudo-labels, weight 0.5) | 0.743 | **0.657** | .592 | **.839** | >1000 COLLISION catastrophic 0.071 → **0.013** |
| XN2 (3 seeds) | E2 + NEXAR expansion | 0.742 | 0.676 | .593 | .835 | worse ENTRY; lowers the ensemble |
| XN4e0 (3 seeds) | expansion, COLLISION-only (ENTRY masked) | 0.739 | 0.654 | .605 | .823 | no standalone gain |
| M1 (3 seeds) | P2 loss + both motion | 0.730 | 0.615 | .589 | .813 | |

### Tried and rejected (3-seed CV, matched controls)
| idea | result |
|---|---|
| geometry-head spatial features (lane/vehicle grids, corridor occupancy; 83-d per frame) | +0.005 alone, −0.005 with motion; ENTRY unchanged |
| 14×25 high-resolution frozen tokens (vs 7×10) | **−0.026** (0/5 folds, overfits) |
| ENTRY loss weight 2 | −0.008 |
| sharper targets σ = 0.5 positions | −0.015 (0/5 folds) |
| event-window attribute stacker (logistic regression on geometry/motion windows) | side 0.937 → 0.896, evasion 0.716 → 0.667 |
| per-attribute family / threshold selection on other folds | −0.003 (unstable choices) |
| event-specific family subsets (ENTRY / COLLISION chosen on other folds) | +0.002 with 3-seed families, noisy with 8 |
| residual motion as a late decode cue on v5 | −0.003 |
| decode-time ENTRY/COLLISION bias shift | no systematic bias exists (median signed error ≈ 0) |

## NEXAR expansion details
* R2 `stage2/manifests/nexar_candidates.csv` holds Nexar's public `time_of_event` / `time_of_alert` for all 750 positives. On the
  80 labelled clips `time_of_event` matches our COLLISION within 0.3 s for 80 % (88 % after the +0.10–0.12 s median offset,
  estimated per fold on training clips only). `time_of_alert` is not our ENTRY (30 %).
* ENTRY pseudo-labels: per fold, that fold's E4+E2 teacher (trained without the fold) picks ENTRY within 2 s before the anchored
  COLLISION. On labelled NEXAR clips this rule scores 0.588 vs 0.525 unconstrained.
* The 670 videos are decoded directly (no JPEG frames). Features verified against the labelled-clip caches: DINO token cosine 0.996,
  global motion corr 0.998–1.000, residual corr 0.997. This needed a fix: `cv2.phaseCorrelate` modifies its inputs in place.
* Each epoch = all labelled clips + an equal random draw of expansion clips; attributes masked for expansion clips.

## Recommendation for the next submission (not built)
v6 Stage 2 = equal-weight ensemble of **E4 + E2 + XN4**, all seeds, retrained on all 349 labelled clips (+ the 670 expansion clips
for XN4, labels from the full-data teacher), v5 decoding with β = 0.5 (the CV choice in 4/5 folds). Inference needs residual
motion features (CPU; sparse Lucas–Kanade + RANSAC on 160×90 frames) in addition to the current inputs; no FPS.
Expected: +0.03 CV over v5 (0.769 → 0.80), with the largest gains on long clips / NEXAR COLLISION.
