## 1. Points lost per component (score = .35 ENTRY + .35 COLL + .15 side F1 + .15 evasion F1)

| arm | cond | score | lost: ENTRY | COLL | side | evasion |
|---|---|---|---|---|---|---|
| v8 | k1 | 0.764 | 0.126 | 0.055 | 0.010 | 0.044 |
| v8 | k2 | 0.750 | 0.137 | 0.062 | 0.011 | 0.041 |
| v8 | k3 | 0.710 | 0.161 | 0.070 | 0.013 | 0.045 |
| v8 | crop50 | 0.751 | 0.133 | 0.062 | 0.011 | 0.044 |
| v8 | crop25 | 0.731 | 0.166 | 0.048 | 0.011 | 0.044 |
| v8 | dacon | 0.754 | 0.127 | 0.063 | 0.012 | 0.045 |
| v8+D5H | k1 | 0.775 | 0.118 | 0.053 | 0.012 | 0.042 |
| v8+D5H | k2 | 0.764 | 0.123 | 0.062 | 0.012 | 0.039 |
| v8+D5H | k3 | 0.722 | 0.153 | 0.070 | 0.013 | 0.042 |
| v8+D5H | crop50 | 0.746 | 0.137 | 0.060 | 0.011 | 0.046 |
| v8+D5H | crop25 | 0.735 | 0.164 | 0.046 | 0.010 | 0.046 |
| v8+D5H | dacon | 0.766 | 0.117 | 0.060 | 0.012 | 0.045 |

## 2. By source (v8+D5H; v8 in brackets): ENTRY / COLL / side F1 / evasion F1

| source | n | cond | ENTRY | COLL | side F1 | evasion F1 |
|---|---|---|---|---|---|---|
| AIHUB | 58 | k1 | 0.431 (0.448) | 0.741 (0.776) | 0.948 (0.948) | 0.841 (0.843) |
| AIHUB | 58 | k3 | 0.466 (0.466) | 0.707 (0.672) | 0.895 (0.913) | 0.822 (0.773) |
| AIHUB | 58 | dacon | 0.500 (0.448) | 0.690 (0.690) | 0.913 (0.913) | 0.824 (0.843) |
| CCD | 69 | k1 | 0.754 (0.725) | 0.870 (0.870) | 0.922 (0.906) | 0.606 (0.590) |
| CCD | 69 | k3 | 0.377 (0.304) | 0.710 (0.710) | 0.906 (0.906) | 0.666 (0.619) |
| CCD | 69 | dacon | 0.754 (0.725) | 0.870 (0.870) | 0.922 (0.906) | 0.606 (0.590) |
| MMAU | 76 | k1 | 0.868 (0.855) | 0.974 (0.947) | 0.921 (0.947) | 0.724 (0.704) |
| MMAU | 76 | k3 | 0.895 (0.868) | 0.974 (0.974) | 0.934 (0.934) | 0.726 (0.726) |
| MMAU | 76 | dacon | 0.895 (0.829) | 0.961 (0.974) | 0.934 (0.934) | 0.744 (0.744) |
| NEXAR | 80 | k1 | 0.562 (0.512) | 0.800 (0.775) | 0.900 (0.925) | 0.684 (0.669) |
| NEXAR | 80 | k3 | 0.487 (0.475) | 0.787 (0.812) | 0.900 (0.899) | 0.669 (0.664) |
| NEXAR | 80 | dacon | 0.487 (0.512) | 0.762 (0.725) | 0.912 (0.924) | 0.639 (0.635) |

## 3. ENTRY error structure (v8+D5H)

* k1: misses 96/284 | late 0.83 / early 0.17 | median |err| 0.67s | near misses 0.3-0.6s 0.47 | >1s 0.29 | >2s 0.09 | pulled toward COLLISION 38 | ENTRY predicted at/after COLLISION 6
* k3: misses 124/284 | late 0.38 / early 0.62 | median |err| 0.60s | near misses 0.3-0.6s 0.56 | >1s 0.30 | >2s 0.10 | pulled toward COLLISION 17 | ENTRY predicted at/after COLLISION 7
* dacon: misses 95/284 | late 0.68 / early 0.32 | median |err| 0.57s | near misses 0.3-0.6s 0.54 | >1s 0.25 | >2s 0.08 | pulled toward COLLISION 30 | ENTRY predicted at/after COLLISION 7

| gap bin | n | ENTRY k1 | ENTRY k3 | ENTRY dacon | median signed err (misses, k1) | v8 ENTRY k1 / k3 / dacon |
|---|---|---|---|---|---|---|
| 0-0.5 s | 51 | 1.000 | 0.902 | 0.961 | +nans | 1.000/0.902/0.941 |
| 0.5-1 s | 80 | 0.800 | 0.588 | 0.725 | -0.33s | 0.750/0.512/0.675 |
| 1-1.5 s | 92 | 0.587 | 0.478 | 0.609 | +0.48s | 0.565/0.457/0.587 |
| 1.5-2.5 s | 46 | 0.348 | 0.435 | 0.478 | +0.87s | 0.370/0.457/0.457 |
| 2.5-99 s | 15 | 0.200 | 0.200 | 0.267 | +1.72s | 0.133/0.200/0.267 |

## 4. ENTRY vs position in the clip and clip length (native, v8+D5H)

* GT ENTRY at 0.00-0.30 of the clip: native n 26 ENTRY 0.577 | crop25 n 180 ENTRY 0.489
* GT ENTRY at 0.30-0.50 of the clip: native n 179 ENTRY 0.615 | crop25 n 51 ENTRY 0.588
* GT ENTRY at 0.50-0.70 of the clip: native n 63 ENTRY 0.810 | crop25 n 33 ENTRY 0.636
* GT ENTRY at 0.70-0.85 of the clip: native n 15 ENTRY 0.733 | crop25 n 18 ENTRY 0.611
* GT ENTRY at 0.85-1.01 of the clip: native n 1 ENTRY 1.000 | crop25 n 2 ENTRY 0.500
* clip length 0-150 frames: n 121 ENTRY 0.843 COLL 0.917
* clip length 150-300 frames: n 73 ENTRY 0.493 COLL 0.795
* clip length 300-750 frames: n 11 ENTRY 0.636 COLL 0.909
* clip length 750-1000000 frames: n 79 ENTRY 0.544 COLL 0.785

## 5. COLLISION errors (v8+D5H)

* k1: misses 43 | late 0.51 early 0.49 | median |err| 0.72s | 0.3-0.6s 0.40 | >1s 0.26 | by source AIHUB 15, CCD 9, MMAU 2, NEXAR 16
* k3: misses 57 | late 0.37 early 0.63 | median |err| 0.50s | 0.3-0.6s 0.63 | >1s 0.25 | by source AIHUB 17, CCD 20, MMAU 2, NEXAR 17
* dacon: misses 49 | late 0.24 early 0.76 | median |err| 0.58s | 0.3-0.6s 0.57 | >1s 0.22 | by source AIHUB 18, CCD 9, MMAU 3, NEXAR 19
* native: COLLISION misses that also miss ENTRY: 24/43

## 6. Attributes (native, v8+D5H)

* side: F1 0.922 | confusion (gt,pred) {(0, 0): 122, (1, 1): 140, (0, 1): 13, (1, 0): 9} | errors with ENTRY miss 0.50 vs overall ENTRY miss 0.34 | F1 k3 0.912 dacon 0.922
* evasion: F1 0.717 | confusion (gt,pred) {(1, 0): 31, (0, 0): 79, (1, 1): 127, (0, 1): 47} | errors with ENTRY miss 0.35 vs overall ENTRY miss 0.34 | F1 k3 0.723 dacon 0.702

## 7. Actor dependence (native ENTRY)

* actor ok: n 179 | v8+D5H 0.676 | v8 0.670
* actor wrong / missing: n 105 | v8+D5H 0.638 | v8 0.590
* ranker margin 0.01-0.89: v8+D5H 0.620 | v8 0.606
* ranker margin 0.89-1.88: v8+D5H 0.535 | v8 0.521
* ranker margin 1.88-3.44: v8+D5H 0.775 | v8 0.761
* ranker margin 3.44-8.01: v8+D5H 0.718 | v8 0.676

## 8. Seeds (per-seed ENTRY native / 1/3 / dacon, score native)

* v8: s0: E 0.641/0.493/0.627 S 0.765 | s1: E 0.592/0.503/0.606 S 0.739 | s2: E 0.620/0.525/0.627 S 0.762
* v8+D5H: s0: E 0.651/0.570/0.648 S 0.768 | s1: E 0.609/0.542/0.620 S 0.745 | s2: E 0.648/0.563/0.648 S 0.769

## 9. Hard core: ENTRY missed under every condition (v8+D5H)

* 45 clips (0.16) missed everywhere; 88 always hit
* hard core by source {'CCD': 7, 'NEXAR': 23, 'MMAU': 5, 'AIHUB': 10} (all: {'AIHUB': 58, 'CCD': 69, 'NEXAR': 80, 'MMAU': 76, 'CAUSALCRASH': 1})
* hard core median gap 1.73s vs all 1.00s | median signed err native +0.88s | late share 0.91 | actor ok 0.60 vs all 0.63 | COLL also missed (native) 0.29
* examples: ccd_000475 (gap 1.8s, err +0.9s), ccd_000664 (gap 2.2s, err +0.9s), nexar_00077 (gap 3.2s, err +2.3s), nexar_00225 (gap 1.4s, err -10.1s), mmau_009949 (gap 2.5s, err +1.7s), aihub_bb_1_100826_vehicle_233_35549 (gap 3.2s, err +1.6s), aihub_bb_1_141102_vehicle_126_317 (gap 1.3s, err +4.8s), aihub_bb_1_181004_vehicle_192_029 (gap 1.8s, err +0.9s)

## Deepening Exp 1-2 (3 seeds, same protocol)
* Exp 1, nested-CV attribute thresholds (stage2/actor/attr_threshold.py): negative. Per-fold evasion thresholds .50-.60, score -.001 to -.003
  in every condition for v8 and v8+D5H; side likewise. 0.5 is already near-optimal; evasion errors are not a calibration problem.
* Exp 2, head-level multi-rate TTA (entry_suite --tta W --tta-gates G; views positions[o::s], s = 2, 3): ungated w .25 helps native
  (+.011) and crops but hurts short sequences (DACON view -.004, w .5: -.013). Gated (views only with >= 64 sampled positions),
  v8+D5H w .25: native .775 -> .787 (ENTRY .662 -> .694), crop50 +.009, 1/2, 1/3, crop25 and DACON view unchanged. No effect on the
  50-frame test-like view or the LB-forecast inputs -> not packaged.

## Deepening Exp 3 — free-space evidence for evasion (stage2/actor/freespace.py, freespace_eval.py; nested CV, 284 clips)
* Per-frame road-head free space beside the ego lane (alternative drivable, curb, lane marking, near-field vehicles left / in / right),
  pooled over the PREDICTED ENTRY..COLLISION window, side-relative copies.
* Free-space only (logistic, nested C): evasion macro F1 .695 native / .698 DACON-like vs ensemble .717 / .702 — nearly as good from
  16 geometric numbers, i.e. the signal is real.
* Ensemble + free-space: .713 / .683 (score -.0006 / -.0028); probability average .703 / .707. Disagreements are symmetric (native:
  free-space right & ensemble wrong 40, the reverse 46; probability correlation .48), so no combination beats the ensemble.
  Per source, CCD improves (.606 -> .622) and NEXAR drops (.684 -> .668). Verdict: negative — evasion plateaus at ~.70-.72 with the
  evidence tested so far (the ensemble already encodes the free-space information, or the label is partly subjective).
