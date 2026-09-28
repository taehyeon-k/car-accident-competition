# Stage 2 v13 — robustness report (2026-09-28)

**v13** = v8's three families (E4, E2 boundary head, XN4 with v8's 670 NEXAR extras), each trained with an equal stride mix
(1/1, 1/2, 1/3), unlabelled cross-frame-rate consistency (0.3) and window-crop augmentation (0.3). **No MM-AU/CCD extras** (the
ingredient of v10 / v12, both below v8 on the LB). Package `submit_v13_sbuC-E4E2XN4_V3_acc0p4_steer7.zip`: 12 members (4 seeds x 3),
full-data refit (stop epochs 9 / 8 / 8), runtime identical to v8, parity exact (logit diff 0.0, 70 clips), smoke passes;
Stage 3 = V3 at accel ±0.4 / steer 7° (best measured, v10 0.7475). **v13nc** = same without window crops (attribution).
All numbers duplicate-clean (284 validation clips), plain decoding unless noted.

## 1. Conditions (robustness profile, seeds 0-1, same protocol as reports/stage2_robustness_profile.md)
| recipe | LB | native | 1/2 | 1/3 | 1/4 | crop 0.5 | crop 0.25 | 1/3 + crop 0.5 | worst condition |
|---|---|---|---|---|---|---|---|---|---|
| v7 | 0.5464 | 0.751 | 0.712 | 0.638 | 0.612 | 0.739 | 0.741 | 0.705 | 0.612 |
| **v8** | **0.5929** | 0.749 | 0.734 | 0.699 | 0.701 | 0.735 | 0.729 | **0.750** | 0.699 |
| v10 | 0.5590 | 0.764 | 0.760 | 0.723 | 0.724 | 0.740 | 0.704 | 0.743 | 0.704 |
| v12 | 0.5635 | 0.755 | 0.759 | 0.740 | 0.745 | 0.742 | 0.701 | 0.738 | 0.701 |
| v13nc | — | 0.744 | 0.743 | 0.720 | 0.729 | 0.739 | 0.707 | 0.746 | 0.707 |
| **v13** | — | 0.752 | 0.745 | 0.707 | 0.713 | 0.731 | 0.724 | 0.742 | **0.707** |
DACON-like view (each clip at the stride nearest 10 fps, 50-frame window): v13 0.758 (ENTRY .641, COLLISION .835), v13nc 0.754.

## 2. ENTRY / COLLISION by task, gap, lead-in, position
| | v8 | v10 | v12 | v13nc | v13 |
|---|---|---|---|---|---|
| ENTRY native / 1/3 | .616 / .518 | .627 / .546 | .609 / .581 | .602 / .567 | .609 / .553 |
| COLLISION native / 1/3 | .838 / .796 | .866 / .827 | .852 / .827 | .838 / .792 | .845 / .782 |
| ENTRY on 25 % windows | **.532** | .458 | .433 | .479 | **.518** |
| ENTRY gap <0.5 / 0.5-1 / 1-1.5 s | .86 / .59 / .47 | .90 / .62 / .51 | .98 / .69 / .52 | .94 / .67 / .50 | .96 / .66 / .51 |
| ENTRY gap 1.5-2.5 / >2.5 s | **.42** / .18 | .44 / .17 | .33 / .20 | .36 / .13 | **.32** / .13 |
| ENTRY lead-in 2-4 s / >4 s | .50 / .46 | .55 / .49 | .57 / .52 | .58 / .48 | .57 / .48 |
| slope ENTRY vs log gap (more negative = worse on long gaps) | −.233 | −.246 | −.304 | −.289 | −.305 |
| slope ENTRY vs log fps (0 = frame-rate invariant) | .139 | .108 | .062 | .084 | .086 |
ENTRY scorecard (3 seeds; pooled native + 1/3): crop-0.25 ENTRY / position slope: v8 .525 / .58, v13 .514 / .54, v13nc .486 / .48;
long-gap (≥1.5 s) median ENTRY bias v8 +0.33 s, v13 +0.53 s, v13nc +0.57 s.

## 3. Sources (native / 1/3 rate)
| | AIHUB | CCD | MMAU | NEXAR | worst |
|---|---|---|---|---|---|
| v8 | .654 / .630 | .777 / .604 | .876 / .873 | .676 / .665 | .654 / .604 |
| v12 | .658 / .703 | .788 / .678 | .856 / .894 | .703 / .674 | .658 / .674 |
| v13 | .648 / .636 | .783 / .621 | .861 / .881 | .694 / .667 | .648 / .621 |

## 4. Held-out source (LOSO, E4 + E2 families, 3 seeds; source-clean: held-out source's consistency clips excluded)
| | native mean / worst | 1/2 mean / worst | 1/3 mean / worst | per source @1/3 (AIHUB / CCD / MMAU / NEXAR) |
|---|---|---|---|---|
| v8 families (E4_sa + E2_sa) | **.699** / **.604** | .668 / .591 | .630 / .531 | .558 / .531 / .810 / .621 |
| v13 families (E4_sbuC + E2_sbuC) | .693 / .564 | **.686** / **.581** | **.659** / **.592** | .615 / .596 / .834 / .592 |
v13 transfers better at reduced frame rate (+0.029 mean, +0.061 worst at 1/3) but slightly worse at native (NEXAR held out .564 vs .606).

## 5. In-domain CV with the submission decoder (goal_eval, 3 seeds, clean)
v13 **0.7675** (ENTRY .630, COLLISION .873), NEXAR **0.723** — v8 recipe 0.7743 (ENTRY .644, COLLISION .866), NEXAR 0.707.

## 6. LB forecast (fits on the 6 scored submissions; ±0.05 at least, see v10 miss)
* 1/3-rate score only: v13 0.565, v8 0.562 (fitted), v12 0.575 (actual 0.5635).
* 1/3-rate score + ENTRY on 25 % windows: **v13 0.589**, v8 0.590 (fitted; actual 0.5929), v12 0.563 (actual 0.5635), v13nc 0.575.

## Assessment
* v13 keeps the frame-rate gains of the later recipes (every stride ≥ v8; LOSO at 1/3 +0.03 mean / +0.06 worst) while window crops
  restore most of the truncation robustness the extras destroyed (ENTRY on 25 % windows .518 vs v8 .532, v12 .433).
* It is **not better than v8 on long ENTRY→COLLISION gaps** (1.5-2.5 s .32 vs .42; late bias +0.53 s vs +0.33 s) and is slightly weaker
  at native frame rate on held-out NEXAR. The equal stride mix + consistency appears to sharpen the short-gap behaviour even without
  the extras (v13nc shows the same).
* Forecast: a tie with v8 (0.589 vs 0.590). v13 is the robust, extras-free candidate; it is not expected to beat v8 by a clear margin.
