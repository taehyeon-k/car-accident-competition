# Stage 2: validation methodology that forecasts the leaderboard (2026-09-29)

Code: `stage2/generalization/lb_forecast.py` (evaluation of forecasters), `stage2/generalization/lb_predict.py` (forecast tool),
`stage2/generalization/test50_view.py` (rejected test-format view). Data: Stage-2-only leaderboard scores reported by the user.

## Evidence
Eight scored recipes, CV proxies at seeds 0-1 (284 duplicate-clean clips, plain decoding):

| recipe | LB | unlabelled pool | NEXAR extras | native | 1/2 | 1/3 | crop25 ENTRY |
|---|---|---|---|---|---|---|---|
| v5 | .5314 | no | no | .724 | .682 | .624 | .560 |
| v6 | .5277 | no | no | .756 | .700 | .622 | .581 |
| v7 | .5464 | no | yes | .751 | .712 | .638 | .556 |
| v8 | .5929 | no | yes | .749 | .734 | .699 | .532 |
| v8 + KDe | .5886 | no | yes | .753 | .738 | .703 | .549 |
| v10 | .5590 | yes | yes | .764 | .760 | .723 | .458 |
| v12 | .5635 | yes | yes | .755 | .759 | .740 | .433 |
| v13 | .5609 | yes | yes | .752 | .745 | .707 | .518 |

Held-out error of candidate forecasters (leave-one-out / leave-two-out MAE, leaderboard units):

| forecaster | all 8 recipes | 5 pool-free recipes |
|---|---|---|
| mean of LB scores | .020 / .021 | .033 / .033 |
| native CV | .027 / .030 | .074 / .066 |
| 1/2-rate CV | .018 / .019 | .017 / .019 |
| **1/3-rate CV** | .018 / .018 | **.005 / .007** (max .008 / .029) |
| 1/3 + crop25 ENTRY (old two-factor) | .010 / .011 | .004 / .013 (max .056) |
| 1/3 + unlabelled-pool flag | .008 / .009 | – |

Rejected: native CV, per-source slices, the CCD-native slice, and a test-format view (10 fps, 50-frame windows, COLLISION at frames
30-41 as in the DACON samples), which ranked v12 > v13 > v10 > v8kde > v8, the reverse of the leaderboard.

## Method (fixed from now on)
1. **Domain gate.** The forecast is valid only for recipes trained on the 349 labelled clips without the MM-AU/CCD/AIHUB unlabelled
   pool. Pool-trained recipes scored ~.03 below it. The labelled-only rule keeps every future recipe inside the gate. Recipes without
   XN4's NEXAR extras are a mild extrapolation (3 of the 5 calibration points contain them).
2. **Metric.** 1/3-frame-rate CV score: native frames[::3], adaptive re-sampling, motion recomputed; duplicate-clean 284 clips; 5-fold
   out-of-fold; plain decoding; entry_suite condition `k3`.
3. **Forecast.** Anchored on v8 at matched seeds: LB = .5929 + b x (S_third(candidate) - S_third(v8)), with b = .770 fitted on the
   five pool-free points.
4. **Uncertainty.** Paired bootstrap over clips of the S_third difference, typically SD .005-.011 in CV, i.e. .004-.009 LB. The
   leaderboard adds its own test-sampling noise: v8 + KDe was forecast .5951 +- .0038 and scored .5886.
5. **Decision rule.** A candidate is expected to beat v8 only if its forecast gain exceeds 2 SDs. Differences below ~.01 LB are not
   resolvable with 284 validation clips.
6. **Prospective check.** Before each submission, record the forecast and SD; afterwards add the result to the calibration. Stop
   trusting the method if two consecutive in-gate submissions miss by more than 2 SDs.

## Current candidates (3 seeds, anchored on v8 at 3 seeds)
| candidate | labelled-only? | forecast LB (+- 1 SD) | verdict |
|---|---|---|---|
| v8 + KDe (scored .5886) | no (XN4 extras) | .5951 +- .0038 | indistinguishable |
| v8 FGk = FGk + FGk_E2 + FGk_XN4 (packaged 09-29) | no (XN4 extras) | .5915 +- .0085 | indistinguishable |
| FG2 + FG2_E2 + XN4_sa | no | .5975 +- .0080 | indistinguishable |
| KD3 + E2_sa + XN4_sa | no | .5978 +- .0043 | indistinguishable |
| E4_sa + E2_sa (v8 without XN4) | **yes** | .5774 +- .0049 | **likely worse** |
| **FG2 + FG2_E2 + FGk** | **yes** | **.5923 +- .0083** | indistinguishable (v8-level) |
| FGk + FGk_E2 + FG2 + FG2_E2 | yes | .5908 +- .0088 | indistinguishable |

Implication: obeying the labelled-only rule by dropping XN4 costs ~.015; the flow-based families (FG2, FG2_E2, FGk) recover v8's
level without any extra data. Nothing tested so far is forecast to beat v8 by a resolvable margin.

## Attempt at a 0.63-forecast model (2026-09-29 07:40-07:52 KST)
Target: forecast .63 needs S_third about +.048 over v8 at matched seeds (~.758 vs .710 at 3 seeds).
* 1/3-heavy training mix (native / 1/2 / 1/3 = .25 / .25 / .5), 3 seeds: E4t .759 / .741 / .710 (ENTRY .644 / .539) vs E4_sa
  .756 / .736 / .689 -> +.021 at 1/3 with no native loss; FG2t (flow) .747 / .737 / .702 vs FG2 .766 / .737 / .714 -> worse.
* Labelled-only ensembles: E4t + FG2 + FGk .779 / .766 / .711 -> forecast .5933 +- .0068; E4t + FG2 + FG2_E2 .774 / .762 / .708 ->
  .5912 +- .0075. Best native / 1/2 labelled-only ensembles so far, but 1/3 saturates at ~.71 for every ensemble: family-level
  1/3 gains (E4t +.021, FG2 +.025) do not add up after averaging.
* Result: no candidate reaches a resolvable gain over v8; .63 is not reached.
