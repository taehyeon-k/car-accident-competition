n# Stage 2: robustness profile of the recent models and what it says about the leaderboard (2026-09-27, late)

Code: `stage2/generalization/robustness_profile.py` (per-clip predictions), `robustness_analyze.py` (slices, drivers, LB correlation).
Data: `stage2/generalization/results/robustness_profile.json` (8 recipes x 7 conditions x 349 clips), `robustness_profile_dacon.json`,
`robustness_analysis.json`, `robustness_position_prior.json`.

## Setup
* Recipes (CV proxies): v5, v6, v7, v8, v9, v10, v11, v12. LB: v5 0.5314, v6 0.5277, v7 0.5464, v8 0.5929, v10 0.5590;
  v9 / v11 / v12 are unscored.
* **Seeds 0-1 of every family for every recipe** (v5's LC families only have two seeds). Ensemble sizes are therefore matched across
  recipes, and absolute numbers sit ~0.003 below the 3-seed numbers in other reports.
* Metrics are on the 284 duplicate-clean clips (`clean_eval.py` exclusion), official 0.3 s rule. Decoding is plain (probability average +
  constrained anchors), not the submission's fusion.
* Conditions: frame rate 1/k (k = 1..4, motion recomputed) and duration crops. A crop is a random window of 50 % or 25 % of the clip that
  still contains ENTRY..COLLISION; 25 % was also run at 1/3 rate (k3 + crop 0.5).
* Slice tables pool the four frame-rate conditions, so each clip counts once per rate (n = 1,136).

## 1. Scores per condition
| recipe | LB | native | 1/2 | 1/3 | 1/4 | crop 0.5 | crop 0.25 | 1/3 + crop 0.5 |
|---|---|---|---|---|---|---|---|---|
| v5 | 0.5314 | 0.724 | 0.682 | 0.624 | 0.595 | 0.719 | 0.708 | 0.645 |
| v6 | 0.5277 | 0.756 | 0.700 | 0.622 | 0.616 | 0.742 | 0.737 | 0.699 |
| v7 | 0.5464 | 0.751 | 0.712 | 0.638 | 0.612 | 0.739 | 0.741 | 0.705 |
| v8 | **0.5929** | 0.749 | 0.734 | 0.699 | 0.701 | 0.736 | 0.729 | 0.750 |
| v9 | — | 0.754 | 0.730 | 0.687 | 0.687 | 0.747 | 0.727 | 0.742 |
| v10 | 0.5590 | **0.764** | 0.760 | 0.723 | 0.724 | 0.740 | **0.704** | 0.743 |
| v11 | — | 0.749 | 0.754 | 0.737 | 0.738 | 0.732 | **0.694** | 0.734 |
| v12 | — | 0.755 | **0.759** | **0.740** | **0.745** | 0.742 | **0.701** | 0.738 |

ENTRY hit rate (the task that moves most):
| recipe | native | 1/2 | 1/3 | 1/4 | crop 0.5 | crop 0.25 |
|---|---|---|---|---|---|---|
| v5 / v6 / v7 | .63 / .64 / .60 | .55 / .53 / .54 | .43 / .38 / .38 | .41 / .36 / .32 | .61 / .63 / .60 | .56 / .58 / .56 |
| v8 / v9 | .62 / .63 | .58 / .59 | .52 / .49 | .49 / .47 | .60 / .61 | .53 / .51 |
| v10 / v11 / v12 | .63 / .59 / .61 | .62 / .62 / .62 | .55 / .58 / .58 | .54 / .59 / .60 | .58 / .55 / .57 | **.46 / .42 / .43** |

COLLISION is much less sensitive (0.62-0.87 across all conditions; v10-v12 ~0.83-0.87). Side (0.87-0.95) and evasion (0.68-0.72) F1
barely move on any axis.

## 2. Robustness along each axis (hit rates pooled over the four frame rates)
**Frame rate (effective fps = native fps / k).** ENTRY hit rate:
| effective fps | <=5 | 6-8 | 10-12 | 15 | 30 |
|---|---|---|---|---|---|
| v7 | .24 | .43 | .54 | .63 | .64 |
| v8 | .40 | .53 | .65 | .61 | .67 |
| v10 | .46 | .59 | .64 | .61 | .68 |
| v12 | **.52** | **.61** | .67 | .60 | .66 |

Stride augmentation (v8) and the equal stride mix + consistency (v11/v12) removed most of the frame-rate cliff. Near 10 fps, all
v8+ models are equal (~.65). COLLISION at <=5 fps: v7 .68, v8 .76, v10-v12 .79-.81.

**ENTRY->COLLISION gap.** This is the largest remaining driver.
| gap | <0.5 s | 0.5-1 | 1-1.5 | 1.5-2.5 | >2.5 s |
|---|---|---|---|---|---|
| ENTRY v7 | .73 | .48 | .38 | .41 | .13 |
| ENTRY v8 | .86 | .59 | .47 | **.42** | .18 |
| ENTRY v10 | .90 | .62 | .51 | .44 | .17 |
| ENTRY v12 | **.98** | **.69** | .52 | **.33** | .20 |
| COLLISION v8 / v10 / v12 | .99 / .99 / 1.0 | .85 / .87 / .86 | .81 / .82 / .83 | .69 / .75 / .72 | .57 / .70 / .62 |

The extras-trained models (v10-v12) are sharper on short gaps and, for v11/v12, worse on 1.5-2.5 s gaps. No model finds ENTRY more
than 2.5 s before COLLISION (.13-.22).

**Duration / frame count.**
| | <4 s | 4-8 s | 8-20 s | >35 s (NEXAR) |
|---|---|---|---|---|
| ENTRY v7 / v8 / v12 | .82 / .92 / .96 | .39 / .52 / .62 | .45 / .47 / .50 | .38 / .46 / .48 |
| COLLISION v7 / v8 / v12 | .97 / .98 / .99 | .79 / .86 / .89 | .75 / .74 / .78 | .74 / .76 / .76 |

Frames the model receives (retained frames): ENTRY is worst below 25 frames (v7 .24, v8 .41, v12 .56) and best at 25-127 frames. It
declines again at >=128 frames (.41-.52): these are long clips, where one out of many candidate events must be picked.

**Context before ENTRY.** Clips with >2 s of lead-in are much harder for ENTRY (.36-.57) than clips with 1-2 s (.74-.86).

**Source (pooled rates), ENTRY / COLLISION.** AIHUB .38-.49 / .57-.77, CCD .31-.58 / .71-.87, MMAU .72-.85 / .87-.97,
NEXAR .39-.51 / .56-.81. MMAU is easy for every model (short gaps, clean labels).

## 3. What drives a miss, per clip (L2 logistic fit, coefficient per SD, + = more hits)
Features: log effective fps, log duration, log gap, log pre-ENTRY context, log post-COLLISION tail. Retained frames and relative
position are exact functions of these, so they are left out.

| ENTRY | eff. fps | duration | gap | pre-ENTRY | AUC |
|---|---|---|---|---|---|
| v7 | **+0.75** | -0.01 | -0.34 | -0.57 | 0.74 |
| v8 | +0.45 | -0.08 | -0.67 | -0.24 | 0.71 |
| v10 | +0.34 | -0.33 | -0.78 | -0.07 | 0.72 |
| v12 | +0.12 | -0.51 | **-1.29** | +0.20 | 0.77 |

**The dominant failure mode moved from frame rate (v7) to the ENTRY->COLLISION gap (v12).** Within MMAU the v12 gap coefficient is
-2.60 (v8 -0.79). The extra MM-AU/CCD clips mostly have short gaps (MMAU median 0.40 s), so their teacher/metadata ENTRY labels teach a
short-gap convention. Within NEXAR, long clips cost ENTRY for every model (duration -0.46 to -1.07).

COLLISION: the gap is the main driver for all models (-0.6 to -0.9), and duration is second. Frame rate matters little after v8.

## 4. Position prior under truncation (`robustness_position_prior.json`)
Slope of predicted vs true ENTRY position inside the window (1 = follows the evidence, lower = pulled to a typical position):
| | native | crop 0.5 | crop 0.25 | ENTRY hit when true ENTRY is in the first quarter of a 25 % window |
|---|---|---|---|---|
| v5 / v6 / v7 | .81 / .83 / .82 | .74 / .72 / .75 | .63 / .64 / .64 | .51 / .53 / .52 |
| v8 / v9 | .71 / .77 | .75 / .74 | .55 / .56 | .47 / .43 |
| v10 | .80 | .68 | .50 | .36 |
| v11 / v12 | .85 / .78 | .69 / .71 | **.42 / .41** | .30 / .33 |

In short windows, the models trained with extras predict ENTRY near the middle of the clip (mean predicted position .45-.47 vs true
.26). Truncation is the only condition in which v10 is clearly worse than v8.

## 5. Which robustness features track the leaderboard?
**113 recipe-level features** were tested against 5 LB points (exact permutation p over the 5! orderings; LOO = range of Pearson r
with one submission left out).
| feature | r | Spearman | LOO range | perm p |
|---|---|---|---|---|
| native CV | +0.30 | 0.00 | -0.28 .. +0.58 | 0.59 |
| 1/2 rate | +0.67 | +0.80 | +0.58 .. +0.91 | 0.20 |
| 1/3 rate | +0.78 | +0.90 | +0.71 .. +1.00 | 0.12 |
| 1/4 rate | +0.77 | +0.60 | +0.71 .. +0.95 | 0.12 |
| drop native -> 1/3 | -0.80 | -0.80 | -0.92 .. -0.71 | 0.09 |
| mean over rates | +0.74 | +0.80 | +0.66 .. +0.95 | 0.13 |
| 10 fps view | +0.63 | +0.80 | +0.52 .. +0.86 | 0.26 |
| DACON-like (10 fps, 50-frame window; pre-registered) | +0.57 | +0.60 | | |
| crop 0.25 score | -0.01 | -0.30 | | 0.97 |
| ENTRY @ crop 0.25 | -0.48 | -0.90 | | 0.40 |
| worst source @ 1/3 | +0.67 | +0.60 | | 0.21 |
| best of 113: ENTRY hit, clips with <1 s before ENTRY (16 clips x 4 rates) | +0.95 | +1.00 | +0.92 .. +0.99 | 0.017 |

* **Nothing survives the multiple comparison.** With 5 points the smallest achievable p is 0.017; by chance ~1.9 of 113 features
  would reach it, and exactly 1 did. Its slice is 16 clips, so it is best treated as noise.
* **Frame-rate robustness explains v5 -> v8 but not v10.** Fitted on v5-v8, LB ~ 0.83 x (1/3-rate score) + c predicts v10 = 0.614;
  the actual score is 0.559 (residual -0.055). Every frame-rate, duration, gap, source and 10 fps view scores v10 >= v8.
* **The pre-registered DACON-like test failed.** Each clip was put at the stride nearest 10 fps and cut to a 50-frame window containing
  both events (the only known test trait). It still ranks v10 (0.759) above v8 (0.741); v11 / v12 score 0.761.
* The only features where v10 is worse than v8 are truncation-related (ENTRY at crop 0.25: .458 vs .532; position slope). These rank
  v5-v7 above v8, so they do not track the LB alone. v11 / v12 are worse than v10 on exactly these features.

## 6. Interpretation
1. **Frame-rate robustness was a real, now largely solved problem.** It is the only axis whose improvement coincides with a large LB
   gain (v7 -> v8, +0.047), and v11/v12 extend it (ENTRY at <=5 fps .40 -> .52).
2. **v10's drop is not visible in any robustness view of our CV.** All frame-rate, duration, gap and DACON-like views score v10 above
   v8. What v10 adds is pseudo-labelled MM-AU/CCD extras. The measurable side effects are a stronger short-gap convention (steeper gap
   coefficient) and a stronger position prior under truncation. The most likely explanation is a **label-convention effect**: the
   extras' ENTRY/COLLISION conventions agree with our validation labels, which come from the same sources, but not with the hidden
   test's labels. In-source CV cannot detect this by construction. This is an inference, not a measurement.
3. **Consequences for the unscored packages.**
   * v9 (no extras) has a v8-like profile: slightly lower frame-rate robustness and the same truncation behaviour. Expected near v8,
     probably a little below.
   * v11 / v12 have the best frame-rate robustness, but carry v10's ingredient more strongly: gap coefficient -1.29 vs -0.78, position
     slope .41 vs .50, ENTRY at 1.5-2.5 s gaps .33 vs .44. Their LB is not forecastable from these data; the risk of repeating v10's
     drop is real.
4. **No offline metric is currently a trustworthy LB forecaster.** Forecasts from 1/3-rate CV should be treated as at least
   +/-0.05 (v10 miss).

## 7. Suggested next steps
* **Separate the two ingredients of v11/v12.** A v11-style ensemble *without* labelled extras is equal stride mix + unlabelled
  consistency, with E4_sbu-type members for all three families. It keeps the frame-rate gains (the part the LB rewarded) without the
  pseudo-labels (the suspected part of v10's loss). Submitting it against v8 is a clean A/B.
* Add **gap robustness** and **truncation / position robustness** (crop 0.25 ENTRY, position slope) to the scorecard next to the
  frame-rate views. A model that improves frame rate but worsens these is the v10 pattern.
* For an ENTRY label audit, the long-gap (>1.5 s) clips are where every model fails and where conventions differ most.
