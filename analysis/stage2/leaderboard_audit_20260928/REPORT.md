# Stage 2 leaderboard prediction reliability audit — 2026-09-28

## Conclusion

**The saved validation calculations reproduce, but no available metric is a reliable absolute leaderboard forecaster across new training recipes.** Reduced-rate validation is a useful screening signal. Adding crop ENTRY accuracy explains existing scores better, but its apparent reliability depends heavily on v8 and on having already observed an extras-trained submission. Treat it as a retrospective hypothesis, not a validated score estimator.

The existing v13 forecast of **0.589 is numerically reproducible**. Removing v8 from calibration changes it to **0.556**. These data do not establish that v13 matches or beats v8's actual **0.59293**.

## Data and reproducibility

- Ten distinct recorded Stage 2 leaderboard results were located: v1, v3, specialist, hybrid, v5, v6, v7, v8, v10, v12. Six have a common robustness evaluation: v5 .5314, v6 .5277, v7 .5464, v8 .59293, v10 .5590, v12 .5635. Scores are local user-reported records in `HANDOFF.md` and the review CSV; this audit did not query the live leaderboard.
- Main analysis uses the six comparable recipes, seven perturbation conditions, and the same **284 clips** after existing duplicate exclusions, from **349 unique predictions per recipe/condition**. Historical fixed-split metrics are evaluated separately; missing robustness measurements are not imputed.
- Recomputed all **113 common legacy features** from saved per-clip predictions. Maximum difference from `robustness_analysis_obl.json`: **0.0**. Verified unique IDs, consistent clip coverage, source identity, and event gap metadata. Saved snapshot SHA-256 is in `data_integrity.json`.
- Uses the repository's scorer: 35% ENTRY within 0.3 s, 35% COLLISION within 0.3 s, 15% side macro-F1, 15% evasion macro-F1. Scores pool predictions and recompute F1. Predictions remain in native frame coordinates, so using native FPS to score stride views is consistent.
- These are saved CV checkpoint ensemble proxies, seeds 0–1, plain decoding. Submitted packages can use more seeds, full-data refits, and different fusion. No fresh GPU inference, retraining, or independent checkpoint-to-prediction parity check was performed. Model-side weights were not changed.

Run from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /venv/main/bin/python analysis/stage2/leaderboard_audit_20260928/audit.py
```

Artifacts: `features.json`, `data_integrity.json`, `validation.json`, `heldout_predictions.csv`, `correlations.json`, `conditional_forecasts.json`, `early_fixed_split.json`, and `heldout_predictions.svg`.

## Actual masking tests

Every withheld leaderboard value is excluded from coefficient fitting. Leave-one-out trains on five submissions. Pair masking trains on four and tests both withheld submissions; all 15 possible pairs were evaluated. Errors are in leaderboard units (0.01 is one percentage point).

| Predictor | Leave-one-out MAE | Largest LOO error | All 15 pair masks MAE | Largest pair-mask error |
|---|---:|---:|---:|---:|
| Training-score mean baseline | .0220 | .0473 | .0221 | .0518 |
| Native CV, linear calibration | .0282 | .0631 | .0355 | .1967 |
| ⅓-rate CV, linear calibration | .0177 | .0396 | .0193 | .0639 |
| Native minus ⅓-rate drop | .0181 | .0387 | .0189 | .0519 |
| Existing 10-fps view | .0171 | .0424 | .0207 | .0451 |
| Corrected equal-clip 10-fps view | .0165 | .0395 | .0192 | .0417 |
| **⅓-rate + crop ENTRY, existing proposal** | **.0141** | .0394 | **.0213** | **.1160** |
| Same two features, standardized ridge α=1 | .0166 | .0451 | .0211 | .0914 |
| ⅓-rate + extras indicator, ridge α=1 | .0137 | .0465 | .0172 | .0516 |

The extras indicator is exploratory and encodes v10/v12's shared change. It is not an independently validated causal penalty; when no extras-trained result is in the training set, its effect cannot be learned. Ridge strength was fixed at 1, not selected on held-out scores.

**The two-metric model's advantage shrinks when masking two results.** It is nearly the mean baseline on average and has a much larger worst miss. Regularization does not resolve the small-sample problem.

Requested examples of masking results in the middle, using the existing two-metric model:

| Masked together | Training results | Prediction | Actual |
|---|---|---:|---:|
| v6 | v5, v8, v10, v12 | .5459 | .5277 |
| v7 | v5, v8, v10, v12 | .5447 | .5464 |
| v7 | v5, v6, v10, v12 | .5335 | .5464 |
| v8 | v5, v6, v10, v12 | .5444 | .5929 |

The first middle mask looks good; the second misses the most successful recipe by .0486. Reporting only the first would give a misleading impression.

## Can it predict the next kind of recipe?

Hold out **both v10 and v12**, train on v5–v8:

| Predictor | v10 predicted / actual | v12 predicted / actual |
|---|---|---|
| ⅓-rate alone | .6138 / .5590 | .6274 / .5635 |
| ⅓-rate + crop ENTRY | **.6249 / .5590** | **.6415 / .5635** |

The added crop feature does not anticipate the extras-family failure. A chronological replay with the two-metric model predicts v10 at .6249; after including v10 it predicts v12 at .5629 (actual .5635). This is encouraging for interpolation within a known family, but the family effect was already represented in calibration.

This replay is **retrospective**: the crop feature itself was proposed after earlier failures, including v10, were visible. Masking coefficients cannot erase that historical feature-selection information. A genuine prospective test requires freezing the full predictor before obtaining a new leaderboard result.

## Audit of existing claims

1. **Existing “LOO” is not prediction validation.** `robustness_analyze.py` removes one observation and recomputes correlation among the remaining points. A stable correlation does not establish accurate prediction of the omitted score. This audit actually predicts omitted scores.
2. **The 113-feature search is underpowered.** On all six known scores, the largest absolute correlation is ENTRY at ⅓-rate with a 50% crop: r=.8833, unadjusted exact permutation p=.0181. Accounting for the entire correlated feature search using a max-|r| permutation test gives **p=.4264**. No feature passes a .05 search-adjusted threshold. Permutation inference also assumes exchangeability, which related evolving submissions do not perfectly satisfy.
3. **Selecting the “best” feature is unstable.** Selecting a single feature by inner leave-one-out MAE, entirely inside each outer training fold, gives outer MAE **.0504**, worse than the mean baseline .0220. This evaluates that specific single-feature search procedure; it does not rule out every possible predictor.
4. **The crop coefficient is almost entirely supported by v8.** Full fit: crop coefficient .7349. Without v8: .0036. The rate coefficient simultaneously falls from 1.1262 to .2670. The apparent second relationship does not replicate across six independent families.
5. **10-fps weighting issue.** `ten_fps_view()` appends both AIHub native and half-rate predictions but one view for every other clip, doubling AIHub's contribution. The corrected audit averages two full-cohort scores, each with one view per clip, differing only in AIHub's chosen rate. LOO MAE improves slightly (.0171 → .0165); this does not solve calibration. Averaging two cohort scores is a defined view average, not an exact 10-fps resampling of 15-fps clips.
6. **Spearman ties.** The old `rankdata()` assigns sequential ranks to tied feature values; this audit uses average ranks. The original report's correlation rankings should be interpreted with that implementation caveat.
7. **Small slices and repeated views are not additional submissions.** A slice containing 16 clips viewed four times still contains 16 unique clips, and six leaderboard targets remain six targets. Clip-driver logistic AUCs in the existing code are in-sample explanatory fits, not held-out leaderboard prediction accuracy.
8. **NEXAR score proximity is not reliable prediction.** Across seven historical recipes with fixed-split measurements, fixed NEXAR has Pearson r=.0114 and LOO MAE .0648, versus .0551 for the mean baseline. Fixed-all MAE is .0473. These historical proxies have differing selection/refit histories and should not be pooled with the common robustness protocol.
9. **The 10-fps offset and ⅓-rate law did fail.** The old v7/v8 constant offset rests on two points; the later recorded v10/v12 failures invalidate universal use. Validation alone does not identify hidden test FPS, source mixture, or label conventions.
10. **Extras are associated with failure, not proven to cause it.** v10 and v12 both underperform v8 despite better rate scores, but they are related recipes with multiple changes. A learned extras penalty may capture the observed family, decoder/proxy differences, or other changes. It is not a transportable law.
11. **Crops are label-conditioned diagnostics.** Their windows are chosen to contain both ground-truth events and can be enlarged to fit the event gap. They are useful controlled stresses, not a sample from the hidden test distribution or an inference-time operation that can use hidden labels.
12. **Duplicate-clean rescoring has limits.** It excludes known problematic evaluation clips; it does not independently retrain the backbone or eliminate checkpoint/recipe selection on validation. These are conditional estimates, not fully nested model-development validation.

## Which metrics to retain

Use a small, fixed scorecard rather than searching for the next best correlation:

- **Rate transfer:** ⅓-rate score, native-to-⅓ drop, and the equal-clip 10-fps view. These show the most consistent directional usefulness across the existing changes, but do not capture every domain shift.
- **ENTRY context sensitivity:** ENTRY hit rate on 25% windows, native-to-crop loss, and long-gap ENTRY errors. These flag changes that improve rate handling while weakening context behavior. Crop accuracy is not independently established as an absolute-score predictor.
- **Source transfer:** matched per-source deltas and leave-one-source-out results using the same families, seeds, and decoder. Current LOSO tables use changing/two-family proxies and lack six matched submitted-recipe pairs, so this audit cannot reliably fit a LOSO-to-leaderboard mapping.
- **Training and deployment provenance:** extras/label conventions, backbone, folds, seed count, full-refit versus OOF, and decoder. These describe when calibration is extrapolating; they should trigger wider uncertainty, not an arbitrary guaranteed penalty.

For selection, inspect paired gains and losses on the same clips, known-duplicate groups, seeds, sources, and rates. A candidate that improves rate robustness while losing cropped/long-gap ENTRY should remain an uncertain tradeoff. More epochs, seeds, or validation rows do not create more independent leaderboard observations.

## Existing numerical forecast: reproducible, conditional only

For the current matched features, the full-data fit is:

`predicted S2 = -0.587960 + 1.126249 × third_rate_score + 0.734867 × crop25_ENTRY_accuracy`

This reproduces the previous v13 forecast:

| Unscored recipe in local records | Conditional point estimate | Range across fits omitting one scored submission |
|---|---:|---:|
| v9 | .5609 | .5503–.5634 |
| v11 | .5501 | .5473–.5633 |
| v13 | .5891 | **.5558–.5914** |
| v13 without crops | .5752 | .5591–.5771 |

These ranges describe sensitivity, **not confidence intervals**. Classical 95% prediction intervals are included in JSON for transparency but assume a fixed selected model, independent errors and stable domain; they omit model search, proxy mismatch and new-family shift. Neither those intervals nor “±.05” has established coverage. Observed errors reach .078 for the withheld extras family and .116 across pair masks for this model.

## Prospective verification protocol

Freeze this six-score dataset, feature definitions, and formula before seeing another score. For the next submissions record the forecast, all three baseline forecasts (mean, rate-only, equal-clip 10-fps), training-family changes, package identity, and actual Stage 2 score. Evaluate every new result, including losses, before changing coefficients. Keep protocol changes versioned and evaluate their future predictions separately.

Prefer tests that isolate one change while matching decoder and package construction. Accumulate results across genuinely different families; continue evaluating forward prediction error and holding entire families out. Until that evidence exists, use the scorecard to prioritize experiments and describe numerical forecasts as conditional estimates. There is currently no supported claim of a reliable .63–.65 prediction or a validated improvement over v8.
