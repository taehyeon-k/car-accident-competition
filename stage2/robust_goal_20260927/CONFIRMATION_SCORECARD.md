# Grouped-fold confirmation scorecard

Complete matched folds: [0, 1, 2, 3, 4]. Seed: 0. All five folds complete.

| Rate | Base | Sharp auxiliary | Broad auxiliary | Broad minus base | Broad minus sharp |
|---|---:|---:|---:|---:|---:|
| k1 | 0.709922 | 0.741372 | 0.747518 | +0.037596 | +0.006146 |
| k2 | 0.701187 | 0.718385 | 0.715214 | +0.014027 | -0.003172 |
| k3 | 0.644718 | 0.680732 | 0.675702 | +0.030984 | -0.005031 |

## Source and sampling-rate tradeoffs

Composite-score deltas versus matched base. Small source slices are descriptive. A positive pooled score does not imply uniform robustness.

| Source | Clips | Sharp native | Sharp half | Sharp third | Broad native | Broad half | Broad third |
|---|---:|---:|---:|---:|---:|---:|---:|
| AIHUB | 88 | +0.074640 | +0.041460 | +0.089834 | +0.076091 | +0.015236 | +0.073360 |
| CAUSALCRASH | 1 | +0.350000 | +0.000000 | +0.000000 | +0.075000 | +0.075000 | -0.275000 |
| CCD | 83 | -0.003080 | +0.010138 | +0.039081 | +0.014716 | +0.007548 | +0.044084 |
| MMAU | 97 | +0.024949 | +0.031914 | +0.030109 | +0.027268 | +0.016596 | +0.027254 |
| NEXAR | 80 | +0.022207 | -0.018753 | -0.018684 | +0.028528 | +0.011869 | -0.023014 |

## Event-error tails

Catastrophic means absolute frame error greater than 10% of available clip length, following the repository metric. Lower is better; this is not a seconds-based threshold.

| Rate | Arm | ENTRY catastrophic | COLLISION catastrophic |
|---|---|---:|---:|
| k1 | base | 0.180516 | 0.068768 |
| k1 | aux_sharp | 0.146132 | 0.080229 |
| k1 | aux_broad | 0.171920 | 0.085960 |
| k2 | base | 0.174785 | 0.080229 |
| k2 | aux_sharp | 0.183381 | 0.097421 |
| k2 | aux_broad | 0.194842 | 0.083095 |
| k3 | base | 0.246418 | 0.123209 |
| k3 | aux_sharp | 0.263610 | 0.117479 |
| k3 | aux_broad | 0.240688 | 0.097421 |

Scores pool clip predictions and recompute macro-F1; they are not averages of fold scores. No candidate is promoted automatically. See confirmation_aggregate.json for source breakdowns, paired uncertainty, and per-fold scores.
