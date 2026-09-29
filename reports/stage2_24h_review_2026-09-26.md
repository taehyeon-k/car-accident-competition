**Stage 2 — review of the last 24 hours**

Review date: 26 September 2026, UTC. The initial inspection window was approximately 25 September 23:36–26 September 23:36; newer completed confirmations encountered during the review are included explicitly below. Sources are the local experiment reports, saved predictions and metrics, evaluation code, and packaged-model configurations. No new training was launched.

**Assessment**

The work produced two convincing advances: learning from residual motion, and training with temporal-rate augmentation. It also exposed how easily small-seed validation results can mislead. My current preference among the built packages is **v8**, with moderate confidence: its frame-rate robustness is clearly stronger, while its advantage over v9 under source shift rests on a two-family proxy rather than the exact submitted three-family ensemble. No v7–v9 leaderboard score was found in the reviewed records.

The highest saved campaign CV score is **0.799453**, with NEXAR approximately **0.721**. This meets the NEXAR >0.70 target but does not meet overall >0.80. The 0.000547 shortfall is less than the score contribution of one extra correct event prediction out of 349 clips (0.35/349 ≈ 0.001003), so crossing that threshold alone would not establish a meaningful improvement.

**What the experiments established**

| Experiment | Evidence | Interpretation |
|---|---|---|
| Phase supervision | Eight-seed gains of approximately +0.005–0.007; reported confidence intervals include zero | A small possible auxiliary benefit, below the intended +0.01 success criterion |
| Old PH recipe | Reproduction approximately 0.690 vs control 0.699; earlier apparent advantage shrank with repeated seed draws | The original result was not reliable evidence for the recipe |
| Phase-based decoding | Transition/global structured decoding generally loses 0.03–0.08 | Keep direct event decoding; phase accuracy is not a substitute for event accuracy |
| Residual motion, M1 | 0.730 vs matched M0 0.707; paired bootstrap Δ +0.023, 95% CI [+0.007,+0.039] | Strongest complementary-input result; largely improves COLLISION |
| Boundary supervision with feature differences | BND2 approximately +0.013, CI [+0.002,+0.024] in screening | Useful representation supervision; a smaller contribution than motion |
| E4: NT loss + global/residual motion | Approximately 0.740 vs A0 0.699 | Strong foundation for the newer ensemble families |
| NEXAR expansion | XN4 approximately 0.743 overall / 0.657 NEXAR; long-clip COLLISION catastrophic rate reported as 0.071 → 0.013 vs E4 | Extra collision supervision is useful, especially for long NEXAR clips |
| Risk progression | R3 approximately −0.005 at eight seeds; no demonstrated ensemble complementarity | Do not prioritize further variants without new evidence |
| Lane intrusion | Approximately +0.010 in three-seed screening, CI includes zero; not specifically an ENTRY gain | Interesting but insufficiently confirmed |
| More global spatial tokens | 14×25 tokens approximately −0.026 | More resolution in this representation did not help |

These are different matched comparisons, not additive gains. For example, boundary supervision on top of the stronger motion baseline gives E2 roughly 0.743 vs E4 0.740, not another full +0.013.

The expansion uses 670 additional NEXAR clips, public collision timestamps, and fold-specific ENTRY pseudo-labels. It is weakly supervised expansion rather than fully human-labeled data. Public `time_of_alert` was found unsuitable as an ENTRY label. The saved campaign's best ensemble has ENTRY accuracy 0.6877, COLLISION accuracy 0.8739, side macro-F1 0.9396, and evasion macro-F1 0.7463.

**Which built recipe looks strongest?**

The following CV comparison uses three seeds per family and the same motion-fusion decoder. Robustness scores use plain decoding and must be compared within their own columns.

| Recipe | Fused CV overall | Fused CV NEXAR | Plain score: native rate | Half rate | One-third rate |
|---|---:|---:|---:|---:|---:|
| v7: E4 + E2 + XN4 | 0.784 | 0.717 | 0.772 | 0.720 | 0.648 |
| **v8: temporal-rate augmentation** | **0.787** | 0.707 | **0.775** | **0.759** | **0.718** |
| v9: augmentation + EMA | 0.783 | 0.707 | 0.771 | 0.747 | 0.696 |

At one-third frame rate, v8 gains approximately **0.070** over v7. Its native-to-one-third degradation is approximately 0.057 rather than 0.124: about a **54% reduction**. This is much stronger evidence than the +0.003 native CV difference. NEXAR native CV decreases by about 0.010, so augmentation is not a universal per-subset improvement.

The earlier 0.7995 campaign result used more CV members than this comparison. It should not be contrasted directly with 0.787 to claim that augmentation costs 0.0125.

EMA illustrates why individual-model and ensemble performance need separate evaluation. In the saved six-seed LOSO confirmation:

| E4 + E2, six seeds per family | Mean individual-model score across sources | Ensemble mean | Ensemble worst source |
|---|---:|---:|---:|
| Temporal-rate augmentation | 0.6727 | **0.7209** | **0.6170** |
| Temporal-rate augmentation + EMA | **0.6967** | 0.7122 | 0.6004 |

EMA improves individual members but loses when these members are averaged. Reduced diversity is a plausible explanation; the reversal alone does not prove the mechanism. E2+augmentation+EMA by itself has a strong six-seed ensemble mean of 0.7217, so “EMA never helps ensembles” would also be incorrect.

**Important qualification:** the LOSO comparison above is E4+E2 with six seeds each. The actual v8 and v9 packages contain **E4+E2+XN4 with four seeds each**. Both have 12 members, but they are different compositions. The existing report's phrase “at submission size” is accurate for count, not for exact recipe. Treat LOSO as supporting evidence for v8, not definitive end-to-end validation of its packaged ensemble.

**New result beyond the living report: wider ENTRY targets (H17)**

The living report still described confirmation as queued. I found and recomputed completed results: six LOSO seeds for sigma_ENTRY=2, and three CV seeds. Against the matching E4+augmentation baseline:

| Protocol | Baseline | Wider ENTRY target | Change |
|---|---:|---:|---:|
| Six-seed LOSO, mean individual score | 0.6679 | 0.6764 | +0.0085 |
| Six-seed LOSO, worst individual-score source | 0.5603 | 0.5664 | +0.0062 |
| Six-seed LOSO, ensemble mean | 0.7035 | 0.7177 | +0.0142 |
| Six-seed LOSO, ensemble worst source | 0.6055 | 0.6186 | +0.0131 |
| Matched three-seed CV, mean individual score | 0.742148 | 0.729827 | **−0.012321** |
| Matched three-seed CV, ENTRY accuracy | 0.604585 | 0.574021 | **−0.030564** |

The six-seed LOSO single-model improvement is positive on all four sources, but the initial three-seed estimate of mean +0.013 / worst +0.023 has shrunk to roughly +0.008 / +0.006. This is a **domain-transfer versus localization-precision trade-off**, not an unconditional improvement. It is worth retaining as a research candidate, but there is no evidence yet to replace all ENTRY targets in v8. Sigma_ENTRY=3 has only three LOSO seeds; compare it only with baseline seeds 0–2.

**What still limits Stage 2**

ENTRY is the largest event-localization bottleneck. In the best saved campaign ensemble, closing its gap to COLLISION accuracy would contribute approximately +0.065 to the total score, with everything else fixed. This is a way to quantify headroom, not a predicted achievable gain.

Two explanations remain plausible. Long-gap ENTRY often precedes a visually salient intrusion in the frozen 7×10 features. Meanwhile, held-out AIHUB tends to have late predictions and held-out NEXAR early predictions, consistent with source-dependent annotation conventions. The current experiments do not cleanly separate visual representation limits, annotation differences, and other domain differences. Claims that either cause has been conclusively established go beyond the evidence. A causal head failing to help only rules out that tested intervention; it does not rule out all architectural improvements.

Evasion also transfers poorly. Threshold adaptation gives only a small score gain, while source-specific discrimination remains limited. A shared-definition label audit and object/free-space representations are more promising next hypotheses than another threshold sweep. Low-resolution robustness was reassuring for the tested transformation; it does not establish robustness to blur, compression, weather, or occlusion.

**Reliability issues found during review**

1. **Downsampled normalized-error diagnostics have inconsistent units.** `generalization/robust_eval.py` preserves native predicted/GT frame numbers but sets `num_available_frames` to the retained frame count. `long_context_v2_experiments/common.py:metrics` divides native-frame error by that reduced count. This inflates normalized error and catastrophic rates under stride. The main score uses native-frame error divided by original FPS and is unaffected. Do not use stride catastrophic-rate comparisons until the denominator is corrected to the native temporal span.
2. **The crop test is limited and not necessarily reproducible across processes.** Its window is constrained using GT to contain both events, and its seed is derived from Python `hash(sample_id)`, which varies across processes unless `PYTHONHASHSEED` is fixed. The result supports tolerance to event-preserving crops; it does not prove absence of clip-position priors or robustness to arbitrary truncation.
3. **CV is used for selection as well as reporting.** Checkpoints use held-out-fold FPS-blind scores; many recipes and ensembles were compared on the same clips. This is useful development validation, but not an untouched final estimate. More seeds reduce training noise, not dataset uncertainty or repeated-selection bias. Existing bootstrap intervals do not correct for searching many candidates.
4. **Reports mix snapshots and protocols.** Distinguish fixed 70-clip validation, 349-clip pooled CV, LOSO, plain decoding, and fusion. The living report's earlier v9 recommendation is superseded by its later v8 recommendation. The goal report's proposed “v6” family recipe subsequently appears as the built v7; identify packages by their configuration, not prose version names alone.
5. **Leaderboard evidence remains sparse.** The reviewed records document v5 Stage 2 at 0.5314 vs v3 0.4618. The length-gated hybrid reached fixed validation 0.802 but leaderboard 0.4154. This strongly cautions against optimizing the small fixed NEXAR slice, but does not establish that hidden test data is literally NEXAR or forecast v8's leaderboard score.

**Recommended next decisions**

1. Keep **v8 as the preferred currently built candidate**, with v9 retained for comparison. Both have saved smoke results showing successful Stage 2 output with networking blocked. The one-clip v8 smoke took about 2.00 seconds and 557 MiB peak allocated GPU memory on RTX 5090; this is a smoke result, not a representative throughput benchmark.
2. Before claiming an exact package-level winner, evaluate matched four-seed, three-family proxies with identical decoding. Use a source-clean expansion policy in LOSO so the held-out source cannot enter through extra data. Quantify paired uncertainty and compare several disjoint seed draws where available.
3. Audit ENTRY and evasion definitions across sources using a small, consistently re-annotated subset. Include long-gap clips and both directions of source-specific ENTRY error. Use those findings to decide between label changes and object-centric features.
4. Treat wider ENTRY targets as a trade-off to investigate under a predeclared domain-shift objective. Do not promote them from the three-seed headline alone.
5. Correct the stride diagnostic units and crop reproducibility before making additional robustness claims. Deprioritize risk-loss sweeps, phase-based decoding, global-resolution increases, and threshold tuning unless new evidence motivates them.

**Evidence locations**

- [Phase study](stage2_phase_supervision_study.md)
- [Complementary signals](stage2_complementary_signals_interim.md)
- [Goal campaign](stage2_goal_campaign.md)
- [Generalization living report](stage2_generalization_research.md)
- [Leaderboard comparison](stage2_leaderboard_vs_nexar.md)
- [Best campaign metrics](../stage2/aux_signal_experiments/results/goal_E4_E2_XN4.json)
- [Six-seed LOSO confirmation](../stage2/generalization/results/loso_final_6seeds.json)
- [v8 robustness](../stage2/generalization/results/robust_sarecipe.json)
- [v9 robustness](../stage2/generalization/results/robust_saema_recipe.json)
- [Wider-target LOSO runs](../stage2/generalization/results/LOSO_E4_sa_se2/)
- [Wider-target CV runs](../stage2/generalization/results/E4_sa_se2/)
- [v8 configuration](/workspace/outputs/submit_v8_robust-E4E2XN4sa_V3_acc0p5_steer3/model/stage2/config.json)
- [v8 smoke result](/workspace/outputs/submit_v8_robust-E4E2XN4sa_V3_acc0p5_steer3_smoke.json)
