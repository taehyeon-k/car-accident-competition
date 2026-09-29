**Stage 2: findings, independent analysis, and next experiments**

**Update — 2026-09-27:** User reports v8 Stage 2 leaderboard **0.59293**, versus v7 **0.5464** (+0.04653). Use v8 as the scored baseline and retain stride augmentation in controls. v9 remains unscored. Read the [updated comprehensive review](STAGE2_EXPERIMENTS_DATASETS_LEADERBOARD_REVIEW_2026-09-27.md) before executing this historical brief; its corrected experiment comparisons, dataset caveats, and priorities supersede earlier status assumptions. The gain supports temporal robustness but does not establish hidden-test FPS.

Prepared 2026-09-26 from the preceding 24-hour experiment review. This is a handoff brief for an AI experiment agent, not a record of experiments already performed. Refresh run completion status before acting: training was still producing results during the review. Paths below are relative to `/workspace/car-accident` unless absolute.

**Mission and operating constraints**

Improve Stage 2 generalization to unseen video sources while preserving temporal-rate robustness and precise ENTRY/COLLISION localization. Prioritize explanations and reproducible improvements over crossing an arbitrary validation threshold.

Read the applicable workspace instructions first. Inspect current processes, queue state, free RAM, GPU use, and existing results before launching jobs. Reuse the existing memory-aware queue and preserve its minimum 10 GB free-RAM policy; coordinate with ongoing Stage 3 work. Do not stop unrelated jobs or overwrite existing checkpoints, caches, labels, reports, or packaged submissions. Put new experiments in a uniquely named campaign directory and record the code revision and effective configuration. Do not launch the entire plan at once: execute the decision gates below in order. Package a finalist only after evaluation; this brief does not request a leaderboard submission.

Preserve the existing FPS/duration/timestamp-blind model-input constraint. FPS may be used for offline evaluation. Sampling may use the established frame-index policy, but do not feed source identity, absolute clip duration, FPS, timestamp, or an inferred temporal prior into the model. Source labels may define splits and diagnostic tables. Avoid source-specific decoding rules and NEXAR position priors.

**Established findings: use these as baselines, not hypotheses to rediscover**

The official offline score is `0.35 * ENTRY@0.3s + 0.35 * COLLISION@0.3s + 0.15 * side_macro_F1 + 0.15 * evasion_macro_F1`. Pooled CV contains 349 clips, including 80 NEXAR; fixed validation contains only 70 clips, including 15 NEXAR. The four major sources are AIHUB, CCD, MMAU, and NEXAR; the pooled results also include one CAUSALCRASH clip. Preserve and explicitly document that clip's treatment.

| Finding | Evidence | Implication |
|---|---|---|
| Learned residual motion helps | M1 0.730 vs matched M0 0.707; paired bootstrap delta +0.023, CI [+0.007,+0.039] | Keep global + residual motion as a strong baseline; most improvement is COLLISION |
| NT loss + both motion signals is strong | E4 approximately 0.740 vs A0 approximately 0.699 | Use E4 for inexpensive controlled ablations |
| Boundary supervision helps modestly | BND2 screening +0.013; E2 on the stronger motion baseline approximately 0.743 vs E4 0.740 | Keep E2 as a complementary family; do not add gains from different controls |
| NEXAR expansion helps collision | XN4 approximately 0.743 overall / 0.657 NEXAR | Preserve the 670-clip expansion as a baseline; public collision timestamps are weak labels, not equivalent to human annotations |
| Phase supervision has a small uncertain gain | Approximately +0.005–0.007 with eight seeds; CIs include zero | Auxiliary-only phase is optional; do not expect the original large PH gain |
| Old PH gain was unreliable | Reproduction approximately 0.690 vs control 0.699 | Small-seed wins require confirmation |
| Phase decoding hurts | Transition/structured decoders often lose 0.03–0.08 | Preserve direct event decoding |
| Stride augmentation helps robustness | v8 one-third-rate score 0.718 vs v7 0.648 | Keep augmentation in the default research control |
| EMA depends on ensemble composition | Better individual members, but E4+E2 six-seed-per-family ensemble is worse with EMA | Judge actual ensembles; do not infer ensemble quality from member means |
| Risk progression did not survive confirmation | R3 approximately −0.005 at eight seeds | Low priority |
| Higher global token resolution failed | 14×25 tokens approximately −0.026 | Do not repeat this as the proposed object-centric experiment |

The best saved campaign ensemble reached **0.799453 overall / approximately 0.721 NEXAR**. Its components were ENTRY 0.6877, COLLISION 0.8739, side F1 0.9396, and evasion F1 0.7463. It did not exceed 0.80 overall. One additional correct event on 349 clips contributes approximately 0.0010, so merely crossing 0.80 is weak evidence of progress.

Matched three-seed-per-family comparison, using saved predictions:

| Recipe | Fused CV overall | Fused CV NEXAR | Plain native-rate score | Plain half-rate | Plain one-third-rate |
|---|---:|---:|---:|---:|---:|
| v7: E4 + E2 + XN4 | 0.783754 | 0.717119 | 0.772 | 0.720 | 0.648 |
| v8: same families + stride augmentation | **0.787419** | 0.707134 | **0.775** | **0.759** | **0.718** |
| v9: same families + stride augmentation + EMA | 0.782980 | 0.706984 | 0.771 | 0.747 | 0.696 |

Fused and plain columns are different protocols. The 0.799453 campaign result uses more members and must not be compared directly with 0.787419 to estimate an augmentation effect. The saved v8/v9 packages each contain three families × four seeds. The existing 12-member LOSO comparison instead uses E4+E2 × six seeds: same count, different composition.

The completed wider-ENTRY-target confirmation is newer than the living report's “queued” description:

| E4 + stride augmentation, sigma_ENTRY 1 → 2 | Baseline | Wider target |
|---|---:|---:|
| Six-seed LOSO mean individual score | 0.6679 | 0.6764 |
| Six-seed LOSO ensemble mean | 0.7035 | 0.7177 |
| Six-seed LOSO ensemble worst source | 0.6055 | 0.6186 |
| Matched three-seed CV mean individual score | **0.742148** | 0.729827 |
| Matched three-seed CV ENTRY accuracy | **0.604585** | 0.574021 |

This is a transfer/precision trade-off. Sigma_ENTRY=3 had only three LOSO seeds at review time; do not compare it against a six-seed control without matching seeds.

**Independent analysis: what is known and what remains uncertain**

My preferred built candidate is **v8**, with moderate confidence. Its robustness advantage is substantial; its small native-CV advantage is not decisive. Exact three-family package behavior under source shift remains insufficiently established. No v7–v9 leaderboard result was found in the reviewed records. The documented v5 Stage 2 leaderboard score was 0.5314, versus v3 0.4618; a length-gated hybrid with fixed-validation score 0.802 scored only 0.4154 on the leaderboard. Do not treat a small NEXAR validation subset as a hidden-test oracle.

ENTRY is the largest event-localization opportunity. Matching its accuracy to COLLISION in the best saved ensemble would add about 0.065 score, but this is headroom arithmetic, not an achievable-gain forecast. Two explanations remain unresolved: small, early vehicle motion may be poorly represented by frozen pooled features, and annotation conventions may differ across sources. Held-out AIHUB errors are predominantly late, while held-out NEXAR has substantial early error. These patterns are consistent with convention shift but do not prove it; visual domain shift could also contribute. The failed causal-head experiment does not rule out every architectural solution.

Evasion appears limited by both representation and possibly label conventions. Threshold adaptation gave little benefit. Generic regularizers and global motion statistics have not supplied a convincing solution. Object trajectories and explicit free-space evidence are promising hypotheses, not established wins.

**Common experimental protocol**

Before running an arm, write its hypothesis, control, changed variables, seed list, data eligibility, primary metric, and promotion rule to a machine-readable manifest. Freeze those choices before examining the arm's results. The numeric gates below are proposed engineering decisions, not statistical guarantees; if a gate is unsuitable, revise it before the relevant run and explain why.

Use the existing five CV folds for comparability. Report pooled out-of-fold metrics per seed, then mean and standard deviation across complete seeds. Do not average fold F1 scores as a substitute for pooled F1. Screen with three complete seeds; extend only finalists to six or eight matched seeds. A completed seed means every required fold/source has finished. Record failed and incomplete runs explicitly rather than silently dropping them.

Use LOSO for unseen-source evaluation, reporting all four sources, their unweighted mean, and worst-source score. Maintain a fixed stopping rule independent of the held-out source. Existing LOSO used epoch 9; keeping it aids comparability, but it is a development choice calibrated from prior work, not a perfectly untouched outer evaluation. If changing training duration or architecture, use training-source-only inner selection equally for control and candidate.

For final claims, recognize that existing CV/LOSO sets have been searched repeatedly. Use a locked new evaluation set if one is available, or nested selection within the available data. Never describe a newly chosen subset of previously inspected clips as an untouched test set. If independent validation is unavailable, state that the result remains development evidence.

Report paired deltas with clip-level bootstrap intervals; resample original-video groups if clips share a parent video. Stratify by source where appropriate, and recompute F1 and score inside each bootstrap replicate. Also report seed variation and per-source changes. These intervals do not correct for searching many hypotheses, and four held-out sources are not a large sample of possible domains.

Evaluate individual members and fixed-size ensembles separately. Keep family weights and seed counts explicit. Use matched decoding, with fusion beta frozen or selected only inside training/other-fold data. Always include native/half/one-third-rate evaluation for finalists. Keep FPS out of model inputs and target-width selection; seconds-based gap/error bins are offline diagnostics only.

**Priority 0 — repair and lock evaluation before more model search**

These changes are justified by code inspection, not speculative model ideas.

1. In `stage2/generalization/robust_eval.py`, predicted and GT frames remain native indices, but `num_available_frames` becomes the retained frame count. `stage2/long_context_v2_experiments/common.py:metrics` divides native-frame errors by that count, inflating normalized errors and catastrophic rates under stride. Store an explicit native-coordinate temporal span for normalization; keep retained-frame count separately for diagnostics. Define full-clip and crop normalization consistently. Do not globally change legacy metric semantics without versioning the output.
2. Replace `hash(sample_id)` crop seeding with a stable digest-derived seed, or otherwise guarantee process-independent seeds. Record the crop intervals.
3. Add focused regression checks: an unchanged native-frame prediction error has unchanged normalized error when only sampling density changes; official score stays unchanged by the diagnostic fix; independent processes generate identical crop intervals.
4. Recompute affected robustness diagnostics into new versioned files. The official score is expected to remain unchanged. Preserve historical files and explain any unexpected differences.
5. Label current crop tests “event-preserving crops”: windows use GT to contain both events. They do not measure arbitrary truncation or prove the absence of position priors.

Deliverable: a short evaluation audit with before/after examples, regression results, and exact metric definitions. Finish this before interpreting catastrophic-rate robustness.

**Priority 1 — close the package-comparison gap with existing predictions first**

Hypothesis: v8's ensemble-level advantage survives matched family composition and seed count.

Compare v7, v8, and v9 CV proxies at three families × four seeds, using identical folds and equal weights. Inventory existing completed runs first and train only missing required runs. Use the package's actual seed IDs where possible; never score full-data-refit checkpoints on clips included in their training. Add fixed-size disjoint draws only where enough completed seeds exist; avoid selecting a lucky draw for the headline.

For LOSO, first compare E4+E2 variants at four seeds each to remove the current six-versus-four discrepancy. A three-family source-clean LOSO extension is useful but needs an explicit policy for XN4: when NEXAR is held out, exclude all NEXAR expansion clips and any teacher trained on NEXAR. That arm cannot be the literal NEXAR-expanded XN4 package recipe. Label it as a source-clean analogue and report the changed training data. Do not claim it is an exact deployment evaluation.

Verify that expansion clips and any extracted segments do not duplicate validation clips or share original-video groups across splits. Teachers, pseudo-labels, timestamp offsets, and calibration must use only eligible training data. A fold-excluded teacher alone does not prevent leakage through a duplicate external clip.

Report paired deltas, source breakdowns, ENTRY/COLLISION errors, diversity/disagreement, robustness, and representative inference costs. Treat the current v8 preference as provisional until this comparison finishes. Retain v8 if alternatives have no repeatable advantage; do not pick a new candidate on a sub-0.005 isolated score change.

**Priority 2 — audit labels and visual evidence before choosing an ENTRY intervention**

Hypothesis: a material portion of ENTRY/evasion failure is explainable by inconsistent definitions, missing visual evidence, or both.

Construct a reproducible diagnostic panel of about 48–64 clips balanced across the four major sources, short/long ENTRY-to-COLLISION gaps, and correct/early/late model predictions. Include a random component, not only dramatic failures. Record the selection rule. Generate contact sheets or short clips around GT and predicted events, with identical presentation across sources.

Find the official ENTRY and evasion definitions in repository task materials. If no operational definition exists, document the ambiguity and request the missing definition through the normal collaboration channel; continue preparing evidence. Do not invent an authoritative definition. Describe proposed alternatives explicitly, such as manoeuvre onset versus physical ego-lane intrusion, only as audit categories.

For each clip record source, original labels, prediction errors, event visibility, likely participant, occlusion, a defensible event interval, uncertainty, and whether the existing label matches the shared definition. Use blinded review where feasible: obtain an initial visual judgment before displaying the old timestamp/model prediction. An AI-generated judgment is a hypothesis, not new ground truth. Human or independent review is needed before making label changes that drive final claims.

Keep original benchmark labels immutable. Produce a separate audit manifest with proposed corrections, rationale, provenance, and confidence. Do not use validation reannotations or validation-derived source offsets to train models that will be evaluated on those clips.

Decision gate: if disagreement clusters around definitions, prioritize uncertainty-aware supervision; if the participant is visible but absent from pooled features, prioritize object-centric features. If neither explanation is supported, do not force the conclusion—report the unresolved cases.

**Priority 3 — object-centric ENTRY representation (highest-potential new model experiment)**

Hypothesis: explicit vehicle appearance and trajectories preserve early intrusion evidence that global 7×10 pooled tokens miss.

First inspect reusable detector/tracker code and cached artifacts from the earlier joint model. Reuse these where feasible. If introducing a detector, record its weights, license, expected cost, and training-data provenance. Prototype extraction on a small balanced panel before processing all videos.

Proposed minimal input: per-vehicle ROI appearance plus normalized box center/size, confidence, visibility/missingness mask, and displacement across retained frames. Optionally express position relative to the ego-lane corridor. Keep track identity within clips; provide a safe masked fallback when detection/tracking fails. Do not choose the “important vehicle” using GT event times or actor annotations unavailable at inference. Do not derive velocities using FPS. Recompute stride-dependent track features consistently for robustness views.

Use a small temporal track encoder with masked attention pooling, fused with E4+stride features. Initially add a residual branch to ENTRY only to isolate the mechanism. Use temporal gradients as training inputs only if computed identically at inference. Avoid a large new architecture or end-to-end backbone fine-tuning in the first experiment.

| Arm | Purpose |
|---|---|
| O0: unchanged E4 + stride augmentation | Matched baseline |
| O1: same added capacity, global-feature input | Distinguish object evidence from extra parameters |
| O2: boxes/trajectories only | Test motion/geometry evidence |
| O3: trajectories + ROI appearance | Test object appearance beyond trajectories |

Measure detection coverage, ID switches on the audit panel, missing-track frequency, source-specific failures, and extraction/runtime cost. Evaluate ENTRY hit rate by offline gap bins, signed timing error, top-K event recall, and catastrophic errors alongside total score. A gain only on small-gap clips does not support the intended mechanism.

Screen O2 before O3 if ROI extraction is expensive. Promote a finalist to six matched seeds only if its three-seed result shows a plausible ENTRY gain (suggested ≥0.02 absolute), improved mean LOSO or broad per-source improvement, no >0.01 overall native-score regression, and no >0.01 one-third-rate regression. Confirm against O1 and test replacement in a fixed-size v8 ensemble. If extraction frequently loses the relevant participant, repair the input pipeline before escalating training.

**Priority 4 — ENTRY uncertainty without sacrificing precise localization**

Hypothesis: separating ambiguous onset evidence from precise event localization recovers some wider-target transfer benefit without its native-CV penalty.

Use the existing sigma_ENTRY=1 and sigma_ENTRY=2 E4+stride runs as controls. First finish only missing matched confirmation runs and inspect where widening helps/hurts. Do not start another broad sigma sweep.

Test one narrowly scoped arm: keep the main ENTRY target at sigma=1 and add an auxiliary broad ENTRY head at sigma=2 on the shared representation, with a predeclared modest weight such as 0.25. Keep COLLISION and attributes unchanged. Decode only the sharp head. Include a same-capacity auxiliary sharp-head control to distinguish broad supervision from another loss/head. This is a new hypothesis, not a known fix; gradients could simply blur the shared representation again.

If the audit provides independently justified training-label intervals, an alternative follow-up is interval likelihood: penalize low total ENTRY probability within the annotated interval, while retaining sharp targets for confident labels. Derive intervals only from training annotations with provenance. Do not infer them from validation residuals, source-specific bias fitting, or arbitrary per-source offsets. Run this follow-up only if such intervals actually exist.

Promotion gate: seek ≥0.01 LOSO ensemble-mean improvement versus sigma=1, no >0.005 native CV score loss, and no >0.01 worst-source or one-third-rate loss, confirmed with matched seeds. If it merely reproduces sigma=2's approximately −0.012 CV trade-off, retain it as a domain-transfer option rather than declaring a new default. Test ensemble complementarity at fixed member count; an enlarged ensemble is not a sufficient control.

**Priority 5 — object-conditioned free-space evidence for evasion**

Run only after the label audit and, preferably, reusable object features. Previous global geometry inputs and event-anchored pooling did not establish a reliable evasion-transfer gain; the new hypothesis must differ materially.

Hypothesis: free space relative to the threatening vehicle and possible left/right escape corridors is more useful than global occupancy averages.

Build a compact evasion-only residual head using per-side drivable/free-space occupancy, tracked obstruction positions, and temporal changes relative to the ego corridor. Use attention or predicted events to select relevant frames at inference. GT-centered windows may be reported as an explicitly labeled diagnostic upper bound, but never as deployable input. Verify that the official evasion definition is compatible with these features before extraction.

Controls: unchanged E4+stride; equal-capacity head on existing global features; prior global-geometry recipe; proposed object-conditioned features. Hold ENTRY/COLLISION and side outputs fixed for the first comparison to isolate evasion effects. Report macro-F1, AUC, calibration, confusion matrices, and per-source class prevalence. Keep threshold 0.5 initially.

Promotion gate: a suggested ≥0.03 mean held-out-source evasion macro-F1 improvement, improvement on at least three sources, and no large collapse on the fourth, with confirmation across seeds. AUC should support improved discrimination rather than only a lucky threshold effect. This corresponds to only about +0.0045 total score if all else stays fixed; account for extraction/runtime cost accordingly.

**Lower-priority work and stop rules**

Do not repeat risk-loss sweeps, monotonic phase terms, phase-based inference, source/length priors, higher-resolution global tokens, generic normalization, or threshold-only tuning without a specific new hypothesis. Frame-rate TTA and semi-supervised consistency already showed weak or adverse trade-offs. Mixed EMA/non-EMA ensembles have already been tried; revisit only with a materially different, preregistered fixed-size composition and a measured diversity rationale.

New blur/compression/occlusion tests can be useful after a finalist emerges. The successful low-resolution test does not cover those corruptions. Do not buy a large augmentation campaign before demonstrating an actual failure mode.

Stop an experiment family after a correctly implemented screen fails its stated mechanism and gate. Record negative findings. Do not keep changing losses, thresholds, seed subsets, or ensemble weights until the same development clips produce a win.

**Required deliverables from the experiment agent**

- A versioned experiment manifest with hypotheses, controls, seeds, folds, data exclusions, decoder settings, and predeclared decision gates.
- An evaluation repair note and focused regression checks for Priority 0.
- Machine-readable results plus a concise Markdown report separating completed, incomplete, failed, and rejected arms.
- Tables for pooled CV, every held-out source, native/half/one-third rate, all four task components, signed ENTRY errors, and runtime/memory on representative short and long clips.
- Separate tables for individual models and fixed-size ensembles, with paired uncertainty and exact membership.
- Label/visual audit artifacts with provenance and confidence; original labels preserved.
- A final recommendation tied to the intended distribution shift, including trade-offs and what remains unverified. If no arm passes, explicitly retain the baseline.

**Files to read and reuse**

| Purpose | Location |
|---|---|
| Full independent review | `reports/stage2_24h_review_2026-09-26.md` |
| Recomputed wider-target confirmation | `reports/stage2_24h_review_2026-09-26_confirmation.json` |
| Phase study and seed-noise analysis | `reports/stage2_phase_supervision_study.md` |
| Complementary signals | `reports/stage2_complementary_signals_interim.md` |
| Expansion and best campaign | `reports/stage2_goal_campaign.md` |
| Living generalization report | `reports/stage2_generalization_research.md` |
| Leaderboard history | `reports/stage2_leaderboard_vs_nexar.md` |
| Training/model/queue | `stage2/aux_signal_experiments/{train.py,model.py,queue.py,jobs.txt}` |
| CV prediction ensembles | `stage2/aux_signal_experiments/{ensemble_test.py,goal_eval.py}` |
| LOSO and robustness evaluation | `stage2/generalization/{loso_analyze.py,robust_eval.py}` |
| Shared metric implementation | `stage2/long_context_v2_experiments/common.py` |
| Generalization results | `stage2/generalization/results/` |
| Saved six-seed LOSO comparison | `stage2/generalization/results/loso_final_6seeds.json` |
| Wider targets | `stage2/generalization/results/{LOSO_E4_sa_se2,E4_sa_se2}/` |
| v8 exact member configuration | `/workspace/outputs/submit_v8_robust-E4E2XN4sa_V3_acc0p5_steer3/model/stage2/config.json` |
| v9 exact member configuration | `/workspace/outputs/submit_v9_gen-E4E2XN4saema_V3_acc0p5_steer3/model/stage2/config.json` |

Start with the evaluation audit and completed-run inventory. Then close the recipe-comparison gap and prepare the label audit. Choose the first new modeling experiment from the resulting evidence; object-centric ENTRY features are the leading representation hypothesis, while uncertainty-aware ENTRY supervision is the cheaper controlled alternative.
