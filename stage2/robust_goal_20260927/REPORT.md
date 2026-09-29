# Stage 2 robustness goal: experiment log

**Current decision:** retain v8 (real leaderboard 0.59293). All five grouped folds and three matched CPU seeds are complete for the baseline versus sharp auxiliary ENTRY comparison. Sharp's mean gain is small and seed-unstable; an independent GPU seed3 replication is mixed and loses on NEXAR. v10 (0.5590) and v12 (0.5635) both scored below v8 despite stronger offline robustness views; both use MM-AU/CCD pseudo-label expansion. The clean ViT-B addition improves offline scores only slightly and regresses on third-rate NEXAR. No new method is validated for a 0.63–0.65 leaderboard expectation.

**Live evidence:** [matched-fold scorecard](CONFIRMATION_SCORECARD.md), `confirmation_status.json`, `confirmation_aggregate.json`. The scorecard currently requires all three arms to finish a fold before including it, checks split identity and native parity, recomputes pooled macro-F1, and reports known-duplicate-group bootstrap uncertainty. An incomplete cohort is explicitly labeled; single-seed results cannot establish seed robustness.


Goal: improve on user-reported v8 leaderboard 0.59293 toward 0.63–0.65. Required absolute gain is 0.03707–0.05707; current local experiments cannot establish that expected leaderboard gain.

## Resource isolation

Existing Stage 2 training/evaluation and Stage 3 training remain untouched. This campaign uses saved predictions or CPU head training/inference, nice 15, and no visible CUDA device. Grouped folds0/1 use one numerical thread; folds2–4 use two, matched across arms within each fold. The streaming robustness run yields below 10.5 GiB available RAM and clears feature memory maps per clip. Experiment artifacts stay in this directory; the dedicated supervisor wrapper/config are recorded below.

## Preregistered inference screen

See manifest.json. Four fixed nine-member recipes, three pooling rules, seeds 0–2 and all three leave-one-seed-out subsets. Deployment motion beta is frozen at 0.5 with native snap; no prior, no source-specific decoder, no fitted hyperparameters. Attributes retain arithmetic averaging.

Duplicate exclusions leave 284 of 349 clips. Exclusion addresses known contaminated evaluation clips; this is not independent validation or duplicate-group retraining. Existing checkpoints were selected on their folds. Confidence intervals do not correct repeated experimentation.

| Recipe / pooling | Clean score | Delta vs v8 | Paired 95% interval | Seed-pair deltas | Screen |
|---|---:|---:|---|---|---|
| v8/arithmetic | 0.774300 | +0.000000 | [+0.000000, +0.000000] | +0.0000, +0.0000, +0.0000 | Control |
| v8/geometric | 0.774300 | -0.000000 | [-0.007394, +0.007394] | +0.0012, +0.0037, +0.0037 | Gate not met |
| v8/median | 0.780462 | +0.006162 | [-0.004930, +0.018486] | +0.0062, +0.0074, +0.0049 | Advance to stress test |
| clean_expansion/arithmetic | 0.775415 | +0.001114 | [-0.014804, +0.016421] | +0.0048, -0.0042, +0.0044 | Gate not met |
| clean_expansion/geometric | 0.774182 | -0.000118 | [-0.016942, +0.015903] | +0.0085, -0.0005, +0.0032 | Gate not met |
| clean_expansion/median | 0.768020 | -0.006280 | [-0.022720, +0.010065] | -0.0001, -0.0005, +0.0007 | Gate not met |
| retain_base_E4/arithmetic | 0.782150 | +0.007849 | [-0.006922, +0.022763] | +0.0152, +0.0050, +0.0036 | Gate not met |
| retain_base_E4/geometric | 0.785847 | +0.011546 | [-0.003932, +0.026985] | +0.0189, +0.0026, +0.0061 | Gate not met |
| retain_base_E4/median | 0.777220 | +0.002920 | [-0.012936, +0.019743] | +0.0103, +0.0050, +0.0036 | Gate not met |
| collision_only_E4/arithmetic | 0.777672 | +0.003372 | [-0.011840, +0.019460] | +0.0136, -0.0118, +0.0014 | Gate not met |
| collision_only_E4/geometric | 0.782602 | +0.008301 | [-0.008034, +0.025196] | +0.0185, -0.0081, +0.0051 | Gate not met |
| collision_only_E4/median | 0.770278 | -0.004023 | [-0.020477, +0.013571] | +0.0049, -0.0130, +0.0027 | Gate not met |

Median pooling of the original v8 recipe is the only arm passing all predeclared screening gates. Its clean native gain is five additional correct event predictions (quantify components in JSON), without adding head/backbone passes. This remains a small selected result, not a demonstrated leaderboard improvement. Expansion-based candidates lose more than 0.01 on at least one main source, despite stronger aggregate scores in some cases. Do not discard those trade-offs by choosing only NEXAR.

## Completed robustness and source confirmation

Streaming CPU evaluation used the same nine OOF members on 284 duplicate-excluded clips. Plain ordered decoding isolates pooling from camera-motion fusion and snapping. CPU timing excludes backbone and video decode. The full run completed in about 121 seconds.

| View | Arithmetic | Median | Delta | 95% paired interval |
|---|---:|---:|---:|---|
| k1_crop0.0 | 0.764441 | 0.773068 | +0.008627 | [-0.003728, +0.020982] |
| k2_crop0.0 | 0.749563 | 0.752027 | +0.002465 | [-0.008658, +0.013556] |
| k3_crop0.0 | 0.710388 | 0.704226 | -0.006162 | [-0.018486, +0.006162] |
| k1_crop0.5 | 0.751094 | 0.763418 | +0.012324 | [-0.001232, +0.025880] |

Median loses at one-third rate despite gains at native/half rate and on event-preserving crops. Crops use GT only to ensure both events remain visible; this is not arbitrary truncation robustness.

Independent-protocol confirmation used existing head-level LOSO E4+E2 checkpoints, six seeds per family (12 members). This is a two-family proxy, not the deployed v8 three-family package. Shared-backbone exposure remains a caveat. Both methods use the same source exclusions.

| Held-out source | Plain median delta | Deployment-decoder median delta |
|---|---:|---:|
| AIHUB | -0.012069 | +0.000000 |
| CCD | -0.015217 | -0.015217 |
| MMAU | -0.013816 | -0.018421 |
| NEXAR | +0.000000 | -0.013125 |

**Decision: reject median pooling as the default.** It fails to preserve strong temporal-rate robustness and loses on several held-out sources. The native CV improvement is insufficient. Keep v8 arithmetic averaging. Geometric pooling and expansion combinations failed at least one initial screening gate; no replacement package was built.

## Cleaner training experiment started

Known duplicate components were moved together in a new isolated five-fold split: 349 clips, 338 groups, 10 moved clips, fold sizes 73/68/71/68/69. The 19 directed known cross-fold duplicate matches become zero. Only best matches and identical video paths are represented; this is not a complete parent-video audit. Existing folds/checkpoints remain unchanged. Do not reuse old-fold pseudo-labels for these new folds.

A sequential CPU-only first-fold screen is running from a local source snapshot:

1. Base E4 with stride probabilities 0.5/0.25/0.25.
2. Same model plus sharp auxiliary ENTRY, weight 0.25 and sigma 1.
3. Same model plus broad auxiliary ENTRY, weight 0.25 and sigma 2.

All use grouped fold0, seed0, the same 30-epoch ceiling/patience7 recipe, and identical decoded heads. The sharp auxiliary control separates extra-capacity/loss effects from uncertain-target supervision. No pseudo labels enter this first screen. A single fold is only feasibility evidence; promotion requires five grouped folds, matched seeds, source and rate stress, and deployment parity.

Resource controls: one CPU thread, nice15, CUDA hidden, only one training process from this campaign at a time; wait for 12.5 GiB available RAM before data preparation and 10.5 GiB before training batches. Logs and status are local. No current GPU job, shared queue, or shared training source was modified.

The goal remains active. **There is currently no evidence that a new candidate will reach 0.63–0.65 on the real leaderboard.** A score forecast must remain uncalibrated until external results support it. No submission has been made.

## Rate-error audit and visual review panel

On the completed nine-member v8 plain-decoder predictions, correctness is classified across native, half, and third rates. This audit identifies sensitivity; it does not imply an oracle can choose the correct rate at inference.

| Event | Correct at all rates | Changes correctness | Wrong at all rates | Native miss recovered at lower rate | Native miss recovered with context crop |
|---|---:|---:|---:|---:|---:|
| entry | 113 | 108 | 63 | 39 | 20 |
| collision | 207 | 50 | 27 | 18 | 9 |

ENTRY rate sensitivity is especially common on AIHUB (39/58 clips) and CCD (34/69); NEXAR has 30/80 persistent ENTRY misses. This motivates testing uncertain-target supervision and inspecting visual/annotation ambiguity, while preserving successes. It does not justify source-specific inference or using FPS as model input.

The [48-clip contact-sheet panel](REVIEW_PANEL.md) contains four deterministically selected clips per source × ENTRY pattern, including successful controls. All 288 requested cached frames were available. These sheets show annotations and predictions, so review is unblinded; they are diagnostic evidence, not corrected labels. Original labels are unchanged. `rate_error_audit.csv` contains signed errors and per-clip patterns.

Training was verified live at baseline epoch8 on this continuation; no auxiliary-arm result is available yet. The initial isolation-import errors were corrected before successful training; the active baseline has completed epochs with a CPU device and no GPU allocation.

## Persistent-failure inspection

Among 63 persistent ENTRY misses, 36 have closest-rate absolute error between 0.3 and 0.6 seconds; 15 exceed 1 second. For COLLISION, 14 of 27 are between 0.3 and 0.6 seconds and 3 exceed 1 second. This separates near-boundary timing failures from larger errors, without assigning new labels.

Four inspected persistent ENTRY examples include early predictions on AIHUB/CCD and late predictions on MM-AU/NEXAR. The sheets suggest onset-definition ambiguity and small/low-visibility actors as hypotheses, not verified causes. See `visual_review_notes.json`. No universal source offset or annotation correction is justified.

`evaluate_training.py` evaluates completed checkpoints on the new grouped fold at rates1/2/3 and requires native predictions to match the training artifact. It deliberately does not evaluate new-fold checkpoints on the old CV partitions.

## Balanced stride policy: matched completed-run screen

Reused completed E4_sb seeds0–2 from the existing queue without starting new training. Compared .34/.33/.33 rate probabilities against .5/.25/.25; all five folds, same 284 duplicate-excluded clips.

| Comparison | Plain delta | Frozen deployment delta |
|---|---:|---:|
| E4_sb | -0.008918 | -0.007686 |
| v8_balanced_E4 | -0.002401 | -0.001168 |

The standalone clean CCD score drops from 0.8042 to 0.7429 under plain decoding. The full-ensemble replacement reduces the aggregate loss but does not establish a better default. Keep existing stride probabilities pending stress evaluation; stronger augmentation is not automatically more robust. Results and intervals: `stride_screen_results.json`.

## First grouped-fold baseline completed

CPU baseline seed0/fold0 finished after 22 epochs, selecting epoch 15 using the unchanged FPS-blind checkpoint criterion. Native re-inference exactly matches all saved ENTRY, COLLISION, side, and evasion predictions.

| Rate | Overall score | ENTRY accuracy | COLLISION accuracy |
|---|---:|---:|---:|
| k1 | 0.730590 | 0.547945 | 0.876712 |
| k2 | 0.739663 | 0.561644 | 0.876712 |
| k3 | 0.694483 | 0.452055 | 0.863014 |

This is one model on 73 clips, not the nine-member v8 OOF score or a leaderboard estimate. Sharp auxiliary control has begun; broad auxiliary follows sequentially. No candidate comparison can be made until those arms finish.

## Balanced stride: completed reduced-rate confirmation

Fresh CPU inference, matched seeds0–2 and 284 duplicate-excluded OOF clips:

| Rate | Existing E4_sa | Balanced E4_sb | Delta | 95% paired interval |
|---|---:|---:|---:|---|
| k1 | 0.756275 | 0.748589 | -0.007686 | [-0.027190, +0.010514] |
| k2 | 0.735687 | 0.727135 | -0.008552 | [-0.029070, +0.011668] |
| k3 | 0.689072 | 0.702076 | +0.013004 | [-0.006449, +0.031936] |

Decision: no default replacement. Third-rate gain comes with native/half-rate loss; intervals include zero. The policy may be useful in a later diverse recipe, but further weight search on the same clips would be optimistic.

Fresh CPU inference changes one native ENTRY versus saved logits (nexar_00456, 561→543; annotated548). Frame grids are identical. Maximum per-member ENTRY-logit differences are about 0.00036–0.00084; fresh float32 and float64 pooling both pick543. This is consistent with numerical sensitivity near competing peaks; the arithmetic precision of pooling alone does not explain it. GPU/runtime parity remains required before deployment. This changes the aggregate by one event (0.001232), insufficient to reverse the overall conclusion. See `stride_parity_diagnosis.json`.

## Sharp auxiliary control completed

Same grouped fold0 and seed0, native parity exact. This is the extra-loss/capacity control for broad supervision, not yet a candidate validated across folds.

| Rate | Base | Sharp auxiliary | Delta |
|---|---:|---:|---:|
| k1 | 0.730590 | 0.753407 | +0.022817 |
| k2 | 0.739663 | 0.726859 | -0.012804 |
| k3 | 0.694483 | 0.726303 | +0.031820 |

Broad auxiliary remains in progress. Sharp control helps native/third rate but hurts half rate; wait for the matched broad comparison and multi-fold confirmation.

## Target-width audit and prepared follow-up

Existing sigma=1 refers to sampled positions. Across third-rate augmentation, median target standard deviation in native-frame units grows by 3.00× for CCD and MM-AU, 2.46× for AIHUB, and 1.69× for NEXAR ENTRY. This measured change is not proof of harm, but it motivates a controlled test. No FPS was used in the calculation. See `target_width_audit.json`.

Prepared target-transport arm: retain the native Gaussian target mass and assign it to the nearest sampled frame of each augmented view; preserve native-view training and the model/decoder. This differs from a global sharper-sigma sweep because it controls the effect of resampling rather than changing all targets. Mass conservation, unchanged-grid identity, and padded native-loss parity passed (maximum numerical difference <1e-6). See `transport_manifest.json` and `transport_checks.json`. It has not been launched and must wait until the current sequential screen finishes.

## Three-arm grouped-fold screen completed

All three CPU-trained models finished and passed exact native prediction parity. The broad auxiliary head is training-only; inference architecture/decoded outputs retain the original main heads.

| Rate | Base | Sharp auxiliary | Broad auxiliary | Broad minus base [95% group bootstrap] |
|---|---:|---:|---:|---|
| k1 | 0.730590 | 0.753407 | 0.749958 | +0.019369 [-0.027654, +0.064073] |
| k2 | 0.739663 | 0.726859 | 0.763604 | +0.023942 [-0.022885, +0.067811] |
| k3 | 0.694483 | 0.726303 | 0.719639 | +0.025156 [-0.022170, +0.076450] |

Broad supervision improves all three aggregate rates versus base in this first fold. Against the same-capacity sharp control, it loses 0.00345 native, gains 0.03674 half-rate, and loses 0.00666 third-rate. Thus this does not yet establish a general advantage of broader supervision over an auxiliary loss. Confirm the three arms on remaining grouped folds rather than promoting from one seed/fold. Full uncertainty details: `training_firstfold_comparison.json`.

The target-transport control has now started sequentially on CPU after native-loss, mixed-batch mass/padding, and finite-gradient checks passed on a real cached CCD example. The waiting check was stopped once to release RAM, then rerun successfully after the prior training completed. Existing GPU jobs and shared queues remain untouched.

## Target transport restart and managed confirmation

The first target-transport launch stopped before epoch1 because its helper incorrectly required strictly increasing sampled frames. The existing sampler repeats some frame positions. The helper now accepts nondecreasing grids, preserves the indexed Gaussian on an identical grid, and divides transported mass equally between duplicate positions on changed grids. All 2,094 targets (349 clips × three views × two events) passed normalization/identity checks; 120 views contain repeated positions. Shared sampling and existing training code were not changed. The restarted arm has progressed through training epochs.

Remaining-fold confirmation is now managed by supervisor service `stage2_robust_confirm`. It waits for the current campaign trainer, evaluates the transport checkpoint if produced, then runs 12 jobs (base/sharp/broad × grouped folds1–4, seed0) sequentially on CPU. It does not edit or control any other queue/service. The wrapper startup was corrected to follow the instance helper conventions and verified RUNNING. Live state: `confirmation_status.json`; protocol: `confirmation_manifest.json`; scripts: `confirm_folds.py` and `evaluate_training.py`. Training and each completed checkpoint evaluation use the existing RAM safeguards.

## Target transport: first grouped-fold result

The corrected arm completed, selecting epoch18, and passed exact native prediction parity.

| Rate | Base | Transport | Delta |
|---|---:|---:|---:|
| k1 | 0.730590 | 0.750222 | +0.019632 |
| k2 | 0.739663 | 0.767231 | +0.027569 |
| k3 | 0.694483 | 0.716794 | +0.022311 |

Aggregate gains at all three rates are promising but heterogeneous: native NEXAR loses 0.0465 while native MM-AU gains 0.0732; at half rate NEXAR gains 0.0737. These are small per-source slices of one fold and cannot establish transfer. Do not promote the method from this screen. Its remaining-fold confirmation can reuse the baseline runs after the current base/sharp/broad campaign, with no concurrent training added.

The supervisor advanced automatically to base/fold1/seed0 after evaluating transport. The remaining-fold campaign is live and uses RAM admission before each job.

## Cached-object feasibility audit

Existing RF-DETR/ByteTrack caches contain candidate road-user boxes on 344/349 annotated ENTRY frames and 267/349 COLLISION frames. Among top-four candidates, 609/1,175 ENTRY boxes and 324/760 COLLISION boxes have letterbox-adjusted area below one cell of the 7×10 pooled feature grid. This does not identify the true opponent or prove absence of useful appearance information; it warns that coarse ROI pooling and missing-detection handling must be checked before a large extraction/training campaign. Event timestamps are used only in this diagnostic, not for inference actor selection. See `roi_feasibility_audit.json`.

## Resource-safe CPU acceleration

A low-priority forward/backward benchmark on a worst-length synthetic batch measured median step times 0.4595s (1 thread), 0.2709s (2), and 0.2358s (4), with process peak RSS around 1.04GiB across settings. This is a microbenchmark under active load, not a full-run speed guarantee. The host exposes16 CPUs and showed low CPU pressure.

Future grouped folds2–4 use two threads consistently for all arms; folds0/1 remain at one thread to preserve within-fold matching. The active fold1 process was not changed or restarted. New configs record cpu_threads; evaluation uses the same setting. Low priority, one active campaign job, CPU-only execution, and RAM backpressure remain in place. Details: `cpu_thread_benchmark.json`.

## Fold1 sharp control completed

The second grouped fold selected sharp-control epoch6 and passed native prediction parity.

| Rate | Base | Sharp auxiliary | Delta |
|---|---:|---:|---:|
| k1 | 0.692305 | 0.713094 | +0.020789 |
| k2 | 0.685996 | 0.715319 | +0.029323 |
| k3 | 0.633088 | 0.644840 | +0.011751 |

This supports further testing of an auxiliary loss but is not evidence that broad targets are necessary. The broad arm is now training on fold1; the all-three-arm matched scorecard remains limited to fold0 until it finishes. No ensemble or leaderboard gain is inferred from these single-model results.

## Two matched grouped folds completed

Fold1 broad targets selected epoch22 and passed exact native prediction parity. Its native/half/third-rate scores are 0.758422/0.724251/0.642407. The supervisor has advanced to base/fold2, with no additional concurrent campaign trainer.

Pooling the 141 held-out predictions from folds0 and1 and recomputing macro-F1 gives:

| Rate | Base | Sharp auxiliary | Broad auxiliary | Broad minus base [95% paired group bootstrap] |
|---|---:|---:|---:|---|
| Native | 0.712249 | 0.734042 | 0.754198 | +0.041949 [+0.005249, +0.079112] |
| Half | 0.713807 | 0.721340 | 0.744952 | +0.031145 [+0.001176, +0.064680] |
| Third | 0.664881 | 0.687085 | 0.682920 | +0.018039 [-0.020022, +0.058816] |

The intervals use 2,000 paired resamples of 135 known-duplicate groups. They are descriptive, conditional on the selected checkpoints and this partial cohort, with no search/sequential-testing correction. They do not establish a full-CV or leaderboard gain.

Broad targets improve native and half-rate scores in all four represented sources, but **third-rate NEXAR loses 0.073014 versus base** (32 clips), despite gains in AIHUB/CCD/MMAU. Broad also trails sharp by 0.004165 on pooled third-rate performance. This prevents a claim of uniform robustness. Continue the already scheduled five-fold comparison; if this NEXAR regression persists, inspect ENTRY and COLLISION errors separately before selecting any ensemble or augmentation remedy. Do not tune a source-specific rule on this small slice.

Decision remains to retain v8 for deployment. See `CONFIRMATION_SCORECARD.md` and `confirmation_aggregate.json` for the updated complete-cohort evidence. Additional seeds and source-transfer confirmation are still required before packaging a replacement.

## Nexar third-rate regression attribution

`audit_confirmation_errors.py` decomposes the matched two-fold scores and saves clip-level hit transitions in `confirmation_error_audit.json`. For broad versus base on the 32 Nexar clips, ENTRY contributes -0.065625 to the score difference, COLLISION -0.021875, side +0.015158, and evasion -0.000672. ENTRY has seven regressions versus one recovery; COLLISION five versus three. Thus improved attributes partially mask degraded localization.

Four of the seven ENTRY regressions miss by 0.34–0.54s, two by 0.77–0.94s, and one by 3.90s. Five are early and two late, so a universal time offset is not justified. Collision regressions include errors of 4.05s and 18.30s, indicating that tolerance-boundary jitter alone cannot explain the loss. These errors warrant checking event selection as well as peak precision if they recur in the complete cohort. No thresholds, labels, decoder offsets, or training recipes were changed in response to this partial validation slice.

The audit also warns against calling native Nexar robust from its +0.008159 composite-score gain: ENTRY hit rate is unchanged, COLLISION hit rate falls by 1/32, and the repository-defined catastrophic normalized ENTRY error rate rises from 5/32 to 9/32. Track event-level tails alongside the composite score in the remaining-fold and multi-seed decision.

## Fold2 sharp control completed

The sharp auxiliary model selected epoch14 and passed exact native prediction parity. Native/half/third scores are 0.738796/0.709348/0.651469, compared with baseline 0.723846/0.681130/0.592876: gains of approximately +0.014950/+0.028218/+0.058593. This is a third fold with improved native performance for the sharp auxiliary control, further motivating the same-capacity comparison before attributing gains specifically to broad targets.

The trainer yielded during transient RAM pressure and resumed without a restart. The managed sequence then advanced to broad/fold2. The matched three-arm scorecard still includes only folds0/1 until that evaluation completes; no incomplete-fold pooled result is used for promotion.

## Three matched grouped folds completed

Broad/fold2 selected epoch8, passed native parity, and scored 0.714410/0.685443/0.642567 at native/half/third rates. Native performance falls below base (0.723846) and sharp (0.738796), so the first two folds' broad-target gains are not uniformly replicated.

Pooled predictions over 212 clips now give:

| Rate | Base | Sharp auxiliary | Broad auxiliary | Broad minus base [95% paired group bootstrap] |
|---|---:|---:|---:|---|
| Native | 0.716308 | 0.735810 | 0.741581 | +0.025273 [-0.004762, +0.056051] |
| Half | 0.703166 | 0.717504 | 0.725739 | +0.022572 [-0.004757, +0.051476] |
| Third | 0.641173 | 0.675577 | 0.670004 | +0.028831 [-0.005830, +0.064012] |

All broad-versus-base intervals now include zero. Broad-versus-sharp differences are +0.005771/+0.008234/-0.005573, also uncertain. Earlier two-fold intervals excluding zero were interim observations, not confirmation.

Across 48 Nexar clips, broad-minus-base is -0.004696 native, +0.004812 half-rate, and **-0.079003 third-rate**. Third-rate ENTRY now has ten regressions and two recoveries; COLLISION seven regressions and three recoveries. The source-specific timing weakness has persisted into the third fold. This materially weakens broad targets as a robustness candidate despite the positive pooled averages. Complete the fixed five-fold experiment before deciding whether sharp targets merit additional seeds; do not automatically extend broad targets solely on its pooled native score. Updated source tables and error transitions are in `CONFIRMATION_SCORECARD.md` and `confirmation_error_audit.json`.

## Fold3 sharp control completed

Sharp auxiliary supervision selected epoch8 and passed exact native parity. Scores versus the matched baseline:

| Rate | Base | Sharp | Delta |
|---|---:|---:|---:|
| Native | 0.725576 | 0.776818 | +0.051241 |
| Half | 0.681108 | 0.764872 | +0.083765 |
| Third | 0.636360 | 0.717487 | +0.081127 |

Sharp now improves native score on each of folds0–3, and both reduced rates on folds1–3 (fold0 half-rate regressed). This makes sharp auxiliary supervision a credible candidate for seed confirmation if the complete study supports it. It is still not uniformly better: native CCD on fold3 loses 0.028860, while Nexar gains 0.079129 native and 0.123530 third-rate. Per-source subsets are small; the single CAUSALCRASH example cannot support a source-level conclusion. Broad/fold3 is now running. The three-arm pooled scorecard remains restricted to complete matched folds0–2.

## Four matched grouped folds completed

Broad/fold3 selected epoch8, passed native parity, and scored 0.771131/0.718760/0.700492. It beats base at all rates but trails sharp at all rates on this fold. Pooled results over 280 clips:

| Rate | Base | Sharp | Broad |
|---|---:|---:|---:|
| Native | 0.720324 | 0.745800 | 0.748849 |
| Half | 0.699426 | 0.729030 | 0.724096 |
| Third | 0.641999 | 0.685839 | 0.677467 |

Broad minus sharp is only +0.003049 native, versus -0.004935 half-rate and -0.008372 third-rate; all paired intervals for this direct comparison include zero. The extra target width has not demonstrated an advantage. Broad's third-rate Nexar regression decreases in magnitude but persists at -0.041007 over 64 clips, with 11 ENTRY regressions versus five recoveries and eight COLLISION regressions versus three recoveries. The single CAUSALCRASH clip is reported separately without interpreting its score change as source-level evidence.

Sharp also retains source tradeoffs: pooled native CCD is -0.022990 versus base, and Nexar is -0.010540 half-rate and -0.023848 third-rate despite +0.023793 native. It is a candidate for further testing, not an established uniformly robust improvement.

The final-fold baseline job has launched and is waiting at its 12.5 GiB memory-admission guard before training. Complete that fold before choosing the seed-confirmation arm. The intervals in `confirmation_aggregate.json` remain exploratory and conditional on checkpoint selection; their movement across interim cohorts illustrates why the earlier two-fold intervals excluding zero were insufficient for confirmation.

## Full five-fold auxiliary-loss confirmation completed

All 15 base/sharp/broad checkpoints (349 held-out clips per arm, seed0) completed and passed native parity, split identity, and complete-ID-coverage checks. These are single-model grouped-CV results, not v8 ensemble comparisons or leaderboard estimates.

| Rate | Base | Sharp | Broad | Sharp minus base | Broad minus base |
|---|---:|---:|---:|---:|---:|
| Native | 0.709922 | 0.741372 | 0.747518 | +0.031449 | +0.037596 |
| Half | 0.701187 | 0.718385 | 0.715214 | +0.017199 | +0.014027 |
| Third | 0.644718 | 0.680732 | 0.675702 | +0.036014 | +0.030984 |

Native paired group-bootstrap intervals: sharp-base [+0.007777, +0.054291], broad-base [+0.015489, +0.059530]. Third-rate intervals also exclude zero; half-rate intervals include zero. These intervals condition on selected checkpoints and one training seed, and do not correct experiment selection. Broad-minus-sharp intervals include zero at all rates; broader targets have not demonstrated a clear mechanism benefit.

Sharp improves native score in all five folds, but half-rate regresses on folds0 and4. On the full cohort, sharp loses 0.003080 native CCD, 0.018753 half-rate Nexar, and 0.018684 third-rate Nexar. Broad improves native and half-rate scores in all four main sources but loses 0.023014 third-rate Nexar. There, broad ENTRY has 14 regressions versus seven recoveries, and COLLISION eight versus four; improved attributes mask some timing loss. Neither candidate is uniformly robust, and neither has been tested across training seeds or as a replacement member of the v8 ensemble.

Next experiment launched: fixed target transport on grouped folds1–4, seed0, reusing the five completed baseline controls. Its first-fold +0.019632/+0.027569/+0.022311 rate gains satisfy its predeclared feasibility screen, but source transfer remains uncertain. The recipe is unchanged; no FPS enters training or inference. This four-run confirmation is smaller than immediately expanding both auxiliary variants to multiple seeds and addresses resampling-induced target broadening directly. Complete it before selecting which candidate merits matched multi-seed/source-transfer confirmation.

The previous supervisor sequence exited successfully. Only its dedicated stopped wrapper was switched to `confirm_transport.py`; the new managed process was verified RUNNING and initially waited at the 12.5 GiB admission guard. See `transport_confirmation_manifest.json`, `transport_confirmation_status.json`, and `train_transport_fold*.log`. Other queues, models, and GPU runs remain untouched.

## Saved-logit native ensemble screen during transport memory wait

`native_aux_ensemble.py` tested fixed 50/50 probability averaging on the already saved grouped-five-fold, seed0 native OOF logits. Its standalone NumPy ordered-pair decoder exactly reproduced each saved model prediction before mixing, including attributes. No new model pass, training, GPU use, or label-fitted weight was involved. The first script run failed on a negative-log-probability initialization and was corrected before producing results.

| Native recipe | All 349 | Nexar 80 |
|---|---:|---:|
| Base | 0.709922 | 0.604805 |
| Sharp | 0.741372 | 0.627012 |
| Broad | 0.747518 | 0.633334 |
| **Sharp + broad, equal weights** | **0.752801** | **0.650457** |
| Base + sharp, equal weights | 0.735760 | 0.627741 |
| Base + broad, equal weights | 0.744428 | 0.627589 |

The sharp+broad blend improves native score by 0.005283 versus broad and 0.011429 versus sharp. This is an exploratory native-only lead: reduced-rate logits were not saved, the models share seed0 training data and backbone representation, and the comparison uses validation-selected checkpoints. Do not infer a leaderboard score or ship this blend. A viable ensemble requires half/third-rate, source-transfer, seed, and deployment-runtime confirmation. Full source breakdowns: `native_aux_ensemble_results.json`.

## Transport: second grouped fold completed

Transport/fold1 selected epoch9, passed exact native parity, and scored 0.735559/0.694072/0.653638 native/half/third, versus base 0.692305/0.685996/0.633088. Pooled over 141 held-out clips from folds0/1, transport versus base is +0.031684 native, +0.018323 half-rate, and +0.021589 third-rate. The 2,000 paired known-duplicate-group bootstrap intervals are [+0.000648,+0.065509], [-0.018736,+0.054022], and [-0.016468,+0.060923], respectively; these are interim, single-seed intervals without a selection correction.

Source transfer remains mixed. At third-rate, the 32 Nexar clips lose 0.03107 and MM-AU loses 0.00893 despite aggregate gain. At native rate, each of the four main source slices improves, but these slices remain small. This repeats the reduced-rate Nexar concern seen with the auxiliary variants, so transport has not established the robustness needed for a 0.63–0.65 leaderboard expectation. The sequential service advanced to transport/fold2. Live status and complete-cohort scorecard: `transport_confirmation_status.json`, `TRANSPORT_SCORECARD.md`.

## Transport: third grouped fold completed

Transport/fold2 selected epoch4 and passed native parity, but loses against its matched base at native/half/third rates: 0.705278/0.678181/0.583804 versus 0.723846/0.681130/0.592876. Across the first three folds (212 clips), transport still has small pooled gains of +0.015814/+0.011806/+0.012077; all paired 95% known-duplicate-group intervals include zero.

The third-rate Nexar deficit has grown to **-0.07944** across 48 clips, and native Nexar is now -0.01535. This contradicts a claim of uniform cross-rate robustness and weakens transport as a candidate despite earlier aggregate gains. The fourth grouped fold is running sequentially. Complete the fixed experiment for an unbiased method comparison, but do not spend multi-seed training on transport solely because its first two folds were positive. See `TRANSPORT_SCORECARD.md` and `transport_aggregate.json`.

## Transport: four matched folds and seed-confirmation preparation

Transport/fold3 selected epoch4, passed parity, and scored 0.740512/0.695511/0.720814, improving all rates versus the matched baseline. Across the first four folds, pooled transport gains are +0.013866 native, +0.010833 half-rate, and +0.027689 third-rate; all paired intervals include zero (third-rate lower endpoint -0.000426). Third-rate Nexar remains -0.02441 across 64 clips. On the same four folds, the sharp auxiliary control had larger pooled gains at all three rates (+0.025477/+0.029604/+0.043839). This is a comparison of these single-model grouped-fold arms, not a leaderboard projection. Transport/fold4 now follows sequentially.

`confirm_extra_seeds.py` is prepared but not running. Its dry run validates 20 sequential jobs for a selected candidate and matched base on seeds1/2 × five grouped folds, with unique output paths and the same RAM/CPU safeguards. `evaluate_training.py` now accepts an optional `--seed` while retaining seed0 as its default, so the current transport evaluator remains compatible. The follow-up runner refuses to start while transport or another campaign trainer is live. Select the candidate after transport's final fold; the current evidence favors sharp auxiliary supervision over target transport for seed confirmation, with source-transfer concerns still unresolved.

## Complete transport comparison and matched-seed launch

Transport/fold4 selected epoch6 and passed parity. Its native/half/third scores 0.695694/0.697728/0.667905 compare with base 0.665087/0.705423/0.652835: half-rate regresses on this fold. Across all 349 held-out clips, the grouped-five-fold, seed0 transport-minus-base gains are +0.017187/+0.007173/+0.025122. The paired group-bootstrap intervals are [-0.003440,+0.037933], [-0.016674,+0.031808], and [+0.000642,+0.049907]. They remain conditional on one seed and selected checkpoints. Third-rate Nexar is -0.009379 and half-rate CCD -0.008921; the method is not uniformly robust.

Sharp-minus-base on the exact same folds was +0.031449/+0.017199/+0.036014. Broad's native advantage over sharp (+0.006146) has an interval including zero, while its reduced-rate scores are lower. **Selected next arm: sharp auxiliary ENTRY** for matched seeds1/2 against base. This is a resource-allocation decision for further testing, not a deployment choice or a leaderboard forecast.

The transport sequence exited successfully. The dedicated stopped supervisor wrapper was switched to `confirm_extra_seeds --candidate aux_sharp` and started; service RUNNING, first job `base/fold0_seed1` reached training epochs. It will run 20 jobs sequentially: two arms × five grouped folds × seeds1/2, then per-checkpoint native/half/third evaluation with parity checks. `extra_seeds_aux_sharp_manifest.json` records the protocol, `extra_seeds_aux_sharp_status.json` the live state, and `EXTRA_SEEDS_AUX_SHARP_SCORECARD.md` recomputes each complete seed separately. Existing GPU jobs and shared queues remain untouched.

## Isolated deployment-loader parity for auxiliary checkpoints

The scored v8 runtime's inference-only `AuxPyramid` does not instantiate the training-only `entry_aux` layer. It therefore cannot strictly load a newly trained sharp checkpoint as-is. An isolated copy of its `lc_models.py`, `aux_models.py`, `sampling.py`, and `runtime.py` lives in `deployment_adapter/`. The copy adds an optional `entry_aux` linear layer to the inference-only model and passes `entry_aux` through `build_member`; the layer is present for strict loading but not called by the decoded path.

A completed sharp checkpoint strictly loaded through this copied runtime builder. On a deterministic CPU tensor with 128 positions, 70×384 cached tokens per position, and 100-dimensional motion, ENTRY, COLLISION, side, and evasion outputs matched the training model **exactly**. This establishes architecture/weight compatibility for one checkpoint, not full submission compatibility. Video preprocessing, actual cached-clip predictions, rate variation, all candidate checkpoints, latency, and packaged submission validation remain to be checked after the seed decision. The scored v8 package was not edited.

The adapter was then checked on **12 actual cached-clip views**: one held-out clip from each main source × native/half/third sampling rates. `verify_deployment_adapter.py` enforced the 10.5 GiB RAM guard, strict-loaded the sharp checkpoint through the copied runtime builder, and compared its ENTRY, COLLISION, side, and evasion tensors with the training implementation. All four tensors matched exactly on every view, and all 12 decoded predictions matched the saved out-of-fold evaluations. Evidence: `deployment_adapter_parity.json`. This checks the model loader on real cached DINO/motion inputs; it still does not validate end-to-end video extraction, packaging, multi-checkpoint ensemble behavior, or latency.

## Seed1, fold0: sharp gains did not replicate

The first matched seed1 pair passed native parity but sharp-minus-base was **-0.029591 native, -0.022569 half-rate, -0.028262 third-rate** on the 73 held-out clips. Native Nexar lost 0.120695 and third-rate Nexar lost 0.085054, despite native AIHUB gaining 0.071803. Thus seed0's fold0 gains cannot be treated as stable, and this independent seed substantially lowers confidence in an expected leaderboard gain. The planned five-fold seed1 comparison continues automatically; an individual fold is not sufficient to reject the full recipe either. Do not promote the candidate or modify its weights/selection criterion in response to this fold. The service advanced to base/fold1_seed1.

## Isolated GPU seed3 replication

The user authorized use of spare training capacity. At launch, the RTX 5090 had 32,109 MiB free and 0% utilization; system RAM had about 25 GiB available, while `/workspace` had only 31 GiB free. An isolated `train_gpu.py` copy and `confirm_gpu_seed3.py` runner used separate `training/gpu_base` and `training/gpu_aux_sharp` paths, seed3, the same grouped folds/recipe, low CPU and idle-class I/O priority, a 12.5/10.5 GiB RAM guard, a 20 GiB GPU-free admission check, and a 40% CUDA allocator cap. The existing `stage2_robust_confirm` service and its training source/outputs remained untouched. The managed fold0 matched-pair probe under `stage2_robust_gpu_seed3` passed initial resource and parity checks before folds1–4 ran. GPU replication is a distinct seed/device test, not an independently calibrated leaderboard prediction.

GPU seed3 fold0 completed: base native/half/third 0.771175/0.704496/0.659977; sharp 0.735783/0.707466/0.712901. Sharp loses 0.035392 native but gains 0.052924 at third rate. Both fold0 checkpoints passed exact native GPU-to-CPU prediction parity, and folds1–4 were launched under the same managed service. On fold1, the strict evaluator detected one numerical edge case: CPU re-inference chose ENTRY frame 17 where the GPU training output had frame 18 for `ccd_001362`. The evaluator now has an opt-in GPU-only rule that records and tolerates at most two one-frame temporal differences per checkpoint; the default CPU-campaign rule remains exact. Fold1 sharp evaluation completed with this single recorded mismatch. Do not describe GPU fold1 as exact-parity or merge GPU and CPU seeds without acknowledging the device change.

The **full grouped five-fold GPU seed3** matched pair is now complete (`gpu_seed3_aggregate.json`, 349 clips, 338 known-duplicate groups). Pooled baseline → sharp scores are native 0.740217 → 0.732501 (Δ −0.007716, paired group interval [−0.029692,+0.014427]), half-rate 0.696645 → 0.692788 (Δ −0.003857 [−0.026357,+0.017936]), and third-rate 0.637489 → 0.647488 (Δ +0.009999 [−0.014769,+0.034659]). No interval excludes zero. Sharp loses on NEXAR at every rate (−0.034715/−0.043525/−0.011068), and on >1000-frame clips at native and half rate (−0.022801/−0.036387). Four GPU fold pairs passed exact native GPU/CPU prediction parity; the fold1 sharp checkpoint had the one adjacent-frame difference above. Fold-level rate effects change sign, so seed0's broad gain does not replicate here. This remains a single-model OOF comparison and provides no calibrated real-LB score or reason to replace v8. The GPU service exited after the final pair; other GPU workloads started near fold4, so no further GPU jobs should launch without rechecking contention.

## Complete CPU seed1 confirmation

The independent CPU seed1 pair finished all five grouped folds with exact native re-inference parity. Baseline → sharp scores are native 0.745592 → 0.723367 (Δ −0.022225, paired group interval [−0.045625,+0.001988]), half-rate 0.703668 → 0.700067 (Δ −0.003601 [−0.027513,+0.020690]), and third-rate 0.661160 → 0.655352 (Δ −0.005808 [−0.030928,+0.018883]). NEXAR lost 0.057323/0.023505/0.022821 at native/half/third rate; >1000-frame clips lost 0.069607/0.036470/0.035778. This directly reverses the all-positive seed0 result under the same CPU recipe. Across the two complete CPU seeds, mean sharp-minus-base deltas are only +0.004612 native, +0.006799 half-rate, +0.015103 third-rate, with sample SDs 0.037954/0.014707/0.029573. These are descriptive; two seeds cannot estimate real-LB variance reliably. The managed service advanced automatically to matched CPU seed2; see `EXTRA_SEEDS_AUX_SHARP_SCORECARD.md`. The GPU seed3 replication provides a third, directionally negative native check but uses another numerical training backend, so keep its evidence separate from the CPU mean.

## Clean-label backbone-diversity screen

The updated leaderboard record in `HANDOFF.md` and `reports/stage2_leaderboard_correlation.md` shows v10 scored 0.5590, **0.03393 below v8**, despite improving the duplicate-clean third-rate CV view. That invalidates the earlier linear third-rate-to-LB forecast and weakens confidence in metadata-anchored MM-AU/CCD pseudo labels shared by v10/v11/v12. No numerical LB forecast is defensible from their offline scores. A cleaner candidate is the existing `E4_sbB` family: original frozen DINOv3 ViT-B features, the same grouped folds/seeds and robust stride recipe as `E4_sb`, and no MM-AU/CCD pseudo labels or consistency pool. The previous robustness campaign found complementarity from ViT-B, but had not isolated a fixed equal-weight ViT-B addition to the *v8* three-family ensemble against a same-backbone E4_sb addition.

`clean_vitb_blend.py` now runs that comparison on the established 284 duplicate-clean out-of-fold clips: v8 plain (`E4_sa+E2_sa+XN4_sa`), v8 plus three `E4_sb` ViT-S heads, and v8 plus three `E4_sbB` ViT-B heads, all equal-weight probability averaged with the same plain decoder. It first evaluates third-rate views, the historically informative but v10-falsified stress condition; native/half rates follow only if the screen has value. It reuses saved CV checkpoints and feature caches, checks exact clip coverage, and writes per-clip predictions to `clean_vitb_blend_results.json`. The CPU-only managed service `stage2_clean_vitb_blend` runs at nice19/idle I/O, one CPU thread, CUDA hidden, and waits for at least 14 GiB MemAvailable per clip. The ongoing CPU seed2 campaign and external GPU processes remain untouched. Any apparent gain still needs source/length breakdown, paired uncertainty, runtime and package parity, and ideally external leaderboard evidence before promotion.

All three rates completed. The ViT-B addition gains only +0.003449/+0.005477/+0.004723 versus v8 plain at native/half/third rate, with paired group intervals spanning zero; versus the same-backbone E4_sb addition, its gains are +0.000163/+0.000986/+0.003351. Third-rate NEXAR and >1000-frame clips regress by 0.016786 and 0.017518. This is a small inconsistent ensemble gain and does not justify the extra backbone inference pass or a leaderboard claim. The CPU-only service exited successfully. Full scorecard: `CLEAN_VITB_BLEND_SCORECARD.md` and `clean_vitb_blend_scorecard.json`.

## Final sharp-seed decision and updated LB evidence

Seed2 finished all five grouped folds with exact native CPU re-inference parity. Sharp-minus-base is +0.018767 native, +0.007254 half-rate, **−0.014473 third-rate**. Across the three complete matched CPU seeds, the mean deltas are only **+0.009330 ± 0.028054 native, +0.006951 ± 0.010403 half-rate, and +0.005245 ± 0.026997 third-rate** (sample SD). Seed1 loses at all rates; seed2 loses at the most frame-rate-stressed view. These are single-model grouped-CV scores; the GPU seed3 run separately loses native and half-rate and NEXAR at every rate. Sharp is rejected as a v8 replacement or added family. Do not spend deployment or leaderboard trials on the seed0 result alone.

The updated `HANDOFF.md` records **v12 real Stage2 LB = 0.5635**, after v10 = 0.5590 and v8 = 0.59293. V12's four-family mixed-backbone package had the best duplicate-clean third-rate CV in its screen (0.745 versus v8 recipe 0.710) but still lost roughly 0.0294 real LB. This is a second prospective failure of the earlier low-rate-CV-to-LB extrapolation for packages trained with metadata-anchored MM-AU/CCD extras. The shared extras are a plausible cause, not proved by these bundled submissions. Numerical forecasts for unscored variants based on that mapping are withdrawn. A clean-label addition to v8 is more defensible than another package containing those extras, but still needs actual external validation.

## Fast GPU inference screens on already trained clean-label heads

At this continuation's resource check another process held 7.7 GiB VRAM but the RTX 5090 had 24.4 GiB free and 0% utilization. The `clean_vitb_blend.py` evaluator now supports `--device cuda` with a 15% PyTorch allocator cap, a 16 GiB initial/12 GiB per-fold GPU-free admission check, and the existing 14 GiB host MemAvailable guard. It uses one low-priority CPU thread and cached features. The other process and all previous Stage2 training services were left untouched.

The first new fixed equal-weight addition, **E4_sbu** (equal stride mix plus unlabelled cross-rate consistency, but no MM-AU/CCD pseudo event labels), scored 0.715024 versus v8 plain 0.710388 and the same-backbone E4_sb capacity control 0.711759 on 284 duplicate-clean OOF clips at third rate. Its paired group interval versus v8 is [−0.004392,+0.014364]. NEXAR falls from 0.685196 to 0.674415 and >1000-frame clips from 0.675673 to 0.664404. This is too small and source-uneven to promote; only the stress-rate screen was run to save time. Reproducible predictions and uncertainty: `clean_consistency_blend_results.json`, `clean_consistency_blend_scorecard.json`.

A more distinct saved head, **E4_sbOA**, adds per-frame tracked-vehicle crop appearance and geometry. Prior single-family work reduced late bias for 1.5–2.5-second ENTRY-to-COLLISION gaps, but its total third-rate score regressed. The GPU-only fixed-addition screen is running on the same duplicate-clean cohort at native/half/third rate, using precomputed object caches and the exact training-time segment pooling. `clean_objappearance_blend_results.json` will contain its per-clip predictions when each rate completes. This is a representation test; deployment would require RF-DETR tracking and crop encoding, so a small CV gain would not justify a package.
