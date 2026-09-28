Do not assume this prompt perfectly matches the current code. Verify everything from the repository.

The previous campaign has narrowed the problem substantially. Do NOT start another broad architecture sweep.

Current important finding:

OB_D, the object-appearance + ego-lane geometry residual ENTRY branch, is the only tested new feature source that clearly improves several difficult ENTRY dimensions at the single-family level:

E4 v8 → OB_D approximately:

native overall .753 → .766
1/3 overall .687 → .713
ENTRY native .623 → .648
ENTRY 1/3 .500 → .581
ENTRY gap 1.5–2.5 s .32 → .43
long-gap late bias +.40 s → +.27 s
gap-OOD-L 1/3 .213 → .311

However, there are two unresolved problems.

First, the gains are source-dependent: OB_D improves AIHUB/MMAU but hurts native CCD/NEXAR.

Second, when OB_D E4 + OB_D E2 are combined with XN4, aggregate performance improves but long-gap ENTRY is only equal to the original v8 ensemble. Therefore OB_D may be correcting errors already handled by E2/XN4 rather than providing sufficiently complementary information.

The goal of this campaign is:

Determine exactly why OB_D works, identify its current bottleneck, and make its correction both source-robust and complementary to the v8 ensemble.

Do not test more generic frame-rate consistency, equal-stride recipes, MM-AU/CCD extras, generic level-0 boundary branches, gap-balanced loss, or counterfactual gap warping unless required as a control. Those questions have already been sufficiently explored.

Experiment A — diagnose causal-vehicle selection before training anything

The current OB_D branch uses the three most prominent vehicles selected approximately by area × centrality.

Test whether this selection rule is missing the vehicle responsible for ENTRY.

Build a per-clip diagnostic for all duplicate-clean validation clips.

For every tracked vehicle around GT ENTRY, save:

track ID
detector confidence
track persistence
bounding-box area
image centrality
box growth
lane overlap
signed lane intrusion depth
change in lane intrusion depth
lateral motion after camera-motion compensation
appearance-change magnitude

For diagnostic purposes only, construct an approximate oracle relevant track. It may use GT ENTRY/COLLISION and future observations because this oracle is NEVER used for inference. For example, identify the track that exhibits the strongest physically plausible increase in ego-lane intrusion/overlap around ENTRY and remains relevant toward COLLISION.

Also manually inspect a compact failure gallery so the automatic oracle itself is not blindly trusted.

Measure:

current area×centrality selector recall@1/3/5 for the oracle track
recall by source
recall by ENTRY→COLLISION gap
recall on clips where OB_D fixes v8
recall on clips where OB_D hurts v8
recall on CCD and NEXAR specifically

Most importantly, compute conditional ENTRY accuracy:

selected relevant track present vs absent
valid lane corridor vs invalid/unreliable
stable track vs fragmented track.

If OB_D performance is much higher when the correct track is present, candidate selection is the primary bottleneck.

If the relevant track is almost always present but OB_D still fails, move attention to lane quality or onset modeling.

Do this analysis before spending GPU time on new models.

Experiment B — replace prominence selection with intrusion-relevance selection

Only if Experiment A shows meaningful selection failures, test improved candidate selection.

Compare the current area×centrality top-3 selector with three controlled alternatives.

SEL-H: a deterministic causal selector using only information available up to the current frame, based on lane-relative relevance such as signed lane distance, rate of lane-distance change, lane-overlap change, lateral velocity, box growth and track persistence.

Do not use GT ENTRY, COLLISION, future frames, FPS, source ID or any future information.

SEL-S: keep more tracks, e.g. top 6–10 detector/tracker candidates, and use a tiny learned soft relevance scorer followed by attention/log-sum-exp aggregation instead of hard top-3 selection.

The scorer should stay very small to avoid overfitting. It should operate on track geometry/motion/confidence features and learn which object matters for ENTRY.

SEL-HS: heuristic pre-filter followed by learned soft relevance attention.

Do not make a large Transformer. This dataset is too small. A small per-track MLP + temporal encoder + masked attention is enough.

The relevant research idea is object-centric retrieval: retrieve the object that explains the temporal event rather than selecting the visually largest object.

Report selector recall diagnostics in addition to normal ENTRY metrics.

Experiment C — test whether lane geometry quality explains the source inconsistency

The present OB_D assumes the road-head-derived ego-lane corridor is reliable.

Test this assumption explicitly.

For each frame/track create lane-quality diagnostics such as:

whether corridor boundaries are defined
road/lane prediction confidence if available
temporal stability of the corridor
lane-width stability
sudden frame-to-frame lane-boundary jumps
fraction of frames with missing geometry.

Compare these statistics across AIHUB, CCD, MMAU and NEXAR and between OB_D successes/failures.

Then test a small reliability-gated OB_D.

Instead of always trusting lane features:

lane_feature_used = q_lane * lane_feature

where q_lane is a confidence/reliability value derived only from current/past geometry quality.

Also try lane-feature dropout during training so the object branch cannot become dependent on perfect lane estimates.

A lightweight residual gate may depend on:

track confidence
track persistence
lane reliability
object-branch confidence/margin.

The desired behavior is:

use OB_D strongly when the object/lane evidence is reliable, but fall back toward the original v8 logits when it is not.

Do not use source identity in the gate.

The main question is whether this removes the native CCD/NEXAR regressions without removing the AIHUB/MMAU gains.

Experiment D — determine whether the OB_D representation contains a better onset than its head extracts

The generic level-0 boundary branch already failed. Do not repeat it.

Instead apply boundary/onset supervision directly to the OB_D object/lane representation.

Run a clean ablation:

OB_D baseline
OB_D + local ranking loss
OB_D + BEFORE/ONSET/AFTER state supervision
OB_D + both

Do NOT combine gap warp in these experiments.

Use a narrow region around GT ENTRY.

Possible ranking constraint:

score(ENTRY) > score(ENTRY ± δ) + margin

and explicitly distinguish:

BEFORE: approaching but not yet entering
ONSET: initial lane-intrusion transition
AFTER: clear intrusion.

If useful, add a small contrastive loss so object/lane features immediately before and after ENTRY are separated.

The goal is to test:

Does OB_D already contain the physical signal earlier than its current onset head fires?

This is substantially different from the failed generic boundary branch because the supervision is now applied to actor-specific lane-relative evidence.

Experiment E — measure complementarity, not just CV improvement

This is essential.

For every candidate produce out-of-fold per-clip predictions and compare against:

E4_sa
E2_sa
XN4_sa
v8 3-family ensemble.

For ENTRY calculate:

rescue = candidate correct AND v8 ensemble wrong

harm = candidate wrong AND v8 ensemble correct

and report rescue/harm separately for:

long-gap clips
crop25
gap-OOD-L
AIHUB
CCD
MMAU
NEXAR.

Also report error correlation / disagreement between the new branch and E2/XN4.

A model that improves E4 but only solves errors already corrected by E2/XN4 is NOT an interesting final candidate.

Prefer a slightly weaker individual branch if it has substantially higher net rescue against the v8 ensemble.

Experiment F — temporal-position debiasing only if the previous experiments still show position dependence

Do not start here.

Only if the best object/lane model still has poor crop25 slope or gap-OOD despite good causal-object selection and lane reliability, test one small position-debiasing intervention.

Add an auxiliary adversarial probe that tries to predict normalized temporal position / coarse position bin from the ENTRY-specific representation, with gradient reversal so the ENTRY representation is discouraged from carrying easily exploitable absolute temporal-position information.

Do not remove useful motion information and do not apply this to COLLISION.

Compare with an identical model where the position probe is trained without gradient reversal.

This is inspired by temporal-grounding work showing that models can exploit annotation-location distributions instead of visual evidence.

Keep this experiment small; it is not the primary hypothesis.

Evaluation protocol

Use the existing duplicate-clean 5-fold protocol.

Use 3 seeds for screening and 6 seeds for any candidate that appears genuinely better.

Initially use the manually labelled/v8 training data only. Do not add MM-AU/CCD extras.

Keep v8's stride mix unless the current repository establishes a more appropriate matched control.

Report at minimum:

overall native / 1/2 / 1/3
ENTRY native / 1/3
ENTRY by gap, especially 1.5–2.5 and >2.5 s
signed long-gap ENTRY bias
crop25 ENTRY and position slope
crop50 ENTRY and slope
gap-OOD-L native / 1/3 and bias
source-wise ENTRY native / 1/3 / crop25
per-seed results
selector recall@K
lane-quality statistics
v8-ensemble rescue/harm.

The strongest candidate should not merely improve normal CV.

At the E4-family level, try to preserve approximately the current OB_D strengths:

1/3 overall around or above .71
1.5–2.5 s ENTRY around or above .43
long-gap bias around or below +.27 s
crop25 near or above v8
gap-OOD-L 1/3 around or above .31.

For the final ensemble, the more important requirement is to improve beyond v8's existing long-gap .43 rather than merely reproduce it, while keeping frame-rate and crop robustness.

Decision logic

At the end, explicitly classify the bottleneck as one of:

candidate-object selection
detector/tracker quality
lane-corridor quality
object/lane representation
onset head/loss
temporal-position prior
or a mixture supported by quantitative evidence.

Do not hide negative experiments.

If an oracle/relevance selector strongly improves results, prioritize better object selection.

If lane-valid clips perform well but lane-invalid clips fail, prioritize reliability gating or a better lane representation.

If the correct vehicle and good lane geometry are present but onset remains late, prioritize object-specific boundary/change-point learning.

If OB_D improves E4 but has almost no rescue over the full v8 ensemble, focus on generating a more complementary representation rather than increasing OB_D's standalone CV.

Produce a final report explaining not only which arm wins, but which failure mechanism the evidence supports and therefore what the next architectural step should be.