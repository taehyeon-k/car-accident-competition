Stage 2 Long-Context / NEXAR Experiment Task

First inspect the repository carefully before changing anything. Do not blindly trust this prompt if the current code, paths, checkpoints, or reports disagree with it.

Important files/directories to inspect first include:

HANDOFF.md

reports/stage2_temporal_pyramid_framecount_experiments.md

reports/stage2_iterative_search.md

stage2/long_video_experiments/REPORT.md

stage2/temporal_pyramid_experiments/

stage2/spotting_experiments/

current Stage 2 submission runtime/model code

existing P2, P10, ASFormer and ensemble results

Do not overwrite previous experiments, checkpoints, reports, or results. Create a new isolated experiment directory, for example:

stage2/long_context_v2_experiments/

The objective is to improve Stage 2 specifically for long-context videos similar to NEXAR, where clips often contain approximately 500–1250+ frames.

The competition model must remain:

FPS-blind

duration-blind

timestamp-blind

FPS may only be used afterward for offline competition-style evaluation if this is already consistent with the existing evaluation code.

1. Current evidence you must verify

Before implementing anything, independently verify the following from the repository results.

The existing experiments suggest:

fixed-128 ASFormer has relatively strong NEXAR COLLISION behavior despite lower aggregate validation score.

adaptive P2 temporal pyramid improves aggregate validation but does not solve long NEXAR failures.

increasing sampling density from 128 to 192/256/320 does not reliably improve long-video performance.

tiled local models did not solve the problem.

coarse global attention P5 did not help.

long-video oversampling P6 did not help.

adaptive ASFormer strongly overfit.

P2 4-seed probability ensemble improved aggregate validation substantially.

P10 EMA showed some useful regularization.

many NEXAR failures are catastrophic event-selection errors rather than small localization errors.

Examples previously observed include predictions hundreds of frames away from the true accident.

The important hypothesis is therefore:

The main long-video bottleneck may be selecting the correct accident event among many plausible distractor events, rather than insufficient temporal sampling resolution.

Do not assume this hypothesis is true. First test it directly.

2. Experiment A — Controlled NEXAR context-length benchmark

This experiment should be performed before designing a large new architecture.

For every NEXAR validation video, construct multiple context-length versions while keeping the same labeled ENTRY and COLLISION events visible.

Target context sizes:

approximately 256 frames

approximately 512 frames

approximately 768 frames

approximately 1024 frames

full original clip

Prefer crops centered so ENTRY/COLLISION remain at approximately the same relative local accident sequence while progressively adding unrelated context around them.

If symmetric cropping cannot be done because the event is near a boundary, use the maximum valid context while recording the actual context length.

Evaluate at least:

existing fixed-128 ASFormer baseline

P2

P10 if available

current P2 ensemble if practical

Report for every context size:

ENTRY hit accuracy

COLLISION hit accuracy

ENTRY frame MAE

COLLISION frame MAE

normalized ENTRY MAE

normalized COLLISION MAE

entry_side macro F1

evasion_space macro F1

overall offline score

number of samples

Also measure prediction drift as context increases.

For each sample compare:

prediction_256 -> prediction_512 -> prediction_768 -> prediction_1024 -> prediction_full

We specifically want to know whether adding unrelated temporal context causes the model to switch to false accident candidates.

Save per-video predictions.

3. Experiment B — Candidate Recall@K analysis

This is extremely important.

For the current temporal models, do not only inspect final argmax predictions.

For each validation video, obtain the event probability/logit sequences before final constrained decoding.

For both ENTRY and COLLISION calculate whether a correct temporal region appears among the top-K local maxima.

Test:

K = 1

K = 2

K = 4

K = 8

K = 16

Use an appropriate non-maximum-suppression radius so adjacent positions belonging to the same peak are not counted as separate candidates.

Report:

ENTRY Recall@K overall

COLLISION Recall@K overall

NEXAR ENTRY Recall@K

NEXAR COLLISION Recall@K

Recall@K by frame-count bin

Frame-count bins should include roughly:

<150

150–300

300–500

500–750

750–1000

>1000

The key diagnostic question is:

Does long-video top-1 accuracy collapse while top-8 candidate recall stays high?

If yes, prioritize candidate re-ranking rather than increasing sampling density.

Also save per-video candidate positions and scores.

4. Experiment C — Temporal pooling ablation

The current temporal pyramid uses average pooling between levels.

Run a tightly controlled ablation.

Keep everything else identical to a strong P2 configuration.

Compare:

C0

Current average pooling.

C1

Max pooling.

C2

Learnable strided depthwise convolution or SGP-style downsampling/local modeling.

Do not substantially increase parameter count.

Keep:

same DINO features

same training split

same loss

same sampling

same hidden dimensions

same optimizer

same EMA policy if applicable

same early stopping rules

Primary metric is not aggregate validation alone.

Report especially:

NEXAR ENTRY

NEXAR COLLISION



1000-frame bin

Candidate Recall@K

Reject an architecture if aggregate validation increases but long-video event selection clearly worsens.

5. Experiment D — Accident-phase auxiliary supervision

Extend the strongest compact temporal model with an auxiliary dense temporal phase task.

Generate labels automatically from ENTRY and COLLISION annotations.

Minimum phase definition:

PRE: before ENTRY

BETWEEN: ENTRY through before COLLISION

POST: after COLLISION

Optionally test a richer formulation if useful:

PRE

ENTRY neighborhood

INTRUSION

COLLISION neighborhood

POST

But start with the simple 3-state version.

The model must simultaneously predict:

ENTRY temporal logits

COLLISION temporal logits

temporal phase logits

Train with:

L_total = L_entry + L_collision + lambda_phase * L_phase + existing attribute losses

Try only a very small number of sensible lambda_phase values, for example:

0.1

0.25

0.5

Do not perform a huge hyperparameter sweep.

The goal is to determine whether global accident-state supervision reduces distant false positives in long clips.

Report whether false high-scoring peaks occur in phase-inconsistent regions.

6. Experiment E — Top-K joint ENTRY/COLLISION pair scorer

This is the most important architectural experiment.

Do not immediately decode:

argmax ENTRY + argmax COLLISION

Instead obtain top-K candidate peaks.

Start with:

K_entry = 8
K_collision = 8

For every valid pair satisfying:

entry <= collision

construct a pair representation.

Use information such as:

ENTRY candidate feature

COLLISION candidate feature

pooled feature between ENTRY and COLLISION

local feature before ENTRY

local feature after COLLISION

temporal phase probabilities

normalized candidate gap

event logits themselves

Do not use FPS, duration, timestamps, source identity or dataset identity.

An example representation is:

z_pair = concat(h_entry, h_collision, pool(entry:collision), pool(before_entry), pool(after_collision), phase_stats, event_scores, normalized_gap)

Then use a very small MLP or equivalent lightweight scorer:

pair_score = f(z_pair)

The final prediction should be the highest-scoring valid pair.

Important:

The candidate generator should optimize recall, while the pair scorer should perform discrimination.

Do not perform hard top-1 coarse localization followed by refinement.

Use multiple candidates to prevent error accumulation.

Training options may include:

cross-entropy over candidate pairs

positive-vs-hard-negative ranking

pairwise margin loss

Prefer the simplest stable implementation first.

A positive pair is the candidate pair nearest the GT ENTRY/COLLISION pair.

Hard negatives should include high-scoring incorrect pairs far from the GT event.

Report:

pair candidate oracle accuracy

final pair selection accuracy

Candidate Recall@8 before re-ranking

accuracy after re-ranking

long NEXAR improvements

false-event corrections

Provide several concrete examples showing cases where:

top-1 original model = wrong

but

GT candidate exists in top-8

and

pair scorer selects the correct pair.

7. Experiment F — Small coarse-scale long-range module

Only do this after Experiments A–E establish that additional long-range reasoning may help.

Do NOT build a large global Transformer.

Use the compact temporal pyramid and modify only coarse temporal levels.

Candidate designs:

F1

Local Conv/SGP at fine levels + small bidirectional SSM/Mamba at the two coarsest levels.

F2

Local Conv/SGP at fine levels + local self-attention at coarse levels.

The idea is:

fine scales preserve exact temporal detail

coarse scales model long-range accident context

Do not run full global attention over all 320 positions.

Keep total trainable parameter count preferably below approximately 1.5M unless there is a strong reason otherwise.

Because the dataset is small, aggressively watch for overfitting.

Compare train vs validation curves.

8. Experiment G — Native-frame candidate refinement

Only do this if Candidate Recall@K is already high.

Do not encode every frame of every long video densely.

Procedure:

Run global coarse model using normalized sparse sampling.

Obtain top candidate ENTRY and COLLISION regions.

Keep several candidate pairs.

Load native frames around each candidate.

Example local windows:

±12 frames

±24 frames

Merge overlapping windows to avoid duplicate DINO computation.

Encode these native frames using the same geometry-tuned DINOv3-S backbone.

Use a small local temporal refinement head to predict precise ENTRY/COLLISION positions.

Possible local head:

2–3 depthwise Conv1D blocks

residual connections

small hidden dimension

optional local attention

Do not use another large video foundation model.

Evaluate:

candidate region recall before refinement

exact-frame error before refinement

exact-frame error after refinement

encoded frames/video

runtime

VRAM

The refinement stage must operate on multiple candidate regions, not one top-1 region.

9. Attribute prediction experiment

Current global pooling for entry_side / evasion_space may be weak for long videos.

After obtaining the selected ENTRY/COLLISION pair, test event-conditioned attribute prediction.

Use something like:

z_event = concat(h_entry, h_collision, pooled_between, local_before, local_after)

Then:

side head from z_event

evasion head from z_event

Compare against current global pooled attribute heads.

Do not reuse validation source information.

10. Regularization

Because previous Stage 2 experiments showed severe overfitting:

keep DINO frozen for the first experiments

use EMA if it remains beneficial

early stop

use dropout

weight decay

avoid large models

use multiple seeds for any promising model

do not conclude from a single seed

For promising experiments run at least:

seed 0

seed 1

seed 2

If inexpensive, seed 3 as well.

Report mean and standard deviation.

Do not use the validation labels to manually tune per-source thresholds or source-specific gates.

11. Validation priorities

Do not select models based only on aggregate 70-video validation score.

Every experiment report must include:

Overall

competition-style offline score

ENTRY accuracy

COLLISION accuracy

side macro F1

evasion macro F1

By source

At minimum:

NEXAR

MM-AU

AIHub

CCD

By frame count

<150

150–300

300–500

500–750

750–1000

>1000

Long-video metrics

Especially:

NEXAR ENTRY hits

NEXAR COLLISION hits

median/mean frame error

Candidate Recall@K

catastrophic miss rate

Define a catastrophic miss as something sensible such as normalized temporal error >10% or another clearly stated threshold.

12. Experimental order

Run experiments in this order:

Phase 1 — diagnostics

context-length controlled benchmark

Candidate Recall@K

Do not skip these.

Phase 2 — cheap architecture changes

average vs max vs SGP pooling

phase auxiliary supervision

top-K pair scorer

Phase 3 — only if justified

small coarse SSM/local-attention module

native-frame candidate refinement

event-conditioned attribute heads

Do not jump directly to the most complicated architecture.

At every stage ask:

What specific failure mode did this experiment fix?

13. Expected strong architecture if evidence supports it

The likely final design is approximately:

Long frame sequence
        |
FPS-blind normalized sparse sampling
        |
Frozen geometry-tuned DINOv3-S
        |
Frame/token features
        |
Fine temporal local modeling
        |
Multi-scale temporal pyramid
  - max/SGP style downsampling
  - optional coarse SSM
        |
ENTRY heatmap
COLLISION heatmap
phase logits
        |
Top-K ENTRY candidates
Top-K COLLISION candidates
        |
All valid ENTRY <= COLLISION pairs
        |
Joint pair scorer
        |
Top-R accident hypotheses
        |
Dense native-frame candidate refinement
        |
Final ENTRY + COLLISION
        |
Event-conditioned side/evasion heads

But only implement this full version if preceding experiments support each component.

14. Compute/resource constraints

Prefer reuse of existing cached geometry-DINOv3-S features for experiments C–F.

Do not recalculate expensive features unnecessarily.

Do not interfere with unrelated running jobs.

Monitor:

GPU VRAM

peak allocated memory

runtime/video

number of encoded frames/video

Avoid unnecessarily large experiments.

15. Deliverables

Create a final report such as:

reports/stage2_long_context_v2_report.md

It should include:

repository state / commit used

exact dataset split

all implemented architectures

parameter counts

training setup

tables of all experiment results

source-wise results

frame-count-wise results

NEXAR-specific results

Candidate Recall@K

controlled context-length degradation

runtime and VRAM

overfitting analysis

representative successful/failed examples

conclusions supported by evidence

recommendation for the next leaderboard submission

Also save machine-readable:

summary.json

per-run metrics

per-video predictions

candidate lists

configs

histories

checkpoints for useful runs

16. Important research discipline

Do not claim improvement because aggregate validation goes up by a few points.

The main target of this experiment is long-video robustness.

A useful model should ideally demonstrate some combination of:

better NEXAR performance

better >1000-frame performance

lower context-length degradation

higher Candidate Recall@K

fewer catastrophic false-event selections

stable multi-seed behavior

no severe regression on short clips

If an experiment fails, keep the result and explain why.

Do not hide negative results.

At the end, clearly separate:

confirmed findings

plausible hypotheses

inconclusive results

recommended next experiment

The final objective is not architecture complexity. It is to identify the smallest architecture that can reliably locate the correct accident event inside a long video containing many temporal distractors.