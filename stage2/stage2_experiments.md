You are working on my Car Accident Competition Stage 2 repository. First inspect the latest code, reports, configs, experiment history, and current best models. Do not blindly trust this prompt if the repository differs; adapt implementation to the actual current structure while preserving the experimental intent.

The current real-leaderboard reference is v8 = 0.59293. Recent experiments indicate that the major remaining Stage 2 weakness is ENTRY localization, especially long ENTRY→COLLISION gaps and temporal-position shortcuts. ODS/object-state supervision was the strongest recent object-centric direction.

The purpose of this task is rapid hypothesis screening, not publication-quality validation.

IMPORTANT: keep experiments cheap

This is a fast draft campaign.

* Do not run several seeds for each method.
* Do not perform large hyperparameter sweeps.
* Do not do full expensive CV unless absolutely required.
* Use one seed, a compact but meaningful validation split, and reduced training where appropriate.
* Prefer cached DINO/motion/detection features whenever possible.
* Stop clearly bad experiments early.
* The objective is to determine whether each idea has enough signal to deserve a later rigorous experiment.
* It is acceptable for absolute scores to be noisy. Compare all candidates under the same protocol.

Use v8 and/or the strongest appropriate ODS baseline under exactly the same fast protocol as controls.

⸻

Five directions to test

Direction 1 — Collision-conditioned causal actor transition

Current object methods largely aggregate top objects. Instead explicitly identify the vehicle that is most likely involved in COLLISION and preserve that object’s identity backward through time.

Pipeline:

video
→ current global/motion model
→ COLLISION estimate
→ identify likely collision-causing vehicle near COLLISION
→ track that same actor backward
→ construct actor-specific temporal sequence
→ predict ENTRY from that actor's state transition

Actor features can include existing available features:

* DINO appearance/crop representation
* bbox center/width/height/area
* lane-relative position
* intrusion depth
* lane overlap
* ego-lane center offset
* temporal derivatives
* residual/camera-compensated motion
* detector/tracker confidence

Do not immediately logsumexp/max-pool across objects. Preserve object identity through temporal modeling.

Also run an inexpensive oracle diagnostic if feasible:

GT COLLISION → likely collision actor → backward actor trajectory → simple state/change-point ENTRY estimate.

This is diagnostic only and must never be mixed into deployable validation results.

⸻

Direction 2 — Dynamic temporal label assignment for ENTRY

Exact ENTRY annotation may contain small temporal ambiguity and source-specific convention differences.

Instead of forcing the model to assign all ENTRY probability to exactly the annotated sampled position, allow the positive boundary to move within a small local neighborhood around GT when another nearby frame is more consistent with the learned transition evidence.

Implement a conservative local dynamic assignment inspired by precise temporal event spotting / DETR-style matching:

[
t^*

\arg\min_{t \in [t_{GT}-r,t_{GT}+r]}
[
\lambda_t |t-t_{GT}| - \lambda_s S_{transition}(t)
]
]

or an equivalent differentiable/soft formulation.

Important constraints:

* small local radius only;
* strong cost for temporal displacement;
* never allow labels to migrate toward COLLISION simply because it produces a stronger visual response;
* compare against the existing fixed ENTRY target;
* this should address local label ambiguity, not become another gap prior.

Measure how much the assigned target actually shifts. If it consistently shifts late, treat that as a warning.

⸻

Direction 3 — Future latent prediction / transition surprise

Use current frozen visual/object features rather than replacing DINO with a large video backbone.

Train a small predictor:

[
\hat z_{t+k} = f(z_{\le t})
]

for one or a few short horizons such as (k\in{1,2,4}).

Then compute prediction error / surprise:

[
r_{t,k}

z_{t+k}-\hat z_{t+k}
]

or

[
s_{t,k}=|z_{t+k}-\hat z_{t+k}|.
]

Feed this signal into the actor ENTRY/onset branch.

Hypothesis:

A vehicle transitioning from normal motion into ego-lane intrusion should produce a stronger prediction residual than ordinary smooth driving.

Keep this module small. Do not introduce a large V-JEPA-like architecture. The purpose is a cheap predictive temporal auxiliary signal on top of the existing frozen representation.

If useful, prediction training may use unlabeled videos because it requires no ENTRY pseudo-labels.

⸻

Direction 4 — Single-transition hazard model for ENTRY

This is one of the most important experiments.

Replace or augment unconstrained ENTRY argmax scoring with a discrete-time hazard model.

Predict:

[
h_t=P(E=t \mid E\geq t,x)
]

and convert it to an ENTRY distribution:

[
P(E=t)

h_t\prod_{j<t}(1-h_j).
]

This naturally models ENTRY as a one-time transition.

Use actor/state representation as the main conditioning signal.

Possible inputs:

* causal actor representation;
* ODS BEFORE/ONSET/AFTER features;
* lane/intrusion geometry;
* motion;
* future-prediction residual from Direction 3;
* collision-conditioned context.

Avoid absolute normalized video position if possible.

Compare:

standard ENTRY logits
vs
hazard ENTRY
vs
standard + hazard ensemble/residual

Inspect whether hazard modeling:

* reduces multiple ENTRY peaks;
* reduces long-gap late bias;
* improves long-gap ENTRY;
* maintains crop robustness.

Do NOT turn this into the previously failed structured phase decoder.

⸻

Direction 5 — High-resolution causal-actor ROI branch

Previous global high-resolution experiments were not successful. This experiment is different.

Keep the normal low-resolution/global representation, but selectively spend resolution on the causal actor.

Architecture:

low-resolution global frame representation
                 +
high-resolution causal-actor crop
                 +
optionally actor + surrounding lane-context crop

Suggested crop:

* detected/tracked causal actor bbox;
* expand bbox roughly 1.5–2× to retain surrounding road/lane context;
* encode crop using the existing DINOv3-S or a lightweight compatible encoder;
* fuse as a residual branch rather than replacing the global pathway.

Goal:

Recover subtle ENTRY cues when the incoming car is still small/far away without paying the cost or overfitting penalty of full-frame high-resolution inference.

Compare:

global only
actor crop only
global + actor ROI

If actor tracking is unavailable for a frame, fall back cleanly to the global branch.

⸻

v15 combined architecture

After individual screening, build one lightweight prototype combining only the components that showed positive signal.

Target architecture:

                    existing v8 global branch
                 DINOv3-S + residual/camera motion
                              |
                         COLLISION head
                              |
                  collision-conditioned actor
                              |
                preserve actor identity backward
                              |
          +-------------------+--------------------+
          |                                        |
 actor geometry/state                         actor ROI
 bbox/lane/motion                       high-res DINO crop
          |                                        |
          +-------------------+--------------------+
                              |
                    actor temporal representation
                              |
                future latent predictor
                              |
                  prediction residual
                              |
             +----------------+----------------+
             |                                 |
      ODS state head                     hazard ENTRY head
 BEFORE / ONSET / AFTER                         |
             |                                  |
             +------------ transition ----------+
                              |
                 dynamic local label assignment
                              |
                  local physical decoding
                              |
                         COST ENTRY
                              |
                 reliability-gated fusion
                    COST ENTRY ↔ v8 ENTRY
                              |
                        final prediction

Do not force every component into v15 if the corresponding isolated experiment is clearly negative.

The purpose of v15 is to combine complementary positive mechanisms, not maximize architectural complexity.

⸻

Lightweight physical decoder

For the actor/hazard version, try a simple local decoder rather than structured phase decoding.

Candidate score:

[
S(e)

w_h \log P_{\text{hazard}}(E=e)
+
w_o S_{\text{ODS}}(e)
+
w_\Delta S_{\text{state-change}}(e)
+
w_g S_{\text{geometry}}(e).
]

Only enforce physically safe constraints:

* ENTRY < COLLISION;
* after ENTRY, AFTER/intrusion state should persist briefly;
* avoid selecting isolated one-frame spikes.

Do not use:

* average training ENTRY→COLLISION gap;
* preferred normalized clip position;
* source-specific timing offsets.

Keep post-processing parameters minimal.

⸻

Optional cheap debiasing experiment

If the existing --pos-grl implementation is still available and simple to enable, include one cheap experiment:

best actor/ODS configuration
vs
same + temporal-position GRL

This is especially useful if the new method still shows late long-gap bias or poor crop-position slope.

Do not spend much compute on this if it requires substantial engineering.

⸻

Evaluation protocol

Use the same single-seed compact validation protocol for all candidates.

At minimum report:

Main

* overall competition-style score;
* ENTRY score;
* COLLISION score.

ENTRY robustness

* native ENTRY;
* 1/3-rate ENTRY;
* crop0.25 ENTRY;
* crop0.5 ENTRY;
* crop-position slope if existing code supports it;
* long-gap late bias.

Gap bins

Report ENTRY hit rate for approximately:

<0.5 s
0.5–1.0 s
1.0–1.5 s
1.5–2.5 s
>2.5 s

The most important bins are 1.5–2.5 s and >2.5 s.

Actor diagnostics

For actor-based methods also report:

* causal actor available at GT ENTRY;
* causal actor track continuity ENTRY→COLLISION;
* actor-selection confidence/margin;
* performance conditional on successful vs failed actor tracking.

This is important to separate:

bad actor selection
from
bad ENTRY localization after correct actor selection

Efficiency

Measure approximate inference cost:

* detector calls/video;
* tracker calls;
* actor crop encoder calls;
* wall-clock inference time or FPS where easy;
* peak VRAM if easy.

The proposed method must remain realistic for submission inference.

⸻

Ablation matrix

Keep this compact. Something approximately like:

ID	Method
B0	v8 fast-protocol baseline
B1	ODS/current strongest object baseline
D1	collision-conditioned actor
D2	+ dynamic label assignment
D3	+ future-prediction surprise
D4	+ hazard ENTRY
D5	+ high-res actor ROI
V15	best compatible combination

You may alter this if repository structure makes a different decomposition cleaner.

Again: one seed is enough for this screening campaign.

Do not rerun methods just because differences are noisy. The objective is to distinguish:

clearly promising
possibly useful
clearly not useful

not establish statistical significance.

⸻

What counts as promising

Do not judge solely by aggregate validation score.

A method is particularly interesting if it gives a clear improvement in one or more of:

* long-gap ENTRY;
* 1/3-rate ENTRY;
* long-gap timing bias;
* crop robustness;
* ENTRY localization while preserving COLLISION;
* complementary rescue cases relative to v8.

A small aggregate gain produced only by short/easy clips is not enough.

For v15, I would consider the prototype highly promising if it shows a meaningful simultaneous improvement in:

ENTRY native
+
ENTRY 1/3-rate
+
1.5–2.5 s gap
+
>2.5 s gap

without a major crop0.25 regression.

⸻

Deliverables

At the end, write a concise report containing:

1. exact code/config changes;
2. experimental protocol;
3. one table containing every tested method;
4. per-gap ENTRY results;
5. crop/rate robustness;
6. inference-speed comparison;
7. actor-tracking diagnostics;
8. qualitative examples of:
    * successful long-gap correction;
    * failed actor selection;
    * ENTRY pulled toward COLLISION;
    * dynamic-label shifts;
9. ranking of the five directions as:
    * strong signal
    * weak/uncertain
    * negative
10. recommendation for the smallest architecture worth running rigorously next.

Do not claim an improvement based only on the fast validation split. Clearly distinguish fast-screening evidence from a result that deserves multi-seed/full-CV validation.

The central research question is:

Can Stage 2 ENTRY be improved by first identifying the collision-causing actor and then treating ENTRY as that actor’s one-time physical state transition, while using local label flexibility, predictive temporal surprise, and selective high-resolution evidence instead of relying on temporal-position or ENTRY→COLLISION-gap priors?