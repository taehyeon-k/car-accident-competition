# Stage 2 goal update: scene-conditioned actor ENTRY pilot

**Decision: stop this model family. Keep submitted v8 as the best verified Stage 2 model (real leaderboard 0.59293).** The target of 0.63–0.65 is not yet supported by a candidate or a calibrated forecast. The leaderboard gain required over v8 is 0.03707–0.05707.

## Why this experiment

Earlier OB_D uses separate temporal encoders for the top three tracked actors and pools their onset scores. It improved some long-gap and one-third-rate ENTRY cases but lost on native CCD and NEXAR. This pilot conditioned actor selection on the scene representation at each position, then used the selected actor's features and their temporal change to adjust only ENTRY. It was intended to retrieve the causal actor rather than favor an isolated per-object onset. The actor features are cached appearance and lane geometry; no source, FPS, duration, or validation label is a model input.

The work used the existing fold0 train/validation split (279/70 clips), seed0, manually labelled data only, and the established 0.5/0.25/0.25 stride mix. Both variants were trained on the free GPU in isolated paths. Other runs, their code, queues, and checkpoints were untouched. This is a **single-fold screen**, not a robust multi-seed estimate. The duplicate-clean evaluation subset has 54 clips and 14 clips with an ENTRY-to-COLLISION gap at least 1.5 seconds.

| Fold0 duplicate-clean model | Native score | Half-rate score | Third-rate score | Native / third long-gap ENTRY hit | Native / third NEXAR ENTRY hit |
|---|---:|---:|---:|---:|---:|
| E4_sa v8-family parent | 0.739 | 0.735 | 0.699 | 0.357 / 0.214 | 0.438 / 0.500 |
| Prior OB_D object branch | 0.732 | 0.725 | 0.723 | 0.286 / 0.429 | 0.438 / 0.438 |
| New actor-scene, trained end to end | 0.727 | 0.707 | 0.713 | 0.286 / 0.429 | 0.563 / 0.563 |
| New actor-scene, E4_sa trunk frozen | 0.732 | 0.735 | 0.699 | 0.357 / 0.214 | 0.438 / 0.500 |

Scores above use the same plain direct decoder, cached inputs, fold, and clips. The end-to-end model selected epoch6 by FPS-blind validation; on all 70 fold0 clips it scored 0.743 versus OB_D 0.760. The frozen-trunk version selected epoch1 and scored 0.755 on all 70; its reduced-rate predictions were essentially the parent model's. The end-to-end version gained two NEXAR ENTRY hits on this small slice but worsened aggregate event performance. It did not improve long-gap ENTRY. The frozen version did not convert that slice gain into a consistent improvement. These data do not justify spending time on more seeds/folds or packaging either model.

The newer `reports/stage2_narrowing_experiments.md` campaign also contains learned actor-selection variants (SEL-S/SEL-HS), so another attention sweep would repeat work. Existing v8 deployment parity was already verified exactly by `submission_tools/v8_stage2/parity_results.json`; repeating that smoke test will not explain the leaderboard gap. The v10 and v12 real leaderboard losses after stronger offline scores also warn against treating a small development-set gain as an expected real gain.

## Next experiment with a credible path to the goal

Obtain **new, independent, human-adjudicated ENTRY labels** for clips that reflect the hidden 10 Hz accident distribution, especially early lane-intrusion and long ENTRY-to-COLLISION gaps. Existing models repeatedly fail those cases; architecture and pseudo-label sweeps have not closed the gap. A compact blinded annotation protocol should specify the first objectively visible lane intrusion, collision contact, side, and evasion, with two annotators and disagreement flags. Keep the new clips entirely outside model and decoder selection for the first measurement. If the target convention differs from current source labels, correct or stratify *training-only* labels with provenance, train a matched v8 recipe, and evaluate the unchanged decoder on the locked set before any leaderboard submission. This is a proposed data experiment, **not a result**; no new labels have been obtained here.

Artifacts: `robust_goal_20260927/actor_interaction.py`, `train_frozen_actor.py`, `eval_actor_interaction.py`, `actor_scene/fold0_robust.json`, and the two isolated `actor_scene*/cv/fold0_seed0` checkpoint directories. Checkpoint configuration identifies the fold, seed, cache, stride mix, and training recipe. `train_frozen_actor.py` additionally needs `ACTOR_BASELINE_CKPT` pointing to the matching E4_sa checkpoint.
