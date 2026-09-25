# Stage 2 length-gated validation experiment — 2026-09-25

## Result

The requested *offline* target is met on the unchanged 279-train / 70-validation manifests. The preferred replicated hybrid scores **0.7104 on the 15 NEXAR validation clips** and **0.8017 on all 70 clips**. It uses exactly the original P2 ensemble prediction on every non-NEXAR validation clip, retaining its **0.8257 non-NEXAR score**. The higher but less stable seed-0 specialist yields NEXAR 0.7438 and overall 0.8089.

The new Stage 2 submission scored **0.437 on the real leaderboard**, versus the previous P2 submission's **0.4618**. This experiment has no leaderboard result. Its offline improvement must not be read as evidence of a hidden-test improvement.

## Fixed protocol

- The train and validation manifests are `/workspace/data/stage2/manifests/train.jsonl` (279) and `val.jsonl` (70), unchanged from prior Stage 2 work. They share no sample IDs. The P2 comparator is the 279-training-video, four-seed ensemble, not the 349-video refit submitted to the leaderboard. The NEXAR specialist heads were also trained on the same 279 videos; their temporal prior was fitted only to the 65 NEXAR training labels.
- On clips with **more than 500 original frames**, predict with the previously trained NEXAR specialist; otherwise predict with the original P2 ensemble. The rule was fixed from the training length distributions: MM-AU is at most 415 frames; NEXAR is at least 570; one CausalCrash training clip has 1109 frames. Thus the rule is imperfect as a source proxy. At inference it reads only native frame count, never source ID, FPS, duration, timestamps, or annotations. All 15 NEXAR validation clips, and no other validation clip, exceed 500 frames.
- The two specialist variants were pre-existing: the best seed-0 three-head ensemble and the equal-weight ensemble across three architectures and three seeds. Their internal training, prior and limitations are documented in `../nexar65_experiments/REPORT.md`. No model was retrained or selected by a threshold sweep in this experiment.
- Offline score is 0.35×ENTRY accuracy within 0.3 seconds + 0.35×COLLISION accuracy within 0.3 seconds + 0.15×side macro-F1 + 0.15×evasion macro-F1. FPS is used only after prediction for this metric.

## Source results

| Slice | n | Original P2 | Hybrid: seed 0 | Hybrid: all seeds |
|---|---:|---:|---:|---:|
| **All validation** | **70** | 0.7567 | **0.8089** | **0.8017** |
| NEXAR | 15 | 0.5004 | **0.7438** | **0.7104** |
| AIHub | 18 | **0.7738** | 0.7738 | 0.7738 |
| CCD | 17 | **0.7950** | 0.7950 | 0.7950 |
| MM-AU | 20 | **0.8902** | 0.8902 | 0.8902 |
| All non-NEXAR | 55 | **0.8257** | 0.8257 | 0.8257 |

The all-seed hybrid gets **11/15 NEXAR ENTRY hits and 10/15 COLLISION hits**; original P2 gets **5/15 and 7/15**. The hybrid's overall ENTRY and COLLISION accuracies are 0.7286 and 0.8714; original P2 gets 0.6429 and 0.8286. Side macro-F1 is 0.8998 versus 0.9000, and evasion macro-F1 is 0.7116 for both.

All specialist seed variants were retained. The same switch with specialist seeds 0, 1 and 2 yields NEXAR 0.7438 / 0.6295 / 0.6461 and overall 0.8089 / 0.7883 / 0.7873. Only seed 0 and the all-seed ensemble clear the stated NEXAR target; all variants clear the overall target. This makes the **all-seed hybrid** the less seed-dependent result, despite its lower best-case score.

## Checks and limitations

- `evaluate.py` composes existing saved predictions by the fixed frame-count rule and stores full per-video results and source metrics in `results.json` and `predictions_*.json`.
- `runtime.Hybrid` executes the same rule on the actual cached DINO features. `verify_runtime.py` confirmed **70/70 exact prediction parity** with the saved hybrid results. The runtime accepts the *native* clip frame count separately from the 128–320 adaptively sampled frame numbers; confusing these would silently route long clips to P2.
- This experiment does not test an independent validation split. Source and duration are strongly confounded, the NEXAR specialists were previously selected using the same 15 validation clips, and the >500 rule perfectly separates validation sources. The preserved non-NEXAR scores follow from routing, not from learning a shared robust model.
- The real leaderboard result suggests its clip construction or domain differs from the NEXAR validation slice. A hidden long clip unlike NEXAR will still be routed to the specialist and may lose performance. Do not predict leaderboard improvement from the offline 0.8017 score.
- No existing submission or checkpoint was replaced, and no new ZIP was made from this research candidate.
