# Source-adversarial Stage 2 transfer pilot — rejected

**Hypothesis.** An E4_sa head that cannot easily encode which of AIHUB, CCD, MMAU, or NEXAR a training clip came from may transfer better to an unseen 10 Hz source. Source ID is a training-only auxiliary target. The deployed model would still use the same frozen DINO features, both motion inputs, FPS-blind sampler, and direct decoder.

**Controlled pilot.** Fold0 seed0, 279 training clips and 70 validation clips, E4_sa's 0.5/0.25/0.25 native/half/third stride mix, 30-epoch ceiling and patience7. A 64-unit classifier reads mean-pooled hidden states; its cross-entropy has weight 0.2, and gradient reversal has scale 0.2. The one CAUSALCRASH training clip is ignored by this auxiliary loss. Other Stage 2/3 runs and source files were not modified. The model checkpoint was selected at epoch9 by the standard FPS-blind validation criterion; all-70 official score was 0.7622. Evaluation below excludes the known duplicate clips, leaving 54.

| Duplicate-clean fold0 | Native | Half rate | Third rate |
|---|---:|---:|---:|
| Matched E4_sa parent | 0.7387 | 0.7346 | 0.6990 |
| Source-adversarial pilot | 0.7448 | 0.7236 | 0.6567 |
| Difference | +0.0061 | −0.0109 | **−0.0423** |

At one-third rate, collision accuracy falls 0.8889 → 0.7778, while ENTRY changes 0.4630 → 0.4815. CCD and NEXAR ENTRY do not improve at that rate (CCD 0.2143 → 0.2143; NEXAR 0.5000 → 0.4375). This fails the transfer mechanism and the rate-robustness requirement. **Do not tune more reversal weights or promote this model.** One fold cannot establish a population effect, but the targeted rate regression is large enough to stop this pilot.

The available DACON Stage 2 sample contains only five CCD clips with collision labels; they are already represented in the local CCD source, and ENTRY/side/evasion labels are absent. It is therefore not an independent hidden-like validation set. The 349 labelled raw videos are present, but the previously documented MM-AU/CCD pseudo-labelled expansion worsened real leaderboard results. A credible next data experiment needs genuinely new, independently annotated clips or adjudication of training-label conventions, kept separate from model selection.

Reproduction: `STAGE2_SOURCE_REVERSE=0.2 STAGE2_SOURCE_LOSS=0.2 python -m stage2.robust_goal_20260927.source_invariant --run-id SOURCE_GRL_A --seed 0 --motion both --stride-aug 0.5,0.25,0.25 --train-split stage2/long_context_v2_experiments/folds/fold0_train.jsonl --val-split stage2/long_context_v2_experiments/folds/fold0_val.jsonl --output stage2/robust_goal_20260927/source_invariant/cv/fold0_seed0`. Evaluate with `python -m stage2.robust_goal_20260927.eval_source_invariant`. Scores: `fold0_robust.json`.
