# Interim DriveDNA validation comparison

Both EMA checkpoints evaluated on the same 1,008 held-out DriveDNA clips (302,408 frames). Combined checkpoint: best snapshot from epoch 76; baseline: saved BATON-only best checkpoint. Decoder: acceleration ±0.25 m/s² and steering 5°. GPU bf16 evaluation, batch 1, CNN chunk 16, no loader workers.

| Metric | BATON only | BATON + DriveDNA | Difference |
|---|---:|---:|---:|
| competition_score | 0.750263 | 0.751136 | +0.000873 |
| acceleration_macro_f1 | 0.712174 | 0.719629 | +0.007454 |
| steering_macro_f1 | 0.839137 | 0.824652 | -0.014485 |

The combined score improves by only 0.000873 (0.087 percentage points). Acceleration improves, while steering declines. This single-split, single-seed comparison does not establish reliable overall superiority. It is an interim result for the frozen epoch-76 checkpoint, not the final 100-epoch model.

Evaluation runtimes: baseline 461.8 s, combined 444.0 s. The separate evaluator used approximately 1.1 GB GPU memory; training plus evaluation stayed around 13 GB at observed checks. The training process was not stopped or restarted.

Machine-readable scores and confusion matrices: `baseline.json`, `combined.json`, and `comparison.json` in this directory.
