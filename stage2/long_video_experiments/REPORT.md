# Stage 2 long-video experiments — 2026-09-23

## Scope

The 349-video manifests (279 train, 70 validation) and existing frozen
geometry-DINOv3-S feature cache were used. No backbone was trained. All model
selection used the FPS-blind 1%-of-clip validation score. The ±0.3-second
competition-style metric below is an offline diagnostic computed after predictions.
Every result is one seed on the same split; score differences are not confirmed
across folds or leaderboard submissions.

The new tiled models operated on **192 cached positions**, not all native frames.
Consequently they test whole-cache search and local/global temporal modeling,
not the proposed full-native-frame implementation.

| Model and view | Selected epoch | FPS-blind selection | Offline score | ENTRY | COLLISION | Side F1 | Evasion F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing soft target, 128 | 14 | 0.4643 | **0.6893** | 0.557 | **0.814** | 0.828 | 0.568 |
| Same checkpoint, 192 | 14 | 0.4199 | 0.6749 | 0.571 | 0.743 | 0.857 | 0.576 |
| Tiled local 32 / step 16, 192 | 15 | 0.4194 | 0.6694 | 0.557 | 0.743 | 0.753 | 0.676 |
| Tiled local 64 / step 32, 192 | 9 | 0.4083 | 0.6483 | 0.529 | 0.729 | 0.726 | 0.663 |
| Exact ENTRY and COLLISION, 128 | 6 | **0.4749** | 0.6849 | **0.571** | 0.743 | **0.886** | 0.614 |
| Exact ENTRY, soft COLLISION, 128 | 28 | 0.4683 | 0.6733 | 0.543 | 0.729 | 0.843 | **0.679** |

### NEXAR validation (15 clips)

| Model | ENTRY hits | COLLISION hits | Median ENTRY error | Median COLLISION error |
|---|---:|---:|---:|---:|
| Existing soft, 128 | 5 | **10** | 41 frames | **5 frames** |
| Existing soft, 192 | 6 | 6 | 21 frames | 11 frames |
| Tiled 32, 192 | 5 | 7 | 18 frames | 11 frames |
| Tiled 64, 192 | 5 | 6 | 17 frames | 13 frames |
| Exact both, 128 | **7** | 8 | 13 frames | 6 frames |
| Exact ENTRY / soft COLLISION, 128 | **7** | 6 | **12 frames** | 17 frames |

The 128-position cache covers the labeled ENTRY within the offline tolerance
for all 15 NEXAR validation clips; nearest cached-frame error is at most six
frames. The main failure is selecting the right event from the long clip,
not a hard sampling-resolution ceiling. Length and source are confounded:
NEXAR is the only consistently long source here.

## Interpretation

- The two tiled prototypes did not improve overall validation or NEXAR event
  hit rates. Increasing tile length worsened overall score. This is evidence
  against these particular heads on the sparse cache, not against a dense
  native-frame model that has not been trained.
- Replacing the 1.5%-of-video soft target with an exact sampled-position target
  improved NEXAR ENTRY from 5/15 to 7/15 and median error from 41 to 13 frames,
  but reduced NEXAR COLLISION from 10/15 to 8/15. The total score stayed within
  0.005 of the baseline. Mixed targets retained the ENTRY gain but damaged
  COLLISION further; they are not a supported improvement.
- Training loss kept falling while validation fluctuated. These single-seed
  differences, especially two clips in a 15-clip source slice, need confirmation.

## Resource isolation

New code and artifacts are confined to `stage2/long_video_experiments/`.
The existing feature cache was read only. Tiled runs peaked at 587 MiB of
PyTorch allocated GPU memory; exact/mixed runs at 616 MiB. GPU runs used a
4%-of-total per-process allocator cap and required at least 2 GiB free at
startup. Their CUDA context/reservation consumed more than the PyTorch peak.
One baseline 192-position GPU evaluation exited with an out-of-memory error
when another job's memory rose; it wrote no result. The comparison was then
completed on CPU. The active Stage 3 processes remained running and their
code/checkpoints were not changed by these experiments.

## Next decision

Keep the existing soft-target checkpoint as the current overall baseline. For
NEXAR-specific research, exact ENTRY is a useful lead, but no tested model
justifies replacing the submission. Before another full training run, inspect
the NEXAR false-event peaks and test a genuinely dense native-frame or
high-recall candidate-refinement design. Pre-register source-specific metrics
and compare over multiple seeds or grouped folds.
