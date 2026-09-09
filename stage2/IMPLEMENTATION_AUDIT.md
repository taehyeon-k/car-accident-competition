# Architecture and original-plan audit

Historical audit: the local integrations and single-GPU smoke checks described as
outstanding below are now covered by [WORKSPACE.md](WORKSPACE.md). Full-data
training, convergence, distributed operation, and submission validation remain separate.

## Conclusion

The earlier implementation was incomplete. Prepared-feature datasets and generic
factory adapters did not implement the planned native-frame workflow. The missing
`cache_geometry.py` was a genuine gap, not just a naming difference. This review
adds that workflow and fixes the errors below, but local pretrained integrations
remain external and unverified.

## Original plan to actual code

| Responsibility | Implementation after review |
| --- | --- |
| YAML entry points and training | `run.py`, `configs/`, `trainer/trainer.py` |
| Native manifests and source splits | Added `scripts/prepare_manifest.py`; `data/dataset.py` |
| Frozen detector/depth cache | Added `data/cache_geometry.py` |
| Window-specific tracking and ROIs | Added `data/preparation.py`; `model/tracking.py`, `geometry.py` |
| Training-only geometry normalization | Added `data/geometry_stats.py`; buffers in `GeometryMLP` |
| Crop, rate augmentation, bins, fine windows | `data/sampling.py`, `data/dataset.py` |
| Clip-consistent photometric transforms | `data/transforms.py` using torchvision |
| Coarse/fine heads and LoRA | `model/model.py`, `modules.py`, `lora.py`, `pipeline.py` |
| End-to-end native inference | Added `model/inference.py`; connected `test.py` |
| Losses, metrics, checkpoint/RNG state | `utils/losses.py`; added `metrics.py`, `checkpoint.py`, `utils.py` |
| Export and Stage 2 benchmarking | Added `scripts/export_submission.py`, `scripts/benchmark.py` |
| Local visual-backbone contract check | Added `scripts/smoke_test_backbones.py`; not run with weights |
| Vendored pretrained implementations | Still external; `third_party/README.md` documents contracts |

The planned `model_utils.py` responsibilities are split across `modules.py`,
`data/preparation.py`, and `data/transforms.py`; no redundant file is needed.
Inference reads configuration from checkpoints instead of a shared `config.yaml`.

## Logical corrections

- Prevent all-masked Transformer NaNs and `0 * -inf` fine losses; clear padded
  tokens and exclude padded DINO frames from encoding.
- Restrict LoRA to the last four attention blocks and validate all projections
  before mutation. Reject incomplete base checkpoint loading.
- Fix coarse positional encoding to the fixed tubelet grid, independently
  initialize cloned Transformer layers, and make ROIAlign autocast-safe.
- Fit separate coarse/fine geometry statistics using observed training objects
  only; checkpoint normalization and validate training source membership.
- Apply crop → FPS augmentation → coarse bins. Fine training reopens consecutive
  native frames with the specified perturbed-bin sampling distribution.
- Preserve original filename IDs through windows and overlap-logit averaging,
  including noncontiguous IDs.
- Reject train/validation source leakage; use standard scalar-label collation
  and per-sample masked losses.
- Correct incomplete gradient-accumulation sample weighting and scheduler update
  counts after loader sharding; save RNG and scaler state for resume.

## Computational corrections

- Cache compact per-detection frozen geometry, not full depth maps or
  context-dependent tracks. Compute depth median/MAD once per frame.
- Vectorize association costs and use SciPy Hungarian assignment with active tracks.
- Chunk only valid DINO frames. This limits forward workspace, not all saved
  activations required for training gradients.
- Share frozen observations/DINO features across overlapping inference windows;
  recompute window-specific tracks and release features after their last consumer.
- Use torchvision resize/normalize/ROIAlign and PyTorch Transformer/DataLoader
  primitives. Retain custom logic for architecture-specific temporal coupling.

No GPU profiling was performed; these eliminate identifiable redundant work but
do not establish the fastest possible implementation on a particular GPU.

## Verification and remaining work

CPU tests cover actual heads/backward passes, padded losses, all-masked attention,
BF16 ROIAlign, strict checkpoint rejection, LoRA targeting/merge equivalence,
cached-vs-direct geometry, sampling, native-ID inference and unique-frame reuse.
Accumulation is tested with a minimal CPU accelerator double, not actual DDP.

Before claiming production training or submission readiness:

1. Supply and verify all four local pretrained implementations/checkpoints,
   including RF-DETR compatibility and depth orientation. Factories are integration
   boundaries, not vendored backends.
2. Check real Accelerate, distributed resume, mixed-precision GPU training and
   final-container offline execution. CPU tests cannot establish these.
3. Evaluate coarse-selected fine windows on real data. Fine trainer validation
   uses ground-truth-centered windows to isolate the fine model; its metrics are
   not full-pipeline localization accuracy.
4. Integrate official submission schema/metrics and Stage 1/3 resource budgets.
   The benchmark currently measures Stage 2 only.
5. Assess the spec's short-clip padding tradeoff: repeated RGB still enters V-JEPA
   tubelets despite downstream masks. The specified sampling policy is retained,
   rather than silently changing the architecture.

New checkpoints use format version 2; old prototype states are not silently
treated as compatible. Load only trusted full training checkpoints.

Previously tracked bytecode was preserved. `.gitignore` prevents new artifacts
but cannot retroactively untrack existing Git entries.
