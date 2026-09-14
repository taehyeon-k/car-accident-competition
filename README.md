# Stage 2 accident-event localizer

This repository implements the Stage 2 joint accident-event architecture: cached
DINOv3 local features plus an online frozen V-JEPA 2.1 global branch, with
RF-DETR detections and Hungarian tracking supplying nine bbox/tracking geometry
channels per object. The live design is in [stage2/JOINT.md](stage2/JOINT.md).
See the [workflow](stage2/README.md). The earlier coarse-to-fine cascade and its
Depth Anything branch were removed on 2026-09-14;
[stage2/Stage2_Architecture.md](stage2/Stage2_Architecture.md) and the
[implementation audit](stage2/IMPLEMENTATION_AUDIT.md) are kept as history.
Native training/inference logic is CPU-tested. Local asset wiring and GPU
smoke-test instructions are in [the workspace guide](stage2/WORKSPACE.md).

The code is organized like `Diffusion-ISP`:

```text
stage2/
  configs/       Training configuration
  data/          Dataset, frame sampling, coordinate transforms
  model/         Backbones, tracking, geometry, learned model heads
  trainer/       Accelerate training loop and checkpoints
  utils/         Loss functions and support code
  tests/         CPU-only tensor and invariant tests
```

## Local weights and data

Weights and datasets are excluded from Git. Put them in local `weights/` and `data/`
directories, then configure their paths in `stage2/configs/joint.workspace.yaml`.

Large artifacts can be versioned deliberately with Git LFS because `.gitattributes`
defines the expected model-file patterns.

## Version-control handoff

Review the staged changes before committing:

```bash
git status
git add .gitignore .gitattributes .dockerignore README.md requirements.txt requirements-dev.txt pyproject.toml Dockerfile stage2
git diff --cached --check
git commit -m "Implement the Stage 2 joint localizer"
git push -u origin <branch-name>
```

Do not commit private datasets, experiment outputs, or unreviewed pretrained weights.
