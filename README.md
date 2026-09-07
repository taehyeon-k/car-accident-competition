# Stage 2 accident-event localizer

This repository implements the Stage 2 coarse-to-fine accident-event architecture.
The fixed design is in [stage2/Stage2_Architecture.md](stage2/Stage2_Architecture.md).

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
directories, then configure their paths in `stage2/configs/coarse.yaml` and
`stage2/configs/fine.yaml`.

Large artifacts can be versioned deliberately with Git LFS because `.gitattributes`
defines the expected model-file patterns.

## Version-control handoff

Review the staged changes before committing:

```bash
git status
git add .gitignore .gitattributes README.md requirements.txt Dockerfile stage2
git diff --cached --check
git commit -m "Implement Stage 2 coarse-to-fine localizer"
git push -u origin <branch-name>
```

Do not commit private datasets, experiment outputs, or unreviewed pretrained weights.
