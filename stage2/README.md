# Stage 2 training

This implements the fixed coarse-to-fine architecture in `Stage2_Architecture.md`.
The training code keeps the `Diffusion-ISP` shape: YAML configuration, `run.py`,
`data/`, `model/`, `trainer/`, and `utils/`.

Run either trainer from this directory:

```bash
python run.py --config configs/coarse.yaml
python run.py --config configs/fine.yaml
```

Each JSONL manifest row must include a `feature_path`. That path is a `torch.save`
dictionary containing labels plus fixed geometry. For coarse head-only verification it
contains `dense [16,24,24,768]`, `boxes_grid [16,12,4]`, `geometry [16,12,9]`,
`object_valid [16,12]`, `bin_valid [32]`, `entry_bin`, `collision_bin`,
`entry_side`, and `evasion`. Fine records use `global_tokens [64,384]`, `dense
[64,24,24,384]`, `boxes_grid [64,12,4]`, `geometry [64,12,9]`, `object_valid
[64,12]`, `time_valid [64]`, `event_type`, and `event_local_index`.

For actual LoRA training, store `coarse_rgb [3,32,384,384]` or `fine_rgb
[64,3,336,336]` instead of visual features, and set the corresponding local factory
and checkpoint path. A factory has the form `your_package.backbones:build_vjepa`
or `...:build_dino`; it constructs the architecture only and returns dense outputs
under the adapter contracts in `model/backbones.py`. Checkpoints are required to be
local. RF-DETR and depth outputs must be cached separately because they are frozen;
do not cache V-JEPA or DINO outputs while LoRA is active.

The detector/depth adapter is intentionally factory-based because the architecture
requires a locally vendored RF-DETR implementation compatible with the target
container. `model/backbones.py` rejects missing files and never downloads weights.
