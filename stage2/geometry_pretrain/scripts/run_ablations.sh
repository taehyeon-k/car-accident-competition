#!/usr/bin/env bash
# Two ablation lanes in parallel; each run is followed by its held-out geometry eval.
cd /workspace/car-accident
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
lane() { for n in "$@"; do
  python -m stage2.geometry_pretrain.train --config stage2/geometry_pretrain/configs/abl_$n.yaml > /workspace/outputs/geometry_pretrain/logs/abl_$n.log 2>&1
  python -m stage2.geometry_pretrain.evaluate geometry --run-dir /workspace/outputs/geometry_pretrain/runs/abl_$n > /workspace/outputs/geometry_pretrain/logs/eval_abl_$n.log 2>&1
done; }
lane full4k no_depth bdd_only &
lane no_flow no_road bdd_tusimple &
wait
