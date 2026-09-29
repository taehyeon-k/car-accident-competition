#!/bin/bash
set -euo pipefail
source /venv/main/bin/activate
cd /workspace/car-accident
run=/workspace/runs/stage3_tcp_transfer/C0_control
if [[ -s "$run/metrics.json" ]]; then
  echo 'C0 already complete.'
  exit 0
fi
args=()
if [[ -s "$run/last.pt" ]]; then
  args+=(--resume)
else
  args+=(--base stage3/experiments/tcp_transfer/control_base.yaml)
fi
exec python -m stage3.experiments.run \
  --name C0_control --root /workspace/runs/stage3_tcp_transfer "${args[@]}"
