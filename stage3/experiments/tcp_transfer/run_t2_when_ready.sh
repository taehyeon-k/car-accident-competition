#!/bin/bash
set -euo pipefail
source /venv/main/bin/activate
cd /workspace/car-accident

t1=/workspace/runs/stage3_tcp_transfer/T1_frozen_tcp/metrics.json
while [[ ! -s "$t1" ]]; do
  sleep 30
done

run=/workspace/runs/stage3_tcp_transfer/T2_tcp_projection
if [[ -s "$run/metrics.json" ]]; then
  echo 'T2 already complete.'
  exit 0
fi
args=(--base stage3/experiments/tcp_transfer/t2_tcp_projection.yaml)
if [[ -s "$run/last.pt" ]]; then
  args=(--resume)
fi
exec python -m stage3.experiments.tcp_transfer.run_tcp \
  --name T2_tcp_projection \
  "${args[@]}" \
  --root /workspace/runs/stage3_tcp_transfer
