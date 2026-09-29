#!/bin/bash
set -euo pipefail
source /venv/main/bin/activate
cd /workspace/car-accident
exec python -m stage3.experiments.tcp_transfer.cache_tcp \
  --manifest /workspace/data/stage3/manifests/all.jsonl \
  --output /workspace/cache/stage3/tcp_repro_rgb_v2 \
  --batch 16 --decoders 4
