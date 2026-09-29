#!/bin/bash
set -euo pipefail
cd /workspace/car-accident
source /venv/main/bin/activate
python -m stage3.scripts.cache_drivedna_parallel \
  --manifest /workspace/data/stage3/baton_drivedna_final/manifests/drivedna_all.jsonl \
  --config runs/stage3_v2/V3_tcnssm_100ep/config.yaml \
  --raw-dir /workspace/data/stage3/drivedna/raw \
  --filter-report /workspace/data/stage3/drivedna/filtered/filter_report.json \
  --selection-report /workspace/data/stage3/drivedna/filtered/selection_report.json \
  --work-dir /workspace/data/stage3/baton_drivedna_final/cache_work \
  --workers 2 --max-vram-mb 19000
