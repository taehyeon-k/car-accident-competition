#!/bin/bash
set -euo pipefail

cd /workspace/car-accident
export PYTHONDONTWRITEBYTECODE=1

python -m stage2.external.collect_metadata
python -m stage2.external.normalize_filter
python -m stage2.external.download_selected_videos --source causalcrash

if [[ "${1:-}" == "--with-mmau-cap-1-10" ]]; then
  df -h /workspace
  du -sh /workspace/data/* 2>/dev/null | sort -h
  python -m stage2.external.download_mmau_archive --archive CAP-DATA_chunks/1-10
fi

python -m stage2.external.validate_videos
python -m stage2.external.deduplicate_candidates
python -m stage2.external.prepare_labeling
python -m stage2.external.build_report
