#!/bin/bash
set -euo pipefail

root="${STAGE2_EXTERNAL_ROOT:-/workspace/data/stage2_external}"
remote="r2:car-accident-dataset/stage2/external"

rclone copy "${root}/metadata" "${remote}/metadata" --checksum --s3-no-check-bucket --progress
rclone copy "${root}/manifests" "${remote}/manifests" --checksum --s3-no-check-bucket --progress
rclone copy "${root}/raw_selected/mmau" "${remote}/mmau/videos" --checksum --s3-no-check-bucket --progress
rclone copy "${root}/raw_selected/causalcrash" "${remote}/causalcrash/videos" --checksum --s3-no-check-bucket --progress
rclone copy "${root}/labeling/annotations.jsonl" "${remote}/annotations" --checksum --s3-no-check-bucket --progress
rclone copy "${root}/labeling/queue.jsonl" "${remote}/annotations" --checksum --s3-no-check-bucket --progress
rclone copy "${root}/reports" "${remote}/reports" --checksum --s3-no-check-bucket --progress
rclone copy /workspace/car-accident/stage2/external "${remote}/scripts" --include '*.py' --include '*.json' --include '*.sh' --include '*.conf' --include 'README.md' --s3-no-check-bucket --progress

for pair in \
  "${root}/metadata|${remote}/metadata" \
  "${root}/manifests|${remote}/manifests" \
  "${root}/raw_selected/mmau|${remote}/mmau/videos" \
  "${root}/raw_selected/causalcrash|${remote}/causalcrash/videos" \
  "${root}/reports|${remote}/reports"
do
  local_path="${pair%%|*}"
  remote_path="${pair#*|}"
  [[ -d "${local_path}" ]] || continue
  rclone check "${local_path}" "${remote_path}" --one-way --size-only
done
rclone check "${root}/labeling" "${remote}/annotations" --one-way --size-only --include 'annotations.jsonl' --include 'queue.jsonl'
rclone check /workspace/car-accident/stage2/external "${remote}/scripts" --one-way --size-only --include '*.py' --include '*.json' --include '*.sh' --include '*.conf' --include 'README.md'

python - "${root}" "${remote}" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
status = {
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "remote": sys.argv[2],
    "verification": "rclone check --one-way --size-only passed for metadata, manifests, videos, annotations, reports, and pipeline scripts",
}
path = root / "reports/r2_backup_status.json"
path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
PY
rclone copyto "${root}/reports/r2_backup_status.json" "${remote}/reports/r2_backup_status.json" --checksum --s3-no-check-bucket --progress
