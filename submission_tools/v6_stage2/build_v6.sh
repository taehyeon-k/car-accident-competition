#!/bin/bash
# Build submit_v6: Stage 1 + Stage 3 (V3, steer 5 deg, accel +-0.5) + DINOv3 vendor/backbone from
# /workspace/outputs/submit_nexar_specialist_steer5_accel0p5.zip (Stage 1 0.953, Stage 3 LB 0.7388), with model/stage2 replaced by
# the event-specific ensemble (stage2/aux_signal_experiments/logs/event_select_7.out, CV 0.7884 / NEXAR 0.7103).
# Output: $DIR (unzipped submission folder) and $OUT (zip with files at the zip root).
set -e
cd /workspace/car-accident; source /venv/main/bin/activate >/dev/null
NAME=${NAME:-submit_v6_ES7-auxens_V3_acc0p5_steer5}
BASE=${BASE:-/workspace/outputs/submit_nexar_specialist_steer5_accel0p5.zip}
DIR=${DIR:-/workspace/outputs/$NAME}; OUT=${OUT:-/workspace/outputs/$NAME.zip}
V6=submission_tools/v6_stage2
rm -rf "$DIR"; mkdir -p "$DIR"
unzip -q "$BASE" -d "$DIR"
S2=$DIR/model/stage2
# keep only the shared Stage 2 assets (DINOv3 vendor code, backbone.pth, sampling.py); everything model-specific is replaced
find "$S2" -mindepth 1 -maxdepth 1 ! -name vendor ! -name backbone.pth ! -name sampling.py -exec rm -rf {} +
cp $V6/{runtime.py,aux_models.py,config.json,README.txt} submission_tools/v5_stage2/lc_models.py "$S2/"
python - "$S2" <<'PY'
import json, sys, torch
from pathlib import Path
s2 = Path(sys.argv[1]); (s2 / "members").mkdir()
for m in json.loads((s2 / "config.json").read_text())["members"]:
    st = torch.load(m["source"], map_location="cpu", weights_only=False)
    torch.save({k: v.detach().cpu().clone() for k, v in st["model"].items()}, s2 / m["file"])
PY
rm -f "$DIR/requirements.py"
find "$DIR" -name __pycache__ -prune -exec rm -rf {} +
rm -f "$OUT"; (cd "$DIR" && zip -q -r -X "$OUT" inference.py requirements.txt model)
echo "folder: $DIR"; echo "zip:    $OUT ($(du -h "$OUT" | cut -f1))"
