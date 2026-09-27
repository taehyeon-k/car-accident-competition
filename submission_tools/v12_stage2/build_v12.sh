#!/bin/bash
# Build submit_v8: Stage 1 + Stage 3 (V3) + DINOv3 vendor/backbone from the specialist zip (Stage 1 0.953), with
# model/stage2 replaced by the v8 robust full-data ensemble (stride-augmented E4 + E2 + XN4, 4 seeds each; runtime from v6_stage2) and
# Stage 3 decoder overrides steering 3.0 deg / acceleration +-0.5.
# Output: $DIR (unzipped submission folder) and $OUT (zip with files at the zip root).
set -e
cd /workspace/car-accident; source /venv/main/bin/activate >/dev/null
NAME=${NAME:-submit_v12_XSbU-E4E2XN4+XSbUB-E4_V3_acc0p4_steer7}
BASE=${BASE:-/workspace/outputs/submit_nexar_specialist_steer5_accel0p5.zip}
DIR=${DIR:-/workspace/outputs/$NAME}; OUT=${OUT:-/workspace/outputs/$NAME.zip}
V8=submission_tools/v12_stage2
rm -rf "$DIR"; mkdir -p "$DIR"
unzip -q "$BASE" -d "$DIR"
S2=$DIR/model/stage2
# keep only the shared Stage 2 assets (DINOv3 vendor code, backbone.pth, sampling.py); everything model-specific is replaced
find "$S2" -mindepth 1 -maxdepth 1 ! -name vendor ! -name backbone.pth ! -name sampling.py -exec rm -rf {} +
cp $V8/{runtime.py,aux_models.py,lc_models.py,config.json,README.txt} "$S2/"; cp /workspace/pretrained/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth "$S2/backbone_vitb.pth"
# Stage 3: V3 unchanged; decoder thresholds steering 7.0 deg, acceleration +-0.4
cp $V8/stage3_decoder_overrides.json "$DIR/model/stage3/decoder_overrides.json"
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
