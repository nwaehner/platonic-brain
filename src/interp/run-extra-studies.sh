#!/usr/bin/env bash
# Run the four INDEPENDENT interp studies sequentially.
#
#   [PYTHON=/path/to/python] ./src/interp/run-extra-studies.sh [--smoke]
#
# Uses the active venv's `python` (or a PYTHON= override). model_stitching / attribute_probing
# are run once per distinct EEG grid (--eeg-ref). attribute_probing needs the Component-1
# feature cache for that grid, so each ref builds it first (interp_features.py --grid <ref>;
# that step needs torch/CLIP — without it the high-level-visual tier is simply absent).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
REFS=(femba neurolm reve steegformer)        # one --eeg-ref per distinct grid (femba covers luna)
echo "repo: $ROOT"
echo "python: $("$PY" -c 'import sys; print(sys.executable)')"

if [ "${1:-}" = "--smoke" ]; then
  echo "===== extra studies — SMOKE ====="
  "$PY" src/interp/studies/intrinsic_dimensionality.py --modalities eeg --models reve \
        --no-local --max-windows 300 --layer-stride 4
  "$PY" src/interp/studies/high_alignment.py --eeg-model reve --vision-arch videomae \
        --llm bloomz-560m --no-video
  "$PY" src/interp/studies/model_stitching.py --eeg-ref reve --no-layer-heatmap
  "$PY" src/interp/studies/attribute_probing.py --eeg-ref reve --smoke \
        --models reve dinov2 bloom --n-layers 3
  echo "===== extra studies SMOKE done ====="
  exit 0
fi

echo "===== intrinsic dimensionality (all models) ====="
"$PY" src/interp/studies/intrinsic_dimensionality.py
echo "===== high alignment (EEG × vision × language) ====="
"$PY" src/interp/studies/high_alignment.py
for ref in "${REFS[@]}"; do
  echo "===== model stitching (eeg-ref=$ref) ====="
  "$PY" src/interp/studies/model_stitching.py --eeg-ref "$ref"
done
for ref in "${REFS[@]}"; do
  echo "===== attribute probing (eeg-ref=$ref) ====="
  "$PY" src/interp/components/interp_features.py --grid "$ref"   # ensure the Comp-1 cache exists
  "$PY" src/interp/studies/attribute_probing.py --eeg-ref "$ref"
done
echo "===== extra studies done. Outputs in src/interp/outputs/ ====="
