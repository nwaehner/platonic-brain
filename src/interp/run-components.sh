#!/usr/bin/env bash
# Run the interp MEDIATION PIPELINE (Components 1–5) sequentially.
#
#   [PYTHON=/path/to/python] ./src/interp/run-components.sh [--smoke]
#
# Uses the active venv's `python` (or a PYTHON= override). Component 1 builds the feature
# caches that Components 2–5 read, so order matters → sequential. Component 1 needs torch
# (CLIP); without it the CLIP tier is skipped and the rest still runs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
echo "repo: $ROOT"
echo "python: $("$PY" -c 'import sys; print(sys.executable)')"

if [ "${1:-}" = "--smoke" ]; then
  echo "===== Components 1–5 — SMOKE ====="
  "$PY" src/interp/run_interp_smoke.py
  exit 0
fi

echo "===== Component 1: feature stacks (all grids) ====="
"$PY" src/interp/components/interp_features.py --grid all
echo "===== Component 2: neighbourhood variance ratio ====="
"$PY" src/interp/components/neighbour_enrichment.py
echo "===== Component 3: feature linear-probing (EEG↔vision) ====="
"$PY" src/interp/components/feature_linear_probing.py
echo "===== Component 5: vision↔language probing (4 s) ====="
"$PY" src/interp/components/vis_lang_probing.py
echo "===== Component 4: caption variance ratio ====="
"$PY" src/interp/components/caption_similarity.py
echo "===== Components 1–5 done. Outputs in src/interp/outputs/ ====="
