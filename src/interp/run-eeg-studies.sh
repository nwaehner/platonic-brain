#!/usr/bin/env bash
# Run the EEG-feature study suite sequentially.
#
#   [PYTHON=/path/to/python] ./src/interp/run-eeg-studies.sh [--smoke]
#
# Pipeline:
#   1. eeg_features.py          — extract NICE features per subject (heavy; skipped if
#                                  outputs already exist, or run separately per subject)
#   2. combine_eeg_features.py  — average across subjects
#   3. eeg_feature_alignment.py — cross-arch alignment vs NICE decodability → summary.npz
#   4. eeg_crossmodal_decodability.py — EEG ↔ vision/LLM alignment → plots
#   5. plot_eeg_perfeature.py   — per-feature scatter grids (reads summary.npz)
#
# For step 1 (raw EEG download per subject), see eeg_features.py --help.
# Steps 2-5 are fast enough to run locally without a GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
echo "repo: $ROOT"
echo "python: $("$PY" -c 'import sys; print(sys.executable)')"

SMOKE=""
if [ "${1:-}" = "--smoke" ]; then
  SMOKE="--smoke"
  echo "===== EEG studies — SMOKE ====="
else
  echo "===== EEG studies — FULL ====="
fi

# ── Step 2: combine per-subject NICE features (idempotent, fast) ──────────────
echo "=== Step 2: combine EEG features ==="
"$PY" src/interp/eeg_features/combine_eeg_features.py

# ── Step 3: cross-arch alignment + decodability → summary.npz ─────────────────
echo "=== Step 3: eeg_feature_alignment ==="
"$PY" src/interp/eeg_features/eeg_feature_alignment.py $SMOKE

# ── Step 4: cross-modal decodability (vision + LLM) ───────────────────────────
echo "=== Step 4: eeg_crossmodal_decodability ==="
"$PY" src/interp/eeg_features/eeg_crossmodal_decodability.py $SMOKE

# ── Step 5: per-feature scatter grids ─────────────────────────────────────────
if [ -z "$SMOKE" ]; then
  echo "=== Step 5: plot_eeg_perfeature ==="
  "$PY" src/interp/eeg_features/plot_eeg_perfeature.py
fi

echo "===== EEG studies done. Outputs in src/interp/outputs/eeg_alignment/ ====="
