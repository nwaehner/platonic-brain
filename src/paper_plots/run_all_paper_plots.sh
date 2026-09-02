#!/usr/bin/env bash
# run_all_paper_plots.sh — compute the three cached artifacts, then render all three blocks.
#
# The compute steps need the EEG/LLM embeddings from HuggingFace (token in tokens/hf_token.txt);
# once the _cache/*.npz exist, the block_*.py steps are fully offline and take seconds.
#
# Env:
#   SMOKE=1        femba+luna only, 2 LLM, K=50  (fast sanity; uses the locally-cached EEG)
#   K=<n>          permutations for calibration (default 100; SMOKE forces 50)
#   PLOTS_ONLY=1   skip the compute steps, just re-render from existing _cache/*.npz
#   HF_HOME=<dir>  HF cache root (defaults to the repo's .hf_cache)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PY:-$ROOT/.venv-interp/bin/python}"; [ -x "$PY" ] || PY=python3
export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"

SMOKE_FLAG=""; K="${K:-100}"
if [ "${SMOKE:-0}" = "1" ]; then SMOKE_FLAG="--smoke"; K=50; fi

cd "$ROOT"
if [ "${PLOTS_ONLY:-0}" != "1" ]; then
  echo "== compute BERT semantic targets =="
  "$PY" src/paper_plots/compute_bert_semantic.py
  echo "== compute decodability (nested subject x time CV, all tiers) =="
  "$PY" src/paper_plots/compute_decodability.py $SMOKE_FLAG
  echo "== compute cross-modal LLM mKNN+CKA (K=$K) =="
  "$PY" src/paper_plots/compute_crossmodal_llm.py $SMOKE_FLAG --K "$K"
  echo "== compute intramodal mKNN+CKA (K=$K) =="
  "$PY" src/paper_plots/compute_intramodal.py $SMOKE_FLAG --K "$K"
fi

echo "== block (a) LLM -> EEG ==";        "$PY" src/paper_plots/block_a.py
echo "== block (b) intra-EEG ==";         "$PY" src/paper_plots/block_b.py
echo "== block (c) cross-modal ANCOVA =="; "$PY" src/paper_plots/block_c.py
echo "== block (b) per-feature ==";        "$PY" src/paper_plots/block_b_disentangled.py
echo "== size vs alignment ==";            "$PY" src/paper_plots/block_size.py
echo "Done. Figures under data/paper-plots/"
