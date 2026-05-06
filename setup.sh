#!/usr/bin/env bash
# Install all dependencies needed by the extract_*.py scripts.
#
# Creates a local virtual environment in ./venv/ to avoid the PEP 668
# "externally-managed-environment" restriction on modern Debian/Ubuntu.
#
# Usage:
#   bash setup.sh
#   source venv/bin/activate         # before running any extract_*.py
#
# Notes:
#   - mamba-ssm (FEMBA only) compiles CUDA kernels at install time, so it must
#     be installed AFTER torch and with --no-build-isolation so it can find
#     the already-installed torch. Requires CUDA toolkit + nvcc on the host.
#   - If you don't need FEMBA, comment out the mamba-ssm line.

set -e

# 1. Create venv if it doesn't already exist
if [ ! -d "venv" ]; then
    echo "[0/3] Creating virtual environment in ./venv/ ..."
    python3 -m venv venv
fi

# 2. Activate venv for this script's process
# shellcheck disable=SC1091
source venv/bin/activate

echo "[1/3] Upgrading pip inside venv..."
pip install --upgrade pip

echo "[2/3] Installing standard dependencies from requirements.txt..."
pip install -r requirements.txt

echo "[3/3] Installing mamba-ssm + causal-conv1d (for FEMBA, needs CUDA toolkit)..."
if command -v nvcc >/dev/null 2>&1; then
    pip install mamba-ssm causal-conv1d --no-build-isolation
else
    echo "  nvcc not found — skipping mamba-ssm (FEMBA will not be available)."
    echo "  To enable FEMBA, install the CUDA toolkit matching your PyTorch CUDA"
    echo "  version, then run: pip install mamba-ssm causal-conv1d --no-build-isolation"
fi

echo ""
echo "Done. Sanity check:"
python -c "import torch, numpy, scipy, transformers, huggingface_hub, mne, cv2; \
print('  torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

echo ""
echo "================================================================"
echo "Activate the venv before running scripts:"
echo "    source venv/bin/activate"
echo "Then:"
echo "    python extract_reve.py    # etc."
echo "================================================================"
