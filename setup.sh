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
# Disk space notes:
#   - mamba-ssm and the HuggingFace model/dataset caches can be large.
#     If /root or /home is small, redirect caches before running scripts:
#       export HF_HOME=/scratch/.cache/huggingface
#       export TMPDIR=/scratch/tmp && mkdir -p $TMPDIR
#   - The apt nvidia-cuda-toolkit (for nvcc) needs ~7 GB on the root
#     partition. Free space there first if needed:
#       apt-get clean && apt-get -o dir::cache=/scratch/apt-cache install nvidia-cuda-toolkit
#
# Notes:
#   - torch is pinned to 2.5.1+cu124. mamba-ssm 2.x uses a c10::Warning
#     ABI that was removed in torch 2.6+, so do not upgrade torch without
#     also verifying mamba-ssm compatibility.
#   - mamba-ssm (FEMBA only) compiles CUDA kernels at install time. It must
#     be installed AFTER torch and causal-conv1d, with --no-build-isolation.
#     Requires CUDA toolkit + nvcc on the host.
#   - If you don't need FEMBA, you can skip the mamba-ssm step.

set -e

# 1. Create venv if it doesn't already exist
if [ ! -d "venv" ]; then
    echo "[0/5] Creating virtual environment in ./venv/ ..."
    python3 -m venv venv
fi

# 2. Activate venv for this script's process
# shellcheck disable=SC1091
source venv/bin/activate

echo "[1/5] Upgrading pip + wheel inside venv..."
pip install --upgrade pip wheel

echo "[2/5] Installing torch 2.5.1 + torchvision (cu124 wheels)..."
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

echo "[3/5] Installing standard dependencies from requirements.txt..."
pip install -r requirements.txt

echo "[4/5] Installing torcheeg (with --no-deps to avoid scipy 1.9 downgrade)..."
pip install torcheeg --no-deps

echo "[5/5] Installing mamba-ssm + causal-conv1d (for FEMBA, needs CUDA toolkit)..."
if command -v nvcc >/dev/null 2>&1; then
    # causal-conv1d must be installed first (mamba-ssm depends on it)
    # TMPDIR override prevents out-of-space errors on small root partitions
    # during the large CUDA kernel compilation step.
    TMPDIR="${TMPDIR:-/tmp}" pip install causal-conv1d --no-build-isolation --no-cache-dir
    TMPDIR="${TMPDIR:-/tmp}" pip install mamba-ssm     --no-build-isolation --no-cache-dir
else
    echo "  nvcc not found — skipping mamba-ssm (FEMBA will not be available)."
    echo "  To enable FEMBA:"
    echo "    1. Install CUDA toolkit: apt-get install nvidia-cuda-toolkit"
    echo "    2. Then run:"
    echo "         TMPDIR=/scratch/tmp pip install causal-conv1d mamba-ssm --no-build-isolation --no-cache-dir"
fi

echo ""
echo "Done. Sanity check:"
python -c "import torch, numpy, scipy, transformers, huggingface_hub, mne, cv2; \
print('  torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

echo ""
echo "================================================================"
echo "Activate the venv before running scripts:"
echo "    source venv/bin/activate"
echo ""
echo "If HuggingFace downloads fail with 'No space left on device':"
echo "    export HF_HOME=/scratch/.cache/huggingface"
echo "    export TMPDIR=/scratch/tmp && mkdir -p \$TMPDIR"
echo ""
echo "Then run e.g.:"
echo "    HF_HOME=/scratch/.cache/huggingface python extract_reve.py"
echo "================================================================"
