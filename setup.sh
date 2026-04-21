#!/bin/bash
# Uttera TTS vLLM — Unified Setup Script
# Version: 0.1.0
# Description: Creates a Python 3.11 venv (or 3.12 — nano-vllm-voxcpm
#              requires <3.13), installs nano-vllm-voxcpm and the FastAPI
#              stack, then hands over to setup_assets.sh for the model +
#              standard voice provisioning.

set -e

echo "🦾 J.A.R.V.I.S. - Starting Uttera TTS vLLM installation..."

echo "[*] Initialising Python Virtual Environment..."
# nano-vllm-voxcpm on PyPI declares Requires-Python >=3.10,<3.13, so we
# prefer 3.11 (widely-available, well-tested), then 3.12, then fall back
# to the system python3 with a warning — on 3.13+ the install will fail
# at the nano-vllm-voxcpm resolution step.
if command -v python3.11 &>/dev/null; then
    PYTHON_BIN=python3.11
    echo "    -> Using python3.11"
elif command -v python3.12 &>/dev/null; then
    PYTHON_BIN=python3.12
    echo "    -> Using python3.12"
else
    PYTHON_BIN=python3
    PY_VER=$(python3 --version)
    echo "    [!] Neither python3.11 nor python3.12 found, falling back to ${PY_VER}"
    case "${PY_VER}" in
        *" 3.13"*|*" 3.14"*)
            echo "    [!!] ${PY_VER} is >=3.13 — nano-vllm-voxcpm will refuse to install (its Requires-Python is <3.13). Install python3.11 (e.g. via pyenv) and re-run." ;;
    esac
fi
$PYTHON_BIN -m venv venv
source venv/bin/activate

echo "[*] Installing build-time dependencies..."
pip install --upgrade pip setuptools wheel

# flash-attn is a transitive dependency of nano-vllm-voxcpm. It has NO
# pre-built wheels on PyPI — it builds from source and requires the host
# nvcc's CUDA major version to match what torch was compiled against.
# On systems with torch-cu128 but system nvcc 13.x the build aborts with
# "The detected CUDA version (13.x) mismatches the version that was used
# to compile PyTorch (12.8)".
#
# To avoid the source build entirely we pre-install an official flash-attn
# release wheel. Those wheels only cover torch up to 2.8.x at the moment
# (flash-attn v2.8.3, 2025-08), so we pin torch to 2.8.x here.
echo "[*] Installing torch 2.8.x + flash-attn build prereqs..."
pip install "torch>=2.8.0,<2.9.0" "torchaudio>=2.8.0,<2.9.0" \
            packaging psutil ninja

# Pre-install flash-attn from its official release wheel, matching the
# torch / python / CXX11-ABI combo we just installed. If no matching
# wheel exists (e.g. torch upgraded past 2.8), we fall through to the
# normal resolver below — which will attempt the source build and may
# fail loudly on CUDA-version mismatch.
echo "[*] Locating matching flash-attn release wheel..."
FLASH_ATTN_VER="2.8.3"
PY_TAG=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
TORCH_MM=$(python -c "import torch, re; m=re.match(r'(\d+)\.(\d+)', torch.__version__); print(f'{m.group(1)}.{m.group(2)}')")
CXX11ABI=$(python -c "import torch; print('TRUE' if torch.compiled_with_cxx11_abi() else 'FALSE')")
FLASH_ATTN_WHEEL="flash_attn-${FLASH_ATTN_VER}+cu12torch${TORCH_MM}cxx11abi${CXX11ABI}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VER}/${FLASH_ATTN_WHEEL}"
echo "    -> ${FLASH_ATTN_WHEEL}"
if pip install "${FLASH_ATTN_URL}"; then
    echo "    -> flash-attn installed from pre-built wheel"
else
    echo "    [!] Could not fetch pre-built flash-attn wheel. The next step"
    echo "        will try to build it from source — which requires the host"
    echo "        nvcc to match torch's CUDA major. If that fails, install"
    echo "        CUDA 12.x toolkit or downgrade the host nvcc."
fi

echo "[*] Installing core dependencies from requirements.txt (may take several minutes)..."
pip install --no-build-isolation -r requirements.txt

if [ -f "./setup_assets.sh" ]; then
    echo "[*] Python environment ready. Handing over to setup_assets.sh..."
    chmod +x setup_assets.sh
    ./setup_assets.sh
else
    echo "[!] WARNING: setup_assets.sh not found. Model + voices will be fetched on first request."
fi

echo "✅ All systems operational."
