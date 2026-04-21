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

# flash-attn is a transitive dependency of nano-vllm-voxcpm and it builds
# from source (no pre-built wheels for every torch/CUDA combo). Its
# setup.py imports torch, packaging, psutil, and ninja. pip's default
# PEP 517 build isolation gives the build a clean sandbox, so those
# imports fail. We install them first, then run the main install with
# --no-build-isolation so flash-attn's build picks them up from the venv.
echo "[*] Pre-installing flash-attn build dependencies..."
pip install "torch>=2.5.0,<2.10.0" "torchaudio>=2.5.0,<2.10.0" \
            packaging psutil ninja

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
