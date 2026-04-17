#!/bin/bash
# Uttera TTS vLLM — Unified Setup Script
# Version: 0.1.0
# Description: Creates a Python 3.12 venv, installs nano-vllm-voxcpm and the
#              FastAPI stack, then hands over to setup_assets.sh for the
#              model + standard voice provisioning.

set -e

echo "🦾 J.A.R.V.I.S. - Starting Uttera TTS vLLM installation..."

echo "[*] Initialising Python Virtual Environment..."
if command -v python3.12 &>/dev/null; then
    PYTHON_BIN=python3.12
    echo "    -> Using python3.12"
else
    PYTHON_BIN=python3
    echo "    [!] python3.12 not found, falling back to $(python3 --version)"
fi
$PYTHON_BIN -m venv venv
source venv/bin/activate

echo "[*] Installing build-time dependencies..."
pip install --upgrade pip setuptools wheel packaging

# flash-attn is a transitive dependency of nano-vllm-voxcpm. Its build
# system requires torch to be importable in the build environment, but
# pip's default PEP 517 build isolation provides an empty env. So we
# install torch first, then the rest with --no-build-isolation so flash-attn
# picks up torch from the venv.
echo "[*] Pre-installing torch (required before flash-attn can build)..."
pip install "torch>=2.5.0,<2.10.0" "torchaudio>=2.5.0,<2.10.0"

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
