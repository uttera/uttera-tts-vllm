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
pip install --upgrade pip setuptools wheel

echo "[*] Installing core dependencies from requirements.txt (may take several minutes)..."
pip install -r requirements.txt

if [ -f "./setup_assets.sh" ]; then
    echo "[*] Python environment ready. Handing over to setup_assets.sh..."
    chmod +x setup_assets.sh
    ./setup_assets.sh
else
    echo "[!] WARNING: setup_assets.sh not found. Model + voices will be fetched on first request."
fi

echo "✅ All systems operational."
