#!/bin/bash
# Uttera TTS vLLM — Asset Provisioning
# Version: 0.1.0
# Description: Pre-downloads the VoxCPM2 model from HuggingFace and the 6
#              standard OpenAI reference voices, so the first request does not
#              pay the ~1.7 GB model download.

set -e

echo "🦾 J.A.R.V.I.S. - Provisioning Uttera TTS assets..."

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ASSETS_DIR="$SCRIPT_DIR/assets"
MODELS_DIR="$ASSETS_DIR/models/huggingface"
VOICES_DIR="$ASSETS_DIR/voices/standard"
ELITE_DIR="$ASSETS_DIR/voices/elite"
CACHE_DIR="$ASSETS_DIR/cache"

mkdir -p "$MODELS_DIR" "$VOICES_DIR" "$ELITE_DIR" "$CACHE_DIR"

export HF_HOME="$MODELS_DIR"

MODEL_NAME="${VOXCPM_MODEL:-openbmb/VoxCPM2}"

if [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"
else
    PYTHON_BIN=python3
fi

echo "[*] Pre-downloading model '$MODEL_NAME' into $MODELS_DIR ..."
"$PYTHON_BIN" - <<EOF
import os
from huggingface_hub import snapshot_download
model = os.environ.get("VOXCPM_MODEL", "$MODEL_NAME")
print(f"    -> Downloading/verifying '{model}'...")
path = snapshot_download(repo_id=model)
print(f"    [✓] Cached at {path}")
EOF

# Standard OpenAI reference voices (public CDN).
VOICE_BASE_URL="https://cdn.openai.com/API/docs/audio"
voices=("alloy" "echo" "fable" "onyx" "nova" "shimmer")

echo "[*] Provisioning 6 standard voices into $VOICES_DIR ..."
for voice in "${voices[@]}"; do
    TARGET="$VOICES_DIR/$voice.wav"
    if [ ! -f "$TARGET" ]; then
        echo "    -> downloading $voice.wav ..."
        # -4: force IPv4. Azure CDN blocks IPv6 from some tunnel broker ASNs.
        curl -L -s -4 -o "$TARGET" "$VOICE_BASE_URL/$voice.wav"
        MIME=$(file --mime-type -b "$TARGET" 2>/dev/null || echo "unknown")
        if [[ "$MIME" != *"audio/"* ]]; then
            echo "    [!] $voice.wav invalid ($MIME), removing."
            rm -f "$TARGET"
        else
            echo "    [✓] $voice.wav ready ($MIME)"
        fi
    else
        echo "    [=] $voice.wav already present, skipping."
    fi
done

echo "✅ Asset provisioning complete."
