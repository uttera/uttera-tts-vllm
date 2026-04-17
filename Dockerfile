# Uttera TTS vLLM — container image
# SPDX-License-Identifier: Apache-2.0
#
# Build:
#   docker build -t uttera-tts-vllm:0.1.0 .
# Run (with GPU):
#   docker run --gpus all --rm -p 5100:5100 \
#       -e VOXCPM_MODEL=openbmb/VoxCPM2 \
#       uttera-tts-vllm:0.1.0

FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-pip \
        git \
        ffmpeg \
        curl \
        ca-certificates \
        file \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3.12 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

COPY requirements.txt /app/
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY . /app/

EXPOSE 5100

ENV XDG_CACHE_HOME=/app/assets/models

CMD ["uvicorn", "main_tts:app", "--host", "0.0.0.0", "--port", "5100"]
