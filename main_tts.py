#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Uttera TTS vLLM Server (Single-Process, Continuous Batching)
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025-2026 Hugo L. Espuny
# Original work created with assistance from Google Gemini and Anthropic Claude
#
# Part of the Uttera voice stack (https://uttera.ai).
# See LICENSE and NOTICE for full terms and attributions.
#
# Package: uttera-tts-vllm
# Version: 1.4.3
# Maintainer: J.A.R.V.I.S. A.I., Hugo L. Espuny
# Description: High-throughput VoxCPM2 TTS server. A single Python process
#              hosts nano-vllm-voxcpm's AsyncVoxCPM2ServerPool; concurrency
#              is handled by the engine's internal continuous batching —
#              no hot/cold pool, no per-request worker spawning.
#
# CHANGELOG:
# - 1.4.3 (2026-04-23): VRAM-usage fix — VLLM_GPU_MEM_UTIL default
#   0.85 → 0.45, zero throughput regression. Previously the 0.85 default
#   on a 32 GB RTX 5090 preallocated ~27.8 GB of VRAM because the engine
#   sizes the KV cache block pool as
#   `num_kvcache_blocks = (total × util - peak) / per_block_size` — it
#   consumes the WHOLE available budget regardless of whether
#   `max_num_seqs × max_model_len` would actually need it. Empirical
#   sweep on sphinx (RTX 5090) with the canonical `uttera-tts-40w`
#   benchmark corpus (2026-04-23):
#
#       util    VRAM    burst-64 wall/rps           burst-256 wall/rps
#       ────    ─────   ────────────────────        ──────────────────
#       0.30    fail    `num_kvcache_blocks>0` assertion on startup
#       0.40    22.0 GB 20.1 s / 3.19 rps           60.9 s / 4.20 rps
#   →   0.45    23.6 GB 20.6 s / 3.11 rps           59.1 s / 4.33 rps   ← new default
#       0.85    27.8 GB 20.8 s / 3.08 rps (baseln)  64.4 s / 3.98 rps
#
#   0.45 preserves throughput (inside variance of baseline 0.85) while
#   freeing ~4.2 GB for other GPU tenants (uttera-stt-hotcold,
#   uttera-sentiment-vllm, comfyui, ...). No API change.
# - 1.4.2 (2026-04-21): setup.sh now pins torch/torchaudio to 2.8.x and
#   pre-installs the official flash-attn 2.8.3 release wheel matching the
#   resolved torch / python / CXX11-ABI combo. Previously the resolver
#   picked up torch 2.9.x and flash-attn then built from source, which
#   breaks on systems where the host nvcc CUDA major doesnt match torch
#   (seen on Ubuntu 25.10 hosts with nvcc 13.x and torch-cu128). The
#   pre-built wheel sidesteps the source build entirely. No runtime code
#   change — identical server behaviour.
# - 1.4.1 (2026-04-21): setup.sh now prefers python3.11 (then 3.12,
#   then falls back to system python3 with a warning). The upstream
#   `nano-vllm-voxcpm` package declares `Requires-Python >=3.10,<3.13`,
#   so installs on py3.13+ systems (e.g. Ubuntu 25.10) hit
#   `No matching distribution found for nano-vllm-voxcpm>=2.0.0`. No
#   runtime code changes — identical server behaviour.
# - 1.4.0 (2026-04-21): Prometheus `/metrics` endpoint. Exposes
#   request counters (by endpoint/method/status), request duration
#   histograms, in-flight gauge, engine-ready gauge, TTS-specific
#   counters (synthesis by response_format + route + cache-decision,
#   characters synthesised), per-op inference duration histograms
#   (synthesis, ffmpeg_encode), voices-loaded gauge, error counters
#   typed by cause, and a build_info gauge with version + engine +
#   model labels. Scrape with Telegraf's inputs.prometheus or any
#   OpenMetrics consumer. Additive — existing endpoints unchanged.
# - 1.3.0 (2026-04-18): Default port migrated from 5100 → 9004 in
#   lockstep with the sibling `uttera-tts-hotcold` v2.3.0. Canonical
#   Uttera-stack port scheme: TTS=9004 (all backends), STT=9005 (all
#   backends). The Gatekeeper routes by service family; swapping
#   hotcold ↔ vllm is a backend ExecStart change, not a port change.
#   The 9000-9099 range is IANA "User Ports" with no canonical
#   assignment and no mainstream collisions. Updated: PORT env
#   default in main_tts.py, Dockerfile EXPOSE/CMD, docker-compose
#   port mapping and healthcheck, .env.example, README, API.md, CI
#   workflow probes, issue template. Migration: set `PORT=5100` in
#   env to preserve the legacy endpoint, else repoint at `:9004`.
# - 1.2.0 (2026-04-18): OpenAI-compat polish sweep. Eight findings
#   uncovered by the full endpoint validation run against v1.1.0 —
#   one CRITICAL bug plus seven polish items. All backward-compatible
#   except the corrected adhoc-cloning path (which was silently broken):
#
#   1. [CRITICAL] Adhoc voice cloning was silently disabled. The
#      `isinstance(spec, UploadFile)` check used `fastapi.UploadFile`
#      but Starlette's form parser returns `starlette.datastructures.UploadFile`
#      which is a DIFFERENT class in FastAPI 0.136+ / Starlette 1.0+
#      (they were identical in older versions). The isinstance check
#      always returned False, so `speaker_wav` never latched and every
#      request silently fell through to the default voice — emitting
#      `X-Route: HOT` (instead of `ADHOC`) and caching the output as a
#      regular request. Fixed by accepting either class (or any
#      file-like object with `read` + `filename`).
#   2. JSON body without `input` raised `pydantic.ValidationError` that
#      bubbled up as HTTP 500 with no body. Now caught and converted
#      to HTTP 422 with the pydantic error detail.
#   3. Bogus `custom_voice_file` (non-audio body) was accepted and
#      silently produced output with the default voice — same root
#      cause as (1). Now rejected with HTTP 400 because the UploadFile
#      latches correctly and `encode_latents` raises a decode error.
#   4. `speed` outside `[0.25, 4.0]` (OpenAI spec) was accepted
#      silently. Now validated → HTTP 422.
#   5. `speed` != 1.0 was silently ignored (the engine doesn't support
#      rate control). Now implemented as a post-process `ffmpeg atempo`
#      filter (chained for values < 0.5 or > 2.0), applied across all
#      output formats including WAV + PCM.
#   6. `cfg_value` outside `[0.5, 5.0]` (VoxCPM safe range) was
#      accepted silently and could produce NaN / garbage. Now
#      validated → HTTP 422.
#   7. HEAD /health returned HTTP 405. Now accepts both GET and HEAD
#      via `@app.api_route(methods=["GET", "HEAD"])`.
#   8. No CORS middleware. Added opt-in `CORSMiddleware` gated on the
#      `CORS_ALLOW_ORIGINS` env var (comma-separated list, or `"*"`).
#      Disabled by default — API-first deployments don't need it.
# - 1.1.0 (2026-04-17): Adhoc voice-cloning field renamed (additively)
#   to `custom_voice_file` — symmetric with uttera-tts-hotcold v2.1.0
#   so the same client code works against either backend. The v1.0.0
#   `speaker_wav` name is accepted as an alias for backward compat;
#   if both fields are present on the same request, `custom_voice_file`
#   wins. New name is format-agnostic (the server still accepts wav /
#   mp3 / flac / any libsndfile-readable format regardless of field
#   name). Docstring on `/v1/audio/speech` updated.
# - 1.0.0 (2026-04-17): First public stable release. Validated end-to-end
#   on RTX 5090 / Blackwell against the 40-prompt Spanish corpus (see
#   uttera/uttera-benchmarks Run 6): 1024/1024 at every burst size, no
#   failures under sustained 2 rps for 5 minutes, aggregate throughput
#   plateaus near 4.3 rps. API surface frozen behind semver — the cache
#   opt-out (body `{"cache": false}` and header `Cache-Control: no-cache`)
#   plus the `X-Cache` response header are now stable.
# - 0.1.4 (2026-04-17): JSON-body cache opt-out. `{"cache": false}` in the
#   request body (or `cache=0/false/no/off` in multipart) skips read +
#   write of the audio cache for that single request. Symmetric with the
#   existing Cache-Control header path and with uttera-tts-hotcold v2.0.3.
# - 0.1.3 (2026-04-17): Per-request cache bypass via the HTTP
#   `Cache-Control: no-cache` header + response header
#   `X-Cache: HIT | MISS | BYPASS | ADHOC | DISABLED` so the cache
#   decision is observable without timing heuristics.
# - 0.1.2 (2026-04-17): setup.sh pre-installs psutil + ninja on top of
#   torch and packaging — flash-attn's setup.py imports all four.
# - 0.1.1 (2026-04-17): setup.sh pre-installs torch before
#   `pip install -r requirements.txt` and runs the main install with
#   `--no-build-isolation`, so flash-attn (transitive dep) can build.
# - 0.1.0 (2026-04-17): Initial scaffold. FastAPI app wrapping
#   nanovllm_voxcpm.models.voxcpm2.server.AsyncVoxCPM2ServerPool.
#   Endpoints: /v1/audio/speech (cached, MP3/WAV/PCM),
#   /v1/audio/speech/stream (uncached, WAV chunked),
#   /v1/voices (list), /admin/reload-voices (rescan voices.json),
#   /v1/models, /health. Voices provisioning follows the same
#   voices.json + assets/voices/{standard,elite}/ layout as
#   uttera-tts-hotcold. Adhoc voice cloning via `speaker_wav` form
#   field on /v1/audio/speech. MD5 audio cache with TTL identical to
#   uttera-tts-hotcold. Redis self-registration carried over from
#   the sibling repos.
#
# --- Architecture Summary (v1.0.0) ---
#
# * SINGLE-PROCESS ENGINE
#   nanovllm_voxcpm's AsyncVoxCPM2ServerPool is instantiated at startup
#   (lifespan) and kept resident. Concurrency comes from continuous
#   batching inside the pool — there is no hot/cold worker pool in this
#   wrapper.
#
# * VOICE REGISTRY (file-based, Model A of the design discussion)
#   voices.json at the repo root maps name -> relative path inside
#   `assets/voices/`. Both `standard/` (OpenAI reference voices) and
#   `elite/` (custom/cloned, persistent on disk) live underneath.
#   Voice latents are precomputed at startup and cached in memory.
#   POST /admin/reload-voices re-reads voices.json and computes latents
#   for any new files, without restarting the engine.
#
# * ADHOC VOICE CLONING (Model C of the design discussion)
#   /v1/audio/speech accepts an optional `speaker_wav` multipart field.
#   When present, the latents are computed on the fly for that single
#   request, no state is persisted, and the audio cache is bypassed
#   (the same text + adhoc audio is not a stable cache key).
#
# * AUDIO CACHE
#   Identical to uttera-tts-hotcold: MD5 of (model, voice, speed,
#   format, params, text). Stored in AUDIO_CACHE_DIR with TTL from
#   CACHE_TTL_MINUTES. Bypassed for adhoc cloning and for the streaming
#   endpoint.
#
# * STREAMING ENDPOINT
#   /v1/audio/speech/stream returns audio/wav chunks as VoxCPM2 emits
#   them. No cache, no format conversion (WAV only). Model A voices
#   only — adhoc cloning on streaming is a future extension.
#

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import struct
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import redis.asyncio as aioredis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile

# Load .env from the project directory or its parent
_base = os.path.dirname(os.path.abspath(__file__))
for _env_path in [os.path.join(_base, ".env"), os.path.join(os.path.dirname(_base), ".env")]:
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        break

# nano-vllm-voxcpm is a heavy import; bring it in after .env so its own
# env vars (if any) are honoured.
from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2ServerPool  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

# -------------------------------
# 1. Global Config & Logging
# -------------------------------

SERVER_VERSION = "1.4.3"

# Validation ranges.
# `speed` — OpenAI spec for /v1/audio/speech is [0.25, 4.0].
SPEED_MIN = 0.25
SPEED_MAX = 4.0
# `cfg_value` — VoxCPM2 classifier-free guidance. Default 2.0. Above 5
# the model frequently degenerates to repetition or NaNs; below 0.5 it
# ignores the reference voice. Mirror the clamp used in the sibling
# uttera-tts-hotcold voxcpm_backend.py.
CFG_MIN = 0.5
CFG_MAX = 5.0

DEBUG = os.environ.get("DEBUG", "false").lower() in ("1", "true", "yes")
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("uttera-tts-vllm")

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

# Model + engine.
VOXCPM_MODEL = os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM2")
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "tts-1")
VLLM_GPU_MEM_UTIL = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.45"))
VLLM_MAX_NUM_SEQS = int(os.environ.get("VLLM_MAX_NUM_SEQS", "32"))
VLLM_MAX_NUM_BATCHED_TOKENS = int(os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "16384"))
VLLM_MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096"))
VOXCPM_INFERENCE_TIMESTEPS = int(os.environ.get("VOXCPM_INFERENCE_TIMESTEPS", "10"))

# Voices.
VOICE_ASSET_DIR = Path(os.environ.get("VOICE_ASSET_DIR", str(ASSETS_DIR / "voices")))
VOICE_ASSET_DIR.mkdir(parents=True, exist_ok=True)
(VOICE_ASSET_DIR / "standard").mkdir(exist_ok=True)
(VOICE_ASSET_DIR / "elite").mkdir(exist_ok=True)
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "alloy")
VOICES_JSON_PATH = BASE_DIR / "voices.json"

# Cache (mirrors uttera-tts-hotcold semantics).
_cache_env = os.environ.get("AUDIO_CACHE_DIR", "").strip()
AUDIO_CACHE_DIR = Path(_cache_env) if _cache_env else (ASSETS_DIR / "cache")
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_MINUTES = int(os.environ.get("CACHE_TTL_MINUTES", "10080"))

# HF model cache.
MODEL_CACHE_DIR = os.environ.get("XDG_CACHE_HOME", str(ASSETS_DIR / "models" / "huggingface"))
os.environ.setdefault("HF_HOME", MODEL_CACHE_DIR)

# VoxCPM2 emits at 48 kHz mono float32.
VOXCPM_SAMPLE_RATE = 48000

# Routing telemetry.
ROUTING_DRAIN_CAP_SECONDS = float(os.environ.get("ROUTING_DRAIN_CAP_SECONDS", "120"))

# Redis self-registration (opt-in).
REDIS_URL = os.environ.get("REDIS_URL", "")
REDIS_NODE_HOST = os.environ.get("NODE_HOST", "localhost")
REDIS_NODE_PORT = int(os.environ.get("NODE_PORT", "9004"))
REDIS_NODE_ID = os.environ.get("NODE_ID", "") or f"{REDIS_NODE_HOST}:{REDIS_NODE_PORT}"
REDIS_KEY = f"tts:nodes:{REDIS_NODE_ID}"
REDIS_PUBLISH_INTERVAL = float(os.environ.get("REDIS_PUBLISH_INTERVAL", "0.5"))
REDIS_TTL = max(2, int(REDIS_PUBLISH_INTERVAL * 3 + 1))

# Supported response formats (from OpenAI spec plus what ffmpeg gives us).
SUPPORTED_FORMATS = {"mp3", "wav", "pcm", "opus", "flac"}

# -------------------------------
# 2. Runtime State
# -------------------------------

_pool: Optional[AsyncVoxCPM2ServerPool] = None
_engine_ready: bool = False
_engine_error: Optional[str] = None

# name -> latents bytes (returned by _pool.encode_latents)
_voice_latents: dict[str, Any] = {}
# name -> resolved absolute path to the .wav used to compute the latents
_voice_wav_paths: dict[str, Path] = {}

_in_flight: int = 0
_total_completed: int = 0
_total_errors: int = 0

_redis: Optional[aioredis.Redis] = None
_redis_task: Optional[asyncio.Task] = None


# -------------------------------
# 2b. Prometheus metrics
# -------------------------------
#
# Naming convention: `uttera_tts_<thing>`. Labels kept deliberately
# low-cardinality — no request_id, no voice name (elite voices are
# an open set), no input text. `endpoint` is clamped to the known
# route list so unknown paths can't blow up cardinality.

_HTTP_REQUESTS_TOTAL = Counter(
    "uttera_tts_requests_total",
    "HTTP requests by endpoint, method and status code",
    ["endpoint", "method", "status"],
)

_HTTP_REQUEST_DURATION = Histogram(
    "uttera_tts_request_duration_seconds",
    "HTTP request wall-clock duration in seconds",
    ["endpoint", "method"],
    buckets=(0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

_INFLIGHT_GAUGE = Gauge(
    "uttera_tts_inflight_requests",
    "Requests currently being processed by the engine",
)

_ENGINE_READY_GAUGE = Gauge(
    "uttera_tts_engine_ready",
    "1 if the TTS engine is loaded and ready, 0 otherwise",
)

_VOICES_LOADED_GAUGE = Gauge(
    "uttera_tts_voices_loaded",
    "Number of voices currently resident (latents precomputed in VRAM)",
)

_SYNTHESIS_TOTAL = Counter(
    "uttera_tts_synthesis_total",
    "Synthesis requests broken down by output format, lane, and cache decision",
    ["response_format", "route", "cache"],
    # response_format ∈ {mp3, wav, pcm, opus, flac}
    # route           ∈ {HOT, CACHE, ADHOC}
    # cache           ∈ {HIT, MISS, BYPASS, ADHOC, DISABLED}
)

_CHARACTERS_SYNTHESISED_TOTAL = Counter(
    "uttera_tts_characters_synthesised_total",
    "Total input characters successfully synthesised (billing / throughput proxy)",
    ["response_format"],
)

_INFERENCE_DURATION = Histogram(
    "uttera_tts_inference_duration_seconds",
    "Per-call inference latency in seconds, by op",
    ["op"],                         # synthesis | ffmpeg_encode
    buckets=(0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

_ERRORS_TOTAL = Counter(
    "uttera_tts_errors_total",
    "Errors by type",
    ["type"],                       # decode | validation | model | encoding
)

_BUILD_INFO = Gauge(
    "uttera_tts_build_info",
    "Build metadata (label values carry version, engine and served model id)",
    ["version", "engine", "model"],
)

# Known HTTP routes — used to normalise the `endpoint` label so
# cardinality stays bounded even if someone probes unknown paths.
_KNOWN_ENDPOINTS = {
    "/v1/audio/speech",
    "/v1/audio/speech/stream",
    "/v1/voices",
    "/admin/reload-voices",
    "/v1/models",
    "/health",
    "/metrics",
}


# -------------------------------
# 3. Voice registry helpers
# -------------------------------

def _load_voices_json() -> dict[str, str]:
    """Load {name: relative_path} from voices.json. Falls back to a single
    alloy mapping if the file is missing (safety for bare installs)."""
    if VOICES_JSON_PATH.exists():
        try:
            return json.loads(VOICES_JSON_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"voices.json invalid: {e}; falling back to alloy default.")
    return {"alloy": "standard/alloy.wav"}


async def _compute_and_cache_voice(name: str, rel_path: str) -> bool:
    """Compute voice latents for a single voice; store in _voice_latents.

    Returns True if computed successfully.
    """
    assert _pool is not None
    full_path = (VOICE_ASSET_DIR / rel_path).resolve()
    if not full_path.is_file():
        log.warning(f"voice '{name}': wav not found at {full_path}")
        return False
    try:
        wav_bytes = full_path.read_bytes()
        wav_format = full_path.suffix.lstrip(".").lower() or "wav"
        latents = await _pool.encode_latents(wav=wav_bytes, wav_format=wav_format)
        _voice_latents[name] = latents
        _voice_wav_paths[name] = full_path
        return True
    except Exception as e:
        log.warning(f"voice '{name}': encode_latents failed: {e}")
        return False


async def _reload_all_voices() -> dict[str, Any]:
    """Re-read voices.json and encode latents for every entry. Returns a
    small status dict suitable for returning from /admin/reload-voices."""
    mapping = _load_voices_json()
    succeeded: list[str] = []
    failed: list[str] = []
    for name, rel_path in mapping.items():
        if await _compute_and_cache_voice(name, rel_path):
            succeeded.append(name)
        else:
            failed.append(name)
    # Drop cached latents for voices no longer listed.
    for stale in list(_voice_latents.keys()):
        if stale not in mapping:
            _voice_latents.pop(stale, None)
            _voice_wav_paths.pop(stale, None)
    return {"loaded": sorted(succeeded), "failed": sorted(failed), "total": len(mapping)}


# -------------------------------
# 4. Audio cache helpers (mirrors uttera-tts-hotcold)
# -------------------------------

def _cache_key(text: str, voice: str, speed: float, fmt: str, params: dict) -> str:
    """Deterministic MD5 over the inputs that materially affect output."""
    canon = (
        f"model={SERVED_MODEL_NAME}|voice={voice}|speed={speed:.4f}|"
        f"format={fmt}|params={json.dumps(params, sort_keys=True)}|text={text}"
    )
    return hashlib.md5(canon.encode("utf-8")).hexdigest()


def _cache_path(key: str, fmt: str) -> Path:
    return AUDIO_CACHE_DIR / f"{key}.{fmt}"


def _cache_hit(path: Path) -> bool:
    if CACHE_TTL_MINUTES <= 0 or not path.is_file():
        return False
    age_s = time.time() - path.stat().st_mtime
    return age_s <= CACHE_TTL_MINUTES * 60


# -------------------------------
# 5. Audio encoding helpers
# -------------------------------

def _float32_to_int16_pcm(arr: np.ndarray) -> bytes:
    clipped = np.clip(arr, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _wav_header(pcm_bytes: int, sample_rate: int = VOXCPM_SAMPLE_RATE,
                channels: int = 1, bits: int = 16) -> bytes:
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + pcm_bytes, b"WAVE",
        b"fmt ", 16,
        1, channels, sample_rate,
        byte_rate, block_align, bits,
        b"data", pcm_bytes,
    )


def _streaming_wav_header(sample_rate: int = VOXCPM_SAMPLE_RATE,
                          channels: int = 1, bits: int = 16) -> bytes:
    """0xFFFFFFFF in both length fields — RIFF spec 'unknown length' for streams."""
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF, b"WAVE",
        b"fmt ", 16,
        1, channels, sample_rate,
        byte_rate, block_align, bits,
        b"data", 0xFFFFFFFF,
    )


def _encode_audio(pcm_bytes: bytes, fmt: str, speed: float = 1.0) -> bytes:
    """Convert raw int16 PCM to the requested output format.

    When `speed` != 1.0 we route through ffmpeg's `atempo` filter even
    for PCM and WAV (which otherwise skip ffmpeg), so speed support is
    consistent across every response_format.
    """
    atempo = _atempo_chain(speed)
    if fmt == "pcm" and not atempo:
        return pcm_bytes
    if fmt == "wav" and not atempo:
        return _wav_header(len(pcm_bytes)) + pcm_bytes

    # ffmpeg path (all formats go through this when atempo is needed,
    # or for any fmt that requires an encoder).
    codec_args: dict[str, list[str]] = {
        "mp3":  ["-codec:a", "libmp3lame", "-qscale:a", "2"],
        "opus": ["-codec:a", "libopus", "-b:a", "64k"],
        "flac": ["-codec:a", "flac"],
        # For PCM + WAV with speed != 1 we re-encode raw int16 back out;
        # ffmpeg produces identical format, just time-scaled.
        "wav":  ["-codec:a", "pcm_s16le"],
        "pcm":  ["-codec:a", "pcm_s16le", "-f", "s16le"],
    }
    if fmt not in codec_args:
        raise ValueError(f"Unsupported response_format: {fmt}")
    out_format = {"mp3": "mp3", "opus": "ogg", "flac": "flac", "wav": "wav", "pcm": "s16le"}[fmt]
    cmd = [
        "ffmpeg", "-y",
        "-f", "s16le", "-ar", str(VOXCPM_SAMPLE_RATE), "-ac", "1",
        "-i", "pipe:0",
        *atempo,
        *codec_args[fmt],
        "-f", out_format,
        "pipe:1",
    ]
    proc = subprocess.run(cmd, input=pcm_bytes, capture_output=True, check=True)
    return proc.stdout


# -------------------------------
# 6. Lifespan — engine + voices + Redis
# -------------------------------

async def _publish_to_redis_loop() -> None:
    global _redis
    while True:
        try:
            await asyncio.sleep(REDIS_PUBLISH_INTERVAL)
            if _redis is None:
                continue
            load = min(1.0, _in_flight / max(1, VLLM_MAX_NUM_SEQS))
            accepts = bool(_engine_ready) and load < 1.0
            payload = json.dumps({
                "load_score":       load,
                "accepts_requests": accepts,
                "host":              REDIS_NODE_HOST,
                "port":              REDIS_NODE_PORT,
                "version":           SERVER_VERSION,
                "engine":            "nano-vllm-voxcpm",
                "model":             SERVED_MODEL_NAME,
                "ts":                time.time(),
            })
            try:
                await _redis.set(REDIS_KEY, payload, ex=REDIS_TTL)
            except Exception as e:
                log.debug(f"Redis publish failed (non-fatal): {e}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.warning(f"Redis publish loop error: {e}")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _pool, _engine_ready, _engine_error, _redis, _redis_task

    log.info(f"Starting Uttera TTS vLLM v{SERVER_VERSION} — model={VOXCPM_MODEL}")

    try:
        model_path = snapshot_download(repo_id=VOXCPM_MODEL)
        log.info(f"Model cached at {model_path}")
        _pool = AsyncVoxCPM2ServerPool(
            model_path=model_path,
            inference_timesteps=VOXCPM_INFERENCE_TIMESTEPS,
            max_num_batched_tokens=VLLM_MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=VLLM_MAX_NUM_SEQS,
            max_model_len=VLLM_MAX_MODEL_LEN,
            gpu_memory_utilization=VLLM_GPU_MEM_UTIL,
            devices=[0],
        )
        await _pool.wait_for_ready()
        log.info("AsyncVoxCPM2ServerPool ready.")

        status = await _reload_all_voices()
        log.info(f"Voices: loaded {len(status['loaded'])}/{status['total']} — {status['loaded']}")
        if status["failed"]:
            log.warning(f"Voices failed: {status['failed']}")
        if DEFAULT_VOICE not in _voice_latents:
            log.warning(f"DEFAULT_VOICE='{DEFAULT_VOICE}' is not loaded. Clients must pass a valid 'voice'.")

        _engine_ready = True
    except Exception as e:
        _engine_error = str(e)
        log.exception("Engine init failed — server will serve /health 503")

    if REDIS_URL:
        try:
            _redis = aioredis.from_url(REDIS_URL, decode_responses=False)
            await _redis.ping()
            _redis_task = asyncio.create_task(_publish_to_redis_loop())
            log.info(f"Redis registered at {REDIS_KEY}")
        except Exception as e:
            log.warning(f"Redis unavailable, skipping self-registration: {e}")
            _redis = None

    yield

    log.info("Shutting down…")
    if _redis_task:
        _redis_task.cancel()
        try:
            await _redis_task
        except Exception:
            pass
    if _redis:
        try:
            await _redis.delete(REDIS_KEY)
        except Exception:
            pass
        try:
            await _redis.aclose()
        except Exception:
            pass
    if _pool is not None:
        try:
            await _pool.stop()
        except Exception:
            pass


app = FastAPI(
    title="Uttera TTS vLLM Server",
    version=SERVER_VERSION,
    lifespan=_lifespan,
)

# Opt-in CORS middleware. API-first deployments don't need CORS, so it
# stays off by default. Set CORS_ALLOW_ORIGINS to a comma-separated list
# of origins, or "*" to allow all.
_cors_origins_env = os.environ.get("CORS_ALLOW_ORIGINS", "").strip()
if _cors_origins_env:
    _cors_origins = ["*"] if _cors_origins_env == "*" else [
        o.strip() for o in _cors_origins_env.split(",") if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Route", "X-Cache"],
    )


# Prometheus middleware — tracks every HTTP request generically.
# Endpoint-specific labels (response_format, route, cache, char
# count) are attached inside the endpoint handlers for richer
# breakdowns.

class _PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.url.path
        method = request.method
        if path == "/metrics":
            return await call_next(request)
        endpoint = path if path in _KNOWN_ENDPOINTS else "other"
        t0 = time.monotonic()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            elapsed = time.monotonic() - t0
            _HTTP_REQUESTS_TOTAL.labels(
                endpoint=endpoint, method=method, status=str(status)
            ).inc()
            _HTTP_REQUEST_DURATION.labels(
                endpoint=endpoint, method=method
            ).observe(elapsed)

app.add_middleware(_PrometheusMiddleware)

# Build_info is a static gauge — set once at module import.
_BUILD_INFO.labels(
    version=SERVER_VERSION,
    engine="nano-vllm-voxcpm",
    model=os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM2"),
).set(1)


def _is_upload_file(value: Any) -> bool:
    """Return True if `value` is a file-upload object.

    FastAPI 0.100+ and Starlette 1.0+ ship distinct `UploadFile` classes
    (`fastapi.datastructures.UploadFile` vs `starlette.datastructures.UploadFile`),
    and starlette's form parser always returns the Starlette flavour. An
    `isinstance(spec, fastapi.UploadFile)` check against the Starlette
    instance silently returns False — which is how adhoc voice cloning
    was broken up to v1.1.0. Match both classes explicitly; fall back to
    duck-typing (has `read` + `filename`) so any future divergence keeps
    working.
    """
    if isinstance(value, (UploadFile, StarletteUploadFile)):
        return True
    return (
        not isinstance(value, (str, bytes))
        and hasattr(value, "read")
        and hasattr(value, "filename")
    )


def _validate_synthesis_params(speed: float, cfg_value: float) -> None:
    """Validate params that the engine doesn't police itself.

    VoxCPM2 doesn't natively support `speed`, so we apply it post-hoc
    via ffmpeg `atempo`; outside [0.25, 4.0] we reject per OpenAI spec.
    Above cfg_value ~5 the model frequently degenerates to repetition
    or NaN, below 0.5 it ignores the reference voice.
    """
    if not (SPEED_MIN <= speed <= SPEED_MAX):
        raise HTTPException(
            status_code=422,
            detail=f"speed {speed} out of range. Must be in [{SPEED_MIN}, {SPEED_MAX}].",
        )
    if not (CFG_MIN <= cfg_value <= CFG_MAX):
        raise HTTPException(
            status_code=422,
            detail=f"cfg_value {cfg_value} out of range. Must be in [{CFG_MIN}, {CFG_MAX}].",
        )


def _atempo_chain(speed: float) -> list[str]:
    """Build a `-filter:a` chain for ffmpeg `atempo`.

    atempo accepts [0.5, 2.0] per invocation; for wider ranges we chain
    (e.g. 0.25 → two 0.5 filters, 4.0 → two 2.0 filters). Fractional
    values outside that band are split to stay in range.
    """
    if abs(speed - 1.0) < 1e-6:
        return []
    parts: list[float] = []
    remaining = speed
    while remaining > 2.0 + 1e-6:
        parts.append(2.0)
        remaining /= 2.0
    while remaining < 0.5 - 1e-6:
        parts.append(0.5)
        remaining /= 0.5
    parts.append(remaining)
    return ["-filter:a", ",".join(f"atempo={p:.6f}" for p in parts)]


# -------------------------------
# 7. Synthesis core
# -------------------------------

async def _synthesize_to_pcm(text: str, latents: Any, cfg_value: float) -> bytes:
    """Run the engine for a complete request and return raw int16 PCM bytes."""
    assert _pool is not None
    chunks: list[np.ndarray] = []
    async for chunk in _pool.generate(
        target_text=text,
        ref_audio_latents=latents,
        cfg_value=cfg_value,
    ):
        if hasattr(chunk, "cpu"):
            arr = chunk.squeeze().cpu().numpy()
        else:
            arr = np.asarray(chunk).squeeze()
        chunks.append(arr)
    if not chunks:
        return b""
    audio = np.concatenate(chunks).astype("float32")
    return _float32_to_int16_pcm(audio)


async def _latents_for_request(voice: Optional[str],
                                speaker_wav: Optional[UploadFile]) -> tuple[Any, bool]:
    """Return (latents, is_adhoc). Adhoc latents come from the uploaded file;
    named latents come from the pre-cached registry. Raises HTTPException on
    missing voice name. Adhoc is preferred over voice name when both present."""
    assert _pool is not None
    if speaker_wav is not None:
        wav_bytes = await speaker_wav.read()
        if not wav_bytes:
            raise HTTPException(
                status_code=400,
                detail="custom_voice_file is empty — upload a valid audio body.",
            )
        wav_format = (speaker_wav.filename or "").rsplit(".", 1)[-1].lower() or "wav"
        try:
            latents = await _pool.encode_latents(wav=wav_bytes, wav_format=wav_format)
        except Exception as e:
            # nano-vllm-voxcpm raises via a remote-call proxy that wraps the
            # real error in a multi-line stringified traceback. We keep the
            # final line (the actual cause: "Format not recognised.", etc.)
            # and drop the stack — clients shouldn't see our library tree.
            msg = str(e).strip().splitlines()[-1] or "encode_latents failed"
            log.warning(f"custom_voice_file encode failed (trimmed): {msg}")
            raise HTTPException(
                status_code=400,
                detail=(
                    "Failed to decode custom_voice_file — not a valid audio "
                    f"stream or unsupported codec ({msg})."
                ),
            )
        return latents, True

    name = (voice or DEFAULT_VOICE).lower()
    if name not in _voice_latents:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown voice '{name}'. Available: "
                f"{sorted(_voice_latents.keys())}. Pass an uploaded 'speaker_wav' "
                f"for adhoc cloning, or add the voice to voices.json and hit "
                f"POST /admin/reload-voices."
            ),
        )
    return _voice_latents[name], False


class SpeechRequest(BaseModel):
    model: Optional[str] = SERVED_MODEL_NAME
    voice: Optional[str] = None
    input: str
    response_format: str = "mp3"
    speed: float = 1.0
    # VoxCPM-specific (accepted, passed through).
    cfg_value: float = 2.0
    # Opt out of the server-side audio cache for this specific request. When
    # False the server neither reads nor writes the MD5-keyed audio cache;
    # the response carries `X-Cache: BYPASS`. Omit (None) to fall back to
    # the server default (driven by `CACHE_TTL_MINUTES`).
    cache: Optional[bool] = None


# -------------------------------
# 8. Endpoints
# -------------------------------

@app.get("/metrics")
async def metrics():
    """Prometheus-format scrape endpoint.

    Scrape with Telegraf's `inputs.prometheus` plugin, Prometheus
    itself, or any OpenMetrics-compatible consumer. Cardinality is
    bounded by design (fixed endpoint list, no per-request-id labels,
    voices are counted rather than labelled).
    """
    _ENGINE_READY_GAUGE.set(1 if _engine_ready else 0)
    _INFLIGHT_GAUGE.set(_in_flight)
    _VOICES_LOADED_GAUGE.set(len(_voice_latents))
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/audio/speech")
async def create_speech(request: Request):
    """OpenAI-compatible speech synthesis.

    Accepts either a JSON body (OpenAI style) or multipart/form-data. The
    multipart form is required for **stateless adhoc voice cloning**: the
    caller uploads a reference audio file and the server clones the voice
    for that single request without persisting anything. The canonical
    field name is `custom_voice_file`; `speaker_wav` is accepted as a
    backward-compatible alias for v1.0.0 clients.
    """
    global _in_flight, _total_errors, _total_completed
    if not _engine_ready:
        raise HTTPException(status_code=503, detail="Engine not ready")

    content_type = (request.headers.get("content-type") or "").lower()
    speaker_wav: Optional[UploadFile] = None
    if "application/json" in content_type:
        body = await request.json()
        try:
            req = SpeechRequest(**body)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
    else:
        # multipart or urlencoded
        form = await request.form()
        _raw_cache = form.get("cache")
        _cache_field: Optional[bool] = None
        if _raw_cache is not None:
            _cache_field = str(_raw_cache).strip().lower() not in ("0", "false", "no", "off")
        try:
            req = SpeechRequest(
                model=form.get("model") or SERVED_MODEL_NAME,
                voice=form.get("voice"),
                input=form.get("input") or "",
                response_format=form.get("response_format") or "mp3",
                speed=float(form.get("speed") or 1.0),
                cfg_value=float(form.get("cfg_value") or 2.0),
                cache=_cache_field,
            )
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
        # Canonical field name is `custom_voice_file`. `speaker_wav` kept as
        # alias for v1.0.0 / Coqui-style clients. If both are sent, the
        # canonical one wins. See `_is_upload_file` — in FastAPI 0.136+ the
        # fastapi.UploadFile and starlette.UploadFile classes diverged, and
        # a straight `isinstance(spec, UploadFile)` would silently fail.
        spec = form.get("custom_voice_file") or form.get("speaker_wav")
        if _is_upload_file(spec):
            speaker_wav = spec

    if not req.input:
        raise HTTPException(status_code=422, detail="'input' must be a non-empty string.")
    fmt = req.response_format.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=f"response_format '{fmt}' not supported. "
                   f"Use one of: {sorted(SUPPORTED_FORMATS)}",
        )
    _validate_synthesis_params(req.speed, req.cfg_value)

    params = {"cfg_value": req.cfg_value}
    voice_name = (req.voice or DEFAULT_VOICE).lower() if speaker_wav is None else "adhoc"
    adhoc = speaker_wav is not None

    # Cache opt-out for this specific request. Two equivalent mechanisms:
    #   1. `{"cache": false}` (or 0) in the JSON body — first-class API field.
    #   2. `Cache-Control: no-cache` / `no-store` request header — standard HTTP.
    # Either one turns off both the read and the write side of the cache for
    # this single request, without affecting `CACHE_TTL_MINUTES`.
    cc = (request.headers.get("Cache-Control") or "").lower()
    bypass_cache = (req.cache is False) or any(tok in cc for tok in ("no-cache", "no-store"))
    cache_file: Optional[Path] = None
    if not adhoc and CACHE_TTL_MINUTES > 0 and not bypass_cache:
        key = _cache_key(req.input, voice_name, req.speed, fmt, params)
        cache_file = _cache_path(key, fmt)
        if _cache_hit(cache_file):
            log.debug(f"cache hit: {cache_file.name}")
            _SYNTHESIS_TOTAL.labels(
                response_format=fmt, route="CACHE", cache="HIT"
            ).inc()
            # Cache hits don't bill characters (the caller already
            # paid on the original MISS that populated the cache).
            return FileResponse(
                cache_file,
                media_type=f"audio/{fmt}",
                headers={"X-Route": "CACHE", "X-Cache": "HIT"},
            )

    latents, _ = await _latents_for_request(req.voice, speaker_wav)

    _in_flight += 1
    _INFLIGHT_GAUGE.inc()
    try:
        with _INFERENCE_DURATION.labels(op="synthesis").time():
            pcm = await _synthesize_to_pcm(req.input, latents, req.cfg_value)
    except Exception:
        _total_errors += 1
        _ERRORS_TOTAL.labels(type="model").inc()
        raise
    finally:
        _in_flight -= 1
        _INFLIGHT_GAUGE.dec()

    try:
        with _INFERENCE_DURATION.labels(op="ffmpeg_encode").time():
            encoded = _encode_audio(pcm, fmt, speed=req.speed)
    except subprocess.CalledProcessError as e:
        _ERRORS_TOTAL.labels(type="encoding").inc()
        raise HTTPException(
            status_code=500,
            detail=f"Audio encoding failed: ffmpeg exited {e.returncode}",
        )

    # Write to cache on the way out (fire-and-forget shape).
    if cache_file is not None:
        try:
            cache_file.write_bytes(encoded)
        except Exception as e:
            log.warning(f"Failed to write cache {cache_file.name}: {e}")

    _total_completed += 1
    if bypass_cache:
        x_cache = "BYPASS"
    elif adhoc:
        x_cache = "ADHOC"
    elif CACHE_TTL_MINUTES <= 0:
        x_cache = "DISABLED"
    else:
        x_cache = "MISS"
    route = "ADHOC" if adhoc else "HOT"
    _SYNTHESIS_TOTAL.labels(
        response_format=fmt, route=route, cache=x_cache
    ).inc()
    _CHARACTERS_SYNTHESISED_TOTAL.labels(response_format=fmt).inc(len(req.input))
    return Response(
        content=encoded,
        media_type=f"audio/{fmt}",
        headers={"X-Route": route, "X-Cache": x_cache},
    )


@app.post("/v1/audio/speech/stream")
async def create_speech_stream(request: Request):
    """Streaming TTS. Returns audio/wav chunked. No cache. No adhoc cloning on this endpoint."""
    global _in_flight, _total_errors, _total_completed
    if not _engine_ready or _pool is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        body = await request.json()
        try:
            req = SpeechRequest(**body)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
    else:
        form = await request.form()
        try:
            req = SpeechRequest(
                model=form.get("model") or SERVED_MODEL_NAME,
                voice=form.get("voice"),
                input=form.get("input") or "",
                response_format="wav",
                speed=float(form.get("speed") or 1.0),
                cfg_value=float(form.get("cfg_value") or 2.0),
            )
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
    if not req.input:
        raise HTTPException(status_code=422, detail="'input' must be non-empty.")
    # Streaming currently ignores `speed` — the stream is emitted at the
    # engine's native rate, and atempo would require buffering which
    # would defeat the point of streaming. `speed != 1.0` is accepted
    # for parity with /v1/audio/speech (same validation) but not applied.
    _validate_synthesis_params(req.speed, req.cfg_value)

    latents, _ = await _latents_for_request(req.voice, None)

    async def _stream():
        global _in_flight, _total_errors, _total_completed
        _in_flight += 1
        _INFLIGHT_GAUGE.inc()
        t0 = time.monotonic()
        try:
            yield _streaming_wav_header()
            async for chunk in _pool.generate(
                target_text=req.input,
                ref_audio_latents=latents,
                cfg_value=req.cfg_value,
            ):
                if hasattr(chunk, "cpu"):
                    arr = chunk.squeeze().cpu().numpy()
                else:
                    arr = np.asarray(chunk).squeeze()
                yield _float32_to_int16_pcm(arr.astype("float32"))
            _total_completed += 1
            _SYNTHESIS_TOTAL.labels(
                response_format="wav", route="HOT", cache="DISABLED"
            ).inc()
            _CHARACTERS_SYNTHESISED_TOTAL.labels(response_format="wav").inc(len(req.input))
            _INFERENCE_DURATION.labels(op="synthesis").observe(time.monotonic() - t0)
        except Exception:
            _total_errors += 1
            _ERRORS_TOTAL.labels(type="model").inc()
            log.exception("stream failed")
            raise
        finally:
            _in_flight -= 1
            _INFLIGHT_GAUGE.dec()

    return StreamingResponse(_stream(), media_type="audio/wav", headers={"X-Route": "HOT"})


@app.get("/v1/voices")
async def list_voices():
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "voice",
                "wav_path": str(path.relative_to(VOICE_ASSET_DIR))
                             if path.is_absolute() and VOICE_ASSET_DIR in path.parents
                             else str(path),
            }
            for name, path in sorted(_voice_wav_paths.items())
        ],
        "default": DEFAULT_VOICE,
    }


@app.post("/admin/reload-voices")
async def reload_voices():
    """Re-read voices.json and recompute latents. Does not touch the engine."""
    if not _engine_ready:
        raise HTTPException(status_code=503, detail="Engine not ready")
    status = await _reload_all_voices()
    return {"status": "ok", **status}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": SERVED_MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "uttera",
        }],
    }


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    load = min(1.0, _in_flight / max(1, VLLM_MAX_NUM_SEQS))
    accepts = bool(_engine_ready) and load < 1.0
    body = {
        "status": "ok" if _engine_ready else "starting",
        "version": SERVER_VERSION,
        "engine": "nano-vllm-voxcpm",
        "model": VOXCPM_MODEL,
        "served_as": SERVED_MODEL_NAME,
        "engine_ready": _engine_ready,
        "engine_error": _engine_error,
        "voices_loaded": sorted(_voice_latents.keys()),
        "default_voice": DEFAULT_VOICE,
        "routing": {"load_score": load, "accepts_requests": accepts},
        "metrics": {
            "in_flight": _in_flight,
            "total_completed": _total_completed,
            "total_errors": _total_errors,
            "max_num_seqs": VLLM_MAX_NUM_SEQS,
            "max_model_len": VLLM_MAX_MODEL_LEN,
            "gpu_memory_utilization": VLLM_GPU_MEM_UTIL,
        },
    }
    return JSONResponse(status_code=200 if _engine_ready else 503, content=body)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "9004"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("main_tts:app", host=host, port=port, log_level="debug" if DEBUG else "info")
