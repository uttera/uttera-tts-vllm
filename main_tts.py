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
# Version: 1.5.0
# Maintainer: Hugo L. Espuny
# Description: High-throughput VoxCPM2 TTS server. A single Python process
#              hosts nano-vllm-voxcpm's AsyncVoxCPM2ServerPool; concurrency
#              is handled by the engine's internal continuous batching —
#              no hot/cold pool, no per-request worker spawning.
#
# CHANGELOG:
# - 1.5.0 (2026-09-24): Engine-hardening sweep, ported from the production
#   build. All additive except the two noted defaults. This is a
#   standalone single-process server; there is no service discovery or
#   external coordination.
#     * Engine circuit breaker. N consecutive ENGINE (5xx) failures mark
#       the node not-ready so /health reports 503 instead of serving from
#       a dead engine. 4xx (caller's fault) never trip it. A later success
#       closes it. `ENGINE_FAIL_THRESHOLD` (default 3).
#     * Recovery self-probe. While the breaker is open, a background task
#       issues a tiny in-process synthesis every `ENGINE_PROBE_SECONDS`
#       (default 30) so a transient failure self-heals without a manual
#       restart. Zero cost while the engine is healthy.
#     * Correct HTTP status codes. Oversized text → 413, GPU out-of-memory
#       → 503 (busy, not broken — and it does NOT trip the breaker),
#       malformed JSON body → 400. Previously all three surfaced as 500,
#       and the 500s tripped the breaker on the caller's behalf. Tracebacks
#       are stripped from error bodies.
#     * Voice latents: preload + lazy load + LRU cap. Only VOICE_PRELOAD
#       voices are resident at startup; the rest are computed on first use
#       and capped at VOICE_CACHE_MAX (VoxCPM2 latents live outside the
#       vLLM VRAM budget, ~0.5 GB each, so an unbounded set starves the
#       GPU). `torch.cuda.empty_cache()` is called on eviction so the freed
#       VRAM is actually returned to other GPU tenants.
#     * Reference-sample trimming. An uploaded cloning sample longer than
#       REF_MAX_SECONDS (default 20 s) is trimmed — leading silence removed
#       first — so a very long recording can't OOM the node. Best-effort:
#       a failure here falls back to the original sample.
#     * Text normalization before synthesis. VoxCPM2 degenerates on raw
#       line breaks (they become babble) and on text that ends mid-sentence
#       (it loops until the token budget runs out). Line breaks become a
#       comma/space and a terminating period is appended when missing. Not
#       part of the cache key, on purpose.
#     * Audio-cache sweep. `CACHE_TTL_MINUTES` only decides whether a file
#       is SERVED; a background sweep now deletes expired files from disk
#       (`CACHE_SWEEP_SECONDS`, default 300) so the TTL is a real retention
#       bound, not just a read gate.
#     * Response integrity headers. `X-Audio-Duration` (seconds of audio
#       returned) and `Content-Digest: sha-256=:…:` (RFC 9530) plus the
#       hex `X-Audio-SHA256` of the exact bytes served.
#     * Cache key upgraded MD5 → SHA-256 (it is only a filename, never a
#       secret, but SHA-256 costs the same and is one less audit note).
#     * DEFAULT CHANGE — CACHE_TTL_MINUTES default 10080 (7 days) → 60
#       (1 hour). Synthesized audio may contain personal data; a short
#       retention is the privacy-friendly default. Set CACHE_TTL_MINUTES
#       to restore any value.
#     * Optional offline mode. Set UTTERA_OFFLINE=1 to force
#       transformers / huggingface_hub / modelscope to use only the local
#       cache (a validated model won't silently re-fetch on restart). OFF
#       by default so a fresh install can download the model.
#     * Removed the optional Redis self-registration loop. The server is
#       now purely standalone; put any load balancing in front of it.
# - 1.4.3 (2026-04-23): VRAM-usage fix — VLLM_GPU_MEM_UTIL default
#   0.85 → 0.45, zero throughput regression. Previously the 0.85 default
#   on a 32 GB RTX 5090 preallocated ~27.8 GB of VRAM because the engine
#   sizes the KV cache block pool as
#   `num_kvcache_blocks = (total × util - peak) / per_block_size` — it
#   consumes the WHOLE available budget regardless of whether
#   `max_num_seqs × max_model_len` would actually need it. Empirical
#   sweep on a 32 GB RTX 5090 with a 40-prompt benchmark corpus:
#
#       util    VRAM    burst-64 wall/rps           burst-256 wall/rps
#       ────    ─────   ────────────────────        ──────────────────
#       0.30    fail    `num_kvcache_blocks>0` assertion on startup
#       0.40    22.0 GB 20.1 s / 3.19 rps           60.9 s / 4.20 rps
#   →   0.45    23.6 GB 20.6 s / 3.11 rps           59.1 s / 4.33 rps   ← new default
#       0.85    27.8 GB 20.8 s / 3.08 rps (baseln)  64.4 s / 3.98 rps
#
#   0.45 preserves throughput (inside variance of baseline 0.85) while
#   freeing ~4.2 GB for other GPU tenants. No API change.
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
#   backends). A fronting proxy can route by service family; swapping
#   hotcold ↔ vllm is a backend ExecStart change, not a port change.
#   The 9000-9099 range is IANA "User Ports" with no canonical
#   assignment and no mainstream collisions. Migration: set `PORT=5100`
#   in env to preserve the legacy endpoint, else repoint at `:9004`.
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
#   on RTX 5090 / Blackwell against a 40-prompt Spanish corpus:
#   1024/1024 at every burst size, no failures under sustained 2 rps for
#   5 minutes, aggregate throughput plateaus near 4.3 rps. API surface
#   frozen behind semver — the cache opt-out (body `{"cache": false}` and
#   header `Cache-Control: no-cache`) plus the `X-Cache` response header
#   are now stable.
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
#   uttera-tts-hotcold.
#
# --- Architecture Summary ---
#
# * SINGLE-PROCESS ENGINE
#   nanovllm_voxcpm's AsyncVoxCPM2ServerPool is instantiated at startup
#   (lifespan) and kept resident. Concurrency comes from continuous
#   batching inside the pool — there is no hot/cold worker pool in this
#   wrapper.
#
# * VOICE REGISTRY (file-based)
#   voices.json at the repo root maps name -> relative path inside
#   `assets/voices/`. Both `standard/` (OpenAI reference voices) and
#   `elite/` (custom/cloned, persistent on disk) live underneath.
#   VOICE_PRELOAD voices are precomputed at startup; the rest are computed
#   lazily on first use and kept under an LRU cap (see VOICE_CACHE_MAX).
#   POST /admin/reload-voices re-reads voices.json without restarting the
#   engine.
#
# * ADHOC VOICE CLONING
#   /v1/audio/speech accepts an optional `custom_voice_file` multipart
#   field. When present, the latents are computed on the fly for that
#   single request, no state is persisted, and the audio cache is bypassed
#   (the same text + adhoc audio is not a stable cache key).
#
# * AUDIO CACHE
#   SHA-256 of (model, voice, speed, format, params, text). Stored in
#   AUDIO_CACHE_DIR with TTL from CACHE_TTL_MINUTES and swept from disk
#   once expired. Bypassed for adhoc cloning and for the streaming
#   endpoint.
#
# * STREAMING ENDPOINT
#   /v1/audio/speech/stream returns audio/wav chunks as VoxCPM2 emits
#   them. No cache, no format conversion (WAV only). Named voices only —
#   adhoc cloning on streaming is a future extension.
#

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os

# --- Optional offline mode (opt-in) -------------------------------------------
# A model that has already been validated should not silently re-fetch from the
# Hub on a restart. Set UTTERA_OFFLINE=1 to force transformers / huggingface_hub
# / modelscope to use ONLY the local cache. These are read at import time, so
# they must be set BEFORE the heavy imports below. OFF by default so a fresh
# install can download the model on first run.
if os.environ.get("UTTERA_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("MODELSCOPE_OFFLINE", "1")

import re
import struct
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
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

SERVER_VERSION = "1.5.0"

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

# ── Voice latents: short preload + lazy load ────────────────────────────────
# Each VoxCPM2 voice latent takes ~0.5 GB of VRAM and lives OUTSIDE the vLLM
# accounting (it is computed with _pool.encode_latents(), it does not go through
# the gpu_memory_utilization budget). Keeping a large catalogue resident starves
# the GPU of the transient headroom a synthesis needs. So: only VOICE_PRELOAD is
# loaded at startup and never evicted; every other voice is computed on first
# request and kept under an LRU cap of VOICE_CACHE_MAX.
VOICE_PRELOAD = [v.strip().lower() for v in
                 os.environ.get("VOICE_PRELOAD", "alloy").split(",") if v.strip()]
VOICE_CACHE_MAX = int(os.environ.get("VOICE_CACHE_MAX", "4"))

# Cache.
_cache_env = os.environ.get("AUDIO_CACHE_DIR", "").strip()
AUDIO_CACHE_DIR = Path(_cache_env) if _cache_env else (ASSETS_DIR / "cache")
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Audio-cache lifetime, in minutes.
#
# 60 = ONE HOUR. Synthesized audio may contain personal data, so a short
# retention is the privacy-friendly default; raise it if your use case needs a
# longer-lived cache.
#
# ⚠ This value only controls whether a file is SERVED (the check is
# `age_min < CACHE_TTL_MINUTES` at read time). Deleting it from disk is done by
# the periodic sweep (_cache_sweep_loop); without the sweep, lowering this
# number does NOT reduce the real retention.
CACHE_TTL_MINUTES = int(os.environ.get("CACHE_TTL_MINUTES", "60"))
# How often the on-disk sweep runs.
CACHE_SWEEP_SECONDS = int(os.environ.get("CACHE_SWEEP_SECONDS", "300"))

# HF model cache.
MODEL_CACHE_DIR = os.environ.get("XDG_CACHE_HOME", str(ASSETS_DIR / "models" / "huggingface"))
os.environ.setdefault("HF_HOME", MODEL_CACHE_DIR)

# VoxCPM2 emits at 48 kHz mono float32.
VOXCPM_SAMPLE_RATE = 48000

# Supported response formats (from OpenAI spec plus what ffmpeg gives us).
SUPPORTED_FORMATS = {"mp3", "wav", "pcm", "opus", "flac"}

# -------------------------------
# 2. Runtime State
# -------------------------------

_pool: Optional[AsyncVoxCPM2ServerPool] = None
_engine_ready: bool = False
_engine_error: Optional[str] = None

# name -> latents (returned by _pool.encode_latents)
_voice_latents: dict[str, Any] = {}
# Usage order of the NON-preloaded voices, so we can evict the oldest.
_voice_lru: list[str] = []
_voice_lock: Any = None          # asyncio.Lock, created in the lifespan
# name -> resolved absolute path to the .wav used to compute the latents
_voice_wav_paths: dict[str, Path] = {}

_in_flight: int = 0
_total_completed: int = 0
_total_errors: int = 0


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
    "Total input characters successfully synthesised (throughput proxy)",
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
        if name not in VOICE_PRELOAD:
            _touch_lru(name)
            _evict_if_over()
        return True
    except Exception as e:
        log.warning(f"voice '{name}': encode_latents failed: {e}")
        return False


def _touch_lru(name: str) -> None:
    """Mark the voice as most-recently-used (non-preloaded voices only)."""
    if name in VOICE_PRELOAD:
        return
    if name in _voice_lru:
        _voice_lru.remove(name)
    _voice_lru.append(name)


def _evict_if_over() -> None:
    """Keep at most VOICE_CACHE_MAX NON-preloaded latents resident.

    ⚠ Dropping the reference does not hand the VRAM back to the driver: PyTorch
    keeps it in its own pool. empty_cache() is needed so the freed room becomes
    available to the OTHER services on the GPU, which is the whole point.
    """
    if VOICE_CACHE_MAX <= 0:
        return
    freed = 0
    while len(_voice_lru) > VOICE_CACHE_MAX:
        old = _voice_lru.pop(0)
        _voice_latents.pop(old, None)
        _voice_wav_paths.pop(old, None)
        freed += 1
        log.info("voice '%s' evicted from the latents cache", old)
    if freed:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


async def _reload_all_voices() -> dict[str, Any]:
    """Re-read voices.json and encode latents for the preloaded voices. The rest
    stay AVAILABLE (still listed in voices.json) but their latent is computed on
    first request. Returns a small status dict for /admin/reload-voices."""
    mapping = _load_voices_json()
    succeeded: list[str] = []
    failed: list[str] = []
    for name, rel_path in mapping.items():
        if name.lower() not in VOICE_PRELOAD:
            continue
        if await _compute_and_cache_voice(name, rel_path):
            succeeded.append(name)
        else:
            failed.append(name)
    # Drop cached latents for voices no longer listed.
    for stale in list(_voice_latents.keys()):
        if stale not in mapping:
            _voice_latents.pop(stale, None)
            _voice_wav_paths.pop(stale, None)
    return {"loaded": sorted(succeeded), "failed": sorted(failed),
            "total": len(mapping), "lazy": sorted(n for n in mapping
                                                  if n.lower() not in VOICE_PRELOAD)}


# -------------------------------
# 4. Audio cache helpers
# -------------------------------

def _cache_key(text: str, voice: str, speed: float, fmt: str, params: dict) -> str:
    """Deterministic SHA-256 over the inputs that materially affect output.

    The digest is only a filename, not a secret, but MD5 has been broken for
    collisions for twenty years and it is the first thing an audit flags.
    Same cost, one less thing to explain.
    """
    canon = (
        f"model={SERVED_MODEL_NAME}|voice={voice}|speed={speed:.4f}|"
        f"format={fmt}|params={json.dumps(params, sort_keys=True)}|text={text}"
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


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
# 6. Engine health: circuit breaker + recovery probe
# -------------------------------

# ── Engine circuit breaker ──────────────────────────────────────────────────
# `_engine_ready` used to reflect only STARTUP state and was never revalidated.
# If the engine died mid-life (a CUDA error, the GPU running out of memory
# because of another process), the service kept returning /health 200 and
# serving requests to a dead engine.
#
# Rule: N consecutive ENGINE failures mark the node not-ready, so /health
# reports 503. A single later success restores it, so a transient failure does
# not take the node out forever.
#
# ⚠ Only failures the service itself classifies as 5xx count. 4xx are the
# caller's fault (unreadable audio, unsupported request) and must NOT knock the
# node out: if they did, a client could take a node down by sending three
# corrupt files in a row.
ENGINE_FAIL_THRESHOLD = int(os.environ.get("ENGINE_FAIL_THRESHOLD", "3"))
_engine_fail_streak = 0


def _engine_failure(http_status: int, exc: Exception | None = None) -> None:
    """Record an engine failure. Opens the circuit at the threshold."""
    global _engine_fail_streak, _engine_ready, _engine_error
    if http_status < 500:
        return                      # caller's fault, the engine is fine
    _engine_fail_streak += 1
    if _engine_fail_streak >= ENGINE_FAIL_THRESHOLD and _engine_ready:
        _engine_ready = False
        _engine_error = ("circuit_breaker: %d consecutive engine failures; last: %s"
                         % (_engine_fail_streak,
                            f"{type(exc).__name__}: {exc}" if exc else "unknown"))[:300]
        log.error("CIRCUIT BREAKER OPEN: %s — the node stops accepting requests",
                  _engine_error)


def _engine_ok() -> None:
    """A success closes the circuit and resets the streak."""
    global _engine_fail_streak, _engine_ready, _engine_error
    _engine_fail_streak = 0
    if not _engine_ready:
        _engine_ready = True
        _engine_error = None
        log.warning("CIRCUIT BREAKER CLOSED: the engine responds again")


# ── Recovery self-probe ─────────────────────────────────────────────────────
# Without this the breaker is one-way: once open, the node stops taking traffic,
# so the success that would close it can never arrive, and a transient failure
# would keep the node out until a manual restart.
#
# The probe runs ONLY while the circuit is open: every ENGINE_PROBE_SECONDS it
# fires a tiny in-process request against the app itself (ASGITransport, no
# network, no port). If it returns 200, the success hook already in the endpoint
# closes the circuit; the probe never touches the state directly.
#
# While the circuit is closed it costs nothing: a boolean check each cycle.
ENGINE_PROBE_SECONDS = int(os.environ.get("ENGINE_PROBE_SECONDS", "30"))


def _probe_wav(seconds: float = 1.0, hz: int = 440, sr: int = 16000) -> bytes:
    """A mono 16 kHz WAV holding a tone. A tone and not silence: pure silence
    can make voice-expecting models fail, and a failing probe would leave the
    circuit open forever."""
    import math, struct as _struct, wave as _w, io as _bio
    n = int(seconds * sr)
    samples = b"".join(_struct.pack("<h", int(12000 * math.sin(2 * math.pi * hz * i / sr)))
                       for i in range(n))
    buf = _bio.BytesIO()
    with _w.open(buf, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr); f.writeframes(samples)
    return buf.getvalue()


async def _cache_sweep_loop() -> None:
    """Delete expired files from disk. Runs every CACHE_SWEEP_SECONDS.

    CACHE_TTL_MINUTES only decides whether a file is SERVED; without this sweep
    an expired file stops being served but STAYS ON DISK indefinitely, which
    turns the TTL into an empty promise for any retention policy.
    """
    while True:
        try:
            await asyncio.sleep(CACHE_SWEEP_SECONDS)
            # With the cache disabled (0) everything left is removed: if nothing
            # is served, there is no reason to keep anything.
            cap = CACHE_TTL_MINUTES * 60 if CACHE_TTL_MINUTES > 0 else 0
            now = time.time()
            removed = 0
            for root, _dirs, files in os.walk(str(AUDIO_CACHE_DIR)):
                for name in files:
                    path = os.path.join(root, name)
                    try:
                        if now - os.path.getmtime(path) >= cap:
                            os.unlink(path)
                            removed += 1
                    except OSError:
                        pass
            if removed:
                log.info("cache: swept %d expired files (TTL=%d min)",
                         removed, CACHE_TTL_MINUTES)
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.warning("cache: sweep failed: %s", e)


async def _engine_probe_loop() -> None:
    import httpx
    while True:
        try:
            await asyncio.sleep(ENGINE_PROBE_SECONDS)
            if _engine_ready:
                continue
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://probe",
                                         timeout=120.0) as cli:
                r = await cli.post("/v1/audio/speech",
                                   json={"model": "tts-1", "input": "ok", "voice": DEFAULT_VOICE},
                                   headers={"Cache-Control": "no-cache"})
            if r.status_code == 200:
                # The endpoint will have called _engine_ok(): the circuit closes
                # without the probe touching the state.
                log.info("probe: the engine responds again")
            else:
                log.warning("probe: the engine is still down (HTTP %s)", r.status_code)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("probe: failed to probe: %s", e)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _pool, _engine_ready, _engine_error, _voice_lock

    log.info(f"Starting Uttera TTS vLLM v{SERVER_VERSION} — model={VOXCPM_MODEL}")
    _voice_lock = asyncio.Lock()

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
        if DEFAULT_VOICE not in _voice_latents and DEFAULT_VOICE.lower() not in \
                {k.lower() for k in _load_voices_json()}:
            log.warning(f"DEFAULT_VOICE='{DEFAULT_VOICE}' is not listed. Clients must pass a valid 'voice'.")

        _engine_ready = True
    except Exception as e:
        # ⚠ Reverting the flag matters: without it, a failure AFTER
        # `_engine_ready = True` but inside this try would leave the service
        # advertising itself healthy while the engine is dead.
        _engine_error = str(e)
        log.exception("Engine init failed — server will serve /health 503")
        _engine_ready = False

    _probe_task = asyncio.create_task(_engine_probe_loop())
    _sweep_task = asyncio.create_task(_cache_sweep_loop())

    yield

    log.info("Shutting down…")
    _probe_task.cancel()
    _sweep_task.cancel()
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


# ── The right error code, not a blanket 500 ─────────────────────────────────
# A 500 says "I broke". When the request is at fault we should say so with a
# 4xx, and it is not cosmetic: 5xx COUNT toward the circuit breaker, so a client
# sending oversized text could otherwise knock the node out for everyone.
#   · truncated JSON body            -> 400 (was 500)
#   · text above max_model_len       -> 413 (was 500)
#   · GPU out of memory on a busy node -> 503 (busy, not broken; was 500)
import json as _json_err
from fastapi.responses import JSONResponse as _RespErr

_SIGNS_TOO_LONG = ("max_model_len", "prompt_len", "context length", "too long",
                   "maximum context", "exceeds", "excede")
_SIGNS_OOM = ("out of memory", "outofmemoryerror", "cuda error: out of memory",
              "cublas_status_alloc_failed")


def _useful_line(exc):
    """Pull out the line that explains the limit and DROP the traceback.

    ⚠ A traceback in the response leaks the server's internal paths and package
    names to the client, and it is unreadable. We return just the line that
    mentions the limit."""
    for line in reversed(str(exc).splitlines()):
        b = line.lower()
        if any(s in b for s in _SIGNS_TOO_LONG) and "file \"" not in b:
            # drop the leading "ValueError:" — it tells the client nothing
            return re.sub(r"^[A-Za-z_]+Error:\s*", "", line.strip())[:300]
    return "the request exceeds the model's limit"


def _classify_error(exc):
    """Return (code, detail) by looking at the error text."""
    t = str(exc).lower()
    if any(s in t for s in _SIGNS_OOM):
        # BUSY, not broken. A 503 does not count toward the breaker and tells
        # the client to retry, which is the truth.
        return 503, "the node has no free memory right now; retry"
    if any(s in t for s in _SIGNS_TOO_LONG):
        return 413, _useful_line(exc)
    return None, None


def _install_error_handlers(app):
    @app.exception_handler(_json_err.JSONDecodeError)
    async def _err_json(request, exc):
        return _RespErr(status_code=400,
                        content={"detail": "malformed JSON body: %s" % exc})

    async def _err_generic(request, exc):
        code, detail = _classify_error(exc)
        if code == 503:
            return _RespErr(status_code=503, headers={"Retry-After": "30"},
                            content={"detail": detail})
        if code:
            return _RespErr(status_code=code, content={"detail": detail})
        # Anything we cannot classify STAYS a 500: we don't disguise a possible
        # server fault as the client's mistake.
        return _RespErr(status_code=500,
                        content={"detail": "%s: %s" % (type(exc).__name__, str(exc)[:300])})

    app.add_exception_handler(ValueError, _err_generic)
    app.add_exception_handler(RuntimeError, _err_generic)   # includes OutOfMemoryError


_install_error_handlers(app)
# ────────────────────────────────────────────────────────────────────────────

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
        expose_headers=["X-Route", "X-Cache", "X-Audio-Duration",
                        "X-Audio-SHA256", "Content-Digest"],
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

#: Line breaks: they must be removed BEFORE synthesis.
#:
#: 🔴 VoxCPM2 degenerates when the text carries line breaks (it slips into
#: babble). Replacing a line break with a comma keeps the pause the break
#: implied without inventing a sentence end the author did not write.
#:
#: ⚠ This is NOT a length problem: in prose the model stays intact for well
#: over a thousand characters. What breaks it is line breaks and text that ends
#: mid-sentence (see below).
_LINE_BREAKS = re.compile(r"\s*\n+\s*")
#: A line break already following punctuation carries its own pause: no comma.
_BREAK_AFTER_TEXT = re.compile(r"(?<=[^\s.,;:!?…\-—])\s*\n+\s*")


def _text_for_engine(text: str) -> str:
    """Normalize text the way the model reads it best.

    Applied on the synthesis path and NOT in the cache key on purpose: two
    requests that differ only by a line break are two distinct cache entries.
    That is correct — the client sent different text — and the price is one
    extra synthesis, never a wrong answer.
    """
    text = _BREAK_AFTER_TEXT.sub(", ", text)
    text = _LINE_BREAKS.sub(" ", text).strip()
    # 🔴 Text that ends MID-SENTENCE leaves the model without a stop signal and,
    # if it is long, it starts REPEATING until the token budget runs out. A
    # trailing period fixes it. The risk grows with length; short text is safe.
    if text and text[-1] not in ".!?…":
        text += "."
    return text


async def _synthesize_to_pcm(text: str, latents: Any, cfg_value: float) -> bytes:
    """Run the engine for a complete request and return raw int16 PCM bytes."""
    assert _pool is not None
    chunks: list[np.ndarray] = []
    async for chunk in _pool.generate(
        target_text=_text_for_engine(text),
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


#: Maximum length of the reference voice SAMPLE, in seconds.
#:
#: Cloning guides recommend 6-12 s: below 6 tonal depth is lost, above 12 it
#: does not improve proportionally, it only adds latency. Trimmed to 20 to leave
#: comfortable margin over the recommendation.
#:
#: ⚠ This is not a convenience: it is what stops a long recording from taking
#: the node down. A very long sample makes the engine ask for tens of GB of GPU
#: and die with OutOfMemory, and the client gets an error that blames their
#: codec. Many people don't know how to trim audio; trimming it is our job.
REF_MAX_SECONDS = float(os.environ.get("REF_MAX_SECONDS", 20.0))


def _trim_sample(wav_bytes: bytes, wav_format: str) -> tuple[bytes, str, bool]:
    """Return (bytes, format, trimmed). If the sample is longer than
    REF_MAX_SECONDS, keep the first REF_MAX_SECONDS seconds.

    Uses ffmpeg, already a dependency. On any failure the original sample is
    returned: trimming is an improvement, and a failure here must not block a
    clone that would otherwise have worked.
    """
    if REF_MAX_SECONDS <= 0:
        return wav_bytes, wav_format, False
    src = dst = None
    try:
        fd, src = tempfile.mkstemp(suffix="." + (wav_format or "wav"))
        with os.fdopen(fd, "wb") as f:
            f.write(wav_bytes)
        dur = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", src],
            capture_output=True, text=True, timeout=20).stdout.strip()
        if not dur or float(dur) <= REF_MAX_SECONDS:
            return wav_bytes, wav_format, False
        fd2, dst = tempfile.mkstemp(suffix=".wav")
        os.close(fd2)
        # ⚠ Leading silence is removed BEFORE the cut. Without this, a recording
        # that opens with several seconds of room tone would spend half the
        # sample on nothing and clone worse from the same file. `silenceremove`
        # only acts at the start, with a conservative -45 dB threshold that lets
        # quiet speech through. If there is no leading silence, it does nothing.
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", src,
             "-af", "silenceremove=start_periods=1:start_silence=0.1:"
                    "start_threshold=-45dB",
             "-t", str(REF_MAX_SECONDS), "-ac", "1", "-ar", "24000", dst],
            capture_output=True, timeout=60, check=True)
        with open(dst, "rb") as f:
            trimmed = f.read()
        # ⚠ Safety net: if silence removal ate almost everything (a very quiet
        # recording, a threshold that doesn't fit it) re-cut WITHOUT the filter.
        # A sample with some leading silence beats an empty one.
        if len(trimmed) < 16000:          # under ~0.3 s at 24 kHz 16-bit
            log.warning("silence removal left the sample at %d bytes; "
                        "re-cutting without the filter", len(trimmed))
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", src,
                 "-t", str(REF_MAX_SECONDS), "-ac", "1", "-ar", "24000", dst],
                capture_output=True, timeout=60, check=True)
            with open(dst, "rb") as f:
                trimmed = f.read()
        log.info("custom_voice_file trimmed: %.1fs -> %.1fs (leading silence removed)",
                 float(dur), REF_MAX_SECONDS)
        return trimmed, "wav", True
    except Exception as e:
        log.warning("could not trim the sample (%r); using it whole", e)
        return wav_bytes, wav_format, False
    finally:
        for p in (src, dst):
            if p:
                try:
                    os.unlink(p)
                except Exception:
                    pass


async def _latents_for_request(voice: Optional[str],
                                speaker_wav: Optional[UploadFile]) -> tuple[Any, bool]:
    """Return (latents, is_adhoc). Adhoc latents come from the uploaded file;
    named latents come from the registry (loaded lazily if not resident).
    Raises HTTPException on a missing voice name. Adhoc is preferred over voice
    name when both are present."""
    assert _pool is not None
    if speaker_wav is not None:
        wav_bytes = await speaker_wav.read()
        if not wav_bytes:
            raise HTTPException(
                status_code=400,
                detail="custom_voice_file is empty — upload a valid audio body.",
            )
        wav_format = (speaker_wav.filename or "").rsplit(".", 1)[-1].lower() or "wav"
        wav_bytes, wav_format, _trimmed = _trim_sample(wav_bytes, wav_format)
        try:
            latents = await _pool.encode_latents(wav=wav_bytes, wav_format=wav_format)
        except Exception as e:
            # nano-vllm-voxcpm raises via a remote-call proxy that wraps the
            # real error in a multi-line stringified traceback. We keep the
            # final line (the actual cause: "Format not recognised.", etc.)
            # and drop the stack — clients shouldn't see our library tree.
            msg = str(e).strip().splitlines()[-1] or "encode_latents failed"
            log.warning(f"custom_voice_file encode failed (trimmed): {msg}")
            # ⚠ Don't blame the codec without knowing. A GPU OutOfMemory is not
            # a bad file: report it as 503 (retry), not 400.
            if "OutOfMemory" in msg or "CUDA out of memory" in msg:
                raise HTTPException(
                    status_code=503,
                    detail=("The node ran out of GPU memory while encoding "
                            "custom_voice_file. Retry in a few seconds."),
                )
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode custom_voice_file ({msg}).",
            )
        return latents, True

    name = (voice or DEFAULT_VOICE).lower()
    if name not in _voice_latents:
        # LAZY LOAD: the voice exists in voices.json but its latent is not
        # resident. Compute it now (one-time cost) instead of keeping every
        # voice's latent in VRAM forever.
        mapping = {k.lower(): v for k, v in _load_voices_json().items()}
        if name in mapping:
            global _voice_lock
            if _voice_lock is None:
                _voice_lock = asyncio.Lock()
            # The lock stops two concurrent requests for the same voice from
            # computing the latent twice.
            async with _voice_lock:
                if name not in _voice_latents:
                    log.info("voice '%s' not resident: computing latent", name)
                    await _compute_and_cache_voice(name, mapping[name])
        if name not in _voice_latents:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown voice '{name}'. Available: "
                    f"{sorted(mapping.keys())}. Pass an uploaded 'speaker_wav' "
                    f"for adhoc cloning, or add the voice to voices.json and hit "
                    f"POST /admin/reload-voices."
                ),
            )
    _touch_lru(name)
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
    # False the server neither reads nor writes the audio cache; the response
    # carries `X-Cache: BYPASS`. Omit (None) to fall back to the server default
    # (driven by `CACHE_TTL_MINUTES`).
    cache: Optional[bool] = None


# -------------------------------
# 8. Endpoints
# -------------------------------

# ── Integrity headers of the served audio ───────────────────────────────────
# `X-Audio-Duration` (seconds of audio returned) and, for content integrity as
# it travels, the SHA-256 of the exact bytes served — both as the hex
# `X-Audio-SHA256` and as the standard `Content-Digest` (RFC 9530, which any
# modern HTTP library understands). They match the EXACT bytes: re-compressed or
# re-cut, they won't.

def _audio_file_duration(path: str) -> float:
    """Duration in seconds of an already-encoded audio file. 0.0 on failure.

    A `.dur` sidecar is written next to the audio when it is cached, so a cache
    hit doesn't pay an ffprobe; if it's missing (older entries) we probe.
    """
    sidecar = path + ".dur"
    try:
        with open(sidecar) as f:
            v = float(f.read().strip())
        if v > 0:
            return v
    except Exception:
        pass
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, timeout=20)
        v = float(p.stdout.strip())
        if v > 0:
            try:
                with open(sidecar, "w") as f:
                    f.write("%.3f" % v)
            except Exception:
                pass
            return v
    except Exception:
        pass
    return 0.0


def _annotate_cache_duration(path: str, seconds: float) -> None:
    """Leave the duration next to the cached file for future hits."""
    if seconds and seconds > 0:
        try:
            with open(path + ".dur", "w") as f:
                f.write("%.3f" % seconds)
        except Exception:
            pass


def _content_digest(hex_sha: str) -> str:
    """The SHA-256 in the standard's format, for the `Content-Digest` header.

    RFC 9530 "Digest Fields" defines `Content-Digest: sha-256=:<base64>:` for
    the integrity of the content as it travels — exactly our case, the bytes on
    the wire. Sent ALONGSIDE `X-Audio-SHA256`, not instead of it: Content-Digest
    is understood by any modern HTTP library without reading our docs, while the
    hex `X-Audio-SHA256` is what you read at a glance in a log or a console.
    """
    import base64, binascii
    try:
        return "sha-256=:%s:" % base64.b64encode(binascii.unhexlify(hex_sha)).decode()
    except Exception:
        return ""


def _sha_bytes(data: bytes) -> str:
    try:
        return hashlib.sha256(data).hexdigest()
    except Exception as e:
        log.warning("could not compute the audio sha: %s", e)
        return ""


def _sha_file(path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    except Exception as e:
        log.warning("could not compute the audio sha: %s", e)
        return ""


def _cache_hit_headers(path) -> dict:
    """Headers for a cache hit, with the audio duration and integrity digest."""
    h = {"X-Route": "CACHE", "X-Cache": "HIT"}
    d = _audio_file_duration(str(path))
    if d > 0:
        h["X-Audio-Duration"] = str(round(d, 2))
    # The hit serves the SAME file, so the digest is the same.
    _hh = _sha_file(path)
    if _hh:
        h["X-Audio-SHA256"] = _hh
        _cd = _content_digest(_hh)
        if _cd:
            h["Content-Digest"] = _cd
    return h


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
            return FileResponse(
                cache_file,
                media_type=f"audio/{fmt}",
                headers=_cache_hit_headers(cache_file),
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
        _engine_failure(500)
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

    # Duration of the generated audio, computed on the raw PCM (int16 mono,
    # 2 bytes/sample) before encoding.
    try:
        audio_seconds = round(len(pcm) / 2.0 / float(VOXCPM_SAMPLE_RATE), 2)
    except Exception:
        audio_seconds = 0.0

    # Write to cache on the way out (fire-and-forget shape).
    if cache_file is not None:
        try:
            cache_file.write_bytes(encoded)
            _annotate_cache_duration(str(cache_file), audio_seconds)
        except Exception as e:
            log.warning(f"Failed to write cache {cache_file.name}: {e}")

    _total_completed += 1
    _engine_ok()
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

    _hdrs = {"X-Route": route, "X-Cache": x_cache}
    if audio_seconds > 0:
        _hdrs["X-Audio-Duration"] = str(audio_seconds)
    _hh = _sha_bytes(encoded)
    if _hh:
        _hdrs["X-Audio-SHA256"] = _hh
        _cd = _content_digest(_hh)
        if _cd:
            _hdrs["Content-Digest"] = _cd
    return Response(
        content=encoded,
        media_type=f"audio/{fmt}",
        headers=_hdrs,
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
                target_text=_text_for_engine(req.input),
                ref_audio_latents=latents,
                cfg_value=req.cfg_value,
            ):
                if hasattr(chunk, "cpu"):
                    arr = chunk.squeeze().cpu().numpy()
                else:
                    arr = np.asarray(chunk).squeeze()
                yield _float32_to_int16_pcm(arr.astype("float32"))
            _total_completed += 1
            _engine_ok()
            _SYNTHESIS_TOTAL.labels(
                response_format="wav", route="HOT", cache="DISABLED"
            ).inc()
            _CHARACTERS_SYNTHESISED_TOTAL.labels(response_format="wav").inc(len(req.input))
            _INFERENCE_DURATION.labels(op="synthesis").observe(time.monotonic() - t0)
        except Exception:
            _total_errors += 1
            _ERRORS_TOTAL.labels(type="model").inc()
            _engine_failure(500)
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
    accepts = bool(_engine_ready) and not _engine_error and load < 1.0
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
    return JSONResponse(status_code=200 if (_engine_ready and not _engine_error) else 503,
                        content=body)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "9004"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("main_tts:app", host=host, port=port, log_level="debug" if DEBUG else "info")
