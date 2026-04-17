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
# Version: 1.0.0
# Maintainer: J.A.R.V.I.S. A.I., Hugo L. Espuny
# Description: High-throughput VoxCPM2 TTS server. A single Python process
#              hosts nano-vllm-voxcpm's AsyncVoxCPM2ServerPool; concurrency
#              is handled by the engine's internal continuous batching —
#              no hot/cold pool, no per-request worker spawning.
#
# CHANGELOG:
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
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

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

SERVER_VERSION = "1.0.0"

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
VLLM_GPU_MEM_UTIL = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.85"))
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
REDIS_NODE_PORT = int(os.environ.get("NODE_PORT", "5100"))
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


def _encode_audio(pcm_bytes: bytes, fmt: str) -> bytes:
    """Convert raw int16 PCM to the requested output format."""
    if fmt == "pcm":
        return pcm_bytes
    if fmt == "wav":
        return _wav_header(len(pcm_bytes)) + pcm_bytes
    # ffmpeg path for mp3/opus/flac.
    codec_args: dict[str, list[str]] = {
        "mp3":  ["-codec:a", "libmp3lame", "-qscale:a", "2"],
        "opus": ["-codec:a", "libopus", "-b:a", "64k"],
        "flac": ["-codec:a", "flac"],
    }
    if fmt not in codec_args:
        raise ValueError(f"Unsupported response_format: {fmt}")
    cmd = [
        "ffmpeg", "-y",
        "-f", "s16le", "-ar", str(VOXCPM_SAMPLE_RATE), "-ac", "1",
        "-i", "pipe:0",
        *codec_args[fmt],
        "-f", {"mp3": "mp3", "opus": "ogg", "flac": "flac"}[fmt],
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
        wav_format = (speaker_wav.filename or "").rsplit(".", 1)[-1].lower() or "wav"
        try:
            latents = await _pool.encode_latents(wav=wav_bytes, wav_format=wav_format)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"speaker_wav encode failed: {e}")
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

@app.post("/v1/audio/speech")
async def create_speech(
    request: Request,
    file: Optional[UploadFile] = File(None, alias="speaker_wav"),
):
    """OpenAI-compatible speech synthesis.

    Accepts either a JSON body (OpenAI style) or multipart/form-data (to
    allow adhoc voice cloning via the `speaker_wav` file field, which has
    no JSON analogue). The JSON fields are the same in both cases.
    """
    global _in_flight, _total_errors, _total_completed
    if not _engine_ready:
        raise HTTPException(status_code=503, detail="Engine not ready")

    content_type = (request.headers.get("content-type") or "").lower()
    speaker_wav: Optional[UploadFile] = None
    if "application/json" in content_type:
        body = await request.json()
        req = SpeechRequest(**body)
    else:
        # multipart or urlencoded
        form = await request.form()
        _raw_cache = form.get("cache")
        _cache_field: Optional[bool] = None
        if _raw_cache is not None:
            _cache_field = str(_raw_cache).strip().lower() not in ("0", "false", "no", "off")
        req = SpeechRequest(
            model=form.get("model") or SERVED_MODEL_NAME,
            voice=form.get("voice"),
            input=form.get("input") or "",
            response_format=form.get("response_format") or "mp3",
            speed=float(form.get("speed") or 1.0),
            cfg_value=float(form.get("cfg_value") or 2.0),
            cache=_cache_field,
        )
        spec = form.get("speaker_wav")
        if isinstance(spec, UploadFile):
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
            return FileResponse(
                cache_file,
                media_type=f"audio/{fmt}",
                headers={"X-Route": "CACHE", "X-Cache": "HIT"},
            )

    latents, _ = await _latents_for_request(req.voice, speaker_wav)

    _in_flight += 1
    try:
        pcm = await _synthesize_to_pcm(req.input, latents, req.cfg_value)
    except Exception:
        _total_errors += 1
        raise
    finally:
        _in_flight -= 1

    try:
        encoded = _encode_audio(pcm, fmt)
    except subprocess.CalledProcessError as e:
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
    return Response(
        content=encoded,
        media_type=f"audio/{fmt}",
        headers={"X-Route": "ADHOC" if adhoc else "HOT", "X-Cache": x_cache},
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
        req = SpeechRequest(**body)
    else:
        form = await request.form()
        req = SpeechRequest(
            model=form.get("model") or SERVED_MODEL_NAME,
            voice=form.get("voice"),
            input=form.get("input") or "",
            response_format="wav",
            speed=float(form.get("speed") or 1.0),
            cfg_value=float(form.get("cfg_value") or 2.0),
        )
    if not req.input:
        raise HTTPException(status_code=422, detail="'input' must be non-empty.")

    latents, _ = await _latents_for_request(req.voice, None)

    async def _stream():
        global _in_flight, _total_errors, _total_completed
        _in_flight += 1
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
        except Exception:
            _total_errors += 1
            log.exception("stream failed")
            raise
        finally:
            _in_flight -= 1

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


@app.get("/health")
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
    port = int(os.environ.get("PORT", "5100"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("main_tts:app", host=host, port=port, log_level="debug" if DEBUG else "info")
