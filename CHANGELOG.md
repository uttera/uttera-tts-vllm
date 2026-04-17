# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-17

First scaffold release. Pre-alpha — active development. API surface may
still change before v1.0.0.

### Added
- **Single-process FastAPI server** embedding nano-vllm-voxcpm's
  `AsyncVoxCPM2ServerPool` in-process. Concurrency handled entirely by
  the engine's internal continuous batching — no hot/cold worker pool.
- **OpenAI-compatible endpoints**:
  - `POST /v1/audio/speech` — JSON body (OpenAI classic) or
    `multipart/form-data` (enables adhoc voice cloning via
    `speaker_wav` file upload). Response formats: `mp3`, `wav`, `pcm`,
    `opus`, `flac`. Cache-key identical to uttera-tts-hotcold: MD5 of
    `(model, voice, speed, format, params, text)`. Cache is bypassed
    for adhoc requests.
  - `POST /v1/audio/speech/stream` — chunked `audio/wav` streaming.
    Uses `AsyncVoxCPM2ServerPool.generate()` directly; emits a
    0xFFFFFFFF-length WAV header immediately, then raw PCM chunks as
    the engine produces them.
  - `GET /v1/voices` — lists every voice whose latents are currently
    resident in memory, plus the configured default.
  - `POST /admin/reload-voices` — re-reads `voices.json` and recomputes
    latents for new entries without restarting the engine.
  - `GET /v1/models`, `GET /health`.
- **File-based voice registry** (design Model A): `voices.json` at the
  repo root maps `name → relative path` inside
  `assets/voices/{standard,elite}/`. Latents are precomputed at startup
  and kept in memory.
- **Adhoc voice cloning** (design Model C): `POST /v1/audio/speech` with
  a `speaker_wav` multipart file field clones the voice just for that
  request. No persistence. Cache is bypassed.
- **Audio cache** with identical semantics to uttera-tts-hotcold:
  `AUDIO_CACHE_DIR` + `CACHE_TTL_MINUTES`. Set TTL to 0 to disable.
- **Optional Redis self-registration** (parity with every other Uttera
  repo). When `REDIS_URL` is set, publishes `{load_score,
  accepts_requests, host, port, version, engine="nano-vllm-voxcpm",
  model, ts}` to `tts:nodes:{NODE_ID}`.
- **Engine tuning env vars**: `VLLM_GPU_MEM_UTIL`,
  `VLLM_MAX_NUM_SEQS`, `VLLM_MAX_NUM_BATCHED_TOKENS`,
  `VLLM_MAX_MODEL_LEN`, `VOXCPM_INFERENCE_TIMESTEPS`. Defaults tuned
  for RTX 5090 (32 GB).
- **Asset pre-provisioning**: `setup_assets.sh` pulls the VoxCPM2 model
  (~1.7 GB) and the 6 standard OpenAI reference voices
  (alloy/echo/fable/onyx/nova/shimmer) before the first request.
- **OSS scaffolding** shared with the rest of the Uttera stack:
  `LICENSE` (Apache-2.0), `NOTICE`, `AUTHORS.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, `CONTRIBUTING.md`, `CODEOWNERS`, `.github/` templates,
  `docs/img/` banner, Dockerfile + `docker-compose.yml` with NVIDIA GPU
  passthrough, systemd unit (`uttera-tts-vllm.yml`), CI workflow
  (lint + structure + optional GPU smoke).

### Not implemented (yet)
- `tests/` with a benchmark harness. Will be ported from
  `uttera-benchmarks` in a later release.
- Dynamic voice registry (design Model B: `POST /v1/voices` to upload +
  persist cloned voices, `DELETE /v1/voices/{id}` to remove). Documented
  in ROADMAP.
- The GPU smoke CI job is defined but gated off (`if: false`) because
  no self-hosted GPU runner is configured yet.
