# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-04-18

OpenAI-compatibility polish sweep. Driven by a full endpoint validation
run against v1.1.0. Found one **CRITICAL bug** (adhoc voice cloning
was silently broken) plus seven polish items, all fixed. Behaviour is
backward-compatible *except* for adhoc cloning which now actually
works — v1.1.0 clients that thought they were cloning a voice were in
fact getting the default voice.

### Fixed

1. **[CRITICAL] Adhoc voice cloning was silently disabled.** The
   `isinstance(spec, UploadFile)` check imported
   `fastapi.datastructures.UploadFile` but Starlette's form parser
   returns `starlette.datastructures.UploadFile` — in FastAPI 0.136+ /
   Starlette 1.0+ these are distinct classes (they were aliases in
   earlier versions). The `isinstance` check always returned False, so
   `speaker_wav` never latched, the handler silently used the default
   voice, and the request hit the regular audio cache. Responses
   carried `X-Route: HOT` / `X-Cache: MISS|HIT` instead of the
   documented `X-Route: ADHOC` / `X-Cache: ADHOC`. Fixed by matching
   either class (plus a duck-type fallback for future-proofing) — see
   `_is_upload_file()`. **Users who relied on v1.1.0 adhoc cloning:
   upgrade to v1.2.0 to actually get cloned voices.**
2. **Bogus `custom_voice_file` bodies** (non-audio, empty) were
   accepted and silently produced default-voice audio — same root
   cause as (1). Now rejected with HTTP 400 and a trimmed decode
   error (the nano-vllm-voxcpm traceback no longer leaks to clients).
3. **JSON body without `input`** raised a `pydantic.ValidationError`
   that bubbled up as HTTP 500 with no body. Now caught and converted
   to HTTP 422 with the pydantic error detail.
4. **`speed` range validation.** Values outside `[0.25, 4.0]` (OpenAI
   spec) were silently accepted. Now → HTTP 422 with an explicit range
   message.
5. **`speed` is now actually applied.** The engine doesn't support
   variable-rate synthesis natively, so `speed != 1.0` was silently
   ignored up to v1.1.0. `_encode_audio()` now routes through an
   ffmpeg `atempo` filter chain (chained for values < 0.5 or > 2.0)
   for every output format — `mp3`, `wav`, `pcm`, `opus`, `flac`.
6. **`cfg_value` range validation.** Values outside `[0.5, 5.0]`
   (VoxCPM2 safe range) were accepted and could produce NaN / garbage
   from the diffusion solver. Now → HTTP 422.
7. **`HEAD /health`** returned HTTP 405. Now accepts both GET and
   HEAD via `@app.api_route(methods=["GET", "HEAD"])`.
8. **No CORS middleware.** Added opt-in `CORSMiddleware` gated on
   the `CORS_ALLOW_ORIGINS` env var (comma-separated list, or `"*"`).
   Disabled by default — API-first deployments don't need it, and
   enabling it unconditionally broadens the attack surface.

### Changed

- **`SERVER_VERSION` bumped to `1.2.0`.**

### Verified

- 128-concurrent regression burst: 128/128 OK, ~12 rps (same order of
  magnitude as v1.1.0; minor variance from cold cache in the run).
- Adhoc voice cloning via `custom_voice_file` and `speaker_wav` alias
  both emit `X-Route: ADHOC` and generate audio in the uploaded voice.
- `speed=2.0` produces ~half-duration audio; `speed=0.5` produces
  ~double-duration audio (verified with `ffprobe`).
- CORS preflight + actual POST emit the expected headers when
  `CORS_ALLOW_ORIGINS` is set.

### Breaking?

- **Clients that unknowingly relied on v1.1.0's silent fallback to
  default voice** will now receive real cloned audio from their
  uploaded `custom_voice_file`. That is the documented contract; if
  the upload fails decode, the server now returns HTTP 400 instead of
  quietly using the default voice.

## [1.1.0] - 2026-04-17

### Added
- **Canonical adhoc-cloning field renamed to `custom_voice_file`**,
  symmetric with `uttera-tts-hotcold` v2.1.0. The same client code —
  `curl -F custom_voice_file=@sample.wav ...` — now works against
  either backend. The v1.0.0 name `speaker_wav` is still accepted as
  an alias (this is a v1.x additive change, not a breaking rename);
  if both field names are present on the same request, the canonical
  `custom_voice_file` wins.

### Clarified
- The field is format-agnostic. Any libsndfile-readable file (wav,
  flac, mp3, ogg, m4a) works — the old `speaker_wav` name was a
  misleading Coqui carry-over.

## [1.0.0] - 2026-04-17

First public stable release. `uttera-tts-vllm` graduates from pre-alpha
after end-to-end validation on NVIDIA RTX 5090 (Blackwell, 32 GB)
against the 40-prompt Spanish corpus in
[`uttera-benchmarks` Run 6](https://github.com/uttera/uttera-benchmarks/tree/master/results/2026-04-17-run6-vllm-tts40w):

  latency    20/20    p50  1.8 s  /  p95   2.5 s
  burst@8     8/8     p50  3.3 s
  burst@64   64/64    p50 11.7 s
  burst@256 256/256   p50 33.9 s
  burst@512 512/512   p50 64.2 s
  burst@1024 1024/1024 p50 123 s     ← zero failures at every N
  sustained  600/600  p50  3.3 s / p95  4.0 s   (2 rps × 5 min)

Throughput saturates near 4.3 rps from N = 256 upwards; sustained at
50 % of burst@64 capacity stays flat with no drift over the window.

### API surface — now stable (semver)
- `POST /v1/audio/speech` — OpenAI-compatible JSON body or multipart
  form. Supports adhoc voice cloning via `speaker_wav` file field.
- `POST /v1/audio/speech/stream` — chunked `audio/wav` streaming.
  Starts emitting PCM as soon as the engine produces it; no caching.
- `GET /v1/voices`, `POST /admin/reload-voices`, `GET /v1/models`,
  `GET /health`.
- Audio cache keyed by MD5 of `(model, voice, speed, format, params,
  text)`. Client opt-out per-request via `{"cache": false}` body or
  `Cache-Control: no-cache` header. Every response carries
  `X-Cache: HIT | MISS | BYPASS | ADHOC | DISABLED`.
- `X-Route: CACHE | HOT | ADHOC` on non-cache responses.

### Engine tuning (env vars, unchanged since 0.1.x)
- `VLLM_GPU_MEM_UTIL` (default 0.85)
- `VLLM_MAX_NUM_SEQS` (default 64)
- `VLLM_MAX_NUM_BATCHED_TOKENS`, `VLLM_MAX_MODEL_LEN`,
  `VOXCPM_INFERENCE_TIMESTEPS`

### Post-1.0 compatibility contract
- The endpoint paths, request/response schemas, env var names and
  their defaults, and the `X-Cache` / `X-Route` header values are
  frozen — any breaking change to these requires a v2.0.0.
- Additive extensions (new optional body fields, new
  `X-Cache`/`X-Route` values) are v1.x minor releases.
- Bug fixes are v1.0.x patch releases.

## [0.1.4] - 2026-04-17

### Added
- JSON-body cache opt-out: `{"cache": false}` in the request body skips
  both the read and the write side of the audio cache for that single
  request. Symmetric with the existing `Cache-Control: no-cache` HTTP
  header support and with `uttera-tts-hotcold` v2.0.3. Multipart/form
  submissions accept `cache=0` / `false` / `no` / `off`.

## [0.1.3] - 2026-04-17

### Added
- Per-request cache bypass via the standard `Cache-Control: no-cache`
  (or `no-store`) request header. Response header `X-Cache: HIT |
  MISS | BYPASS | ADHOC | DISABLED` documents the cache decision on
  every `/v1/audio/speech` response. Symmetric with the feature added
  in `uttera-tts-hotcold` v2.0.2 so clients can use the same
  bench/retry logic across backends.

## [0.1.2] - 2026-04-17

### Fixed
- `setup.sh` pre-install list was still incomplete. `flash-attn`'s
  `setup.py` imports `torch`, `packaging`, `psutil`, and `ninja`;
  v0.1.1 covered only torch and packaging, so the build died on
  `ModuleNotFoundError: No module named 'psutil'`. Added `psutil` and
  `ninja` to the pre-install list.

## [0.1.1] - 2026-04-17

### Fixed
- `setup.sh` failed during `pip install -r requirements.txt` because
  `flash-attn` (a transitive dep of `nano-vllm-voxcpm`) requires
  `torch` at build time, but pip's default PEP 517 build-isolation
  sandbox does not have it. `setup.sh` now pre-installs torch and
  torchaudio, then runs `pip install --no-build-isolation -r
  requirements.txt` so flash-attn picks up the torch in the venv.
  `requirements.txt` drops the torch pins (they live in the
  pre-install step).

v0.1.0 never made it past the first install on a clean machine.

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
