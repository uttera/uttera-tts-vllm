# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-04-21

Prometheus `/metrics` endpoint. Additive only — all existing
endpoints unchanged.

### Added

- **`GET /metrics`** — OpenMetrics-format scrape endpoint using the
  default `prometheus_client` global registry. Scrape with Telegraf's
  `inputs.prometheus` plugin, Prometheus itself, or any other
  OpenMetrics-compatible consumer.
- **HTTP-level metrics** (bounded cardinality — unknown paths fall
  into `"other"`):
  - `uttera_tts_requests_total{endpoint, method, status}`
  - `uttera_tts_request_duration_seconds{endpoint, method}` —
    buckets 25 ms → 60 s
  - `uttera_tts_inflight_requests` — Gauge reflecting `_in_flight`
- **TTS-specific metrics**:
  - `uttera_tts_synthesis_total{response_format, route, cache}` —
    Counter broken down by the output format
    (`mp3` / `wav` / `pcm` / `opus` / `flac`), lane
    (`HOT` / `CACHE` / `ADHOC`) and cache decision
    (`HIT` / `MISS` / `BYPASS` / `ADHOC` / `DISABLED`). The labels
    match the `X-Route` / `X-Cache` response headers exactly.
  - `uttera_tts_characters_synthesised_total{response_format}` —
    Counter summing `len(req.input)` for every successful
    synthesis. Billing / throughput proxy. Cache hits do NOT
    re-bill (the caller already paid when the entry was first
    populated).
  - `uttera_tts_inference_duration_seconds{op}` — Histogram per
    model call kind: `synthesis` (nano-vllm-voxcpm generation) and
    `ffmpeg_encode` (output-format transcoding). Separates GPU
    time from CPU-encoder time.
  - `uttera_tts_voices_loaded` — Gauge of voice names resident in
    VRAM (latents precomputed), refreshed on every `/metrics`
    scrape.
- **State gauges** (refreshed on every `/metrics` scrape so they're
  always current):
  - `uttera_tts_engine_ready` — 1 once the engine has passed
    startup, 0 during load.
- **`uttera_tts_errors_total{type}`** — Counter of errors by cause.
  Types: `model` (uncaught synthesis exception), `encoding` (ffmpeg
  transcode failure). Generic 4xx errors stay visible via the
  `status` label on `requests_total`.
- **`uttera_tts_build_info{version, engine, model}`** — Gauge set
  to `1` with the running `SERVER_VERSION`, engine
  (`nano-vllm-voxcpm`), and the actual `VOXCPM_MODEL` as labels, so
  dashboards can show version + model in the field without a
  separate lookup.

### Instrumentation notes

- The streaming endpoint (`/v1/audio/speech/stream`) increments
  `synthesis_total` with `route="HOT"` + `cache="DISABLED"` and
  records the total stream duration under the `synthesis` op
  (streaming bypasses the cache entirely and emits audio as the
  engine generates it).
- Cache hits increment `synthesis_total{route="CACHE",cache="HIT"}`
  but do NOT tick `characters_synthesised_total`.

### Changed

- **New runtime dep**: `prometheus-client>=0.20.0`.
- **`SERVER_VERSION` bumped to `1.4.0`.**

### Not changed

- `/v1/audio/speech`, `/v1/audio/speech/stream`, `/v1/voices`,
  `/admin/reload-voices`, `/v1/models`, `/health` behave identically
  to v1.3.0. The `/health` body still reports `in_flight` /
  `total_completed` / `total_errors` for callers that have them
  hardcoded; Prometheus counters are the new canonical observability
  path.

## [1.3.0] - 2026-04-18

### Changed

- **Default port migrated from `5100` → `9004`** in lockstep with
  the sibling `uttera-tts-hotcold` v2.3.0. Canonical Uttera-stack
  scheme: TTS services on `9004`, STT services on `9005`. The
  Gatekeeper and clients route by service family; swapping
  hotcold ↔ vllm is a backend change, not a port change.

  **Why not keep `5100`:** pairing TTS=5100 with STT=9005 (STT had
  to move off 5000 due to macOS AirPlay / Docker Registry v2
  collisions) was asymmetric. Both families now live in the
  `9000-9099` range (IANA "User Ports", no canonical assignment,
  no mainstream collisions).

  Artefacts updated: `PORT` env default in `main_tts.py`,
  `Dockerfile` `EXPOSE`/`CMD`, `docker-compose.yml` port mapping +
  healthcheck, `.env.example`, `README.md`, `API.md`,
  `.github/workflows/ci.yml`, issue template health-probe URL.

### Migration

Deployments with explicit `PORT` env var: no change required.
Deployments on the old default (`:5100`):
- Repoint your Gatekeeper / reverse proxy at `:9004`.
- Or set `PORT=5100` in your env to preserve the old endpoint.
- Docker users: update your `-p` flag or `docker-compose.yml`.

### Related

- `uttera-tts-hotcold` v2.3.0 adopts the same `9004` port.
- `uttera-stt-hotcold` v2.3.0 and `uttera-stt-vllm` v1.3.0 adopt
  `9005` for the STT pair.

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
