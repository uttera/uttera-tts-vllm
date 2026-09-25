# uttera-tts-vllm

<p align="center">
  <a href="https://uttera.ai">
    <img src="docs/img/banner.png" alt="uttera.ai — The voice layer for your AI" width="800">
  </a>
</p>

High-throughput **Text-to-Speech** server built on
[nano-vllm-voxcpm](https://github.com/openbmb/nano-vllm-voxcpm)'s
continuous-batching engine. VoxCPM2 today, OpenAI-compatible API,
adhoc voice cloning on day one.

> **Status**: v1.5.0 — stable. The API surface (endpoints, cache opt-out
> semantics, `X-Cache` header values, canonical port `9004`) is frozen
> under SemVer; no breaking changes inside `1.x`. The v1.0.0 baseline
> was validated end-to-end on NVIDIA RTX 5090 (Blackwell, 32 GB) against
> a 40-prompt Spanish corpus — 1024/1024 OK at every burst profile,
> 600/600 OK under 5-minute sustained load, throughput plateau near
> 4.3 rps. v1.5.0 adds an engine-hardening sweep: a circuit breaker + an
> in-process recovery self-probe, correct HTTP status codes (413 / 503 /
> 400 instead of a blanket 500), voice preload + lazy load + LRU cap,
> reference-sample trimming, text normalization for VoxCPM2, an on-disk
> cache sweep, response integrity headers (`Content-Digest`, RFC 9530),
> and an optional offline mode. The server is standalone — put any load
> balancing in front of it.
> See [CHANGELOG.md](CHANGELOG.md) for the full release history.

## Positioning

| Use case | This repo | Sibling repo |
|---|---|---|
| Cloud, multi-tenant, large GPU (≥24 GB per process) | ✅ [uttera-tts-vllm](https://github.com/uttera/uttera-tts-vllm) | — |
| Home-lab, personal, small/mid GPU (8–16 GB) | — | [uttera-tts-hotcold](https://github.com/uttera/uttera-tts-hotcold) |

**Choose `uttera-tts-vllm` when**:
- You serve many concurrent users from a single large-VRAM GPU.
- Continuous batching matters for throughput.
- You're OK with the model resident in VRAM 24/7.
- **You have 32 GB+ of VRAM** (the engine reserves
  `VLLM_GPU_MEM_UTIL × total` at startup and keeps it for the process
  lifetime).

**Choose `uttera-tts-hotcold` when**:
- You have consumer GPUs (RTX 4070, 4080) and don't want to dedicate
  VRAM 24/7.
- Personal or single-user deployment.
- Low baseline load with occasional bursts.
- **You have 8–24 GB of VRAM.**

See [`uttera-benchmarks`](https://github.com/uttera/uttera-benchmarks)
for the canonical head-to-head numbers against `uttera-tts-hotcold`
(Coqui XTTS-v2 and VoxCPM2 backends) on the same corpus and GPU.

## Architecture

A **single Python process** hosts:

- `nanovllm_voxcpm.models.voxcpm2.server.AsyncVoxCPM2ServerPool` — the
  model + continuous batcher.
- A thin FastAPI layer (`main_tts.py`) that exposes the endpoints
  Uttera expects — `/v1/audio/speech`, `/v1/audio/speech/stream`,
  `/v1/voices`, `/admin/reload-voices`, `/v1/models`, `/health` — plus
  the SHA-256 audio cache and file-based voice registry.

**What is here (current release)**:

*Voices and synthesis*
- 6 standard OpenAI reference voices (alloy / echo / fable / onyx /
  nova / shimmer) precomputed at startup.
- Elite/custom voices via file-based registry (`voices.json` +
  `assets/voices/elite/`), reloadable without a server restart via
  `POST /admin/reload-voices`.
- **Adhoc voice cloning** via a multipart `custom_voice_file` upload on
  `/v1/audio/speech` — the one feature the Whisper-stack siblings don't
  have. The legacy field name `speaker_wav` is accepted as an alias.
- 5 response formats: MP3, WAV, PCM, Opus, FLAC.
- Chunked streaming via `/v1/audio/speech/stream`.

*Control plane*
- **`speed` parameter is actually applied** (ffmpeg `atempo`, chained for
  values <0.5 or >2.0). Validated range `[0.25, 4.0]` per the OpenAI
  spec — out-of-range → HTTP 422.
- **`cfg_value`** (VoxCPM2-specific sampling knob) validated range
  `[0.5, 5.0]` — out-of-range → HTTP 422.
- **Correct HTTP status codes.** Oversized text → 413, GPU out-of-memory
  → 503 (busy, not broken), malformed JSON body → 400. Tracebacks are
  stripped from error bodies. Only genuine 5xx trip the circuit breaker.
- **Engine circuit breaker + recovery self-probe.** Repeated engine
  (5xx) failures mark the node not-ready (`/health` 503); an in-process
  probe self-heals it when the engine recovers, no restart needed. 4xx
  (the caller's fault) never trip it.

*Privacy and observability*
- On-disk SHA-256 audio cache with **per-request opt-out** for
  privacy-sensitive calls — three equivalent ways to request it (JSON
  body `cache:false`, multipart form field, or the standard
  `Cache-Control: no-cache` header). See
  [**Cache opt-out**](#cache-opt-out--per-request-privacy-control).
  A background sweep deletes expired files from disk so `CACHE_TTL_MINUTES`
  is a real retention bound, not just a read gate.
- `X-Cache` response header — `HIT | MISS | BYPASS | ADHOC | DISABLED`
  — so clients can verify the cache decision without timing heuristics.
- `X-Route` response header — `HOT | CACHE | ADHOC`.
- Response integrity headers — `X-Audio-Duration`, and the SHA-256 of the
  exact bytes served as both `X-Audio-SHA256` (hex) and `Content-Digest`
  (RFC 9530).

*Operations*
- **Standalone** — no service discovery or external coordination. Put any
  load balancing in front of it.
- Voice **preload + lazy load + LRU cap** so a large voice catalogue can't
  starve the GPU (latents live outside the vLLM VRAM budget).
- **Optional offline mode** (`UTTERA_OFFLINE=1`) — a validated model won't
  silently re-fetch from the Hub on restart.
- `HEAD /health` accepted for uptime probes (in addition to `GET`).
- Opt-in `CORSMiddleware` gated on the `CORS_ALLOW_ORIGINS` env var
  (disabled by default — API-first deployments don't need it).
- Canonical Uttera-stack port `9004` (TTS family; STT family uses
  `9005`). A reverse proxy can route by service family, so swapping
  `hotcold ↔ vllm` is a backend change only.

**What is *not* here**:
- Dynamic voice registry (`POST` / `DELETE /v1/voices`) — the current
  registry is file-based (`voices.json` + disk layout). A dynamic
  registry would be an additive minor if there is demand.
- Adhoc voice cloning on the streaming endpoint — `/v1/audio/speech/stream`
  uses registered voices only. Adhoc streaming would require latent
  computation on the request's critical path before the first chunk.
- In-repo benchmark harness. The canonical numbers live in
  [`uttera-benchmarks`](https://github.com/uttera/uttera-benchmarks).

See [API.md](API.md) for endpoint details, [HISTORY.md](HISTORY.md) for
why there are two TTS repos.

## Quickstart

```bash
git clone https://github.com/uttera/uttera-tts-vllm.git
cd uttera-tts-vllm
cp .env.example .env      # tweak VOXCPM_MODEL, VLLM_* if needed
./setup.sh                # creates venv, installs nano-vllm-voxcpm,
                          # pre-downloads the model and 6 voices
source venv/bin/activate
uvicorn main_tts:app --host 0.0.0.0 --port 9004
```

Then:

```bash
# Standard voice
curl -X POST http://localhost:9004/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hola mundo","voice":"alloy"}' \
  -o hello.mp3

# Adhoc voice cloning
curl -X POST http://localhost:9004/v1/audio/speech \
  -F "input=Hola mundo" \
  -F "speaker_wav=@my_voice.wav" \
  -o hello.wav
```

## Cache opt-out — per-request privacy control

By default the server caches synthesised audio on disk to accelerate repeated requests, keyed by `MD5(model | voice | speed | format | params | text)`. For **privacy-sensitive workloads** (medical/legal dictation, personal messages, one-off text a user does not want persisted on the server), a client can opt the single request out of both the read and the write paths of the cache. The synthesised audio still reaches the caller, but the server writes nothing to disk about that specific request.

Three equivalent ways to request it — use whichever the client finds most natural:

```bash
# (1) JSON body field — OpenAI-style extension
curl -X POST http://localhost:9004/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Notas privadas","voice":"alloy","cache":false}' \
  -o out.mp3

# (2) Multipart form field — accepts 0 / false / no / off
curl -X POST http://localhost:9004/v1/audio/speech \
  -F input='Notas privadas' -F voice=alloy -F cache=false -o out.mp3

# (3) Standard HTTP header — no body changes needed
curl -X POST http://localhost:9004/v1/audio/speech \
  -H 'Cache-Control: no-cache' -H 'Content-Type: application/json' \
  -d '{"input":"Notas privadas","voice":"alloy"}' \
  -o out.mp3
```

Every response carries an `X-Cache` header so the client can verify the decision — `HIT`, `MISS`, `BYPASS`, `ADHOC`, or `DISABLED`. Full reference, including the exact semantics of each `X-Cache` value and a note on what the opt-out does **not** cover (upstream logging outside this server), lives in [API.md](API.md#cache-opt-out--per-request-privacy-control).

The opt-out is per-request; the operator's `CACHE_TTL_MINUTES` default (see [Configuration](#configuration)) is unaffected.

## Configuration

All tuning is env var driven. See [.env.example](.env.example) for the
full surface. The most common overrides:

| Variable | Default | Notes |
|---|---|---|
| `VOXCPM_MODEL` | `openbmb/VoxCPM2` | HF repo of the model. |
| `SERVED_MODEL_NAME` | `tts-1` | Advertised via `/v1/models`. |
| `DEFAULT_VOICE` | `alloy` | Fallback when client omits `voice`. |
| `VLLM_GPU_MEM_UTIL` | `0.45` | Fraction of VRAM the engine is allowed to claim. |
| `VLLM_MAX_NUM_SEQS` | `32` | Maximum in-flight sequences. |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `16384` | Batching budget per decoder step. |
| `VOXCPM_INFERENCE_TIMESTEPS` | `10` | VoxCPM2-specific denoising steps. |
| `VOICE_PRELOAD` | `alloy` | Voices resident at startup (comma-separated); the rest load lazily. |
| `VOICE_CACHE_MAX` | `4` | LRU cap on lazily-loaded voice latents. |
| `AUDIO_CACHE_DIR` | `assets/cache` | SHA-256 audio cache location. |
| `CACHE_TTL_MINUTES` | `60` (1 hour) | 0 to disable. |
| `REF_MAX_SECONDS` | `20` | Cap on the adhoc-cloning reference sample. |
| `ENGINE_FAIL_THRESHOLD` | `3` | Consecutive 5xx that open the circuit breaker. |
| `UTTERA_OFFLINE` | `0` | `1` forces local-cache-only model loading. |
| `PORT` | `9004` | HTTP port. |

## Observability (`/metrics`)

`GET /metrics` returns Prometheus-format metrics for direct scraping
by Prometheus, Telegraf's `inputs.prometheus` plugin, or any other
OpenMetrics-compatible consumer. Metrics are prefixed with
`uttera_tts_` and use low-cardinality labels (no voice names, no
input text, no request IDs).

```toml
[[inputs.prometheus]]
  urls = ["http://tts-host:9004/metrics"]
  interval = "15s"
```

Key series:

| Metric | Type | Use |
|---|---|---|
| `uttera_tts_requests_total{endpoint,method,status}` | Counter | Per-endpoint request rate + status mix |
| `uttera_tts_request_duration_seconds{endpoint,method}` | Histogram | HTTP p50/p95/p99 (total RTT) |
| `uttera_tts_inflight_requests` | Gauge | Live load |
| `uttera_tts_synthesis_total{response_format,route,cache}` | Counter | Traffic mix across format × lane × cache decision (same semantics as `X-Route`/`X-Cache` headers) |
| `uttera_tts_characters_synthesised_total{response_format}` | Counter | Input chars synthesised — throughput proxy. Cache hits are not re-counted |
| `uttera_tts_inference_duration_seconds{op}` | Histogram | Per-call latency, `op` in `{synthesis, ffmpeg_encode}` — separates GPU time from CPU-encoder time |
| `uttera_tts_voices_loaded` | Gauge | Count of voices resident in VRAM |
| `uttera_tts_engine_ready` | Gauge | 1 once engine is warmed up |
| `uttera_tts_errors_total{type}` | Counter | Typed errors (`model` / `encoding`) |
| `uttera_tts_build_info{version,engine,model}` | Gauge | Version + model in the field (value always `1`) |

## Deployment

- **Docker**: `docker compose up -d` (GPU passthrough configured in
  `docker-compose.yml`).
- **systemd**: `uttera-tts-vllm.yml` is a ready-to-adapt unit file;
  install at `/etc/systemd/system/uttera-tts-vllm.service`.

## Hardware requirements

- GPU: NVIDIA with 32 GB+ VRAM recommended for VoxCPM2 at high
  concurrency. Smaller GPUs can run with reduced
  `VLLM_GPU_MEM_UTIL` / `VLLM_MAX_NUM_SEQS`.
- Blackwell (RTX 5090) supported with CUDA 12.8.
- `ffmpeg` on the system PATH for mp3/opus/flac encoding.

## 🛡 License

**Server source code**: [Apache License 2.0](LICENSE). Commercial use permitted.

**VoxCPM2 model weights** (OpenBMB): check the model card on
[HuggingFace](https://huggingface.co/openbmb/VoxCPM2) for the license
terms you must honour when deploying this server commercially. See
[NOTICE](NOTICE) for a consolidated attributions summary.

Created and maintained by [Hugo L. Espuny](https://github.com/fakehec),
with contributions acknowledged in [AUTHORS.md](AUTHORS.md).

## ☕ Community

If you want to follow the project or get involved:

- ⭐ Star this repo to help discoverability.
- 🐛 Report issues via the [issue tracker](../../issues).
- 💬 Join the conversation in [Discussions](../../discussions).
- 📰 Technical posts at [blog.uttera.ai](https://blog.uttera.ai).
- 🌐 Uttera Cloud: [https://uttera.ai](https://uttera.ai) (EU-hosted,
  solar-powered, subscription flat-rate).

---

*Uttera /ˈʌt.ər.ə/ — from the English verb "to utter" (to speak aloud, to
pronounce, to give audible expression to). Formally, the name is a backronym
of **U**niversal **T**ext **T**ransformer **E**ngine for **R**ealtime **A**udio
— reflecting the project's origin as a STT/TTS server and its underlying
Transformer architecture.*
