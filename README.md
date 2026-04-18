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

> **Status**: v1.0.0 — first stable release. Validated end-to-end on
> NVIDIA RTX 5090 (Blackwell, 32 GB) against the 40-prompt Spanish
> corpus: 1024/1024 OK at every burst profile, 600/600 OK under
> 5-minute sustained load, aggregate throughput plateaus near 4.3 rps
> (see [`uttera-benchmarks` Run 6](https://github.com/uttera/uttera-benchmarks/tree/master/results/2026-04-17-run6-vllm-tts40w)).
> API surface is now frozen behind semver — see [CHANGELOG.md](CHANGELOG.md)
> for the full list and [ROADMAP.md](ROADMAP.md) for v1.x plans.

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
  the MD5 audio cache, voice registry, and Redis self-registration
  protocol shared with the other Uttera repos.

**What is here in v1.0.0**:
- 6 standard OpenAI reference voices (alloy/echo/fable/onyx/nova/shimmer)
  precomputed at startup.
- Elite/custom voices via file-based registry (`voices.json` +
  `assets/voices/elite/`), reloadable without a server restart.
- **Adhoc voice cloning** via a `speaker_wav` file upload on
  `/v1/audio/speech` — the one feature the Whisper-stack siblings don't
  have.
- MP3 / WAV / PCM / Opus / FLAC response formats.
- Chunked streaming via `/v1/audio/speech/stream`.
- On-disk MD5 audio cache identical to `uttera-tts-hotcold`, with
  per-request opt-out for privacy-sensitive calls (see
  [**Cache opt-out**](#cache-opt-out--per-request-privacy-control)).

**What is *not* here** (yet — see [ROADMAP.md](ROADMAP.md)):
- Dynamic voice registry (POST/DELETE `/v1/voices`) — scheduled for
  a future v1.x minor.
- In-repo benchmark harness in `tests/`. The canonical numbers live in
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
| `VLLM_GPU_MEM_UTIL` | `0.85` | Fraction of VRAM the engine is allowed to claim. |
| `VLLM_MAX_NUM_SEQS` | `32` | Maximum in-flight sequences. |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `16384` | Batching budget per decoder step. |
| `VOXCPM_INFERENCE_TIMESTEPS` | `10` | VoxCPM2-specific denoising steps. |
| `AUDIO_CACHE_DIR` | `assets/cache` | MD5 audio cache location. |
| `CACHE_TTL_MINUTES` | `10080` (7 days) | 0 to disable. |
| `PORT` | `9004` | HTTP port. |
| `REDIS_URL` | _(empty)_ | Optional; enables self-registration for a router. |

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
