# API

`uttera-tts-vllm` exposes an OpenAI-compatible speech-synthesis surface, extended with file-based voice registry and adhoc voice cloning.

Base URL (default): `http://localhost:5100`

## `POST /v1/audio/speech`

Generate audio from text. Accepts either a **JSON body** (OpenAI classic) or **multipart/form-data** (required when using adhoc voice cloning via `custom_voice_file`).

### Common fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `input` | string (required) | — | Text to synthesise. UTF-8. |
| `voice` | string | `alloy` | Name of a voice in `voices.json` (both `standard/` and `elite/`). |
| `response_format` | string | `mp3` | `mp3`, `wav`, `pcm`, `opus`, `flac`. |
| `speed` | float | `1.0` | Included in the cache key. |
| `cfg_value` | float | `2.0` | VoxCPM2-specific tuning knob. |
| `cache` | bool | `null` | Per-request cache opt-out. `false` (or `0`) tells the server neither to read from nor write to the audio cache for this request — the response will be freshly synthesised and nothing about the request is persisted on disk afterwards. `true` is the explicit opt-in; `null`/omitted follows the server default (cache on whenever `CACHE_TTL_MINUTES > 0`). See **Cache opt-out** below. |
| `model` | string | `tts-1` | Ignored; the model is fixed by `VOXCPM_MODEL` at startup. |

### Multipart-only fields

| Field | Type | Notes |
|---|---|---|
| `custom_voice_file` | file | **Canonical** name for the adhoc voice-cloning upload. When present, the voice is cloned from this audio file for this single request — the server does **not** persist the sample or any derived latents. Cache is bypassed. Accepts any libsndfile-readable format (wav, flac, mp3, ogg, m4a, …). |
| `speaker_wav` | file | Legacy alias of `custom_voice_file`, kept for v1.0.0 compatibility and for clients coming from the Coqui ecosystem. Identical semantics. If both fields are present on the same request, `custom_voice_file` wins. |

### Example — JSON body

```bash
curl -X POST http://localhost:5100/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","voice":"alloy","input":"Hola mundo","response_format":"mp3"}' \
  -o hello.mp3
```

### Example — adhoc voice cloning (stateless)

The upload is consumed in-request and never persisted on the server:

```bash
curl -X POST http://localhost:5100/v1/audio/speech \
  -F "input=Hola mundo" \
  -F "custom_voice_file=@my_voice.wav" \
  -F "response_format=wav" \
  -o hello.wav
```

`-F "speaker_wav=@my_voice.wav"` works identically (alias).

### Response headers

- `X-Route: HOT` — synthesised by the engine now.
- `X-Route: ADHOC` — synthesised by the engine now, cache was bypassed because the request used adhoc cloning.
- `X-Route: CACHE` — served from the on-disk audio cache.

- `X-Cache: HIT` — the bytes came from the on-disk cache (no synthesis ran).
- `X-Cache: MISS` — cache is enabled, but this entry had to be synthesised.
- `X-Cache: BYPASS` — the client asked us to skip the cache for this request.
- `X-Cache: ADHOC` — adhoc voice-cloning request, cache was never eligible.
- `X-Cache: DISABLED` — the operator has disabled the cache globally (`CACHE_TTL_MINUTES <= 0`).

### Cache opt-out — per-request privacy control

The server's audio cache speeds up repeated requests by storing a synthesised file on disk keyed by `MD5(model | voice | speed | format | params | text)`. For some workloads — sensitive text, medical or legal dictation, one-off personal messages — a client may want a guarantee that the server writes **nothing** to disk about the request. The cache opt-out provides that:

**Effect when opt-out is requested**

- The cache read path is skipped (no HIT is served and none is looked for).
- The cache write path is skipped (no file lands in `AUDIO_CACHE_DIR`).
- The synthesised audio is returned directly from a temp file that is unlinked as soon as the HTTP response is flushed.
- The response carries `X-Cache: BYPASS` so the client can verify the decision.

**Three equivalent ways to request it** (use whichever fits the client):

1. **JSON body** — OpenAI-style extension:
   ```bash
   curl -X POST http://localhost:5100/v1/audio/speech \
     -H 'Content-Type: application/json' \
     -d '{"input":"Notas privadas del paciente","voice":"alloy","cache":false}' \
     -o out.mp3
   ```

2. **Multipart form** — same contract, accepts any of `0`, `false`, `no`, `off`:
   ```bash
   curl -X POST http://localhost:5100/v1/audio/speech \
     -F input='Notas privadas del paciente' \
     -F voice=alloy -F cache=false -o out.mp3
   ```

3. **HTTP header** — standard `Cache-Control` semantics (works from any HTTP client without touching the body):
   ```bash
   curl -X POST http://localhost:5100/v1/audio/speech \
     -H 'Cache-Control: no-cache' -H 'Content-Type: application/json' \
     -d '{"input":"Notas privadas del paciente","voice":"alloy"}' \
     -o out.mp3
   ```

`Cache-Control: no-store` is accepted equivalently.

**Notes**

- The opt-out is per-request; the operator's `CACHE_TTL_MINUTES` default is not affected.
- Adhoc voice-cloning requests (with `custom_voice_file` or `speaker_wav`) are implicitly opt-out regardless of the `cache` field — these always return `X-Cache: ADHOC`.
- The opt-out does not control upstream logging of the request (if you run the server behind a reverse proxy or have request-body logging enabled in your own app, those still apply). This server itself logs only the standard uvicorn access line — method, path, status, response time — no request-body content is logged.

## `POST /v1/audio/speech/stream`

Stream the audio as `audio/wav` chunks as soon as the engine emits them. No cache. Registered voices only (adhoc cloning on streaming will land in a later release — see `ROADMAP.md`).

```bash
curl -N -X POST http://localhost:5100/v1/audio/speech/stream \
  -H 'Content-Type: application/json' \
  -d '{"voice":"nova","input":"Long text to stream…"}' \
  -o stream.wav
```

The first bytes are a 44-byte WAV header with the RIFF and data size fields set to `0xFFFFFFFF` (RIFF "unknown length" sentinel for streaming). Everything after is raw int16 PCM at 48 kHz mono.

## `GET /v1/voices`

Lists every voice whose latents are currently resident in memory, plus the configured default.

```json
{
  "object": "list",
  "data": [
    {"id": "alloy",   "object": "voice", "wav_path": "standard/alloy.wav"},
    {"id": "echo",    "object": "voice", "wav_path": "standard/echo.wav"},
    {"id": "fable",   "object": "voice", "wav_path": "standard/fable.wav"},
    {"id": "jarvis",  "object": "voice", "wav_path": "elite/jarvis.wav"},
    {"id": "nova",    "object": "voice", "wav_path": "standard/nova.wav"},
    {"id": "onyx",    "object": "voice", "wav_path": "standard/onyx.wav"},
    {"id": "shimmer", "object": "voice", "wav_path": "standard/shimmer.wav"}
  ],
  "default": "alloy"
}
```

## `POST /admin/reload-voices`

Re-reads `voices.json` and computes latents for any new entries (without restarting the engine). Drops cached latents for names that have been removed. Does not delete the WAV files themselves.

Use this after editing `voices.json` or dropping a new `.wav` into `assets/voices/elite/`.

## `GET /v1/models`

```json
{
  "object": "list",
  "data": [{"id": "tts-1", "object": "model", "created": 1776390000, "owned_by": "uttera"}]
}
```

`id` is controlled by the `SERVED_MODEL_NAME` env var (default `tts-1`, matching OpenAI's public identifier so unmodified SDK clients work).

## `GET /health`

```json
{
  "status": "ok",
  "version": "1.0.0",
  "engine": "nano-vllm-voxcpm",
  "model": "openbmb/VoxCPM2",
  "served_as": "tts-1",
  "engine_ready": true,
  "engine_error": null,
  "voices_loaded": ["alloy", "echo", "fable", "nova", "onyx", "shimmer"],
  "default_voice": "alloy",
  "routing": {"load_score": 0.12, "accepts_requests": true},
  "metrics": {
    "in_flight": 4,
    "total_completed": 1042,
    "total_errors": 0,
    "max_num_seqs": 32,
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.85
  }
}
```

`routing` matches `uttera-tts-hotcold`'s `/health` so an upstream router can consume both backends with the same schema.

## Authentication

No authentication in this repo by design. Deploy behind the Uttera gatekeeper (or any reverse proxy) for API keys, quotas, and rate limits.
