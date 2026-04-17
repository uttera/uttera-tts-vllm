# API

`uttera-tts-vllm` exposes an OpenAI-compatible speech-synthesis surface, extended with file-based voice registry and adhoc voice cloning.

Base URL (default): `http://localhost:5100`

## `POST /v1/audio/speech`

Generate audio from text. Accepts either a **JSON body** (OpenAI classic) or **multipart/form-data** (required when using adhoc voice cloning via `speaker_wav`).

### Common fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `input` | string (required) | — | Text to synthesise. UTF-8. |
| `voice` | string | `alloy` | Name of a voice in `voices.json` (both `standard/` and `elite/`). |
| `response_format` | string | `mp3` | `mp3`, `wav`, `pcm`, `opus`, `flac`. |
| `speed` | float | `1.0` | Included in the cache key. |
| `cfg_value` | float | `2.0` | VoxCPM2-specific tuning knob. |
| `model` | string | `tts-1` | Ignored; the model is fixed by `VOXCPM_MODEL` at startup. |

### Multipart-only fields

| Field | Type | Notes |
|---|---|---|
| `speaker_wav` | file | Optional. When present, the voice is cloned from this audio file for this single request (adhoc path, Model C in the design doc). Cache is bypassed. |

### Example — JSON body

```bash
curl -X POST http://localhost:5100/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","voice":"alloy","input":"Hola mundo","response_format":"mp3"}' \
  -o hello.mp3
```

### Example — adhoc voice cloning

```bash
curl -X POST http://localhost:5100/v1/audio/speech \
  -F "input=Hola mundo" \
  -F "speaker_wav=@my_voice.wav" \
  -F "response_format=wav" \
  -o hello.wav
```

### Response headers

- `X-Route: HOT` — synthesised by the engine now.
- `X-Route: ADHOC` — synthesised by the engine now, cache was bypassed because the request used adhoc cloning.
- `X-Route: CACHE` — served from the on-disk audio cache.

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
  "version": "0.1.0",
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
