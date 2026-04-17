# Roadmap

`uttera-tts-vllm` shipped v1.0.0 on 2026-04-17 after end-to-end
validation on NVIDIA RTX 5090 / Blackwell against
[uttera-benchmarks Run 6](https://github.com/uttera/uttera-benchmarks/tree/master/results/2026-04-17-run6-vllm-tts40w)
(1024/1024 OK at every burst profile, 600/600 OK on the 5-minute
sustained run, aggregate throughput plateaus near 4.3 rps). The API
surface is now frozen behind semver. This document tracks what is in
flight for v1.x and what is out of scope.

## ✅ Shipped in v1.0.0

- Single-process FastAPI wrapper around nano-vllm-voxcpm's `AsyncVoxCPM2ServerPool`.
- Endpoints: `/v1/audio/speech`, `/v1/audio/speech/stream`, `/v1/voices`,
  `/admin/reload-voices`, `/v1/models`, `/health`.
- File-based voice registry (Model A) with 6 standard OpenAI voices
  precomputed at startup, elite voices loaded from `voices.json`.
- Adhoc voice cloning (Model C) via `speaker_wav` multipart upload on
  `/v1/audio/speech`.
- MD5 audio cache keyed by `(model, voice, speed, format, params, text)`
  with TTL from `CACHE_TTL_MINUTES`.
- Per-request cache opt-out via `{"cache": false}` body field *and*
  `Cache-Control: no-cache` header, with `X-Cache: HIT | MISS | BYPASS
  | ADHOC | DISABLED` response header.
- Redis self-registration (parity with the other Uttera repos).
- CI workflow (lint + structure + optional GPU smoke), Dockerfile,
  `docker-compose.yml`, systemd unit, `.env.example`.

## v1.1 — dynamic voice registry (Model B)

- [ ] `POST /v1/voices` (multipart): upload `name` + `sample`, the
      server persists the WAV to `assets/voices/elite/<name>.wav`,
      computes latents, caches them, and extends `voices.json`.
- [ ] `DELETE /v1/voices/{name}`: remove an elite voice (standard
      voices are immutable).
- [ ] Read-write lock on the registry so synthesis and registry
      mutations don't race.
- [ ] Abuse controls: max samples per caller, max total bytes on
      disk, optional API-key gating for the admin verbs.

## v1.2 — production hardening

- [ ] Prometheus `/metrics` endpoint (in-flight, ema_rps,
      total_completed, total_errors, cache hit ratio, vram_free_gb).
- [ ] Structured logging with request IDs and synthesis duration.
- [ ] Streaming-with-adhoc-cloning: currently
      `/v1/audio/speech/stream` only accepts the registered voices
      because an adhoc upload inside a streaming response complicates
      the lifespan of the file handle. Rework once the registry work
      in v1.1 settles.
- [ ] `/health` deep-check: synthesise a one-sentence probe
      periodically to detect a hung engine earlier than a missing
      token update.
- [ ] In-repo benchmark harness in `tests/bench.py` that mirrors
      `uttera-benchmarks/bench.py` for CI smoke runs. Canonical
      numbers will continue to live in the benchmarks repo.

## Not planned

- **Hot/cold worker pool.** Continuous batching replaces it; the
  point of this repo is to avoid that complexity. If you need
  multi-model co-tenancy on one GPU use `uttera-tts-hotcold`.
- **Non-VoxCPM backends.** When vLLM upstream gains native support
  for another streaming TTS model, that will be a separate repo or a
  plugin, not a fork of this one.
- **CPU-only inference.** This is the GPU-oriented sibling; CPU
  users should use `uttera-tts-hotcold` with a light Coqui config.
