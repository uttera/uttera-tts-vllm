# Roadmap

`uttera-tts-vllm` is pre-alpha. This document tracks what is in flight and what is planned.

## v0.1.x — stabilise the scaffold

- [x] Single-process FastAPI wrapper around nano-vllm-voxcpm's `AsyncVoxCPM2ServerPool`.
- [x] `/v1/audio/speech`, `/v1/audio/speech/stream`, `/v1/voices`, `/admin/reload-voices`, `/v1/models`, `/health`.
- [x] File-based voice registry (Model A) with 6 standard OpenAI voices precomputed at startup.
- [x] Adhoc voice cloning (Model C) via `speaker_wav` multipart upload on `/v1/audio/speech`.
- [x] MD5 audio cache identical to `uttera-tts-hotcold`.
- [x] Redis self-registration (parity with the other Uttera repos).
- [x] CI workflow (lint + structure + optional GPU smoke), Dockerfile, systemd unit, `.env.example`.
- [ ] End-to-end smoke test script (`tests/smoke.sh`) — `curl` round-trip with one of the standard voices.
- [ ] Concurrency benchmark against `uttera-benchmarks/bench.py` and publication of the results in
      `uttera-benchmarks/results/`.

## v0.2 — dynamic voice registry (Model B)

- [ ] `POST /v1/voices` (multipart): upload `name` + `sample`, the server persists the WAV to
      `assets/voices/elite/<name>.wav`, computes latents, caches them, and extends `voices.json`.
- [ ] `DELETE /v1/voices/{name}`: remove an elite voice (standard voices are immutable).
- [ ] Concurrency story: read-write lock on the registry so synthesis and registry mutations
      don't race.
- [ ] Abuse controls: max samples per caller, max total bytes on disk, optional API-key gating
      for the admin verbs.

## v0.3 — production hardening

- [ ] Prometheus `/metrics` endpoint (in-flight, ema_rps, total_completed, total_errors,
      cache hit ratio, vram_free_gb).
- [ ] Structured logging with request IDs and synthesis duration.
- [ ] Streaming-with-adhoc-cloning: currently `/v1/audio/speech/stream` only accepts the
      registered voices because an adhoc upload inside a streaming response complicates the
      lifespan of the file handle. Rework once the registry work in v0.2 settles.
- [ ] `/health` deep-check: synthesise a one-sentence probe periodically to detect a hung
      engine earlier than a missing token update.

## v1.0 — parity with the hotcold sibling

- [ ] Feature-parity checklist vs. `uttera-tts-hotcold`: same endpoints, same response shapes,
      same env var surface where applicable.
- [ ] Deployment notes (single-tenant vs. multi-tenant, GPU sizing guide, head-to-head with
      hotcold on the same workload).
- [ ] Release v1.0.0 with a semver guarantee around `/health`, `/v1/audio/speech`, and
      `/v1/audio/speech/stream`.

## Not planned

- **Hot/cold worker pool.** Continuous batching replaces it; the point of this repo is to
  avoid that complexity.
- **Non-VoxCPM backends.** When vLLM upstream gains native support for another streaming
  TTS model, that will be a separate repo or a plugin, not a fork of this one.
- **CPU-only inference.** This is the GPU-oriented sibling; CPU users should use
  `uttera-tts-hotcold` with a light Coqui config.
