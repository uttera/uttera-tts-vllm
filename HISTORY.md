# History

`uttera-tts-vllm` is the high-throughput sibling of [`uttera-tts-hotcold`](https://github.com/uttera/uttera-tts-hotcold). Both solve the same problem — a self-hosted, OpenAI-compatible text-to-speech API — but make opposite assumptions about the runtime.

## Why two repos?

- **`uttera-tts-hotcold`** targets home-lab, personal, and small-to-mid GPU deployments (8–16 GB). It runs a plugin pattern over `coqui-tts` (XTTS-v2) or VoxCPM2 on a single GPU, with a custom *hot worker + on-demand cold pool* architecture that spawns Python subprocesses dynamically. Great on a single 4080; works on a 4060.

- **`uttera-tts-vllm`** targets cloud, multi-tenant, and large-GPU deployments (≥24 GB). It delegates concurrency to **nano-vllm-voxcpm's continuous batching** — a single engine process serves many parallel TTS requests at near-optimal GPU utilisation by dynamically batching in-flight sequences on every decode step. The hot/cold pool disappears; the engine is the concurrency primitive.

Same API shape (OpenAI-compatible), same voice registry layout, same cache semantics; different engine, different concurrency model.

## Genesis

The first Uttera TTS server was `uttera-tts-hotcold`, which hit the natural limits of a single-process hot worker around 2–3 concurrent requests on a 24 GB GPU. Benchmarks on an RTX 5090 with VoxCPM2 + nano-vllm-voxcpm demonstrated ~17 rps at 400 concurrent requests — roughly 4× what the hotcold architecture achieves on the same hardware, at the cost of a permanent VRAM reservation. That trade-off mirrors exactly the one published for STT in [`uttera-stt-vllm`](https://github.com/uttera/uttera-stt-vllm) vs [`uttera-stt-hotcold`](https://github.com/uttera/uttera-stt-hotcold); see [`uttera-benchmarks`](https://github.com/uttera/uttera-benchmarks) for the canonical numbers once they land for TTS as well.

Rather than retrofit a second backend into hotcold (two code paths, two test matrices, two ways to reason about latency), we split: `uttera-tts-hotcold` keeps its niche, `uttera-tts-vllm` owns the high-throughput story.

## Acknowledgments

- **OpenBMB** for [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), the default TTS model here.
- **nano-vllm-voxcpm** maintainers for the async batching engine that makes this server's concurrency story possible.
- **OpenAI** for the audio API surface we mirror (`/v1/audio/speech` with `voice`, `response_format`, `speed`…).
- **HuggingFace** for model hosting and `huggingface_hub`.
- Contributors to sibling Uttera repos whose OSS scaffolding (LICENSE, NOTICE, community files, Docker/systemd templates) landed here verbatim.
