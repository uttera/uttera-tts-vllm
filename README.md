# uttera-tts-vllm

<p align="center">
  <img src="docs/img/banner.png" alt="uttera.ai — The voice layer for your AI" width="800">
</p>

High-throughput Text-to-Speech server built on **vLLM continuous batching**.
Optimized for cloud deployment and multi-tenant serving on large GPUs.

> **Status**: pre-alpha skeleton. Active development. See
> [ROADMAP.md](ROADMAP.md) for what's planned. First release expected
> Q2 2026.

## Positioning

| Use case | This repo | Sibling repo |
|---|---|---|
| Cloud, multi-tenant, large GPU (≥24 GB VRAM per process) | ✅ [uttera-tts-vllm](https://github.com/uttera/uttera-tts-vllm) | — |
| Home-lab, personal, small/mid GPU (8–16 GB) | — | [uttera-tts-hotcold](https://github.com/uttera/uttera-tts-hotcold) |

**Choose `uttera-tts-vllm` when**:
- You serve many concurrent users from a single large-VRAM GPU.
- Continuous batching matters for throughput.
- You're OK with the model resident in VRAM 24/7.

**Choose `uttera-tts-hotcold` when**:
- You have consumer GPUs (RTX 4070, 4080) and don't want to dedicate
  VRAM all the time.
- Personal or single-user deployment.
- Low baseline load with occasional bursts.

## Architecture

Built on [vLLM](https://github.com/vllm-project/vllm) and, for VoxCPM2
specifically, [nano-vllm-voxcpm](https://github.com/openbmb/nano-vllm-voxcpm).

- Continuous batching: multiple requests share the same forward pass.
- Paged KV-cache: efficient memory usage under concurrency.
- OpenAI-compatible API: drop-in for `openai.audio.speech.create()`.
- Multi-tenant features are opt-in via env vars (no code changes needed
  for single-tenant use).

## Benchmarks (preview)

Empirical results on 1× RTX 5090 (Blackwell sm_120, 32 GB VRAM), VoxCPM2
via nano-vllm-voxcpm, 400 concurrent requests:

| Config | Throughput | p50 latency | p95 latency |
|---|---|---|---|
| `max_num_seqs=48` (sweet spot) | **16.87 req/s** | 14.9 s | 24.2 s |
| `max_num_seqs=32` | 15.69 req/s | 14.9 s | 24.2 s |
| `max_num_seqs=64` | 15.21 req/s | 17.1 s | 26.0 s |

~10× the throughput of the hot/cold Coqui server on the same hardware.
See [benchmarks/](benchmarks/) for the methodology.

## Quickstart (coming soon)

```bash
git clone https://github.com/uttera/uttera-tts-vllm.git
cd uttera-tts-vllm
./setup.sh                       # installs vLLM + nano-vllm-voxcpm + deps
./scripts/run.sh                 # starts the server on port 9004
```

## Hardware requirements

- GPU: NVIDIA with 24 GB+ VRAM (RTX 3090, 4090, 5090, A6000, H100…).
- Blackwell (RTX 5090) supported with CUDA 12.8 + PyTorch 2.11 +
  flash-attn built for sm_120. See [docs/blackwell.md](docs/blackwell.md)
  for the build procedure.

## License

[Apache License 2.0](LICENSE). The VoxCPM2 model is also Apache-2.0
(permissive for commercial use). See [NOTICE](NOTICE) for full attributions.

---

*Uttera /ˈʌt.ər.ə/ — from the English verb "to utter" (to speak aloud).
Also the backronym **U**niversal **T**ext **T**ransformer **E**ngine for
**R**ealtime **A**I.*
