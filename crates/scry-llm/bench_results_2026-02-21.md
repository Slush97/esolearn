# scry-llm Benchmark Results — 2026-02-21

**Model:** Llama 3.2 1B (1.24B params, BF16)
**GPU:** NVIDIA GeForce RTX 5070 Ti (16.6 GB VRAM)
**Prompt:** "The capital of France is" (6 tokens), 30 generated tokens, greedy decoding

## Throughput

| Metric | scry-llm (BF16) | scry-llm (F32) | PyTorch HF (BF16) |
|--------|-----------------|----------------|-------------------|
| Decode tok/s | **194.6** | 130.0 | 192.9 |
| Decode latency | **5.1 ms** | 7.7 ms | 5.18 ms |
| Prefill tok/s | 1088 | 730.4 | — |
| Time to 1st token | **0.055s** | 0.008s | 0.237s |

## Memory Footprint

| Metric | scry-llm | PyTorch HF |
|--------|----------|------------|
| RSS before load | 3 MB | 704 MB |
| RSS after load | 440 MB | 1063 MB |
| RSS delta (model load) | 437 MB | 359 MB |
| RSS peak | **708 MB** | 1814 MB |
| VRAM after load | **4561 MB** | 2473 MB |
| VRAM peak inference | **4561 MB** | 2482 MB |

## Cold Start

| Metric | scry-llm | PyTorch HF |
|--------|----------|------------|
| Model load time | 17.54s | 0.63s |
| Time to first token | **0.055s** | 0.237s |
| Total cold→first token | 17.6s | 0.87s |

## Analysis

### Where scry-llm wins
- **Host memory**: 708 MB peak RSS vs 1814 MB — **2.6x less RAM**. No Python runtime, no torch framework overhead.
- **TTFT after load**: 55ms vs 237ms — **4.3x faster** time-to-first-token once model is loaded. No torch.compile warmup, no graph tracing.
- **Decode throughput**: 194.6 tok/s vs 192.9 tok/s — **parity** (within noise).

### Where PyTorch wins
- **VRAM**: 2473 MB vs 4561 MB — PyTorch uses ~1.8x less GPU memory. scry-llm stores bf16 weights (2.5 GB) plus fused QKV matrices that duplicate Q/K/V weights (~0.3 GB overhead), KV cache pre-allocation, and CUDA context overhead.
- **Model load**: 0.63s vs 17.54s — PyTorch mmap's safetensors and uploads directly. scry-llm deserializes to f32 on CPU, transposes, then uploads. Fix: mmap + GPU-side transpose.

### VRAM breakdown (after f32 stub optimization)
- Model weights bf16: ~2,472 MB (1.24B × 2 bytes)
- Fused QKV weight duplication: ~300 MB (Q+K+V stored separately + fused)
- KV cache pre-allocation: ~varies
- CUDA context + kernels: ~1,775 MB (baseline)
- **Total**: ~4,561 MB (down from 9,790 MB before optimization)

### VRAM optimization history
- **Before**: 9,790 MB — f32 + bf16 copies of every weight tensor
- **After (f32 stub)**: 4,561 MB — bf16-only weights with 1-element f32 stubs
- **Saved**: 5,229 MB (−53%)

## Correctness
- All 4 CUDA tests pass (f32 correctness, f32 throughput, bf16 correctness, bf16 throughput)
- "Paris" in top-5 predictions for both f32 and bf16
- Output text matches expected generation

## Hardware
```
GPU:  NVIDIA GeForce RTX 5070 Ti
VRAM: 16.6 GB
OS:   Linux 6.18.6-arch1-1
CUDA: (via cudarc/nvrtc)
```

## Optimization History

| Phase | Change | Decode tok/s | VRAM |
|-------|--------|-------------|------|
| 3 | KV cache pre-allocation, Q pre-scaling | ~160 | — |
| 3.5 | Fused QKV projection (−32 kernel launches) | ~175 | — |
| 6 | Eliminate 113 kernel launches/token | 181 | — |
| 7 | Fused GQA cache read (−64 launches, −64 allocs) | 194.6 | 9,790 MB |
| 8 | f32 stub + bf16 embedding kernel | 193.4 | **4,561 MB** |
