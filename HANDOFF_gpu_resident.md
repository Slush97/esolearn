# GPU-Resident scry-llm — Session Handoff (Round 2, 2026-05-08)

You are continuing work on branch `gpu-resident/scry-vision`. This doc supersedes the previous handoff (the breakdown experiment it called for has been done). Read in this order:

1. This doc.
2. Memory entry `project_scry_vision_gpu_residency.md` (durable cross-session context).
3. `crates/scry-llm/HACKING_GPU_BREAKDOWN.md` (research that justifies the next move).

## What's done

M1 (ScryGpuStorage enum, persistent matmul on Wgsl) and M2 (1/6 — GELU shader on Wgsl) shipped earlier. Recent investigation phase landed:

```
66fa7dc docs(scry-llm): competitive comparison — cuBLAS, PyTorch, and the gap
190f5d9 fix(scry-gpu): bring CUDA backend up to cudarc 0.19.3 API
08e00a5 bench(scry-llm): batched-dispatch POC validates 22-29% wins at common sizes
7b1abc5 bench(scry-llm): GPU breakdown bench identifies dispatch as dominant cost
```

## The big finding

PyTorch matmul on RTX 5070 Ti ≡ scry-gpu cuBLAS within 1% (both ~24.5 TFLOPS at 1024³). Our WGSL kernel: 10.69 TFLOPS. **The kernel-quality gap is the dominant source of our 2.6–6× slowdown vs PyTorch end-to-end.** scry-gpu already has a working cuBLAS path (`Device::cublas_matmul` at `crates/scry-gpu/src/device.rs:600`); it's just not plumbed into the *persistent* matmul.

## Next work — plumb cuBLAS into the persistent matmul path on CUDA

Today, `gpu_matmul_persistent` (in `crates/scry-llm/src/backend/scry_gpu.rs`, lines ~360–415) early-returns `None` on cuBLAS at line 377. The cuBLAS path uses the legacy `gpu_matmul` which uploads → computes → downloads on every call — defeating M1's residency guarantee.

To get cuBLAS users on the GPU-resident path:

1. **Add a cuBLAS arm to `gpu_matmul_persistent`.** Replace the early-return with a call to `ctx.dev.cublas_matmul(...)`. That function already takes `Buffer<f32>` inputs and writes to a `&mut Buffer<f32>`; no allocation work to do. Wrap the result in `ScryGpuStorage::Gpu`.
2. **Wire a CUDA transpose shader.** WGSL transpose lives at `crates/scry-gpu/src/shaders.rs` (used by `gpu_transpose`). Add an NVRTC-compiled CUDA equivalent. Set `transpose: Some(...)` for the cuBLAS branch of `init_scry_context` (currently `None` there).
3. **Wire a CUDA GELU shader.** Same pattern — WGSL GELU already exists, add CUDA, populate `gelu: Some(...)` for cuBLAS init. The GELU shader exists in scry-gpu's `shaders::elementwise::GELU` which today is WGSL only.
4. **Validate.** Re-run `gpu_breakdown` with `--features scry-gpu-cuda`. At 1024³ we should approach PyTorch's ~90 µs chain (down from 220 µs unbatched WGSL). At small sizes we expect more modest gains since launch overhead doesn't change.
5. **Update HACKING_GPU_BREAKDOWN.md** with the post-cuBLAS numbers and refresh the path-forward analysis.

### Why this and not batched dispatch

The batched-dispatch POC (commit 08e00a5) validated 22–29% savings on Vulkan. cuBLAS plumbing closes ~80% of the gap to PyTorch on NVIDIA. **CUDA is bigger leverage on the hardware most users have.** Once the cuBLAS persistent path works, batched dispatch becomes the right next move — it primarily helps Vulkan platforms (Apple via MoltenVK, AMD on Linux, integrated GPUs) where there's no CUDA fallback.

### Build & validation commands

```bash
# Build CUDA path. Env var is required: cudarc 0.19 is pinned to cuda-13010
# features but the local CUDA toolkit is 13.2. Not a code issue.
CUDARC_CUDA_VERSION=13010 cargo build -p scry-llm --features scry-gpu-cuda --release

# Re-run breakdown on CUDA after each step
CUDARC_CUDA_VERSION=13010 cargo bench -p scry-llm \
    --features scry-gpu-cuda --bench gpu_breakdown

# Cross-backend reference (real cuBLAS numbers, no scry-llm involved)
CUDARC_CUDA_VERSION=13010 cargo run -p scry-gpu \
    --example bench_cuda_compare --features "cuda vulkan" --release

# PyTorch reference (already set up at /tmp/torch-bench)
/tmp/torch-bench/.venv/bin/python /tmp/torch-bench/bench.py

# Standard tests / lint
cargo nextest run -p scry-llm --features scry-gpu
cargo clippy -p scry-llm --features scry-gpu-cuda
cargo fmt -p scry-llm -p scry-gpu
```

## Gotchas

- **`stash@{0}`** has unrelated WIP from a prior session (scry-learn async backward + PendingBackward; scry-llm chat features in `SamplingConfig` + `examples/chat.rs`). **Do not pop it onto this branch** — it belongs on separate branches off main. Sort it later.
- **`CUDARC_CUDA_VERSION=13010`** required for any CUDA build on this system. Until cudarc ships a `cuda-13020` feature, this stays.
- **scry-stt is missing a binary** that breaks workspace-wide builds. Use `-p` targeting always.
- **fp32 tolerance for chained ops** must be relative: `1e-4 * value.abs().max(1.0)`, not absolute.
- **GPG signing** may prompt for passphrase mid-commit. Stage messages in `/tmp/commit_msg` and use `git commit -F /tmp/commit_msg`.
- **Don't bump cudarc version** without explicit go-ahead — it could cascade through the CUDA backend's API surface (last bump rotted two call sites; see commit 190f5d9).

## Stop conditions

Stop and ask before:
- Adding heavy deps (candle-core, ort, tch) — multi-hour decisions.
- Touching scry-vision model files (resnet.rs, vit.rs, clip.rs) — M3/M4, comes after the M2 kernels.
- Modifying the `MathBackend` trait surface — affects scry-llm's whole LLM inference path.
- Merging this branch to main — wait for measurable wins vs PyTorch on real models, not just operators.

Otherwise: commit small, run the bench between steps, keep the tree clean.
