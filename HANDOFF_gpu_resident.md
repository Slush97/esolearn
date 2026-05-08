# GPU-Resident scry-llm — Session Handoff (Round 4, 2026-05-09)

You are continuing work on branch `gpu-resident/scry-vision`. This round closed the async-dispatch refactor the previous handoff called for. Read in this order:

1. This doc.
2. Memory entry `project_scry_vision_gpu_residency.md` (durable cross-session context).
3. `crates/scry-llm/HACKING_GPU_BREAKDOWN.md` (research + post-cuBLAS + post-async sections at the bottom).

## What's done

M1 (ScryGpuStorage enum, persistent matmul on Wgsl) and M2 (1/6 — GELU shader on Wgsl) shipped earlier. Recent commits and in-flight work:

```
3413894 docs: handoff Round 2 — next move is cuBLAS into persistent matmul
66fa7dc docs(scry-llm): competitive comparison — cuBLAS, PyTorch, and the gap
190f5d9 fix(scry-gpu): bring CUDA backend up to cudarc 0.19.3 API
08e00a5 bench(scry-llm): batched-dispatch POC validates 22-29% wins at common sizes
7b1abc5 bench(scry-llm): GPU breakdown bench identifies dispatch as dominant cost
```

Round 3 (cuBLAS plumbing) and Round 4 (async dispatch) ship together in this handoff's commit. cuBLAS is now wired into `gpu_matmul_persistent` with CUDA-compiled `TRANSPOSE_CUDA` / `GELU_CUDA` helpers; per-call `stream.synchronize()` is gone from the persistent path. Storage-boundary sync moved to `CudaBuffer::read_back` — required because cudarc's `clone_dtoh` on plain `Vec<T>` does not block (`SyncOnDrop::Sync(None)` is a no-op).

Headlines:
- **cuBLAS:** matmul+gelu chain at 1024³ dropped 238 µs (WGSL) → 128 µs — 1.85×, vs PyTorch's 90 µs. At small (128×256×128) the chain dropped 4×.
- **Async dispatch:** chained-total at small dropped another −19% (34.09 → 27.46 µs). Medium/large unchanged because download dominates the bench. Per-stage matmul/gelu numbers are now misleading under async — only the chained total is meaningful.

## Strategic direction — CUDA first

**Decision (2026-05-08):** scry-llm/scry-vision optimization work focuses primarily on CUDA. PyTorch is also CUDA-first, so that's where parity is reachable; Apple/AMD users have other inference tooling (llama.cpp + Metal, etc.) and the Vulkan substrate gap is structural, not solvable by more engineering. WGSL kernels stay correct and supported, but performance investment goes into CUDA.

Concrete consequences:
- New ops (M2 backlog: softmax, layernorm, batchnorm, pool, conv2d) ship CUDA-first; WGSL versions can lag or stay CPU-fallback indefinitely if they're not in a hot path.
- The Vulkan batched-dispatch port (validated by `gpu_batched_poc`, 22–29% wins) is **off the active backlog**. The POC bench stays for reference but isn't the next move anymore.
- The yardstick for "is this fast enough" is PyTorch on the same NVIDIA GPU, not self-vs-self.

## Round 5 (also in this commit) — alloc_uninit

Discovered that the proposed "workspace pool" was solving the wrong problem. cudarc's `Stream::alloc_zeros` already hits CUDA's stream-ordered memory pool (`cuMemAllocAsync`) — actual alloc churn is sub-microsecond. The wasted work was the `cuMemsetD8Async` dispatch that `alloc_zeros` adds: a separate kernel that zero-fills buffers we're about to overwrite.

Fix: added `Backend::alloc_uninit` (CUDA overrides to skip the memset; default impl falls back to `alloc`) and `Device::alloc_uninit<T>`. scry-llm's persistent path uses it for all outputs the kernel fully overwrites — matmul C, GELU output, transpose output. Tests pass; numerical equivalence preserved.

Bench delta (small chain): 27.46 → 25.39 µs, −7.5%. Per-stage GELU at small dropped −45% (3.27 → 1.80 µs) — the zero-init dispatch was nearly half the stage time.

Cumulative wins this branch (cuBLAS already in place at start, async dispatch + alloc_uninit on top): **34.09 → 25.39 µs at small, −25.5%**.

A full Drop-wrapper buffer pool was deferred — the alloc-churn-on-top-of-CUDA-pool win is marginal once zero-init is gone, and the wrapper would invade `ScryGpuStorage::Gpu` and every consumer that derefs `Arc<Buffer<f32>>`. Revisit only if a real-model bench shows alloc as a meaningful fraction of forward-pass time.

## Next work — tune `GPU_MIN_ELEMENTS` on CUDA

The 65,536-element cutoff was picked when small-matmul was 0.18 TFLOPS; on cuBLAS it's now 0.70+ TFLOPS (4× faster). With async dispatch + uninit alloc the per-call overhead is also lower. The threshold is almost certainly too conservative.

Approach: write a small sweep bench that times CPU vs GPU at sizes spanning 1K–256K elements (M=K=N varies), pick the crossover where GPU starts beating CPU+BLAS for matmul and CPU for GELU. Adjust both `GPU_MIN_ELEMENTS` (matmul) and `GPU_ELEMENTWISE_MIN` (GELU) constants.

Should be a 1-hour task. The reward is small (workloads in the 16K–64K element range start using GPU and getting cuBLAS speeds) but it's a clean closeout before moving to M2 kernels.

## Backlog after threshold tuning

In rough priority order, all CUDA-focused:

1. **M2 CUDA kernels** — softmax, layernorm, batchnorm, then conv2d (the hard one). Each unblocks a class of vision/transformer models. WGSL versions can follow opportunistically.
2. **Real model bench**: pick a small-but-real workload (BERT-base, ViT-small, ResNet-18) and time end-to-end on CUDA vs PyTorch. Operator-level wins don't count until they show up in a real forward pass.

## Image-gen path (longer horizon)

Strategic context: this branch is phase 1 of the broader image-generation roadmap (per `project_scry_vision_gpu_residency.md` and `project_gpu_persistent_tensors.md`). After M2 kernels land, the natural next milestone is a tiny VAE decoder driven by scry-llm's autoregressive loop — the smallest end-to-end "image in → image out" target that exercises the full GPU-resident pipeline. conv2d is the gating kernel.

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
- Adding async-sync semantics across scry-gpu's backend trait — talk through the API shape first; mirroring Vulkan-style fences vs CUDA streams cleanly is a design call.
- Merging this branch to main — wait for measurable wins vs PyTorch on real models, not just operators.

Otherwise: commit small, run the bench between steps, keep the tree clean.
