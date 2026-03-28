# Next Session Prompt

Copy-paste this to start the next conversation:

---

I'd like to continue work on the scry-gpu crate. Read CLAUDE.md first for workspace context.

## What was done last session

The crate was hardened for production (commit 8c4fea0): thread safety via Mutex<SubmissionContext>/Queue/CommandPool, batch gets own fence, Arc<VulkanKernelInner> for batch kernel retention, removed device_wait_idle from Drop, 5s fence timeout, structured BackendOp errors, push constant size reflection from naga, VkPipelineCache persisted to disk, fixed batch pool overflow bug, checked_mul overflow guards.

All 29 tests pass, clippy clean, benchmarks run. RTX 5070 Ti numbers: ~35µs kernel compile, ~8µs batched dispatch, matmul peaks at 14.7 TFLOPS (26.6% of 55.4 TFLOPS FP32 peak).

## What to do next (pick one or more, in priority order)

### 1. 8×8 register-tiled matmul shader (highest value)
Add a new matmul shader to `crates/scry-gpu/src/shaders.rs` with 8×8 thread coarsening (64 outputs per thread) and vec4 global loads. This should roughly double GFLOPS from ~25% to ~45% of peak. Key challenge: NVIDIA's SPIR-V compiler spills `array<f32, 32+>` — you may need to use explicit scalar variables instead of arrays for the 64-element accumulator. Test with `cargo run -p scry-gpu --example bench_compute --release`. The existing `COARSE_64X64` shader in shaders.rs shows the 4×4 pattern to extend from.

### 2. Staging buffer pool
Add a buffer pool to `VulkanBackend` that reuses host-visible staging buffers across upload/download calls instead of alloc+free every time. Key file: `crates/scry-gpu/src/backend/vulkan.rs`. The `upload()`, `read_back()`, and `free_buffer()` methods are the touch points. A simple `Vec<(u64, VulkanBuffer)>` sorted by size with a "find best fit" lookup would suffice.

### 3. Async dispatch API
Add a `Device::dispatch_async()` that returns a `GpuFuture` (fence handle) instead of blocking. The caller can poll/wait later. This enables CPU-GPU overlap in inference pipelines. Bigger refactor — touches the public API.

## Key files
- `crates/scry-gpu/src/backend/vulkan.rs` — Vulkan backend (1300 lines, the core)
- `crates/scry-gpu/src/shaders.rs` — built-in WGSL shaders
- `crates/scry-gpu/src/device.rs` — public Device API
- `crates/scry-gpu/src/error.rs` — GpuError + BackendOp
- `crates/scry-gpu/examples/bench_compute.rs` — benchmark suite

## How to verify
```bash
cargo test -p scry-gpu                                    # 29 tests
cargo clippy -p scry-gpu                                  # should be clean
cargo run -p scry-gpu --example bench_compute --release   # full benchmarks
cargo run -p scry-gpu --example bench_kernel --release    # compile overhead
```
