# Changelog

All notable changes to this crate are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] — 2026-05-07

Initial public release.

### Added

- **Vulkan compute backend** via `ash` 0.38 and `gpu-allocator` 0.27. Compute-only — no graphics state, no swapchain, no surfaces.
- **CUDA backend** via `cudarc` 0.19 (`cuda` feature). NVRTC source compilation through `Device::compile_cuda` and cuBLAS matmul through `Device::cublas_matmul`.
- **Device API.** `Device::auto()` picks the fastest available backend; `Device::with_backend(BackendKind)` pins one explicitly.
- **Typed buffers.** `Buffer<T: bytemuck::Pod>` with safe `upload`/`download`, alignment and staging handled internally. `Device::alloc::<T>(count)` for device-resident allocations.
- **Kernel compilation.** WGSL → SPIR-V through `naga` 24, with on-disk pipeline cache at `~/.cache/scry-gpu/<vendor>-<device>.bin`. `Device::compile` and `Device::compile_named` for named entry points.
- **Auto-dispatch.** `Device::dispatch(shader, &[buffers], invocations)` derives workgroup count from the shader's `@workgroup_size`. Manual dispatch via `Device::dispatch_configured` and `DispatchConfig` for 2D / 3D launches.
- **Push constants.** `Device::run_with_push_constants` and `DispatchConfig::push_constants` for shader uniforms without descriptor binding.
- **Batch dispatch.** `Device::batch()` records multiple kernels into one command buffer with a single fence wait, eliminating per-dispatch submission overhead. Internally pools Vulkan resources across batches.
- **Async dispatch.** `Batch::submit_async()` returns a `Ticket` for non-blocking submission; `Ticket::is_ready()` and `Ticket::wait()` allow CPU/GPU overlap.
- **Built-in shaders** in `scry_gpu::shaders`:
  - `matmul::TILED_16X16` — shared-memory tiled matmul
  - `matmul::COARSE_64X64` — coarse-tile variant (1 output per thread group, 64×64 tiles)
  - `matmul::COARSE_8X8` — vec4-accumulator variant for 128×128 tiles
  - Pairwise squared Euclidean distance
  - Element-wise activations (ReLU, sigmoid, tanh) for ML workloads
  - Subgroup reduction (sum)
  - Each WGSL shader has an equivalent CUDA C source under the `cuda` feature.
- **Subgroup queries.** `Device::subgroup_size()` reports the subgroup size for kernel tuning.
- **Staging buffer pool.** Upload/download paths reuse staging allocations across calls.
- **Cross-backend benchmarks** (`bench_wgpu_compare`, `bench_cuda_compare`) and per-kernel benchmarks (`bench_compute`, `bench_kernel`).

### Status

This is a 0.x release. Public API may change between minor versions. The Vulkan backend is the most exercised path; CUDA is functional but less widely tested. Metal is on the roadmap.

[Unreleased]: https://github.com/Slush97/esolearn/compare/scry-gpu-v0.1.0...HEAD
[0.1.0]: https://github.com/Slush97/esolearn/releases/tag/scry-gpu-v0.1.0
