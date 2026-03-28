# CLAUDE.md — esolearn workspace

## Workspace Overview

Rust workspace with 11 crates in two families. Edition 2021, MSRV 1.83.0.

**scry family** (AI/ML inference):
- `scry-gpu` — Compute-only GPU backend (Vulkan/ash, WGSL→SPIR-V via naga)
- `scry-learn` — ML toolkit: trees, forests, boosting, neural nets, clustering, SVMs, TreeSHAP
- `scry-llm` — Llama inference (CUDA, BLAS, MKL, quantization, safetensors)
- `scry-stt` — Whisper speech-to-text (zero-copy models, live mic, dictation)
- `scry-vision` — Vision inference (ResNet, CLIP, SCRFD, ArcFace, ONNX)

**esoc family** (Graphics/visualization):
- `esoc-chart` — High-level charting API (grammar-of-graphics, SVG/PNG output)
- `esoc-gfx` — SVG-first 2D vector graphics engine
- `esoc-gpu` — wgpu GPU rendering (instanced marks, SDF, MSDF text)
- `esoc-scene` — Arena scene graph with typed visual marks
- `esoc-color` — OKLab/OKLCH perceptual color math (zero dependencies)
- `esoc-geo` — Map projections, GeoJSON, bundled geometries

## Build Commands

```bash
cargo check --workspace                    # full workspace check
cargo test -p <crate>                      # crate-specific tests
cargo clippy -p <crate>                    # lint (pedantic + nursery enabled)
cargo fmt --all -- --check                 # format check
cargo build -p scry-gpu --examples         # build GPU benchmarks
```

scry-stt has a missing binary (`mel_filters_80.bin`) that causes workspace-wide build to fail — use `-p` targeting instead.

## Key Feature Flags

- `scry-gpu`: `vulkan` (default), `bench-wgpu`
- `scry-learn`: `csv`, `serde`, `scry-gpu`, `polars`, `mmap`, `experimental`
- `scry-llm`: `cuda` (default), `blas`, `mkl`, `scry-gpu`, `safetensors`, `bf16`, `quantize`
- `scry-vision`: `cuda`, `blas`, `mkl`, `scry-gpu`, `safetensors`, `onnx`, `gpu-preprocess`

GPU features are optional — standard tests run without Vulkan/CUDA.

## Code Conventions

- **Errors**: Custom `#[non_exhaustive]` enums per crate with `thiserror`. Type alias `Result<T>`.
- **Docs**: `#![warn(missing_docs)]` on all crates. SPDX headers: MIT OR Apache-2.0.
- **Safety**: `#![deny(unsafe_code)]` on pure-Rust crates (esoc-chart, esoc-gfx, scry-learn). Unsafe allowed in GPU/FFI crates.
- **Clippy**: Workspace pedantic config with numerical-code exceptions (cast_precision_loss, many_single_char_names).
- **Modules**: `pub mod` for public API, `pub(crate) mod` for internals. Domain-organized (e.g., scry-learn: tree, ensemble, cluster, neural, metrics).

## System Requirements

- **Vulkan drivers** — required for scry-gpu tests/benchmarks
- **CUDA 13.0+** — optional, for scry-llm cuda feature
- **BLAS/MKL/DNNL** — optional compute backends

## Testing

- scry-learn has extensive test suites: correctness (sklearn reference), convergence, edge cases, golden reference, mathematical invariants
- scry-gpu tests require a Vulkan GPU (fail gracefully with `NoDevice` if unavailable)
- Benchmarks: `cargo run -p scry-gpu --example bench_compute --release`

## scry-gpu Architecture

The GPU crate is the shared compute backend. Key design:
- `Device` → `Backend` trait → `VulkanBackend` (Metal planned)
- Thread-safe: `Mutex<SubmissionContext>` serializes dispatch, `Mutex<Queue>` serializes submission
- `Buffer<T>` — typed, device-local, staging-based upload/download
- `Kernel` — precompiled pipeline (inner `Arc<VulkanKernelInner>` for batch retention)
- `Batch` — multi-dispatch single command buffer, own fence
- Pipeline cache persisted to `~/.cache/scry-gpu/<vendor>-<device>.bin`
- Built-in shaders: tiled matmul (16x16, coarse 64x64), pairwise Euclidean distance
