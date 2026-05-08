# CLAUDE.md — esolearn workspace

Rust workspace, edition 2021, MSRV 1.85.0. 11 active crates in two families: `scry-*` for AI/ML inference and computer vision, `esoc-*` for graphics/visualization. `scry-app` and `scry-stt` are excluded from the workspace; `esoc-gpu` lives under `archived/`.

## Global Commands

```bash
cargo nextest run -p <crate>           # tests for one crate (use nextest, not cargo test)
cargo nextest run                      # all tests in parallel
cargo test --doc -p <crate>            # doc-tests (nextest doesn't run these)
cargo check --workspace                # full workspace check
cargo clippy -p <crate>                # lint (pedantic + nursery)
cargo fmt --all -- --check             # format check
```

## Build Speed Setup

Configured for fast incremental compiles and tests:

- **`.cargo/config.toml`** — uses **mold** linker via clang (~2-5× faster linking).
- **Workspace `Cargo.toml`** — `[profile.dev.package."*"]` and `[profile.test.package."*"]` set `opt-level = 1` and `debug = "line-tables-only"` for dependencies. Heavy deps (wgpu, lyon, regex) run faster, link smaller, panic backtraces still readable. `[profile.dev]` uses `split-debuginfo = "unpacked"` for faster Linux linking.
- **`.config/nextest.toml`** — default profile warns on tests >30s, kills hangs >2min; `ci` profile retries flaky tests.

**Always prefer `cargo nextest run` over `cargo test`** — nextest runs each crate's test binary in parallel rather than serially. Doc-tests aren't covered by nextest; run those occasionally with `cargo test --doc`.

## Code Conventions

- **Errors** — `#[non_exhaustive]` enums per crate with `thiserror`. Type alias `Result<T>`.
- **Docs** — `#![warn(missing_docs)]` on all crates. SPDX headers: MIT OR Apache-2.0.
- **Safety** — `#![deny(unsafe_code)]` on pure-Rust crates (esoc-chart, esoc-gfx, scry-learn, esoc-color). Unsafe allowed in GPU/FFI crates only.
- **Modules** — `pub mod` for public API, `pub(crate) mod` for internals. Domain-organized.
- **Clippy** — workspace pedantic config with numerical-code exceptions (`cast_precision_loss`, `many_single_char_names`, etc.).

## System Requirements

- **Vulkan drivers** — required for scry-gpu tests/benchmarks
- **CUDA 13.0+** — optional, for scry-llm `cuda` feature
- **BLAS / MKL / DNNL** — optional compute backends

---

## scry family — AI/ML inference

### scry-gpu

Compute-only GPU backend (Vulkan/ash, WGSL→SPIR-V via naga). Shared compute substrate for the rest of the scry crates.

- **Architecture** — `Device` → `Backend` trait → `VulkanBackend` (Metal planned). Thread-safe: `Mutex<SubmissionContext>` serializes dispatch, `Mutex<Queue>` serializes submission.
- **Types** — `Buffer<T>` (typed, device-local, staging upload/download); `Kernel` (precompiled pipeline, inner `Arc<VulkanKernelInner>` for batch retention); `Batch` (multi-dispatch single command buffer with own fence).
- **Pipeline cache** — persisted to `~/.cache/scry-gpu/<vendor>-<device>.bin`.
- **Built-in shaders** — tiled matmul (16×16, coarse 64×64), pairwise Euclidean distance.
- **Features** — `vulkan` (default), `cuda`, `bench-wgpu`.
- **Testing** — requires Vulkan GPU; tests fail gracefully with `NoDevice` if unavailable.
- **Benchmarks** — `cargo run -p scry-gpu --example bench_compute --release`.

### scry-learn

ML toolkit: trees, forests, boosting, neural nets, clustering, SVMs, TreeSHAP.

- **Features** — `csv`, `serde`, `scry-gpu`, `cuda` (scry-gpu + cuBLAS), `polars`, `mmap`, `experimental`.
- **Testing** — extensive suites: correctness (sklearn reference), convergence, edge cases, golden reference, mathematical invariants.
- **Safety** — `#![deny(unsafe_code)]`.

### scry-llm

Llama / GPT-2 inference — CUDA, BLAS, MKL, quantization, safetensors.

- **Scope** — inference only; no training, no autograd.
- **Features** — `cuda` (default), `blas`, `mkl`, `scry-gpu`, `scry-gpu-cuda` (scry-gpu + cuBLAS), `safetensors`, `bf16`, `quantize`.
- **Testing** — CUDA smoke tests fail without GPU hardware (pre-existing, not blocking).

### scry-vision

Vision inference: ResNet, CLIP, SCRFD, ArcFace, ONNX.

- **Features** — `cuda`, `blas`, `mkl`, `scry-gpu`, `scry-gpu-cuda`, `safetensors`, `onnx`, `gpu-preprocess`.

### scry-cv

Classical computer vision — feature detection (ORB, BRISK), descriptor matching, optical flow (Farneback), registration, stereo, segmentation. Targets verified gaps in the Rust CV ecosystem.

- **Features** — `image-interop` (image crate), `skia-interop` (tiny-skia).

---

## esoc family — Graphics / visualization

### esoc-color

OKLab / OKLCH perceptual color math. **Zero dependencies.** Foundation crate for the rest of esoc.

- f32 linear RGBA (GPU-native), OKLab perceptual space.
- **Safety** — `#![deny(unsafe_code)]`.

### esoc-scene

Arena scene graph with typed visual marks. Generational indices, 9 mark primitives, `BatchAttr<T>` for instanced rendering.

- **Depends on** — esoc-color.
- Data is f64 until mapped through Scale to f32 visual coords.

### esoc-chart

High-level charting API — grammar-of-graphics, SVG/PNG output.

- **Modules** — `grammar/`, `express/`, `compile/` (new); legacy `Figure` API preserved.
- **Examples** — require `--features scry-learn` to compile.
- **Safety** — `#![deny(unsafe_code)]`.

### esoc-gfx

SVG-first 2D vector graphics engine.

- **Modules** — `scene_svg` walks `SceneGraph` → SVG; legacy `Canvas` API preserved.
- **Safety** — `#![deny(unsafe_code)]`.

### esoc-geo

Map projections, GeoJSON, bundled geometries.

---

## Archived

- **`archived/esoc-gpu`** — wgpu rendering crate, beta; out of workspace pending Phase 6 (GPU rendering completion). Required `unsafe_code = "allow"` for bytemuck Pod/Zeroable impls; WGSL shaders per pass (rect/point/line/rule/text/tess); SDF AA in fragment shaders.
