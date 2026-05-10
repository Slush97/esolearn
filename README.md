# esolearn

A Rust ecosystem for machine learning, AI inference, and data visualization.

## Crates

The workspace is split into two families:

### scry — Machine Learning & AI Inference

| Crate | Description |
|-------|-------------|
| **scry-learn** | ML toolkit — decision trees, random forests, gradient boosting, neural networks, clustering, dimensionality reduction, SVMs, calibration, TreeSHAP. Pure Rust with optional GPU acceleration via scry-gpu. |
| **scry-llm** | Llama / GPT-2 inference engine with CUDA, BLAS, MKL, DNNL, and scry-gpu backends. Supports safetensors loading and quantization. |
| **scry-vision** | Vision inference — ResNet, CLIP, SCRFD face detection, ArcFace embeddings, SAM. ONNX runtime support and GPU preprocessing. |
| **scry-cv** | Classical computer vision — feature detection (ORB, BRISK), descriptor matching, optical flow (Farneback), registration, stereo, segmentation. Targets verified gaps in the Rust CV ecosystem. |
| **scry-diffusion** | Stable Diffusion inference — CLIP text encoder, UNet with cross-attention, VAE decoder, DDIM scheduler. Runs SD-1.5 on the scry-gpu CUDA backend. |
| **scry-stt** | Whisper speech-to-text with zero-copy model loading. Live microphone and dictation modes. (Workspace-excluded.) |
| **scry-gpu** | Compute-only GPU backend (Vulkan via ash). Shader compilation from WGSL to SPIR-V via naga. Built-in tiled matmul + custom CUDA kernels. Shared substrate for scry-learn, scry-llm, scry-vision, and scry-diffusion. |

### esoc — Graphics & Visualization

| Crate | Description |
|-------|-------------|
| **esoc-chart** | High-level charting API (histogram, scatter, bar, box plot, pie, heatmap, etc.) with express and grammar interfaces. |
| **esoc-gfx** | SVG-first 2D vector graphics engine. Optional PNG rasterization via resvg/tiny-skia. |
| **esoc-scene** | Arena-based scene graph with typed visual marks. Shared IR between chart logic and renderers. |
| **esoc-color** | OKLab/OKLCH perceptual color math, CVD simulation, palettes, gamut clipping. Zero dependencies. |
| **esoc-geo** | Map projections (Mercator, Equal Earth, Natural Earth, Albers USA), GeoJSON parsing, polygon simplification, bundled world/US geometries. |

## Building

```sh
# Check the full workspace
cargo check --workspace

# Run tests for one crate (use nextest, not cargo test)
cargo nextest run -p scry-learn

# Run all tests in parallel
cargo nextest run

# Run with GPU features
cargo check -p scry-learn --features scry-gpu
cargo check -p scry-llm --features cuda
```

MSRV is **Rust 1.85.0**. The workspace uses the [mold](https://github.com/rui314/mold) linker for ~2-5× faster linking — see `.cargo/config.toml`.

### Feature highlights

**scry-learn**: `csv`, `serde`, `polars`, `mmap`, `scry-gpu`, `cuda`, `experimental`

**scry-llm**: `cuda`, `blas`, `mkl`, `dnnl`, `scry-gpu`, `scry-gpu-cuda`, `safetensors`, `tokenizer`, `bf16`, `quantize`

**scry-vision**: `cuda`, `blas`, `mkl`, `scry-gpu`, `scry-gpu-cuda`, `safetensors`, `onnx`, `gpu-preprocess`, `decode`

**scry-cv**: `image-interop`, `skia-interop`, `ndarray-interop`, `serde`, `rayon`, `stereo`, `flow`, `background`, `segmentation`

**scry-diffusion**: `safetensors`, `decode`, `scry-gpu`, `scry-gpu-cuda`, `scry-gpu-bf16`, `profile`

**scry-stt** (workspace-excluded): `blas` (default), `cuda`, `mkl`, `scry-gpu`, `safetensors`, `live`, `dictate`

**esoc-chart**: `png`, `scry-learn`

**esoc-gfx**: `png`

**esoc-geo**: `geojson`, `bundled`

## Project structure

```
crates/
  scry-gpu/          GPU compute backend (Vulkan + CUDA kernels)
  scry-learn/        Machine learning toolkit
  scry-llm/          Llama / GPT-2 inference
  scry-vision/       Vision inference (ResNet, CLIP, SCRFD, SAM)
  scry-cv/           Classical computer vision (ORB, optical flow, stereo)
  scry-diffusion/    Stable Diffusion inference
  scry-stt/          Speech-to-text (workspace-excluded)
  esoc-chart/        Charting API
  esoc-gfx/          2D graphics engine (SVG)
  esoc-scene/        Scene graph IR
  esoc-color/        Color system (OKLab/OKLCH)
  esoc-geo/          Geographic utilities (projections, GeoJSON)
datasets/            Sample CSV datasets
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
