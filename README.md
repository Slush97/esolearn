# esolearn

A Rust ecosystem for machine learning, AI inference, and data visualization.

## Crates

The workspace is split into two families:

### scry — Machine Learning & AI Inference

| Crate | Description |
|-------|-------------|
| **scry-learn** | ML toolkit — decision trees, random forests, gradient boosting, neural networks, clustering, dimensionality reduction, SVMs, calibration. Pure Rust with optional GPU acceleration via scry-gpu. |
| **scry-llm** | Llama inference engine with CUDA, BLAS, MKL, DNNL, and scry-gpu backends. Supports safetensors loading and quantization. |
| **scry-stt** | Whisper speech-to-text with zero-copy model loading. Live microphone and dictation modes. |
| **scry-vision** | Vision inference — ResNet, CLIP, SCRFD face detection, ArcFace embeddings. ONNX runtime support and GPU preprocessing. |
| **scry-gpu** | Lightweight GPU compute layer (Vulkan via ash, optional wgpu). Shader compilation from WGSL to SPIR-V via naga. Shared backend for scry-learn, scry-llm, scry-stt, and scry-vision. |

### esoc — Graphics & Visualization

| Crate | Description |
|-------|-------------|
| **esoc-chart** | High-level charting API (histogram, scatter, bar, box plot, pie, heatmap, etc.) with express and grammar interfaces. |
| **esoc-gfx** | SVG-first 2D vector graphics engine. Optional PNG rasterization via resvg/tiny-skia. |
| **esoc-gpu** | wgpu GPU rendering backend — instanced mark rendering, SDF anti-aliasing, MSDF text. |
| **esoc-scene** | Arena-based scene graph with typed visual marks. Shared IR between chart logic and renderers. |
| **esoc-color** | OKLab/OKLCH perceptual color math, CVD simulation, palettes, gamut clipping. Zero dependencies. |
| **esoc-geo** | Map projections (Mercator, Equal Earth, Natural Earth, Albers USA), GeoJSON parsing, polygon simplification, bundled world/US geometries. |

## Building

```sh
# Check the full workspace
cargo check

# Run scry-learn tests
cargo test -p scry-learn

# Run with GPU features
cargo check -p scry-learn --features scry-gpu
cargo check -p scry-llm --features cuda
```

### Feature highlights

**scry-learn**: `csv`, `serde`, `polars`, `mmap`, `scry-gpu`, `experimental`

**scry-llm**: `cuda` (default), `blas`, `mkl`, `dnnl`, `scry-gpu`, `safetensors`, `bf16`, `quantize`

**scry-stt**: `blas` (default), `cuda`, `mkl`, `scry-gpu`, `safetensors`, `live`, `dictate`

**scry-vision**: `cuda`, `blas`, `mkl`, `scry-gpu`, `safetensors`, `onnx`, `gpu-preprocess`, `decode`

**esoc-chart**: `png`, `scry-learn`

**esoc-gfx**: `png`

**esoc-geo**: `geojson`, `bundled`

## Project structure

```
crates/
  scry-gpu/          GPU compute backend
  scry-learn/        Machine learning
  scry-llm/          LLM inference
  scry-stt/          Speech-to-text
  scry-vision/       Vision inference
  esoc-chart/        Charting API
  esoc-gfx/          2D graphics engine
  esoc-gpu/          GPU chart rendering
  esoc-scene/        Scene graph IR
  esoc-color/        Color system
  esoc-geo/          Geographic utilities
datasets/            Sample CSV datasets
research/            Design docs and roadmaps
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
