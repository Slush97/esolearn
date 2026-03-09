# esoc-chart: Performance Benchmarks & Targets

> Concrete, measurable targets. Every claim of "beating" a competitor must be backed by numbers.

---

## 1. Rendering Performance Targets

### 1.1 SVG Output (esoc-gfx)

Target: publication-quality static output. Optimize for output quality and file size, not render speed.

| Metric | Target | Rationale |
|--------|--------|-----------|
| 100 points → SVG | < 1 ms | Instant feedback |
| 1K points → SVG | < 5 ms | Standard scatter/line |
| 5K points → SVG | < 25 ms | Practical SVG limit |
| 10K points → SVG | < 50 ms | Aggressive; most libs choke here |
| SVG file size (1K scatter) | < 80 KB | Plotly outputs ~300 KB for equivalent |
| SVG file size (100-pt bar) | < 15 KB | Compact markup |
| Auto-downsample to SVG | LTTB at 5K threshold | Prevent bloated SVGs |

**Measurement**: `std::time::Instant` around `compile_chart()` + `render_scene_svg()`.

### 1.2 GPU Rendering (esoc-gpu)

Target: beat every JS library; match Datoviz/native GPU state-of-the-art.

| Metric | Target | Competitor Reference |
|--------|--------|---------------------|
| 1K points | 60 FPS, < 0.5 ms/frame | All libs achieve this |
| 10K points | 60 FPS, < 1 ms/frame | Canvas libs start struggling |
| 100K points | 60 FPS, < 2 ms/frame | WebGL libs (deck.gl) achieve this |
| 1M points | 60 FPS, < 8 ms/frame | ChartGPU: 1M @ 60 FPS |
| 10M points | 30+ FPS | Datoviz: 10M @ 250 FPS (Vulkan) |
| Pan/zoom latency | < 16 ms (60 FPS) | Must feel instant |
| Cold start (first frame) | < 100 ms | wgpu device init + first render |

**Measurement**: `wgpu` timestamp queries or `Instant` around `renderer.render()`.

### 1.3 Compile Pipeline (Chart → SceneGraph)

| Metric | Target | Notes |
|--------|--------|-------|
| Simple chart (1 layer, 1K pts) | < 1 ms | Just scale mapping + mark creation |
| Complex chart (5 layers, 10K pts) | < 10 ms | Multiple layers + stats |
| With Bin stat (100K values) | < 5 ms | O(n) binning |
| With KDE stat (10K values) | < 20 ms | O(n × bandwidth) |
| With LTTB (1M → 5K) | < 5 ms | O(n) linear scan |

---

## 2. Memory Usage Targets

| Scenario | Target | Competitor Reference |
|----------|--------|---------------------|
| Empty chart overhead | < 1 KB | Struct sizes only |
| 1K scatter (SVG path) | < 500 KB total | Plotly: several MB |
| 10K scatter (GPU) | < 5 MB | uPlot: 12 MB for 3.6K streaming |
| 100K scatter (GPU) | < 20 MB | Chart.js: 77 MB for 3.6K (!) |
| 1M scatter (GPU) | < 100 MB | GPU buffer = 1M × 64 bytes = 64 MB |
| Scene graph (10K nodes) | < 1 MB | Arena = flat Vec, no heap alloc per node |

**Key insight**: GPU buffer memory is predictable: `count × sizeof(Instance)`. For `PointInstance` (64 bytes), 1M points = 64 MB GPU memory. CPU-side overhead should be minimal since we map data directly to GPU buffers.

---

## 3. Binary / Compile Size

| Metric | Target | Notes |
|--------|--------|-------|
| esoc-color (standalone) | < 50 KB | Zero deps, pure math |
| esoc-scene (standalone) | < 100 KB | Arena + mark types |
| esoc-chart + esoc-gfx (SVG only) | < 500 KB | No GPU, no image deps |
| esoc-chart + esoc-gpu (full) | < 15 MB | wgpu is ~8-10 MB |
| Clean compile (full workspace) | < 90 s | wgpu is the bottleneck |
| Incremental compile (chart change) | < 5 s | Only esoc-chart recompiles |

**Competitor reference**: Plotly.js = 3.5 MB (minified), ECharts = 1 MB. Our SVG-only binary should be < 500 KB.

---

## 4. SVG Output Quality Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Nature single-col (90mm, 300dpi) | 1063 × auto px | Standard figure |
| Nature double-col (180mm, 300dpi) | 2126 × 2008 px max | Full-width figure |
| Font size after scaling | >= 6 pt | Nature minimum is 5 pt |
| Line width minimum | >= 0.5 pt | Visible in print |
| Color count per palette | <= 8 distinct | Distinguishable in print |
| All default palettes WCAG AA | >= 4.5:1 contrast | Against both light/dark bg |
| All default palettes CVD-safe | Distinguishable under deutan/protan/tritan sim | Use OKLab L channel separation |

---

## 5. API Ergonomics Targets

"Time to first chart" — how many lines for common tasks:

| Task | Target (lines) | ggplot2 | Plotly Express | Vega-Lite |
|------|----------------|---------|----------------|-----------|
| Basic scatter | 1 | 1 | 1 | ~5 (JSON) |
| Scatter + color | 1 | 1 | 1 | ~8 |
| Scatter + title + labels | 1 (chained) | 3 | 2 | ~10 |
| Grouped bar | 2 | 3 | 1 | ~10 |
| Multi-layer (scatter + line) | 3 | 3 | N/A | ~15 |
| Faceted scatter | 2 | 2 | 1 | ~8 |
| Full custom (grammar) | 5-10 | 5-10 | N/A | 15-30 |

**Current state**: Basic scatter = 1 line (`scatter(x, y).to_svg()?`). We match Plotly Express for simple cases.

---

## 6. Downsampling (LTTB) Benchmarks

Target: match compiled C/Cython implementations.

| Input Size | Output Size | Target Time | Reference (C) |
|------------|-------------|-------------|---------------|
| 5K → 500 | 500 | < 15 us | C: 10.6 us |
| 50K → 1K | 1K | < 100 us | extrapolated |
| 500K → 2K | 2K | < 1 ms | extrapolated |
| 5M → 5K | 5K | < 10 ms | extrapolated |
| 50M → 5K | 5K | < 100 ms | extrapolated |

**GPU compute shader target** (Phase 4):
| Input Size | Target Time |
|------------|-------------|
| 1M → 5K | < 1 ms |
| 10M → 5K | < 5 ms |

---

## 7. Benchmark Suite Design

### 7.1 Micro-benchmarks (criterion.rs)

```
benches/
  compile_bench.rs    — Chart → SceneGraph for various chart types/sizes
  svg_bench.rs        — SceneGraph → SVG string
  scale_bench.rs      — Scale::map() throughput
  lttb_bench.rs       — LTTB at various input sizes
  color_bench.rs      — Color space conversions (sRGB ↔ OKLab)
  kde_bench.rs        — KDE computation at various bandwidths
  bin_bench.rs        — Histogram binning
  treemap_bench.rs    — Treemap layout algorithm
```

### 7.2 Integration benchmarks

```
benches/
  full_pipeline.rs    — Data → Chart → SceneGraph → SVG (end-to-end)
  gpu_render.rs       — SceneGraph → GPU frame (headless)
  comparison/         — Generate equivalent charts to compare SVG size vs competitors
```

### 7.3 Stress tests

```
tests/
  stress_1m_points.rs     — 1M point scatter, measure time + memory
  stress_100_layers.rs    — 100 overlapping layers
  stress_deep_facet.rs    — 100-panel faceted chart
```

---

## 8. Correctness Benchmarks

Not just speed — we need to verify visual correctness:

| Test | Method |
|------|--------|
| Scale mapping accuracy | Round-trip: `map(invert(x)) ≈ x` within f32 epsilon |
| Tick generation | Compare tick positions to D3's `d3-scale` for reference inputs |
| Color space accuracy | OKLab round-trip error < 1e-5, CVD simulation matches Machado et al. |
| SVG output validity | Parse with `quick-xml`, validate well-formedness |
| Pixel-level regression | Render PNG, compare against golden images (perceptual diff) |
| Statistical accuracy | Bin counts match numpy.histogram, KDE matches scipy.stats.gaussian_kde |

---

## 9. Competitor Comparison Matrix (Quantitative)

To be filled in as we measure:

| Metric | esoc-chart | Plotly | ggplot2 | Vega-Lite | Chart.js | D3 |
|--------|-----------|--------|---------|-----------|----------|-----|
| 10K scatter SVG (ms) | ? | ~1500 | ~200 | ~300 | N/A(canvas) | ~210 |
| 10K scatter GPU (ms) | ? | ~50(WebGL) | N/A | N/A | ~50(canvas) | N/A |
| 1M scatter GPU (FPS) | ? | ~20 | N/A | N/A | N/A | N/A |
| SVG size 1K pts (KB) | ? | ~300 | ~58 | ~40 | N/A | ~30 |
| Binary/bundle (KB) | ? | 3500 | N/A(R) | 400 | 200 | 290 |
| Lines for scatter | 1 | 1 | 1 | 5 | 10 | 20+ |
| Faceting | no | no | yes | yes | no | manual |
| CVD-safe defaults | yes | no | no | no | no | no |
