# Charting Library Research — Comprehensive Findings

## 1. Academic Charting Standards

### Publisher Figure Specifications

| Publisher | Color DPI | Line Art DPI | Column Width | Font Size | Font |
|-----------|-----------|-------------|--------------|-----------|------|
| Nature | 300 (450 rec.) | 300 | 90mm / 180mm | 5-7pt | Arial, Helvetica |
| Science | 150-300 | 150-300 | — | 6pt min | — |
| IEEE | 300+ | 600+ | 88.9mm / 182mm | 9-10pt | Helvetica, Times, Arial |
| ACS | 300 | 1200 | 82.5mm | 8pt min | Arial, Helvetica |
| Cell Press | 300 | 1000 | 85mm / 174mm | 6-8pt | — |
| PNAS | 300 | 1000-1200 | — | 6-8pt | Arial, Helvetica, Times |
| APA | — | — | — | 8-14pt | Sans serif |
| Elsevier | — | — | — | — | 0.25pt min line |

### File Format Requirements
- **Vector**: .ai, .eps, .pdf, .ps, .svg (preferred for line art)
- **Raster**: .tiff (preferred), .png, .jpg
- **Color space**: RGB for digital, CMYK for print
- **Font embedding**: Required in PDF/EPS; PostScript Type 1 or TrueType

### Foundational Theory

- **Tufte**: Maximize data-ink ratio, eliminate chartjunk, no 3D effects on 2D data
- **Cleveland & McGill perceptual hierarchy**: Position (common scale) > Position (non-aligned) > Length > Angle/Direction > Area > Volume > Color saturation
- **Bertin's visual variables**: Position, size, shape, value, color, orientation, texture
- **Wilkinson's Grammar of Graphics**: Data → Transforms → Scales → Coordinates → Geometry → Aesthetics
- **Stephen Few's criteria**: Usefulness, Completeness, Perceptibility, Truthfulness, Intuitiveness, Aesthetics, Engagement

### Accessibility Standards

- **WCAG AA**: 4.5:1 contrast text, 3:1 large text; never color-only info encoding
- **WCAG AAA**: 7:1 contrast text, 4.5:1 large text
- **Colorblind-safe**: Blue safest; ColorBrewer palettes; viridis/inferno/plasma/magma
- **WAI-ARIA Graphics Module**: SVG `<title>`/`<desc>` with `aria-labelledby`
- **Section 508**: US federal accessible color requirement
- **Chartability**: Emerging standard for chart accessibility auditing
- 4.5% of world population is colorblind (~350M people)

### Statistical Visualization Standards

- **Error bars**: Always label type (SD vs SEM vs CI) explicitly
- **Significance markers**: Brackets with `* p<0.05, ** p<0.01, *** p<0.001`
- **Modern distribution plots**: Raincloud (raw data + KDE + box), superplots, estimation statistics
- **NIST TN-1297**: Measurement uncertainty guidelines
- **Trend**: Bar charts being replaced by violin/raincloud plots in top journals

### Domain-Specific Standards

**Medical/Clinical:**
- CONSORT 2025 flow diagrams (600+ journals)
- PRISMA flow diagrams (systematic reviews)
- STROBE checklists (observational studies)
- Forest plots (meta-analysis)
- Kaplan-Meier survival curves (with risk tables)
- Bland-Altman plots (method comparison)

**Financial:**
- Candlestick/OHLC, Renko, Point-and-Figure
- Volume overlays, technical indicators

**Geographic:**
- OGC standards (WMS/WFS/WMTS)
- ISO 19115 metadata, ISO 19100 series
- FGDC symbology

**Engineering:**
- ISO 128 (technical drawings, 15 parts)
- ISO 3098 (lettering standards)
- ASME Y14 series

**Publication Ethics:**
- COPE image manipulation guidelines
- No undisclosed adjustments; retraction for manipulation affecting conclusions

---

## 2. Current Library Feature Matrix

### API Design Philosophies

| Philosophy | Libraries |
|---|---|
| Grammar of Graphics | ggplot2, plotnine, Altair/Vega-Lite, Gadfly.jl |
| Declarative Config | Plotly, ECharts, Highcharts, Chart.js, Vega |
| Declarative Components | Recharts, Victory, Nivo (React) |
| Imperative OO | Matplotlib, Bokeh, Makie.jl, D3.js, Plotters |
| Meta/Backend-agnostic | Plots.jl, HoloViews |

### Rendering Backend Coverage

| Library | SVG | Canvas | WebGL | PDF | PNG | OpenGL | Other |
|---|---|---|---|---|---|---|---|
| Matplotlib | Y | - | - | Y | Y | - | PGF/LaTeX, PS/EPS, Cairo |
| Plotly | Y | - | Y | Y | Y | - | HTML |
| Bokeh | - | Y | partial | - | Y | - | HTML |
| D3.js | Y | manual | manual | - | - | - | HTML |
| ECharts | Y | Y | Y (gl) | - | Y | - | VML |
| Highcharts | Y | Y (boost) | - | Y | Y | - | VML |
| Chart.js | - | Y | - | - | - | - | - |
| Makie.jl | Y | - | Y | Y | Y | Y | Ray-tracing |
| Plotters (Rust) | Y | Y (WASM) | - | Y | Y | - | Piston, GTK |
| VisPy | - | - | - | - | - | Y | - |
| Nivo | Y | Y | - | - | - | - | SSR |

### Performance Tiers

| Tier | Libraries | Scale |
|---|---|---|
| GPU (millions+) | VisPy, Makie.jl, ECharts-GL, Plotly scattergl | 1M-100M+ |
| Optimized (100K-1M) | Bokeh, ECharts, Highcharts boost, Plotters | 100K-1M |
| Standard (10K-100K) | Matplotlib, Seaborn, ggplot2, D3, Chart.js | 1K-100K |
| Small data | Altair, Pygal, Vega-Lite, Observable Plot | <10K native |
| Billion-scale | HoloViews+Datashader, Tableau | 1B+ with aggregation |

### Unique Standout Features

1. D3.js — unlimited expressiveness; foundation for all JS charting
2. Makie.jl — 4 backends including ray-tracer; GPU-native
3. HoloViews+Datashader — billion-point rendering
4. ECharts — incremental rendering 10M+ points; WebSocket streaming
5. Nivo — server-side rendering (emails, PDFs)
6. Poloto (Rust) — CSS-styleable SVG output
7. Highcharts — sonification + WCAG accessibility
8. Mathematica — symbolic plot expressions; largest chart type vocabulary
9. Victory — React Native cross-platform parity
10. Plots.jl — `@recipe` type-dispatch plot definitions

---

## 3. Main Complaints

### Universal: The 80/20 Problem
80% trivial, last 20% customization disproportionately painful.
High-level libs make first 80% easy, last 20% impossible.
Low-level libs make everything possible, nothing easy.

### Matplotlib
- Two competing APIs (pyplot vs OO); SO answers are a "crapshoot"
- Layout system broken: tight_layout and constrained_layout conflict silently
- Threading: GUI backends require main thread; breaks Django/Flask/async
- Font embedding off by default for SVG/PDF
- No smart label placement (NP-hard, no solution)
- Documentation well-written but horribly organized

### Plotters (Rust)
- Maintainer abandonment crisis (recovered but trust damaged)
- Only 4 chart types built-in
- Cannot load system fonts; manual registration required
- Rust devs shell out to Python/matplotlib instead

### Seaborn
- Objects interface perpetually "experimental"
- No stacked bar, no dedicated scatter
- Zero interactivity
- Leaky abstraction to matplotlib

### Plotly
- 3-6MB bundle size; heap memory issues
- Freezes at 100K+ points (regression from v1)
- Dash memory leaks; RAM balloons with callbacks
- Jupyter compatibility constantly breaking

### D3.js
- 100+ lines for a bar chart
- Learning curve: SVG + data binding + enter/update/exit + scales + axes
- "Most teams overestimate their need for D3"

### ggplot2
- 5+ minutes for complex plots; geom_sf: 10s for 1K points
- NSE makes programmatic use nightmarish
- "A different language inside R"
- R lock-in; ports incomplete

### Cross-Cutting
- Large dataset performance: ALL fail (except GPU-accelerated)
- Font handling: cross-platform nightmare everywhere
- Date/time axes: overlapping labels, timezone hell
- Accessibility: ALL inadequate except Highcharts
- Leaky abstractions: Seaborn→mpl, Plotly.py→Plotly.js, Recharts→D3

---

## 4. Critical Gaps (Opportunity Space)

| Gap | Severity | Best Current |
|-----|----------|-------------|
| Cross-chart linking/brushing | Critical | Only Vega-Lite |
| Smart label anti-collision | Critical | Basic in ECharts |
| Accessibility (screen readers, keyboard) | Critical | Only Highcharts |
| Statistical annotations | High | Zero JS; R/ggpubr only |
| Design token integration | High | None support W3C tokens |
| RTL layout | High | Poor everywhere |
| Small multiples/faceting | High | Only Vega-Lite, Observable Plot |
| Adaptive responsive layouts | Medium | Resize yes, adapt no |
| Undo/redo for interactions | Medium | Non-existent |
| Print-quality export (CMYK, bleed) | Medium | Only Datawrapper |
| Built-in data transforms | Medium | Only Vega-Lite, Observable Plot |

### Missing Chart Types (hard/impossible in most libs)
- Chord diagrams (only D3 native)
- Waffle charts (no major library)
- Beeswarm plots (require force simulation)
- Raincloud plots (DIY from violin + box + strip)
- Horizon charts (absent from all major JS libs)
- Marimekko/mosaic plots (unsupported everywhere)
- Network/graph overlaid on charts (separate ecosystem)
- Gantt/timeline (only Highcharts module)
- Sparklines (no dedicated mode anywhere)

### Emerging Needs
- GPU/WebGPU rendering: ChartGPU 35M pts@72fps, early-stage
- WASM portable rendering: unrealized for viz
- AI-assisted chart creation: commercial only
- Collaborative primitives: no library has CRDT integration

---

## 5. Existing Codebase Assets

### esoc-chart (esolearn)
- High-level charting API; matplotlib-equivalent for Rust ML
- Chart types: Line, Scatter, Bar, BoxPlot, ErrorBar, Heatmap, Histogram
- Built on esoc-gfx (SVG vector engine)
- Feature-gated scry-learn interop

### esoc-gfx (esolearn)
- Low-level 2D vector graphics engine
- SVG output (zero deps); optional PNG via resvg
- Canvas, Layer, Element, Path, Text, Transform, Color, Palette, Style
- Coordinate transforms, viewport transforms, axis transforms

### esocidae
- GPU-rendered terminal emulator (wgpu)
- Arena scene graph (flat Vec, O(1) slot reuse)
- Instanced quad rendering (#[repr(C)] Pod)
- User shader pipelines (WGSL)
- Offscreen post-processing (vignette, grain, scanlines, CRT, chromatic aberration)
- Damage tracking for dirty regions
- Dual atlas allocators (shelf for glyphs, slab for images)
- Font rendering infrastructure (eso_font)
