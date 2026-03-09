# esoc-chart: Complete Specification

> Single source of truth for what esoc-chart IS and what "done" means.
> Every feature is tagged: DONE, PARTIAL, TODO, or STRETCH.

---

## 1. Design Philosophy

**Mission**: The fastest, most correct, most ergonomic charting library in any language.

**Core principles**:
1. **Simple things simple, complex things possible** — 3-level API (Express / Grammar / Scene Graph)
2. **Correct by default** — perceptual color, accessible palettes, proper statistical transforms
3. **GPU-first architecture** — write once, render to SVG (publishing) or GPU (exploration)
4. **Zero-copy where possible** — f64 data stays borrowed until scale mapping to f32
5. **Rust-native** — no FFI, no runtime, no GC; compile-time guarantees

**Non-goals** (explicitly out of scope):
- Web framework (no JS/WASM target in v1)
- General-purpose 2D game engine
- GIS/mapping system (basic geo projections yes, full GIS no)
- Dashboard layout manager (single chart focus; composition via external tools)

---

## 2. Architecture

```
User Code
    │
    ▼
┌─────────────────────────────────────────────┐
│  esoc-chart                                 │
│  ┌───────────┐ ┌──────────┐ ┌────────────┐ │
│  │  Express   │ │ Grammar  │ │  Compile   │ │
│  │ scatter()  │ │ Chart    │ │ chart→scene│ │
│  │ line()     │ │ Layer    │ │ axis_gen   │ │
│  │ bar()      │ │ Encoding │ │ mark_gen   │ │
│  │ ...        │ │ Stat     │ │ layout     │ │
│  └─────┬─────┘ └────┬─────┘ └─────┬──────┘ │
│        └─────────────┴─────────────┘        │
└────────────────────┬────────────────────────┘
                     │ SceneGraph
          ┌──────────┼──────────┐
          ▼          ▼          ▼
    ┌──────────┐ ┌────────┐ ┌───────┐
    │ esoc-gfx │ │esoc-gpu│ │ JSON  │
    │ SVG/PNG  │ │ wgpu   │ │ serde │
    └──────────┘ └────────┘ └───────┘
          │          │
          ▼          ▼
    ┌──────────┐ ┌────────┐
    │esoc-scene│ │  wgpu  │
    │  arena   │ │ shaders│
    └────┬─────┘ └────────┘
         │
         ▼
    ┌──────────┐
    │esoc-color│
    │  OKLab   │
    └──────────┘
```

---

## 3. Mark Primitives (esoc-scene)

9 mark types. Every chart decomposes into these.

| # | Mark | Status | Description |
|---|------|--------|-------------|
| 1 | `PointMark` | DONE | center, size, shape, fill, stroke |
| 2 | `LineMark` | DONE | polyline, stroke, interpolation (Linear/Step/Monotone) |
| 3 | `RectMark` | DONE | bounding box, fill, stroke, corner_radius |
| 4 | `AreaMark` | DONE | upper/lower boundaries, fill, stroke |
| 5 | `ArcMark` | DONE | center, inner/outer radius, start/end angle, fill, stroke |
| 6 | `RuleMark` | DONE | line segments, stroke |
| 7 | `TextMark` | DONE | position, text, font, fill, angle |
| 8 | `PathMark` | DONE | MoveTo/LineTo/CubicTo/QuadTo/Close, fill, stroke |
| 9 | `ImageMark` | DONE | bounding box, pixel data |

**Batch rendering**: `MarkBatch` with `BatchAttr<T>` (Uniform/Varying) — DONE for Points, Rects, Rules.

**Missing batch types**: TODO — LineBatch, AreaBatch, ArcBatch, TextBatch, PathBatch.

---

## 4. Scales

### 4.1 Scale Types

| Scale | Status | Domain → Range | Notes |
|-------|--------|---------------|-------|
| **Linear** | DONE | continuous → continuous | `Scale::Linear { domain, range }` |
| **Log** | DONE | continuous → continuous | `Scale::Log { domain, range, base }` |
| **Band** | DONE | discrete → continuous bands | `Scale::Band { domain, range, padding }` |
| **Time** | DONE | temporal (i64 epoch) → continuous | `Scale::Time { domain, range }` |
| **Sqrt** | TODO | continuous → continuous | power(0.5) transform |
| **Power** | TODO | continuous → continuous | arbitrary exponent |
| **Symlog** | TODO | continuous → continuous | handles zero/negatives |
| **Point** | TODO | discrete → point positions | like Band with bandwidth=0 |
| **Ordinal** | TODO | discrete → discrete (e.g., color) | categorical color mapping |
| **Quantize** | TODO | continuous → N discrete bins | equal-interval discretization |
| **Quantile** | TODO | continuous → N discrete bins | equal-count discretization |
| **Threshold** | TODO | continuous → discrete by breakpoints | user-defined thresholds |
| **Identity** | TODO | pass-through | data values = visual values |
| **Reverse** | TODO | invert any scale | wraps another scale |
| **Diverging** | TODO | continuous → bipolar color | midpoint-centered |
| **Sequential** | TODO | continuous → unipolar color | for heatmaps etc. |

### 4.2 Scale Features

| Feature | Status |
|---------|--------|
| `map(f64) → f32` | DONE |
| `map_band(category) → (f32, f32)` | DONE |
| `invert(f32) → f64` | DONE |
| `ticks(count) → Vec<f64>` | DONE (linear, log) |
| `format_tick(f64) → String` | DONE |
| Nice domain extension | DONE (linear) |
| Clamping | TODO |
| Custom tick values | TODO |
| Custom tick formatter | TODO |
| Date/time tick formatting | TODO |

---

## 5. Coordinate Systems

| Coordinate System | Status | Use Cases |
|-------------------|--------|-----------|
| **Cartesian** | DONE | All standard x/y charts |
| **Cartesian flipped** | TODO | Horizontal bar charts |
| **Cartesian fixed-ratio** | TODO | Square scatter, geographic |
| **Polar** | TODO | Pie, donut, radar, nightingale rose, radial bar |
| **Ternary** | STRETCH | Composition plots (chemistry, geology) |
| **Parallel** | STRETCH | Parallel coordinates |
| **Geographic** | STRETCH | Choropleth, bubble map (requires proj4) |

---

## 6. Statistical Transforms

### 6.1 Core Stats (required for common charts)

| Stat | Status | Output | Used By |
|------|--------|--------|---------|
| **Identity** | DONE | pass-through | scatter, line, bar (explicit values) |
| **Bin** | PARTIAL (type exists) | edges + counts | histogram |
| **BoxPlot** | PARTIAL (type exists) | median, Q1, Q3, whiskers, outliers | box plot |
| **Smooth** | PARTIAL (type exists) | fitted y + CI band | trend lines |
| **Aggregate** | PARTIAL (type exists) | count/sum/mean/median/min/max | bar (stat=count) |
| **KDE** | TODO | density curve (x, density) | density, violin, ridgeline, raincloud |
| **Stack** | TODO | cumulative y offsets | stacked bar, stacked area, stream |
| **Normalize** | TODO | proportional y (0-1) | 100% stacked bar |
| **Dodge** | TODO | x-offset per group | grouped bar |
| **Jitter** | TODO | random positional noise | strip, beeswarm |
| **ECDF** | TODO | sorted cumulative proportion | ECDF plot |
| **Quantile** | TODO | quantile values | Q-Q plot |
| **Rank** | TODO | rank ordering | bump chart |
| **Cumulative** | TODO | running sum | waterfall |

### 6.2 Advanced Stats

| Stat | Status | Used By |
|------|--------|---------|
| **Regression** (linear, poly, loess) | TODO | trend lines, confidence bands |
| **KDE 2D** | TODO | contour plots, 2D density |
| **Hexbin** | TODO | hexagonal binning |
| **Contour** (marching squares) | TODO | isopleth maps |
| **Correlation** | TODO | correlogram |
| **Bootstrap CI** | STRETCH | estimation uncertainty |
| **Kaplan-Meier** | STRETCH | survival curves |

### 6.3 Layout Algorithms (non-statistical transforms)

| Algorithm | Status | Used By |
|-----------|--------|---------|
| **Treemap** (squarified) | TODO | treemap |
| **Partition** | TODO | sunburst, icicle |
| **Pack** (circle packing) | TODO | circle packing |
| **Tree** (tidy/cluster) | TODO | dendrogram |
| **Sankey** (Gauss-Seidel) | TODO | Sankey diagram |
| **Chord** | TODO | chord diagram |
| **Force-directed** | TODO | network graphs, beeswarm |
| **Beeswarm** (dodge) | TODO | beeswarm plot |
| **Word cloud** (spiral pack) | STRETCH | word cloud |
| **Geo projection** | STRETCH | maps |

---

## 7. Position Adjustments

| Adjustment | Status | Description |
|------------|--------|-------------|
| **Identity** | DONE | No adjustment (default) |
| **Stack** | TODO | Stack y values cumulatively |
| **Fill** | TODO | Stack + normalize to 100% |
| **Dodge** | TODO | Side-by-side within groups |
| **Jitter** | TODO | Random noise (width, height, seed) |
| **Nudge** | TODO | Fixed offset (for labels) |
| **JitterDodge** | TODO | Combined jitter + dodge |

---

## 8. Chart Type Coverage

### Tier 1: Core (must ship in v1)

| Chart Type | Mark Composition | Stat | Position | Status |
|------------|-----------------|------|----------|--------|
| Scatter | Point | Identity | Identity | DONE |
| Line | Line | Identity | Identity | DONE |
| Bar (vertical) | Rect | Identity | Identity | DONE |
| Bar (horizontal) | Rect | Identity | Identity | TODO (needs coord flip) |
| Area | Area | Identity | Identity | TODO (mark_gen missing) |
| Step | Line (StepBefore/After) | Identity | Identity | DONE (interpolation exists) |
| Histogram | Rect | Bin | Identity | TODO |
| Box Plot | Rect+Rule+Point | BoxPlot | Dodge | TODO |
| Pie | Arc | Proportion | Identity | TODO |
| Donut | Arc (inner_radius>0) | Proportion | Identity | TODO |
| Stacked Bar | Rect | Identity | Stack | TODO |
| Grouped Bar | Rect | Identity | Dodge | TODO |
| Stacked Area | Area | Identity | Stack | TODO |
| Error Bars | Rule | Identity | Identity | TODO |
| Confidence Band | Area | Smooth | Identity | TODO |
| Heatmap | Rect | Identity | Identity | TODO |
| Scatter + Color | Point | Identity | Identity | DONE |
| Multi-line | Line (multi-layer) | Identity | Identity | DONE |
| Lollipop | Rule+Point | Identity | Identity | TODO |
| Rug | Rule (short ticks) | Identity | Identity | TODO |

### Tier 2: Statistical (v1.1)

| Chart Type | Mark Composition | Stat | Status |
|------------|-----------------|------|--------|
| Density | Area | KDE | TODO |
| Violin | Area (mirrored) | KDE | TODO |
| Ridgeline | Area (offset) | KDE | TODO |
| Raincloud | Area+Rect+Point | KDE+BoxPlot | TODO |
| Strip/Jitter | Point | Jitter | TODO |
| Beeswarm | Point | Beeswarm layout | TODO |
| ECDF | Line (step) | ECDF | TODO |
| Q-Q Plot | Point+Rule | Quantile | TODO |
| Hexbin | Path | Hexbin | TODO |
| 2D Density | Path/Image | KDE 2D | TODO |
| Correlogram | Rect+Text | Correlation | TODO |
| Bubble | Point (size channel) | Identity | TODO |

### Tier 3: Composition & Domain (v1.2)

| Chart Type | Mark Composition | Layout/Stat | Status |
|------------|-----------------|-------------|--------|
| Sunburst | Arc (nested) | Partition | TODO |
| Treemap | Rect (nested) | Treemap | TODO |
| Waffle | Rect (grid) | Proportion grid | TODO |
| Stacked Area (stream) | Area | Stack(center) | TODO |
| 100% Stacked Bar | Rect | Normalize+Stack | TODO |
| Waterfall | Rect+Rule | Cumulative | TODO |
| Funnel | Rect (centered) | Proportion | TODO |
| Bullet | Rect (layered) | Identity | TODO |
| Candlestick | Rect+Rule | Identity | TODO |
| OHLC | Rule | Identity | TODO |
| Slope | Line+Point+Text | Identity | TODO |
| Bump | Line+Point | Rank | TODO |
| Dumbbell | Rule+Point | Identity | TODO |
| Tornado/Butterfly | Rect (mirrored) | Identity | TODO |
| Forest Plot | Point+Rule+Rect | Identity | TODO |
| Gantt | Rect | Identity | TODO |
| Calendar Heatmap | Rect (grid) | Identity | TODO |
| Sparkline | Line (no chrome) | Identity | TODO |

### Tier 4: Advanced (v2.0 / STRETCH)

| Chart Type | Layout Algorithm | Status |
|------------|-----------------|--------|
| Sankey | Sankey layout | STRETCH |
| Chord | Chord layout | STRETCH |
| Alluvial | Alluvial layout | STRETCH |
| Dendrogram | Tree layout | STRETCH |
| Circle Packing | Pack layout | STRETCH |
| Icicle / Flame | Partition layout | STRETCH |
| Node-Link Graph | Force layout | STRETCH |
| Parallel Coordinates | Parallel axes | STRETCH |
| Radar/Spider | Polar + Path | STRETCH |
| Nightingale Rose | Polar + Arc | STRETCH |
| Horizon | Band folding | STRETCH |
| Word Cloud | Spiral pack | STRETCH |
| Choropleth | Geo projection | STRETCH |
| Ternary | Barycentric coords | STRETCH |
| Kaplan-Meier | KM stat + Step | STRETCH |
| Marimekko/Mosaic | Variable-width layout | STRETCH |

---

## 9. Aesthetic Channels (Encoding)

### 9.1 Currently Implemented

| Channel | Status | Type |
|---------|--------|------|
| `X` | DONE | position (quantitative, temporal) |
| `Y` | DONE | position (quantitative) |
| `Color` | DONE | categorical color mapping |
| `Size` | PARTIAL (type exists, not wired) | quantitative → point size |
| `Shape` | PARTIAL (type exists, not wired) | categorical → marker shape |
| `Opacity` | PARTIAL (type exists, not wired) | quantitative → alpha |
| `Text` | PARTIAL (type exists, not wired) | string → label |

### 9.2 TODO Channels

| Channel | Priority | Description |
|---------|----------|-------------|
| `Fill` | HIGH | separate from stroke color |
| `Stroke` | HIGH | outline color |
| `StrokeWidth` | MEDIUM | line/border width |
| `StrokeDash` | MEDIUM | dash pattern |
| `X2` / `Y2` | HIGH | range encoding (bars, bands, error bars) |
| `XOffset` / `YOffset` | HIGH | dodge/jitter offset |
| `Angle` | MEDIUM | rotation (text, arrows) |
| `Label` | HIGH | text content for text marks |
| `Tooltip` | MEDIUM | hover text (GPU interactive mode) |
| `Group` | HIGH | line/area grouping without color |
| `Facet` | HIGH | faceting variable |
| `Order` | MEDIUM | drawing order / stack order |
| `Detail` | LOW | grouping without encoding |
| `Href` | STRETCH | clickable links (SVG only) |

---

## 10. Faceting

| Feature | Status | Description |
|---------|--------|-------------|
| `facet_wrap(field)` | TODO | 1D ribbon wrapped into grid |
| `facet_grid(row, col)` | TODO | 2D row × column grid |
| Free/fixed scales | TODO | `scales: "free" / "free_x" / "free_y" / "fixed"` |
| Free/fixed space | TODO | Panel sizes proportional to data range |
| Strip labels | TODO | Facet variable labels above/beside panels |
| `ncol` / `nrow` control | TODO | Control wrap dimensions |
| Drop empty levels | TODO | Skip panels with no data |
| Axis sharing | TODO | Show axes on all panels or margins only |

---

## 11. Guides (Axes & Legends)

### 11.1 Axes

| Feature | Status |
|---------|--------|
| X-axis (bottom) | DONE |
| Y-axis (left) | DONE |
| Axis title | DONE |
| Tick labels | DONE |
| Tick marks | TODO (only labels, no tick lines) |
| Grid lines | DONE |
| Minor grid lines | TODO |
| Log-scale ticks (sub-ticks at 2-9) | TODO |
| Date/time tick formatting | TODO |
| Rotated tick labels | TODO |
| Secondary axis (top / right) | TODO |
| Axis limits (zoom) | TODO |
| Custom breaks/labels | TODO |
| Reversed axis | TODO |

### 11.2 Legends

| Feature | Status |
|---------|--------|
| Color legend (categorical) | TODO |
| Color gradient bar (continuous) | TODO |
| Size legend | TODO |
| Shape legend | TODO |
| Legend title | TODO |
| Legend position (top/bottom/left/right/inside) | TODO |
| Legend direction (h/v) | TODO |
| Combined legends (shared aesthetic) | TODO |
| Hide legend | TODO |

---

## 12. Theme System

### 12.1 Current

| Feature | Status |
|---------|--------|
| `NewTheme` struct | DONE |
| `light()` preset | DONE |
| `dark()` preset | DONE |
| `publication()` preset | DONE |
| Background color | DONE |
| Foreground color | DONE |
| Palette (series colors) | DONE |
| Grid color/width/visibility | DONE |
| Font sizes (title/label/tick/legend) | DONE |
| Font family | DONE |
| Line width, point size | DONE |

### 12.2 TODO

| Feature | Priority |
|---------|----------|
| Per-element overrides (axis.title.x, strip.text.y, etc.) | MEDIUM |
| Margin control (plot, panel, legend) | HIGH |
| Border/frame styling | MEDIUM |
| Strip (facet label) styling | MEDIUM (needs faceting first) |
| Legend key styling | MEDIUM (needs legends first) |
| Preset themes: `minimal`, `classic`, `void`, `economist`, `fivethirtyeight` | LOW |
| Custom theme builder | MEDIUM |

---

## 13. Annotations

| Feature | Status |
|---------|--------|
| Title | DONE |
| X-axis label | DONE |
| Y-axis label | DONE |
| Subtitle | TODO |
| Caption | TODO |
| Tag (e.g., "(a)") | TODO |
| `hline(y)` — horizontal reference line | TODO |
| `vline(x)` — vertical reference line | TODO |
| `abline(slope, intercept)` — diagonal line | TODO |
| `rect_band(x1, x2)` — reference band | TODO |
| `text_annotation(x, y, label)` | TODO |
| `arrow(x1, y1, x2, y2)` | TODO |
| `bracket(x1, x2, y, label)` — significance bracket | STRETCH |

---

## 14. Data Input

### 14.1 Current

| Input | Status |
|-------|--------|
| `&[f64]` slices | DONE |
| `Vec<f64>` | DONE |
| `Vec<String>` (categories) | DONE |
| scry-learn Dataset (feature-gated) | DONE |

### 14.2 TODO

| Input | Priority | Notes |
|-------|----------|-------|
| Iterator/IntoIterator | HIGH | avoid allocation |
| Tuples `(x, y)` | HIGH | ergonomic |
| `HashMap<String, Vec<f64>>` (columnar) | MEDIUM | named columns |
| CSV/TSV reader | MEDIUM | direct file → chart |
| Polars DataFrame | MEDIUM | feature-gated |
| ndarray | LOW | feature-gated |
| Arrow RecordBatch | STRETCH | zero-copy interop |
| JSON data | STRETCH | Vega-like inline data |

---

## 15. Output Formats

| Format | Status | Backend |
|--------|--------|---------|
| SVG string | DONE | esoc-gfx `scene_svg` |
| SVG file | DONE | esoc-gfx `scene_svg` |
| PNG (rasterized SVG) | DONE | esoc-gfx `resvg` (feature-gated) |
| GPU window (wgpu) | PARTIAL | esoc-gpu (skeleton passes) |
| GPU headless (wgpu) | PARTIAL | esoc-gpu `new_headless()` |
| PDF | TODO | direct PDF generation or SVG→PDF |
| EPS | STRETCH | for legacy journal submission |
| JSON spec | TODO | serializable Chart → JSON |
| HTML (embedded SVG) | TODO | SVG with interactive JS |

---

## 16. GPU Rendering (esoc-gpu)

### 16.1 Render Passes

| Pass | Status | Technique |
|------|--------|-----------|
| PointPass | PARTIAL (skeleton) | Instanced billboard quads + SDF shapes |
| LinePass | PARTIAL (skeleton) | Instanced quad segments + miter joins |
| RectPass | PARTIAL (skeleton) | Instanced rectangles + corner radius SDF |
| RulePass | PARTIAL (skeleton) | Instanced thin lines |
| TessPass | PARTIAL (skeleton) | Lyon CPU tessellation → triangle mesh |
| TextPass | PARTIAL (skeleton) | MSDF glyph atlas (planned) |

### 16.2 GPU Features TODO

| Feature | Priority | Description |
|---------|----------|-------------|
| Complete PointPass with all 8 marker shapes | HIGH | SDF per shape in fragment shader |
| Complete LinePass with proper joins/caps | HIGH | Screen-space line expansion |
| Complete RectPass with rounded corners | HIGH | Corner radius SDF |
| MSDF text rendering | HIGH | msdf-atlas-gen → glyph atlas |
| LTTB decimation (compute shader) | HIGH | Real-time downsampling |
| GPU picking (color-encoded FBO) | HIGH | Click/hover identification |
| Anti-aliasing (SDF per-primitive) | MEDIUM | Smooth edges without MSAA |
| Transparency (Weighted Blended OIT) | MEDIUM | Overlapping translucent marks |
| ArcPass (pie/donut) | MEDIUM | SDF arc segments |
| AreaPass (filled regions) | MEDIUM | Tessellation or stencil |
| Smooth transitions (interpolation) | MEDIUM | Animate between states |
| winit integration (pan/zoom/resize) | MEDIUM | Interactive window |
| Damage tracking (incremental redraw) | LOW | Only re-render changed regions |

---

## 17. Color System (esoc-color)

| Feature | Status |
|---------|--------|
| Linear RGBA (f32, GPU-native) | DONE |
| sRGB encode/decode | DONE |
| Hex parsing (#RRGGBB, #RGB) | DONE |
| OKLab perceptual space | DONE |
| OKLch cylindrical space | DONE |
| CVD simulation (deutan/protan/tritan) | DONE |
| WCAG contrast ratio | DONE |
| Gamut clipping | DONE |
| Color palettes (categorical) | DONE |
| ColorScale (linear interpolation) | DONE |
| Named CSS colors | TODO |
| HSL/HSV input | TODO |
| Diverging palettes (blue-white-red etc.) | TODO |
| Sequential palettes (viridis, inferno, etc.) | TODO |
| Perceptually uniform sequential | TODO |
| CVD-safe palette generation | TODO |

---

## 18. Express API Coverage

| Function | Status | Builds |
|----------|--------|--------|
| `scatter(x, y)` | DONE | ScatterBuilder → Chart |
| `line(x, y)` | DONE | LineBuilder → Chart |
| `bar(categories, values)` | DONE | BarBuilder → Chart |
| `area(x, y)` | TODO | AreaBuilder → Chart |
| `histogram(values)` | TODO | HistBuilder → Chart (stat=Bin) |
| `boxplot(categories, values)` | TODO | BoxBuilder → Chart |
| `heatmap(x, y, values)` | TODO | HeatmapBuilder → Chart |
| `pie(categories, values)` | TODO | PieBuilder → Chart |
| `violin(categories, values)` | TODO | ViolinBuilder → Chart |
| `density(values)` | TODO | DensityBuilder → Chart (stat=KDE) |
| `facet_scatter(x, y, facet)` | TODO | FacetScatterBuilder → Chart |
| `candlestick(dates, o, h, l, c)` | TODO | CandlestickBuilder → Chart |
| `pair_plot(columns)` | STRETCH | SPLOM via faceting |

---

## 19. Interactivity (Phase 4)

| Feature | Status | Notes |
|---------|--------|-------|
| Pan (mouse drag) | TODO | Translate viewport |
| Zoom (scroll wheel) | TODO | Scale viewport |
| Hover tooltip | TODO | GPU picking → nearest mark → tooltip |
| Click selection | TODO | GPU picking → mark ID |
| Brush selection (rectangular) | TODO | Drag region → selected marks |
| Lasso selection | STRETCH | Freeform selection |
| Linked views | STRETCH | Selection propagates across charts |
| Keyboard navigation | STRETCH | Accessibility |

---

## 20. Accessibility

| Feature | Status |
|---------|--------|
| OKLab perceptual color space | DONE |
| CVD simulation (preview how colorblind users see) | DONE |
| WCAG contrast ratio calculation | DONE |
| Default palettes pass WCAG AA (4.5:1) | TODO |
| Automatic CVD-safe palette selection | TODO |
| Pattern fills (alternative to color-only encoding) | TODO |
| ARIA attributes in SVG output | TODO |
| Alt text / description in SVG | TODO |
| Screen reader narration | STRETCH |
| Sonification | STRETCH |
| Keyboard-navigable data points | STRETCH |

---

## 21. Publication Export

| Feature | Status | Notes |
|---------|--------|-------|
| Vector SVG output | DONE | Preferred by all journals |
| PNG at configurable DPI | DONE | via resvg (feature-gated) |
| Nature single-col (90mm @ 300dpi) | TODO | Preset size |
| Nature double-col (180mm @ 300dpi) | TODO | Preset size |
| Science single-col (57mm) | TODO | Preset size |
| IEEE single-col (88.9mm) | TODO | Preset size |
| ACS single-col (82.5mm) | TODO | Preset size |
| PDF output | TODO | Direct or SVG→PDF |
| Font embedding in SVG | TODO | Self-contained output |
| CMYK color space export | STRETCH | Required by some print journals |
| EPS output | STRETCH | Legacy format |

---

## Appendix A: Full Chart Type → Mark Primitive Mapping

Every chart type decomposes into these 9 primitives. The intelligence lives in:
1. **Stat transforms** — compute derived data (bins, KDE, layout positions)
2. **Position adjustments** — stack, dodge, jitter
3. **Coordinate transforms** — Cartesian → polar, geographic, etc.
4. **Scale mapping** — data domain → visual range

Mark primitives are deliberately simple and GPU-friendly. Complex charts are compositions, not new primitives.

## Appendix B: Competitor Feature Matrix

| Feature | ggplot2 | Vega-Lite | D3 | Plotly | ECharts | Observable Plot | **esoc-chart** |
|---------|---------|-----------|-----|--------|---------|----------------|---------------|
| Grammar of Graphics | Full | Full | N/A | Partial | N/A | Partial | PARTIAL |
| Faceting | Excellent | Good | Manual | None | None | Good | TODO |
| Polar coords | Yes | Partial | Manual | Yes | Yes | No | TODO |
| Interactivity | No (static) | Selections | Full | Full | Full | No | TODO |
| GPU rendering | No | No | No | WebGL | Canvas | No | PARTIAL |
| 1M+ points | No | No | No | WebGL only | ~50K | No | TODO (target) |
| Accessibility | Partial | Poor | Manual | Poor | Poor | Poor | PARTIAL |
| SVG output | Yes | Yes | Yes | Yes | Yes | Yes | DONE |
| Publication presets | Yes | No | No | No | No | No | PARTIAL |
| CVD-safe colors | Extension | No | No | No | No | No | DONE |
| Perceptual color | No (sRGB) | No | No | No | No | No | DONE |
| Zero dependencies | No | No | No | No | No | No | DONE (color+scene) |

## Appendix C: Implementation Priority Score

Priority = (user demand × competitive gap × architectural readiness) / complexity

**Highest priority TODO items:**
1. Position adjustments (stack/dodge) — unlocks grouped/stacked bars, stacked area
2. Faceting — biggest gap vs. ggplot2, high demand, no JS lib does it well
3. Legends — currently no legend output at all
4. Polar coordinates — unlocks pie, donut, radar, rose
5. Bin stat → histogram — most requested basic chart type not yet working
6. BoxPlot stat → box plot — essential for EDA
7. KDE stat → density/violin — essential for distributions
8. Area mark generation — AreaMark exists but compile pipeline doesn't generate it
9. GPU pass completion — PointPass and LinePass are highest value
10. LTTB downsampling — key differentiator for large data
