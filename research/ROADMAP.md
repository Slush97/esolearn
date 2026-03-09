# esoc-chart: Development Roadmap

> Ordered phases with entry/exit criteria and dependencies.
> Each phase produces a usable, testable increment.

---

## Overview

```
Phase 1-3: DONE (color, scene, GPU skeleton, chart compiler, express API)
    │
    ▼
Phase 4: Grammar Completeness     ← YOU ARE HERE
    │    (stats, positions, faceting, legends, coord systems)
    │
    ▼
Phase 5: Chart Type Expansion
    │    (histogram, boxplot, pie, heatmap, density, violin, ...)
    │
    ▼
Phase 6: GPU Rendering Completion
    │    (all passes functional, MSDF text, LTTB)
    │
    ▼
Phase 7: Interactivity
    │    (pan, zoom, hover, pick, brush, winit)
    │
    ▼
Phase 8: Polish & Ecosystem
         (publication presets, data connectors, benchmarks, docs)
```

---

## Phase 4: Grammar Completeness

**Goal**: The grammar layer can express any Tier 1 chart from the spec.
**Duration estimate**: Large phase, can be split into sub-phases.

### 4A: Position Adjustments & Stacking

**Entry**: Current grammar types exist but compile pipeline only handles Identity.
**Exit**: Grouped bars, stacked bars, stacked area all produce correct SVGs.

| Task | Description |
|------|-------------|
| 4A.1 | Implement `Position` enum: `Identity`, `Stack`, `Fill`, `Dodge`, `Jitter` |
| 4A.2 | Add `position` field to `Layer` struct |
| 4A.3 | Implement stack transform in `compile/stack.rs`: compute cumulative y offsets |
| 4A.4 | Implement dodge transform in `compile/dodge.rs`: compute x offsets per group |
| 4A.5 | Implement fill (normalize) transform |
| 4A.6 | Wire position adjustments into `mark_gen.rs` |
| 4A.7 | Add express API: `stacked_bar()`, `grouped_bar()` |
| 4A.8 | Add examples: grouped bar, stacked bar, 100% stacked bar, stacked area |
| 4A.9 | Tests: verify stacked y-values sum correctly, dodge offsets are symmetric |

**Dependencies**: None (builds on existing grammar types).

### 4B: Core Statistical Transforms

**Entry**: Stat enum exists with Bin/BoxPlot/Smooth/Aggregate but none produce marks.
**Exit**: Histogram, box plot, and aggregated bar chart produce correct SVGs.

| Task | Description |
|------|-------------|
| 4B.1 | Implement `compile/stat_bin.rs`: bin edges, counts, density; Sturges/Scott/FD rules |
| 4B.2 | Implement `compile/stat_boxplot.rs`: median, Q1, Q3, whiskers (1.5×IQR), outliers |
| 4B.3 | Implement `compile/stat_aggregate.rs`: count, sum, mean, median, min, max per group |
| 4B.4 | Wire stats into compile pipeline: stat runs BEFORE scale mapping |
| 4B.5 | `mark_gen` for histogram: Rect marks from bin output |
| 4B.6 | `mark_gen` for boxplot: Rect (box) + Rule (whiskers+median) + Point (outliers) |
| 4B.7 | Add express API: `histogram(values)`, `boxplot(categories, values)` |
| 4B.8 | Add examples: histogram, boxplot, bar with stat=count |
| 4B.9 | Tests: bin edges match expected, boxplot quartiles match known data |

**Dependencies**: None.

### 4C: Area & Arc Mark Generation

**Entry**: AreaMark and ArcMark exist in esoc-scene but compile pipeline doesn't generate them.
**Exit**: Area charts, pie charts, donut charts produce correct SVGs.

| Task | Description |
|------|-------------|
| 4C.1 | `mark_gen` for area: AreaMark with upper=data, lower=baseline |
| 4C.2 | `mark_gen` for arc: compute start/end angles from proportions |
| 4C.3 | `scene_svg.rs`: render AreaMark → SVG `<path>` (fill between upper/lower) |
| 4C.4 | `scene_svg.rs`: render ArcMark → SVG `<path>` (arc segments) |
| 4C.5 | Add express API: `area(x, y)`, `pie(categories, values)` |
| 4C.6 | Add examples: area chart, pie chart, donut chart |
| 4C.7 | Tests: arc angles sum to 2π, area lower boundary at y=0 |

**Dependencies**: None. Can run in parallel with 4A/4B.

### 4D: Legends

**Entry**: No legend output.
**Exit**: Color, size, and shape legends auto-generated from encodings.

| Task | Description |
|------|-------------|
| 4D.1 | Design `Legend` struct: title, entries (label + swatch), position, direction |
| 4D.2 | Implement `compile/legend_gen.rs`: scan encodings → generate legend entries |
| 4D.3 | Color legend: categorical (discrete swatches) |
| 4D.4 | Color legend: continuous (gradient bar) |
| 4D.5 | Size legend: graduated circles |
| 4D.6 | Shape legend: marker shape swatches |
| 4D.7 | Layout: position legend relative to plot area (top/bottom/left/right) |
| 4D.8 | Render legends as TextMark + RectMark/PointMark in scene graph |
| 4D.9 | Tests: legend entries match unique values, correct colors assigned |

**Dependencies**: 4C (arc marks for potential arc-based legends, but not strictly blocking).

### 4E: Faceting

**Entry**: No faceting support.
**Exit**: `facet_wrap` and `facet_grid` produce multi-panel charts with shared/free scales.

| Task | Description |
|------|-------------|
| 4E.1 | Add `Facet` enum to `Chart`: `None`, `Wrap { field, ncol }`, `Grid { row, col }` |
| 4E.2 | Add faceting field to data model (need per-row facet values) |
| 4E.3 | Implement `compile/facet.rs`: split data by facet values → sub-charts |
| 4E.4 | Implement grid layout: compute panel positions, sizes, margins |
| 4E.5 | Scale sharing: `"fixed"` (shared domain) vs `"free"` (per-panel domain) |
| 4E.6 | Strip labels: TextMark headers above/beside each panel |
| 4E.7 | Axis sharing: axes on margins only, or per-panel |
| 4E.8 | Wire into `compile_chart()`: detect faceting → compile sub-charts → arrange |
| 4E.9 | Add express API: `.facet_wrap(field)`, `.facet_grid(row, col)` |
| 4E.10 | Tests: correct number of panels, scales shared/independent as specified |

**Dependencies**: 4D (legends should be shared across facet panels).

### 4F: Additional Scales & Coord Systems

**Entry**: Only Linear, Log, Band, Time scales; only Cartesian coords.
**Exit**: Sqrt, Symlog, Ordinal, Point scales; Polar coordinates.

| Task | Description |
|------|-------------|
| 4F.1 | `Scale::Sqrt`, `Scale::Power { exponent }` |
| 4F.2 | `Scale::Symlog { constant }` (bi-symmetric log) |
| 4F.3 | `Scale::Ordinal { domain: Vec<String>, range: Vec<T> }` (for color mapping) |
| 4F.4 | `Scale::Point` (like Band with zero bandwidth) |
| 4F.5 | Add `CoordSystem` enum to `Chart`: `Cartesian`, `Flipped`, `Polar { theta, radius }` |
| 4F.6 | Polar coord transform in compile: map (x,y) → (angle, radius) → (cx+r*cos, cy+r*sin) |
| 4F.7 | Polar axis generation: circular grid, radial ticks |
| 4F.8 | Scale clamping option |
| 4F.9 | Custom tick values and formatters |
| 4F.10 | Tests: polar scatter matches expected coordinates, symlog handles zero |

**Dependencies**: 4C (polar coords needed for proper pie/donut, but basic arc generation can work without full polar).

### 4G: Annotations

**Entry**: Only title and axis labels.
**Exit**: Reference lines, bands, text annotations, subtitle, caption.

| Task | Description |
|------|-------------|
| 4G.1 | Add `annotations: Vec<Annotation>` to `Chart` |
| 4G.2 | `Annotation::HLine { y, stroke, label }` |
| 4G.3 | `Annotation::VLine { x, stroke, label }` |
| 4G.4 | `Annotation::Band { x1, x2, fill }` or `{ y1, y2, fill }` |
| 4G.5 | `Annotation::Text { x, y, text, font }` |
| 4G.6 | `Annotation::Arrow { x1, y1, x2, y2, stroke }` |
| 4G.7 | Subtitle and caption fields on Chart |
| 4G.8 | Compile annotations → RuleMark / TextMark / AreaMark in scene graph |

**Dependencies**: None.

---

## Phase 5: Chart Type Expansion

**Goal**: All Tier 1 and Tier 2 chart types from the spec produce correct output.
**Entry**: Phase 4 complete (stats, positions, faceting, legends, coords all working).

### 5A: Distribution Charts

| Task | Description | Requires |
|------|-------------|----------|
| 5A.1 | KDE stat implementation (`stat_kde.rs`) | Silverman bandwidth |
| 5A.2 | Density plot (Area from KDE output) | KDE |
| 5A.3 | Violin plot (mirrored Area per category) | KDE + dodge |
| 5A.4 | Ridgeline plot (offset Areas) | KDE + custom y-offset |
| 5A.5 | Raincloud plot (half-violin + box + strip) | KDE + BoxPlot + jitter |
| 5A.6 | Beeswarm layout algorithm | Force-based or exact dodge |
| 5A.7 | Strip/jitter plot | Jitter position |
| 5A.8 | ECDF (step line from sorted cumulative) | Sort + cumulate |
| 5A.9 | Q-Q plot (theoretical quantiles) | Quantile stat |
| 5A.10 | Express API for each | `density()`, `violin()`, etc. |

### 5B: Composition Charts

| Task | Description | Requires |
|------|-------------|----------|
| 5B.1 | Stacked area (with stream graph variant) | Stack position |
| 5B.2 | 100% stacked bar | Fill position |
| 5B.3 | Waffle chart (grid layout) | Proportion → grid allocation |
| 5B.4 | Sunburst (partition layout → nested arcs) | Partition algo + Arc |
| 5B.5 | Treemap (squarified layout → nested rects) | Treemap algo + Rect |

### 5C: Relationship & Domain Charts

| Task | Description | Requires |
|------|-------------|----------|
| 5C.1 | Bubble chart (size channel on scatter) | Size encoding |
| 5C.2 | Heatmap (Rect grid + color encoding) | Color scale |
| 5C.3 | Error bars (Rule marks from y±error) | X2/Y2 encoding |
| 5C.4 | Confidence band (Area from smooth stat) | Smooth stat |
| 5C.5 | Candlestick (Rect body + Rule wick) | OHLC data input |
| 5C.6 | Waterfall (floating Rect + cumulative) | Cumulative stat |
| 5C.7 | Lollipop (Rule stem + Point head) | Composite mark |
| 5C.8 | Dumbbell (Rule + 2 Points) | Composite mark |
| 5C.9 | Slope chart (Line + Point + Text) | Two-point layout |
| 5C.10 | Bullet chart (layered Rect + Rule target) | Layered marks |

---

## Phase 6: GPU Rendering Completion

**Goal**: All mark types render correctly via wgpu. MSDF text. LTTB decimation.
**Entry**: Scene graph is feature-complete from Phases 4-5.

### 6A: Core Passes

| Task | Description |
|------|-------------|
| 6A.1 | PointPass: all 8 marker shapes via SDF in fragment shader |
| 6A.2 | PointPass: per-instance fill + stroke + size |
| 6A.3 | LinePass: screen-space quad expansion, miter/round/bevel joins |
| 6A.4 | LinePass: line caps (butt/round/square) |
| 6A.5 | LinePass: dash patterns |
| 6A.6 | RectPass: rounded corners via SDF |
| 6A.7 | RectPass: per-instance fill + stroke |
| 6A.8 | RulePass: thin line rendering with anti-aliasing |
| 6A.9 | TessPass: lyon tessellation for Area/Path marks |
| 6A.10 | Verify all passes: headless render → PNG, compare against SVG golden images |

### 6B: Text & Arc

| Task | Description |
|------|-------------|
| 6B.1 | MSDF glyph atlas generation (build-time or runtime) |
| 6B.2 | TextPass: MSDF-based text rendering |
| 6B.3 | TextPass: alignment, rotation, font size |
| 6B.4 | ArcPass: pie/donut sectors via SDF |
| 6B.5 | ArcPass: inner/outer radius, start/end angle |

### 6C: Performance Features

| Task | Description |
|------|-------------|
| 6C.1 | LTTB downsampling (CPU, O(n)) |
| 6C.2 | LTTB as wgpu compute shader |
| 6C.3 | Auto-LOD: switch between full data (zoomed in) and downsampled (zoomed out) |
| 6C.4 | Frustum culling: skip marks outside viewport |
| 6C.5 | GPU picking: color-encoded offscreen FBO, readback mark ID |
| 6C.6 | Weighted Blended OIT for transparency |
| 6C.7 | Benchmark suite (criterion.rs) for all passes |

---

## Phase 7: Interactivity

**Goal**: Pan, zoom, hover, click, brush in a winit window.
**Entry**: GPU rendering complete from Phase 6.

| Task | Description |
|------|-------------|
| 7.1 | winit window creation + event loop |
| 7.2 | Mouse pan (drag → translate viewport) |
| 7.3 | Scroll zoom (wheel → scale viewport) |
| 7.4 | Hover detection (GPU pick → nearest mark → highlight) |
| 7.5 | Tooltip rendering (TextMark overlay near cursor) |
| 7.6 | Click selection (mark ID → callback) |
| 7.7 | Rectangular brush selection (drag region → selected mark IDs) |
| 7.8 | Keyboard shortcuts (reset view, toggle grid, etc.) |
| 7.9 | Smooth animated transitions between states |
| 7.10 | Resize handling (window resize → re-layout) |

---

## Phase 8: Polish & Ecosystem

**Goal**: Production-ready library with docs, benchmarks, data connectors.

### 8A: Publication Presets

| Task | Description |
|------|-------------|
| 8A.1 | `Preset::Nature { columns: 1|2 }` — sets width, DPI, font |
| 8A.2 | `Preset::Science { columns: 1|2|3 }` |
| 8A.3 | `Preset::IEEE { columns: 1|2 }` |
| 8A.4 | `Preset::ACS` |
| 8A.5 | `Preset::Poster { size: A0|A1 }` |
| 8A.6 | `Preset::Slide { aspect: 16:9|4:3 }` |
| 8A.7 | PDF output (direct generation or SVG→PDF via printpdf) |

### 8B: Data Connectors

| Task | Description |
|------|-------------|
| 8B.1 | `impl From<&[f64]>` and `IntoIterator` for data input |
| 8B.2 | CSV reader: `Chart::from_csv("file.csv").x("col1").y("col2")` |
| 8B.3 | Polars DataFrame integration (feature-gated) |
| 8B.4 | ndarray integration (feature-gated) |
| 8B.5 | JSON spec serialization (Chart ↔ JSON) |

### 8C: Color System Completion

| Task | Description |
|------|-------------|
| 8C.1 | Named CSS colors (140 colors) |
| 8C.2 | Sequential palettes: viridis, inferno, magma, plasma, cividis |
| 8C.3 | Diverging palettes: RdBu, BrBG, PiYG, coolwarm |
| 8C.4 | Perceptually uniform palette generator |
| 8C.5 | CVD-safe palette auto-selection |
| 8C.6 | Pattern fills (hatching, dots, lines) as alternative to color |

### 8D: Benchmarks & Testing

| Task | Description |
|------|-------------|
| 8D.1 | criterion.rs benchmark suite (per BENCHMARKS.md) |
| 8D.2 | Golden image regression tests (SVG + PNG) |
| 8D.3 | Statistical accuracy tests (vs. known reference values) |
| 8D.4 | Fuzz testing (arbitrary Chart specs via cargo-fuzz) |
| 8D.5 | Competitor comparison script (generate equivalent charts, measure) |

### 8E: Documentation

| Task | Description |
|------|-------------|
| 8E.1 | Crate-level rustdoc with examples |
| 8E.2 | Gallery of all chart types (SVG images in docs) |
| 8E.3 | Cookbook: 20 common chart recipes |
| 8E.4 | Architecture guide (for contributors) |
| 8E.5 | Migration guide (from matplotlib/ggplot2/plotly mental models) |

---

## Phase Dependencies

```
Phase 4A (positions) ──┐
Phase 4B (stats)    ───┤
Phase 4C (area/arc) ───┼──→ Phase 5 (chart types) ──→ Phase 6 (GPU) ──→ Phase 7 (interact)
Phase 4D (legends)  ───┤                                                       │
Phase 4E (faceting) ───┤                                                       ▼
Phase 4F (scales)   ───┤                                                  Phase 8 (polish)
Phase 4G (annot.)   ───┘
```

Phase 4 sub-phases (A through G) are mostly independent and can be worked in parallel.
Phase 5 requires Phase 4 outputs.
Phase 6 can partially overlap with Phase 5 (GPU passes don't need all chart types).
Phase 7 requires Phase 6.
Phase 8 can partially overlap with Phase 7.

---

## Exit Criteria Summary

| Phase | "Done" means |
|-------|-------------|
| **4** | Grammar API can express: grouped bar, stacked bar, stacked area, histogram, boxplot, pie, donut, heatmap, area chart. Faceting works. Legends auto-generate. All produce correct SVGs. |
| **5** | All Tier 1 + Tier 2 charts from SPEC.md produce correct SVGs. Express API covers 15+ chart types. |
| **6** | All mark types render via wgpu at 60 FPS for 100K points. MSDF text is readable. LTTB handles 1M+ points. |
| **7** | Interactive window with pan/zoom/hover/tooltip. GPU picking identifies marks. |
| **8** | Benchmarks prove we beat competitors on stated metrics. Docs and gallery published. Publication presets work. |
