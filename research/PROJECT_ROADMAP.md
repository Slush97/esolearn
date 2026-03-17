# esolearn: Project Roadmap

> High-level roadmap for the esolearn ecosystem.
> Covers new capabilities beyond the existing charting internals roadmap (`ROADMAP.md`).
> Updated periodically as priorities shift.

**Last updated**: 2026-03-17

---

## Overview

```
Stream 1: Geographic Visualization         Stream 2: Advanced Layouts
    │                                           │
    G1  esoc-geo crate (projections,            L1  Treemap layout
    │   GeoJSON, polygon simplification)        │
    │                                           L2  Sankey layout
    G2  Choropleth + bubble maps                │
    │   in esoc-chart grammar                   L3  Force-directed network
    │                                           │
    ▼                                           ▼
    ├───────────────┬───────────────────────────┘
                    │
                    ▼
            Stream 3: Output & Integration
                    │
                    H1  Interactive HTML output (SVG + JS)
                    │
                    H2  Data connectors (CSV, Polars, CRM export)
                    │
                    H3  Business intelligence examples
                    │
                    ▼
                  DONE
```

Streams 1 and 2 are independent and can be developed in parallel.
Stream 3 depends on outputs from both.

---

## Stream 1: Geographic Visualization

### G1 — `esoc-geo` Crate (Foundation)

**Goal**: Standalone crate for geographic data parsing, map projections, and polygon
operations. No dependency on esoc-chart — usable independently.

**Priority**: HIGH — enables an entirely new class of visualizations (choropleth,
bubble map, connection map, hex-bin map).

| Task | Description |
|------|-------------|
| G1.1 | **GeoJSON parser** — parse `FeatureCollection`, `Feature`, `Polygon`, `MultiPolygon`, `Point`, `LineString` into internal geometry types. Use `serde_json` (feature-gated) or a minimal parser for zero-dep builds. |
| G1.2 | **Geometry types** — `GeoPoint { lon, lat }`, `GeoPolygon(Vec<Ring>)`, `GeoMultiPolygon`, `GeoLineString`, `GeoFeature { geometry, properties: HashMap<String, Value> }`, `GeoCollection`. |
| G1.3 | **Map projections** — trait `Projection { fn project(&self, lon: f64, lat: f64) -> (f64, f64); fn invert(...); }`. Implement: `Mercator`, `EqualEarth`, `NaturalEarth1`, `AlbersUsa` (with Alaska/Hawaii insets), `Equirectangular`. |
| G1.4 | **Projection math** — each projection as pure functions, no allocations on the hot path. `AlbersUsa` needs composite projection logic (three conic projections stitched). |
| G1.5 | **Polygon simplification** — Ramer-Douglas-Peucker algorithm for reducing vertex count at lower zoom levels. Configurable tolerance. |
| G1.6 | **Bounding box computation** — `GeoCollection::bounds() -> GeoBounds { min_lon, min_lat, max_lon, max_lat }`. Auto-fit viewport to data. |
| G1.7 | **Bundled geometries** — embed simplified boundary data (world countries, US states) as compressed include bytes. Ship as `esoc-geo/data/` or a `bundled` feature. Target: world countries ~200KB, US states ~100KB after simplification. |
| G1.8 | **Centroid computation** — polygon centroid for label/bubble placement. |
| G1.9 | **Point-in-polygon** — ray casting for spatial joins (assign data points to regions). |
| G1.10 | **Tests** — round-trip projection/inversion accuracy < 1m, simplification preserves topology, AlbersUsa insets positioned correctly. |

**Crate structure**:
```
esoc-geo/
├── Cargo.toml          # optional: serde, bundled
├── src/
│   ├── lib.rs
│   ├── geometry.rs     # GeoPoint, GeoPolygon, GeoFeature, GeoCollection
│   ├── geojson.rs      # GeoJSON parsing
│   ├── projection/
│   │   ├── mod.rs      # Projection trait
│   │   ├── mercator.rs
│   │   ├── equal_earth.rs
│   │   ├── natural_earth.rs
│   │   ├── albers_usa.rs
│   │   └── equirectangular.rs
│   ├── simplify.rs     # Ramer-Douglas-Peucker
│   ├── spatial.rs      # point-in-polygon, centroid, bounds
│   └── bundled.rs      # embedded world/US geometries
└── data/
    ├── world_110m.geojson.zst
    └── us_states_20m.geojson.zst
```

**Dependencies**: None required (pure math). Optional: `serde`/`serde_json` for GeoJSON,
`zstd` for decompressing bundled data.

**Exit criteria**: Can load world GeoJSON, project through Mercator/EqualEarth/AlbersUsa,
simplify polygons, and output projected `(x, y)` coordinates ready for rendering.

---

### G2 — Geographic Charts in `esoc-chart`

**Goal**: Wire `esoc-geo` into the charting grammar. Choropleth and bubble maps via
both grammar and express APIs.

**Entry**: G1 complete. `esoc-chart` Phase 4C (area/arc marks) and 4D (legends)
from `ROADMAP.md` should be done or in progress — choropleth needs color legends.

| Task | Description |
|------|-------------|
| G2.1 | **`GeoMark` in esoc-scene** — new mark type: `GeoMark { polygons: Vec<ProjectedPolygon>, fill, stroke }`. Alternatively, compile geo polygons down to existing `PathMark` batches — evaluate which approach is cleaner. |
| G2.2 | **`MarkType::Geo` in grammar** — add to the `MarkType` enum. A Geo layer takes a `GeoCollection` as data source instead of `&[f64]`. |
| G2.3 | **`CoordSystem::Geographic { projection }` in grammar** — new coordinate system variant. When active, the compile pipeline projects `(lon, lat)` through the specified projection before scale mapping. |
| G2.4 | **Choropleth compile path** — match features to data rows by a key field (e.g., country name, state FIPS code). Map a value field to fill color via a sequential/diverging color scale. |
| G2.5 | **Bubble map compile path** — scatter points at `(lon, lat)` centroids with size encoding. Rendered as PointMark on top of base geography PathMarks. |
| G2.6 | **Connection map** — LineMarks between `(lon, lat)` pairs with optional great-circle interpolation. |
| G2.7 | **Auto-fit viewport** — compute bounding box of all projected geometry, set scale domains automatically. |
| G2.8 | **Express API**: |
|      | `choropleth(geo, values).projection(EqualEarth).color_scale(Sequential::Blues).title("GDP by Country").to_svg()` |
|      | `bubble_map(geo, lons, lats, sizes).projection(NaturalEarth1).to_svg()` |
|      | `us_map(state_values).title("Revenue by State").to_svg()` — convenience for AlbersUsa + bundled US states |
| G2.9 | **SVG rendering** — `PathMark` already renders to SVG `<path>`. Verify polygon winding order (even-odd fill rule). |
| G2.10 | **Examples**: world choropleth (population), US state choropleth (revenue), bubble map (city populations), connection map (flights). |

**Exit criteria**: `choropleth(...)` and `bubble_map(...)` produce publication-quality SVGs
with correct projections, auto-fit viewports, and color legends.

---

## Stream 2: Advanced Layout Algorithms

These add chart types currently marked STRETCH in the spec. Each is a self-contained
layout algorithm that produces standard marks (Rect, Path, Rule, Point, Text).

### L1 — Treemap Layout

**Goal**: Squarified treemap algorithm producing `RectMark` batches. Useful for
hierarchical composition visualization (revenue by sector → sub-sector).

**Priority**: HIGH — directly applicable to sales segment analysis, simple algorithm,
high visual impact.

| Task | Description |
|------|-------------|
| L1.1 | **Hierarchical data input** — `TreeNode { label, value, children }` or flat table with parent column. Conversion from flat → tree. |
| L1.2 | **Squarified treemap algorithm** — Bruls-Huizing-van Wijk (2000). Optimizes aspect ratios of rectangles. Input: tree + bounding rect. Output: `Vec<TreemapCell { label, value, rect, depth }>`. |
| L1.3 | **Nested border padding** — configurable padding between parent and child rects. |
| L1.4 | **Mark generation** — `TreemapCell` → `RectMark` (fill by value or category) + `TextMark` (labels, sized to fit cell). |
| L1.5 | **Color encoding** — color by depth, by value, or by category. Sequential scale for value, categorical for groups. |
| L1.6 | **Express API**: `treemap(labels, values, parents).color_by("sector").title("Revenue Breakdown").to_svg()` |
| L1.7 | **Examples**: file size treemap, revenue by sector/product, budget allocation. |
| L1.8 | **Tests**: total area equals bounding rect area, no overlapping cells, aspect ratios ≤ threshold. |

**Exit criteria**: `treemap(...)` produces correct squarified layouts with labels and
color encoding.

---

### L2 — Sankey Layout

**Goal**: Sankey flow diagram showing flows between stages. Useful for sales funnel
visualization (lead → qualified → proposal → closed).

**Priority**: HIGH — directly applicable to sales pipeline analysis, strong
presentation value.

| Task | Description |
|------|-------------|
| L2.1 | **Flow data input** — `Vec<Flow { source, target, value }>`. Validate: no self-loops, no negative values. |
| L2.2 | **Node positioning** — assign nodes to columns (stages). BFS from sources or explicit stage assignment. Vertical positioning via iterative relaxation (Gauss-Seidel). |
| L2.3 | **Link routing** — cubic Bézier curves between source and target node ports. Link width proportional to flow value. |
| L2.4 | **Mark generation** — nodes: `RectMark` (one per node). Links: `PathMark` with cubic Bézier (filled with transparency). Labels: `TextMark` (node name + value). |
| L2.5 | **Color encoding** — color links by source, target, or a custom field. Nodes colored by stage or category. |
| L2.6 | **Layout tuning** — configurable: node width, node padding, alignment (justify/left/right/center), iterations for relaxation. |
| L2.7 | **Express API**: `sankey(sources, targets, values).title("Sales Funnel").to_svg()` |
| L2.8 | **Examples**: sales funnel, energy flow, website navigation, budget allocation. |
| L2.9 | **Tests**: flow conservation (sum of inputs = sum of outputs per node), no overlapping nodes, links don't cross node boundaries. |

**Exit criteria**: `sankey(...)` produces correct flow diagrams with proper node
ordering, smooth Bézier links, and labels.

---

### L3 — Force-Directed Network Graph

**Goal**: Force-directed layout for node-link diagrams. Useful for client relationship
mapping, referral networks, organizational charts.

**Priority**: MEDIUM — more complex than treemap/sankey, but enables a unique class
of exploration.

| Task | Description |
|------|-------------|
| L3.1 | **Graph data input** — `Vec<Node { id, label, group?, weight? }>`, `Vec<Edge { source, target, weight? }>`. |
| L3.2 | **Force simulation** — Barnes-Hut approximation for n-body repulsion (O(n log n)). Spring forces for edges. Gravity toward center. Configurable: repulsion strength, spring stiffness, damping, iterations. |
| L3.3 | **Collision detection** — prevent node overlap with radius-based collision force. |
| L3.4 | **Mark generation** — nodes: `PointMark` (position from simulation, size from weight). Edges: `RuleMark` or `PathMark` (straight lines or curved). Labels: `TextMark`. |
| L3.5 | **Color encoding** — color nodes by group/community. Edge color by weight or source/target. |
| L3.6 | **Layout variants** — circular layout (for small graphs), hierarchical layout (Sugiyama), radial layout. Force-directed as default. |
| L3.7 | **Express API**: `network(nodes, edges).color_by("group").title("Client Network").to_svg()` |
| L3.8 | **Performance** — target: 1000 nodes in <1s (CPU). Consider GPU compute shader for 10k+ nodes via esoc-gpu. |
| L3.9 | **Examples**: social network, citation graph, client referral network, package dependency graph. |
| L3.10 | **Tests**: all nodes within viewport, no NaN positions, energy monotonically decreasing. |

**Exit criteria**: `network(...)` produces readable node-link diagrams with community
coloring and labels. Handles 500+ nodes without layout collapse.

---

## Stream 3: Output & Integration

### H1 — Interactive HTML Output

**Goal**: Emit self-contained HTML files with embedded SVG + minimal JS for
tooltips, hover highlighting, and pan/zoom. No external dependencies.

**Priority**: HIGH — makes every existing chart type immediately more useful for
presentations and internal dashboards without requiring GPU/winit.

| Task | Description |
|------|-------------|
| H1.1 | **HTML wrapper** — `Chart::to_html() -> String`. Embeds SVG inline with `<style>` and `<script>` blocks. Single self-contained file, no CDN. |
| H1.2 | **Data attributes** — attach `data-*` attributes to SVG marks during rendering (value, label, series, index). Zero overhead for SVG-only output. |
| H1.3 | **Tooltip on hover** — JS: mouseover mark → show floating `<div>` with data values. Style matches chart theme. |
| H1.4 | **Hover highlighting** — CSS: `:hover` opacity change on marks. Dim non-hovered series. |
| H1.5 | **Pan & zoom** — SVG `viewBox` manipulation via mouse drag and wheel. Reset button. |
| H1.6 | **Click-to-filter** — click legend entry → toggle series visibility. |
| H1.7 | **Responsive sizing** — SVG scales to container width, preserves aspect ratio. |
| H1.8 | **Export button** — "Download SVG" / "Download PNG" buttons in the HTML. PNG via canvas `drawImage` + `toDataURL`. |
| H1.9 | **Dark mode** — respect `prefers-color-scheme` or toggle button. |
| H1.10 | **JS size budget** — inline JS must stay under 5KB minified. No frameworks. |

**Exit criteria**: `chart.to_html()` produces a single `.html` file that opens in any
browser with working tooltips, hover, and pan/zoom. File size < SVG size + 10KB.

---

### H2 — Data Connectors

**Goal**: Streamline the path from raw data sources to charts and ML models.

**Priority**: MEDIUM — reduces friction for real-world usage, especially CRM exports.

| Task | Description |
|------|-------------|
| H2.1 | **CSV → Chart direct** — `Chart::from_csv("file.csv").x("revenue").y("close_rate").to_svg()`. Infer column types (numeric, categorical, date). |
| H2.2 | **Polars DataFrame interop** — feature-gated `From<DataFrame>` for both `Dataset` (scry-learn) and chart data input. Column selection by name. |
| H2.3 | **JSON data input** — parse JSON arrays of objects into columnar data. Useful for API responses. |
| H2.4 | **Excel/XLSX reader** — feature-gated via `calamine`. Many CRM exports are Excel. |
| H2.5 | **Dataset splitting utilities** — `Dataset::filter(column, predicate)`, `Dataset::group_by(column)` for segmentation before charting or modeling. |

**Exit criteria**: Can go from a CRM CSV export to a segmented analysis chart in
under 10 lines of code.

---

### H3 — Business Intelligence Examples

**Goal**: End-to-end example applications demonstrating the library for real business
analysis. These serve as both documentation and templates.

**Priority**: MEDIUM — showcases the library's capabilities, provides starting points
for real analyses.

| Example | Description | Crates Used |
|---------|-------------|-------------|
| H3.1 **Sales Lead Classifier** | Load CRM export → preprocess (impute, encode, scale) → train RandomForest/GBM → permutation importance → SHAP analysis → output: ICP report with feature rankings, segment close rates, decision tree visualization. | scry-learn, esoc-chart |
| H3.2 **Pipeline Funnel Dashboard** | Load deal stage data → compute conversion rates per stage → Sankey diagram of flow → bar chart of stage durations → choropleth of regional performance. Output: multi-chart HTML report. | esoc-chart (sankey, choropleth, bar), esoc-geo |
| H3.3 **Customer Segmentation** | Load customer data → KMeans/HDBSCAN clustering → PCA for 2D projection → scatter with cluster coloring → treemap of segment sizes → segment profile table. | scry-learn, esoc-chart |
| H3.4 **Churn Prediction Report** | Load telco churn dataset → train ensemble → ROC/PR curves → confusion matrix → feature importance bar chart → segment-level churn rates heatmap. | scry-learn, esoc-chart |
| H3.5 **Regional Sales Map** | Load sales data with state/country → aggregate by region → choropleth map with revenue → bubble map overlay for deal count → top/bottom region comparison bar chart. | esoc-chart, esoc-geo |

**Exit criteria**: Each example runs end-to-end with `cargo run --example`, reads from
bundled or downloadable datasets, and produces a complete HTML report.

---

## Recommended Execution Order

Based on dependencies, value, and complexity:

### Phase A (immediate — weeks 1-4)
**Build the geographic foundation and treemap.**

These have the highest impact-to-effort ratio and are independent of each other.

| Work Item | Rationale |
|-----------|-----------|
| **G1** (esoc-geo crate) | Foundation for all map work. Pure algorithms, no UI dependencies. Can be developed and tested in isolation. |
| **L1** (treemap layout) | Simple algorithm, immediately useful for sales segmentation, uses existing RectMark. |

### Phase B (weeks 5-8)
**Wire geo into charts, add Sankey, start interactive HTML.**

| Work Item | Rationale |
|-----------|-----------|
| **G2** (choropleth/bubble maps) | Requires G1. High visual impact for presentations. |
| **L2** (Sankey layout) | Independent of geo. Directly applicable to sales funnel analysis. |
| **H1** (interactive HTML) | Makes all existing and new chart types immediately more useful. No dependency on new chart types. |

### Phase C (weeks 9-12)
**Network graphs, data connectors, BI examples.**

| Work Item | Rationale |
|-----------|-----------|
| **L3** (force-directed network) | More complex, benefits from lessons learned in L1/L2. |
| **H2** (data connectors) | Reduces friction for the example applications. |
| **H3** (BI examples) | Requires most prior work to be done. Serves as integration test. |

### Parallel with existing roadmap

The existing `ROADMAP.md` (Phases 4-8) continues independently. Key coordination:

- **Phase 4C** (area/arc marks) should land before or during Phase B — needed for
  polygon fills in maps and Sankey link rendering.
- **Phase 4D** (legends) should land before Phase B — choropleth needs color legends.
- **Phase 4G** (annotations) is nice-to-have for map labels but not blocking.
- **Phase 6** (GPU rendering) is independent — maps render via SVG first, GPU later.

---

## Success Metrics

| Metric | Target |
|--------|--------|
| **Map rendering** | World choropleth SVG in < 100ms |
| **Treemap** | 1000-cell treemap in < 10ms |
| **Sankey** | 50-node flow diagram in < 5ms |
| **Network** | 500-node graph layout in < 1s |
| **HTML output** | Self-contained file < original SVG + 10KB |
| **CSV → chart** | Under 10 lines of user code |
| **BI examples** | Each runs end-to-end with `cargo run --example` |

---

## Non-Goals (for now)

- **Full GIS** — we build visualization-grade projections, not survey-grade geodesy
- **Real-time streaming data** — batch/static analysis first
- **Web framework / WASM target** — native Rust + SVG/HTML output for v1
- **Database connectors** — read from files/DataFrames, not directly from databases
- **Dashboard layout manager** — single charts and multi-chart HTML reports, not drag-and-drop dashboards

---

## Appendix: Technology Decisions

### Map Projections

Implement projections as pure Rust math (no proj4 FFI). The five selected projections
cover 95% of use cases:

| Projection | Use Case | Distortion |
|------------|----------|------------|
| **Mercator** | Web maps, familiar to everyone | Area distortion at poles |
| **Equal Earth** | Global thematic maps | Equal-area, modern (2018) |
| **Natural Earth 1** | World maps for presentation | Compromise, aesthetically pleasing |
| **Albers USA** | US-centric analysis | Equal-area conic + AK/HI insets |
| **Equirectangular** | Quick/debug, trivial math | High distortion everywhere |

More can be added later (Robinson, Winkel Tripel, Lambert) without architectural changes
since they all implement the same `Projection` trait.

### Bundled Geometry Data

Use Natural Earth public domain data, simplified to three resolutions:

| Resolution | Vertices (world) | File Size (zstd) | Use Case |
|------------|-------------------|-------------------|----------|
| 110m (low) | ~5K | ~50KB | Thumbnails, fast preview |
| 50m (medium) | ~25K | ~150KB | Standard charts |
| 10m (high) | ~200K | ~800KB | Publication, zoom |

Default to 110m, let users opt into higher with a feature flag.

### Sankey Layout Algorithm

Use the D3-sankey approach (Mike Bostock / Jason Davies):
1. Assign nodes to columns via longest-path or explicit ordering
2. Initialize vertical positions
3. Iterative relaxation (Gauss-Seidel) to minimize link crossings
4. Resolve overlaps with collision avoidance
5. Route links as cubic Bézier with width proportional to flow value

### Treemap Layout Algorithm

Use squarified treemaps (Bruls, Huizing, van Wijk, 2000):
- Greedy algorithm that optimizes aspect ratios
- O(n log n) for n cells
- Deterministic — same input always produces same layout
- Well-studied, easy to implement correctly

### Force-Directed Layout

Use velocity Verlet integration with:
- Many-body repulsion via Barnes-Hut (O(n log n))
- Spring forces on edges (Hooke's law)
- Center gravity (prevent drift)
- Configurable iteration count (default 300)
- Optional: GPU compute shader for 10k+ node graphs via existing esoc-gpu infrastructure
