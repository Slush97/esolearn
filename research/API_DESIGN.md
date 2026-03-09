# Grammar of Graphics & API Design — Deep Research

## Wilkinson's 7 Orthogonal Components

1. **DATA** — Source dataset + variable operations (filter, sort, derive)
2. **TRANS** — Statistical transforms: bin, smooth, rank, aggregate → new variables
3. **FRAME** — Mathematical space: Cartesian, polar, geographic, nested/faceted
4. **SCALE** — Bijections from data domain to aesthetic range (position, color, size, shape)
5. **COORD** — Geometric transform of frame: transpose, reflect, warp (polar = COORD on Cartesian FRAME)
6. **ELEMENT** — Visual marks: point, line, area, bar, interval, schema
7. **GUIDE** — Annotations: axes, legends, titles (inverses of scales for human reading)

**Key insight**: Components are orthogonal — any valid combination produces a valid graphic.
Bar chart = DATA + TRANS(bin) + FRAME(2D) + SCALE(cat x, linear y) + COORD(rect) + ELEMENT(interval) + GUIDE(axes).
Change COORD to polar → pie chart. Change ELEMENT to point → dot plot.

## How Major Libraries Adapted

### ggplot2 (Wickham 2010)
- Collapsed DATA+TRANS into `data + aes()` mapping
- Merged FRAME+COORD into `coord_*()`
- Made SCALE/GUIDE mostly automatic (inferred from types)
- **Added layers**: each layer = data + mapping + stat + geom + position
- **`+` operator composition**: additive, never removes info, late binding
- **`aes()` symbolic references**: deferred evaluation via NSE — mapping defined before data available

### Vega-Lite (Satyanarayan 2017)
- **Selections**: Formal grammar of interaction — `point`/`interval` selections drive conditional encodings, filter views, parameterize scales
- **View composition algebra**: `layer`, `concat`, `facet`, `repeat`
- **First-class data transforms**: aggregate, bin, timeUnit, filter, calculate, fold, pivot, window
- **Typed encoding channels**: `x`, `y`, `color` etc. carry field name + type (Q/N/O/T) → auto scale/axis/legend

### Observable Plot (Bostock 2021)
- **Marks, not layers**: Self-contained functions (`Plot.dot`, `Plot.line`) — no geom/stat/position decomposition
- **Implicit scales**: Inferred from data types and mark types
- **Transforms as mark options**: `Plot.rectY(data, Plot.binX({y: "count"}, {x: "weight"}))`
- Philosophy: "80% of charts in 20% of effort" — sacrifices some composition for directness

## API Patterns That Work

### Builder Pattern (Rust-native)
```rust
Chart::new()
    .data(&dataset)
    .x("weight", Scale::Linear)
    .y("height", Scale::Linear)
    .mark(Mark::Point)
    .color("species", Scale::Categorical)
    .build()?
```
Type-safe, Rust-idiomatic. Can enforce required fields at compile time (typestate).

### Hybrid: Builders + Serialize/Deserialize
Use builder for programmatic API, `impl Serialize/Deserialize` on spec types for JSON/TOML loading.
Best of both worlds: compile-time safety + declarative flexibility.

### Plotly Express Pattern: One Function = One Chart Archetype
```rust
let chart = Chart::scatter(&data, "x", "y")
    .color("category")
    .size("magnitude")
    .build()?;
```
- DataFrame-first, semantic arguments, smart defaults
- Returns mutable object for further customization (escape hatch)

### Altair-Style Encoding System
```
"field_name:Type"  where Type = Q(uantitative) | N(ominal) | O(rdinal) | T(emporal)
```
- Shorthand strings + full objects interchangeable
- Type annotation drives default scale, axis format, legend, color scheme

## The Escape Hatch Problem (80/20 Cliff)

### Pattern 1: Progressive Disclosure (Layered API)
- Level 1: One-liner functions (like Plotly Express)
- Level 2: Grammar composition (like ggplot2)
- Level 3: Direct scene graph / spec manipulation
- **Each level exposes the level below**

### Pattern 2: Open Configuration Objects
```rust
let mut chart = quick_scatter(&data, "x", "y");
chart.x_axis.tick_count = 5;
chart.marks[0].opacity = 0.7;
```

### Pattern 3: Closure Hooks
```rust
Chart::scatter(&data, "x", "y")
    .customize_axis(|axis| { axis.tick_values = vec![0.0, 0.5, 1.0]; })
```

### Pattern 4: "Eject" to Spec
```rust
let mut spec = Chart::scatter(&data, "x", "y").to_spec();
spec.marks.push(custom_annotation);
render(&spec)?;
```

### Makie.jl's Approach (best-in-class)
- Scene graph tree: each scene has coordinate system, camera, plots
- Plot recipes decompose to ~8 primitives (mesh, lines, scatter, text, image)
- Observables (reactive) drive updates
- **No separate low-level API** — high-level is syntactic sugar over same primitives users can access

## Type-Safe Visualization in Rust

### Typestate Pattern (Compile-Time Validation)
```rust
struct Chart<D, M, E> { _state: PhantomData<(D, M, E)> }
// Only Chart<HasData, HasMark, HasEncoding> can call .render()
```
Powerful but unwieldy with many state parameters.

### Pragmatic Alternative
Runtime validation in `build()` → `Result<Chart, ConfigError>` with descriptive errors.
Most production Rust builders use this (reqwest, tokio, wgpu).

### Builder Best Practices (from Rust ecosystem)
1. Consume `self` by value (prevent reuse of partial builders)
2. `build()` returns `Result` with descriptive error enum
3. `impl Into<T>` for flexible input types
4. Nested builders via closures for sub-configuration
5. `#[derive(Default)]` + `Option<T>` for optional config

### Data Ownership
**Recommended**: Store data in top-level Chart, layers reference by handle/index.
Per-layer data overrides via `Option<DataSource>` (None = use chart default).

### Trait-Based Extensibility
```rust
pub trait MarkRenderer {
    fn required_channels(&self) -> Vec<ChannelType>;
    fn optional_channels(&self) -> Vec<ChannelType>;
    fn render(&self, data: &ResolvedData, scales: &ScaleSet, canvas: &mut impl Canvas) -> Result<()>;
}

pub trait StatTransform { ... }
pub trait ScaleMapping { ... }
pub trait Backend { ... }
```

## Data Abstraction

### Pattern 1: Trait-based
```rust
pub trait DataSource {
    fn column_names(&self) -> Vec<&str>;
    fn column_type(&self, name: &str) -> Option<DataType>;
    fn column_f64(&self, name: &str) -> Option<Vec<f64>>;
    fn column_str(&self, name: &str) -> Option<Vec<String>>;
}
// Feature-gated impls for polars, arrow, etc.
```

### Pattern 2: Internal columnar format + From/Into
```rust
pub struct DataFrame { columns: HashMap<String, Column> }
pub enum Column { Float(Vec<f64>), Integer(Vec<i64>), String(Vec<String>), ... }
```

### Pattern 3: Derive macro
```rust
#[derive(Chartable)]
struct Iris { sepal_length: f64, species: String, ... }
// Generates DataSource impl, each field → column name
```

### Iterator Patterns
```rust
// Direct arrays
Chart::new().mark_point().x(vec![1.0, 2.0]).y(vec![4.0, 5.0])

// Struct iterator with mapping
Chart::new().mark_point().data_from_iter(records.iter(), |r| {
    DataPoint::new().x(r.weight).y(r.height).color(&r.species)
})
```

## Recommended 3-Level Architecture

**Level 1: Express** — `esolearn::scatter(&data, "x", "y").color("cat")`
**Level 2: Grammar** — `Chart::new().data(&d).layer(Layer::new().mark(Point).encode_x(...))`
**Level 3: Spec/SceneGraph** — `chart.to_spec()` → modify directly → `render(&spec)`

Level 1 creates Level 2 objects. Level 2 resolves to Level 3 specs.
Users enter at any level, drop down when needed.
`Backend` trait → SVG, GPU (wgpu), PDF, PNG.
