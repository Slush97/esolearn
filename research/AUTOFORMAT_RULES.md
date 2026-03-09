# Automatic Chart Formatting Rules

Comprehensive, codifiable rules for automatic chart formatting. Every rule includes
concrete numeric thresholds suitable for direct translation into Rust code.

> **Status Legend**
> - ✅ DONE — Implemented correctly, matches spec
> - ⚠️ PARTIAL — Implemented but wrong values, hardcoded, or incomplete
> - ❌ TODO — Not implemented at all
> - 🔶 ACCEPTABLE — Deviates from spec but is a valid alternative
> - 📍 = file location reference

---

## Implementation Summary

| Area | Status | Coverage | Priority |
|------|--------|----------|----------|
| Font hierarchy (ratio-based) | ⚠️ Hardcoded, wrong values | ~20% | **P0** |
| Margin/spacing calculation | ⚠️ Magic numbers, semi-adaptive | ~30% | **P0** |
| Tick generation (Heckbert nice numbers) | ✅ Correct | ~60% | — |
| Tick generation (Extended Wilkinson) | ❌ Missing | 0% | P2 |
| Adaptive tick count | ❌ Hardcoded to 6 | 0% | **P0** |
| Number formatting (SI/commas/auto) | ⚠️ Basic, no SI/commas | ~20% | **P1** |
| Date/time formatting | ❌ Missing | 0% | P2 |
| Label collision detection | ❌ Missing entirely | 0% | **P0** |
| Bar gap ratio | ⚠️ Correct default, no config | ~40% | P1 |
| Histogram bar spacing | ❌ Wrong (20% gap, should ~0) | 0% | **P1** |
| Point size (area vs diameter) | ⚠️ Wrong unit + wrong value | ~10% | **P1** |
| Opacity rules | ❌ Missing | 0% | **P1** |
| Gridline rules (per chart type) | ⚠️ Always both, no logic | ~15% | **P1** |
| Legend placement | ⚠️ Always right, no logic | ~15% | P1 |
| Color palette (Tableau 10) | 🔶 Acceptable alt | ~50% | P2 |
| Okabe-Ito palette | ❌ Missing | 0% | P2 |
| Palette auto-selection | ❌ Missing | 0% | P2 |
| Zero inclusion (bar/area) | ❌ Missing — **misleading charts** | 0% | **P0** |
| Responsive sizing | ❌ Missing entirely | 0% | P2 |
| Aspect ratio banking | ❌ Missing | 0% | P3 |
| LTTB downsampling | ❌ Missing | 0% | P3 |
| Title wrapping | ⚠️ Center only, no wrapping | ~20% | P1 |
| Subtitle styling | ⚠️ Wrong font, no muted color | ~15% | P1 |
| Log scale formatting | ⚠️ Basic, no minor ticks | ~30% | P2 |

### Priority Definitions
- **P0 — Correctness**: Charts are misleading or broken without these (zero inclusion, label collisions, adaptive ticks)
- **P1 — Professional**: Charts look amateur without these (number formatting, opacity, gridline logic, point sizing)
- **P2 — Polished**: Distinguishes a good library from a great one (Wilkinson, responsive, palettes, date axes)
- **P3 — Advanced**: Research-grade features (banking, LTTB, multi-scale)

---

## Table of Contents

1. [Prior Art & Frameworks](#1-prior-art--frameworks)
2. [Typography & Text Rules](#2-typography--text-rules)
3. [Spacing & Layout Rules](#3-spacing--layout-rules)
4. [Axis & Scale Rules](#4-axis--scale-rules)
5. [Color Rules](#5-color-rules)
6. [Mark Sizing Rules](#6-mark-sizing-rules)
7. [Adaptive / Responsive Rules](#7-adaptive--responsive-rules)
8. [Algorithmic References](#8-algorithmic-references)

---

## 1. Prior Art & Frameworks

### 1.1 Vega / Vega-Lite (Gold Standard for Defaults)

Source: `vega-parser/src/config.js` — the single authoritative defaults file.

| Property | Default | Notes |
|---|---|---|
| `padding` | 0 | Canvas edge to chart group |
| `view.continuousWidth` | 200 px | For continuous x-field |
| `view.continuousHeight` | 200 px | For continuous y-field |
| `view.discreteWidth` | 20 px per step | For ordinal x-field (band) |
| `view.discreteHeight` | 20 px per step | For ordinal y-field (band) |
| `view.spacing` | 20 px | Between composed sub-views (facets) |
| `text.fontSize` | 11 px | Default text mark |
| `style.guide-label.fontSize` | 10 px | Tick labels |
| `style.guide-title.fontSize` | 11 px | Axis titles (bold) |
| `style.group-title.fontSize` | 13 px | Chart title (bold) |
| `style.group-subtitle.fontSize` | 12 px | Chart subtitle |
| `symbol.size` | 64 px^2 | Symbol mark area (radius ~ 4.5 px) |
| `style.point.size` | 30 px^2 | Point/circle/square mark area |
| `line.strokeWidth` | 2 px | Line stroke |
| `style.point.strokeWidth` | 2 px | Point stroke |
| `trail.size` | 2 px | Trail width |
| `defaultStrokeWidth` | 2 px | Fallback stroke |
| `axis.domainWidth` | 1 px | Axis domain line |
| `axis.gridWidth` | 1 px | Grid line |
| `axis.tickWidth` | 1 px | Tick mark |
| `axis.tickSize` | 5 px | Tick mark length |
| `axis.labelPadding` | 2 px | Tick label to tick mark |
| `axis.labelLimit` | 180 px | Max tick label width |
| `axis.labelOffset` | 0 px | Extra label offset |
| `axis.tickOffset` | 0 px | Tick offset from axis |
| `axis.titlePadding` | 4 px | Axis title to tick labels |
| `axis.bandPosition` | 0.5 | Band tick at center |
| `axis.minExtent` | 0 px | Min space reserved for axis |
| `axis.maxExtent` | 200 px | Max space reserved for axis |
| `axisBand.tickOffset` | -0.5 | Nudge band ticks |
| `title.offset` | 4 px | Title to chart |
| `title.subtitlePadding` | 3 px | Title to subtitle gap |
| `legend.padding` | 0 px | Legend internal padding |
| `legend.columnPadding` | 10 px | Legend column spacing |
| `legend.rowPadding` | 2 px | Legend row spacing |
| `legend.labelOffset` | 4 px | Symbol to label |
| `legend.titlePadding` | 5 px | Legend title to entries |
| `legend.labelLimit` | 160 px | Max legend label width |
| `legend.titleLimit` | 180 px | Max legend title width |
| `legend.symbolSize` | 100 px^2 | Legend symbol area |
| `legend.symbolStrokeWidth` | 1.5 px | Legend symbol stroke |
| `legend.gradientLength` | 200 px | Gradient legend length |
| `legend.gradientThickness` | 16 px | Gradient legend width |
| `legend.layout.offset` | 18 px | Legend offset from plot |

Vega-Lite mark defaults (from `src/mark.ts`):
- Bar `binSpacing`: 1 px
- Bar `continuousBandSize`: 5 px
- Bar `minBandSize`: 0.25 px
- Bar `timeUnitBandPosition`: 0.5
- Rect `binSpacing`: 0 px
- Tick mark `thickness`: 1 px
- Non-aggregate point/tick/circle/square opacity: 0.7 (aggregate: 1.0)
- Default mark color: `#4c78a8`

### 1.2 ggplot2 Theme Defaults

Base font size: **11 pt** (all ratios relative to this).

| Element | Multiplier | Computed at base=11 |
|---|---|---|
| `plot.title` | `rel(1.2)` | 13.2 pt |
| `plot.subtitle` | `rel(0.9)` | ~9.9 pt |
| `axis.title` | `rel(1.0)` | 11 pt |
| `axis.text` (tick labels) | `rel(0.8)` | 8.8 pt |
| `strip.text` (facet labels) | `rel(0.8)` | 8.8 pt |
| `legend.title` | `rel(1.0)` | 11 pt |
| `legend.text` | `rel(0.8)` | 8.8 pt |
| `lineheight` | 0.9 | Line spacing ratio |

### 1.3 Matplotlib rcParams Defaults

| Parameter | Default | Unit |
|---|---|---|
| `figure.figsize` | 6.4 x 4.8 | inches |
| `figure.subplot.left` | 0.125 | fraction |
| `figure.subplot.right` | 0.9 | fraction |
| `figure.subplot.bottom` | 0.11 | fraction |
| `figure.subplot.top` | 0.88 | fraction |
| `figure.subplot.wspace` | 0.2 | fraction |
| `figure.subplot.hspace` | 0.2 | fraction |
| `axes.xmargin` | 0.05 | fraction |
| `axes.ymargin` | 0.05 | fraction |
| `axes.titlesize` | "large" (~14.4 pt) | — |
| `axes.labelsize` | "medium" (~12 pt) | — |
| `xtick.labelsize` | "medium" (~12 pt) | — |
| `xtick.major.size` | 3.5 | points |
| `xtick.minor.size` | 2 | points |
| `xtick.major.width` | 0.8 | points |
| `xtick.minor.width` | 0.6 | points |
| `xtick.major.pad` | 3.5 | points |
| Constrained layout pad | 3/72 in = 3 pt | — |

### 1.4 Key Academic References

- **Cleveland** "The Elements of Graphing Data" (1985, 1994): Banking to 45 degrees,
  minimize non-data ink, use position over length/angle/area.
- **Tufte** "The Visual Display of Quantitative Information" (1983): Data-ink ratio
  (maximize data pixels / total pixels), eliminate chartjunk, sparklines.
- **Wilkinson** "The Grammar of Graphics" (1999, 2005): Seven components
  (DATA, TRANS, FRAME, SCALE, COORD, GRAPH, GUIDE). Optimization-based labeling.
- **Heckbert** "Nice Numbers for Graph Labels" (Graphics Gems, 1990): Original nice
  number algorithm (see Section 8).
- **Talbot, Lin, Hanrahan** "An Extension of Wilkinson's Algorithm for Positioning
  Tick Labels on Axes" (InfoVis 2010): Extended nice-number search with
  simplicity/coverage/density/legibility scoring (see Section 8).
- **Heer & Agrawala** "Multi-Scale Banking to 45 Degrees" (InfoVis 2006):
  Spectral decomposition + per-frequency banking (see Section 8).

---

## 2. Typography & Text Rules

### 2.1 Font Size Hierarchy — ⚠️ PARTIAL (hardcoded absolutes, no ratio system)

> 📍 `new_theme.rs:41-57` — light theme values: title=16, label=13, tick=11, legend=11
> 📍 `compile/annotation.rs:172` — subtitle uses `label_font_size` (wrong, should be its own)
>
> **Problems**: No `base_font_size` field. All sizes independent, not ratio-derived.
> Title 16px is too large (spec=13px at base 11). No min font enforcement (8px).
> Subtitle reuses axis label size. No muted subtitle color (uses full foreground).

Recommended system using a **base font size** (default 11px, matching Vega/ggplot2):

| Element | Ratio | px at base=11 | px at base=13 |
|---|---|---|---|
| Chart title | 1.2x base | 13 | 16 |
| Chart subtitle | 1.1x base | 12 | 14 |
| Axis title | 1.0x base | 11 | 13 |
| Tick labels | 0.9x base | 10 | 12 |
| Legend title | 1.0x base | 11 | 13 |
| Legend labels | 0.9x base | 10 | 12 |
| Annotation text | 0.9x base | 10 | 12 |
| Tooltip text | 0.85x base | 9 | 11 |

These ratios are derived from cross-referencing Vega (10/11/12/13),
ggplot2 (0.8/1.0/1.2), and Observable Plot conventions.

**Minimum readable font size: 8px** (never go below this for any element).

**Title weight**: bold. **Subtitle weight**: normal. **Subtitle color**: muted
(e.g., 60% opacity or a lighter grey like `#666`).

### 2.2 Title & Subtitle Rules — ⚠️ PARTIAL (center-only, no wrapping, subtitle not muted)

```
RULE: title.max_chars = floor(chart_width / (base_font_size * 0.6))
RULE: if title length > max_chars: wrap at word boundary
RULE: max title lines = 2 (truncate with ellipsis after)
RULE: title positioned at top-left (anchor = "start") for narrative charts
RULE: title positioned at top-center (anchor = "middle") for standalone charts
RULE: subtitle gap from title = 3px (Vega default)
RULE: title offset from chart = 4px (Vega default)
```

### 2.3 Axis Label Rotation Rules — ❌ TODO (P0: labels always horizontal, no collision detection)

Cascade strategy (try in order, stop at first success):

```
1. Horizontal (0 degrees)
   - Fits if: max_label_width < available_space_per_tick
   - available_space_per_tick = axis_length / tick_count

2. Diagonal (45 degrees)
   - Try if horizontal doesn't fit
   - Effective width per label = label_width * cos(45) = label_width * 0.707
   - Fits if: effective_width < available_space_per_tick

3. Vertical (90 degrees)
   - Try if diagonal doesn't fit
   - Effective width = font_height (~= font_size * 1.2)
   - Fits if: font_height < available_space_per_tick

4. Skip labels (show every Nth)
   - If nothing else works, thin labels: show every 2nd, 3rd, etc.
   - N = ceil(max_label_width / available_space_per_tick)
```

**Minimum gap between labels**: 4px (Plotly checks bounding box overlap).

### 2.4 Number Formatting Rules — ⚠️ PARTIAL (P1: no SI prefixes, no commas, wrong thresholds)

> 📍 `scale.rs:362-376` — `format_number()`: scientific at 1e-3 (too aggressive, spec says 1e-6),
> no comma grouping (10K-1M), no SI prefix (1M+), doesn't use step to compute precision.

Choose format based on the magnitude and range of tick values:

```rust
fn auto_number_format(min: f64, max: f64, step: f64) -> Format {
    let abs_max = max.abs().max(min.abs());
    let precision = decimal_places_needed(step);

    if abs_max >= 1e9 {
        // SI prefix: 1.2B, 3.4B
        Format::SiPrefix { precision: 1 }
    } else if abs_max >= 1e6 {
        // SI prefix: 1.2M, 3.4M
        Format::SiPrefix { precision: 1 }
    } else if abs_max >= 1e4 {
        // Comma grouped: 12,000  42,000
        Format::Grouped { decimal_places: 0 }
    } else if abs_max >= 1.0 {
        // Fixed point with minimal decimals
        Format::Fixed { decimal_places: precision.min(2) }
    } else if abs_max >= 0.01 {
        // Fixed point: 0.12, 0.05
        Format::Fixed { decimal_places: precision.min(3) }
    } else if abs_max >= 1e-6 {
        // SI prefix: 1.2m (milli), 3.4u (micro)
        Format::SiPrefix { precision: 2 }
    } else {
        // Scientific: 1.2e-9
        Format::Scientific { precision: 2 }
    }
}

fn decimal_places_needed(step: f64) -> usize {
    // Number of decimal places to distinguish adjacent ticks
    if step >= 1.0 { 0 }
    else { (-step.log10().floor()) as usize }
}
```

### 2.5 Date/Time Formatting Rules — ❌ TODO (P2: no date/time axis support)

D3's multi-resolution time format (the industry standard):

```
if not_round_second(date)    => ".%L"      (milliseconds: ".123")
if not_round_minute(date)    => ":%S"      (seconds: ":45")
if not_round_hour(date)      => "%I:%M"    (minutes: "12:30")
if not_round_day(date)       => "%I %p"    (hours: "2 PM")
if not_round_month(date)     => "%b %d"    (days: "Jan 15")
if not_round_year(date)      => "%B"       (months: "January")
else                         => "%Y"       (years: "2024")
```

For axis tick interval selection given a time range:

| Range | Interval | Format |
|---|---|---|
| < 1 second | 1/5/10/50/100/250/500 ms | ".123" |
| < 1 minute | 1/5/15/30 seconds | ":45" |
| < 1 hour | 1/5/15/30 minutes | "12:30" |
| < 1 day | 1/3/6/12 hours | "2 PM" |
| < 1 month | 1/2/7 days | "Jan 15" / "Mon" |
| < 1 year | 1/3 months | "January" |
| >= 1 year | 1/5/10/25/50/100 years | "2024" |

---

## 3. Spacing & Layout Rules

### 3.1 Margin Calculation Algorithm — ⚠️ PARTIAL (P0: magic numbers, not spec-derived)

> 📍 `compile/layout.rs:1-48` — Magic constants: top=28, bottom=25, left=`tick*4+15`, right=130 (legend) or 15.
> No tick label width measurement. No spec values used (tick_size=5, padding=2, title_pad=4).

Margins should be computed adaptively based on what elements are present:

```
margin_left   = axis_tick_size + tick_label_width + label_padding + axis_title_height + title_padding
margin_bottom = axis_tick_size + tick_label_height + label_padding + axis_title_height + title_padding
margin_top    = chart_title_height + title_offset + subtitle_height + subtitle_padding
margin_right  = if legend_right { legend_width + legend_offset } else { 10 }
```

Concrete fallback values when auto-computation is not possible:

| Component | Pixels |
|---|---|
| Axis tick size | 5 |
| Tick label padding | 2 |
| Axis title padding | 4 |
| Title offset | 4 |
| Subtitle padding | 3 |
| Legend offset | 18 |
| Minimum margin (any side) | 5 |

### 3.2 Plot Area Ratio — ❌ TODO (no validation against 65-80% target)

Target: **the plot area should be 65-80% of total chart dimensions**.

Matplotlib's defaults encode this:
- Horizontal plot fraction: `0.9 - 0.125 = 0.775` (77.5%)
- Vertical plot fraction: `0.88 - 0.11 = 0.77` (77%)

### 3.3 Padding Between Elements — ⚠️ PARTIAL (some padding exists but not from spec values)

| Pair | Pixels | Source |
|---|---|---|
| Title to subtitle | 3 | Vega |
| Title to plot area | 4 | Vega |
| Axis title to tick labels | 4 | Vega |
| Tick mark to tick label | 2 | Vega |
| Legend title to entries | 5 | Vega |
| Legend symbol to label | 4 | Vega |
| Legend column gap | 10 | Vega |
| Legend row gap | 2 | Vega |
| Facet sub-view spacing | 20 | Vega-Lite |
| Subplot h/v spacing | 20% of subplot | Matplotlib |

### 3.4 Legend Placement Rules — ⚠️ PARTIAL (P1: always right, no adaptive logic)

> 📍 `compile/legend_gen.rs:102` — `legend_x = plot_x + plot_w + 15.0` (always right, offset 15 not 18).
> No item count check, no width threshold, no bottom placement, no max constraints.

```
RULE: if num_legend_items <= 5 AND chart_width > 400px:
    place legend at RIGHT
    legend.maxWidth = min(220px, chart_width * 0.25)

RULE: if num_legend_items > 5 OR chart_width <= 400px:
    place legend at BOTTOM
    legend.maxHeight = chart_height * 0.30  // never more than 30%
    layout = horizontal-wrap

RULE: if chart is pie/donut:
    place legend at RIGHT (always)

RULE: legend should never exceed 30% of chart height (right) or 25% of chart width (bottom)
```

### 3.5 Vega-Lite Layout Algorithm (Simplified) — ❌ TODO (reference implementation for our layout rewrite)

```
1. Start with requested width/height (default: 200x200 continuous, 20px/step discrete)
2. Add autopadding = 5px per side
3. For each axis:
   a. Measure tick labels (estimate from data + format)
   b. Reserve max(minExtent, min(measured, maxExtent)) = clamp(0, measured, 200)
   c. Add titlePadding + title height if axis title present
4. For title: add title height + offset (4px)
5. For legend: add legend width/height + offset (18px) in appropriate direction
6. Total = plot_area + margins (computed above)
```

---

## 4. Axis & Scale Rules

### 4.1 Nice Numbers Algorithm (Heckbert 1990) — ✅ DONE

> 📍 `esoc-scene/src/scale.rs:334-359` — Correct thresholds: round {1.5, 3.0, 7.0}, ceil {1.0, 2.0, 5.0}.
> `nice_ticks_linear()` at line 304 implements `loose_label` correctly.

The foundational algorithm used by most libraries. Steps:

```rust
/// Find a "nice" number approximately equal to x.
/// If round=true, round to nearest; if false, ceiling.
fn nice_num(x: f64, round: bool) -> f64 {
    let exp = x.log10().floor();
    let frac = x / 10f64.powf(exp);  // normalized: 1.0 <= frac < 10.0
    let nice = if round {
        // Rounding thresholds: 1.5, 3.0, 7.0
        if frac < 1.5 { 1.0 }
        else if frac < 3.0 { 2.0 }
        else if frac < 7.0 { 5.0 }
        else { 10.0 }
    } else {
        // Ceiling thresholds: 1.0, 2.0, 5.0
        if frac <= 1.0 { 1.0 }
        else if frac <= 2.0 { 2.0 }
        else if frac <= 5.0 { 5.0 }
        else { 10.0 }
    };
    nice * 10f64.powf(exp)
}

/// Generate nice axis bounds and tick spacing.
fn loose_label(data_min: f64, data_max: f64, target_ticks: usize) -> (f64, f64, f64) {
    let range = nice_num(data_max - data_min, false);       // ceiling
    let step = nice_num(range / (target_ticks as f64 - 1.0), true);  // round
    let graph_min = (data_min / step).floor() * step;
    let graph_max = (data_max / step).ceil() * step;
    (graph_min, graph_max, step)
}
```

### 4.2 D3 Tick Step Algorithm — ❌ TODO (P2: alternative to Heckbert, faster)

D3 uses three threshold constants:

```rust
const E10: f64 = 7.0710678118654755;   // sqrt(50)
const E5: f64  = 3.1622776601683795;   // sqrt(10)
const E2: f64  = 1.4142135623730951;   // sqrt(2)

fn tick_step(start: f64, stop: f64, count: usize) -> f64 {
    let step0 = (stop - start).abs() / count.max(1) as f64;
    let power = step0.log10().floor();
    let base = 10f64.powf(power);
    let error = step0 / base;
    let factor = if error >= E10 { 10.0 }
                 else if error >= E5 { 5.0 }
                 else if error >= E2 { 2.0 }
                 else { 1.0 };
    if stop < start { -factor * base } else { factor * base }
}
```

Nice step values form the sequence: `..., 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, ...`

### 4.3 Extended Wilkinson Algorithm (Talbot, Lin, Hanrahan 2010) — ❌ TODO (P2: state-of-the-art)

The state-of-the-art tick labeling algorithm. Optimizes a weighted score over four criteria.

**Nice number sequence Q (in preference order):**
```
Q = [1, 5, 2, 2.5, 4, 3]
```

**Scoring weights:**
```
w = [0.25, 0.2, 0.5, 0.05]  // [simplicity, coverage, density, legibility]
```
Density has the highest weight -- maintaining the requested tick density matters most.

**Scoring functions:**

```rust
fn simplicity(q_index: usize, q_len: usize, j: usize, lmin: f64, lmax: f64, lstep: f64) -> f64 {
    let i = q_index + 1;
    let v = if (lmin % lstep < 1e-10) || ((lstep - lmin % lstep) < 1e-10) {
        // lmin is a multiple of lstep (anchored at nice boundary)
        if lmin <= 0.0 && lmax >= 0.0 { 1 } else { 0 }
    } else { 0 };
    1.0 - (i as f64 - 1.0) / (q_len as f64 - 1.0) - j as f64 + v as f64
}

fn coverage(dmin: f64, dmax: f64, lmin: f64, lmax: f64) -> f64 {
    let range = dmax - dmin;
    1.0 - 0.5 * ((dmax - lmax).powi(2) + (dmin - lmin).powi(2)) / (0.1 * range).powi(2)
}

fn density(k: usize, m: usize, dmin: f64, dmax: f64, lmin: f64, lmax: f64) -> f64 {
    let r = (k as f64 - 1.0) / (lmax - lmin);
    let rt = (m as f64 - 1.0) / (dmax.max(lmax) - dmin.min(lmin));
    2.0 - (r / rt).max(rt / r)
}

fn legibility(_lmin: f64, _lmax: f64, _lstep: f64) -> f64 {
    1.0  // placeholder; can penalize overlapping labels, tiny font, etc.
}
```

**Search procedure (simplified):**
```
for j in 1.. {                              // unit multiplier
    for (qi, q) in Q.iter().enumerate() {   // nice number
        for k in 2.. {                      // tick count
            let delta = (dmax - dmin) / ((k + 1) as f64 * j as f64 * q);
            let z = delta.log10().ceil();
            let step = j as f64 * q * 10f64.powf(z);
            // try candidate start positions
            for start in candidates(dmin, step, j) {
                let lmin = start * step;
                let lmax = lmin + (k - 1) as f64 * step;
                let score = w[0] * simplicity(qi, Q.len(), j, lmin, lmax, step)
                          + w[1] * coverage(dmin, dmax, lmin, lmax)
                          + w[2] * density(k, target_m, dmin, dmax, lmin, lmax)
                          + w[3] * legibility(lmin, lmax, step);
                // keep best score; early-exit when upper bounds < best
            }
        }
    }
}
```

### 4.4 Tick Count Heuristics — ❌ TODO (P0: hardcoded to 6, ignores axis length)

> 📍 `compile/axis_gen.rs:36-37` — `let x_ticks = 6; let y_ticks = 6;`
> Should be `max(2, min(10, floor(axis_length / min_spacing)))` with 80px/40px thresholds.

Default target tick count (before nice-number adjustment):

```
RULE: target_ticks = max(2, min(10, floor(axis_length_px / min_tick_spacing)))
RULE: min_tick_spacing (horizontal x-axis) = 80px  (amCharts: 120px, conservative: 80px)
RULE: min_tick_spacing (vertical y-axis)   = 40px  (amCharts default, widely used)
```

Special cases:
- Time axes: use D3-style interval selection (see Section 2.5)
- Log axes: one tick per power of 10, minor ticks at 2-9x within each decade
- Band/ordinal: one tick per category (thin if too many)

### 4.5 When to Include Zero — ❌ TODO (P0: bar charts can be MISLEADING without this)

> 📍 `compile/mod.rs:288-327` — `compute_resolved_data_bounds()` applies 5% pad uniformly.
> No mark-type check. Bar baseline drawn at y=0 (`mark_gen.rs:222`) but axis domain may not include 0.
> **Result**: If all bars are 50-100, axis shows ~47-105, bars appear to start from bottom = deceptive.

```
RULE: if mark_type is BAR or AREA:
    ALWAYS include zero on the value axis (this is non-negotiable)

RULE: if mark_type is LINE or POINT:
    Include zero if: data_min > 0 AND data_min < 0.25 * data_range
    Otherwise: use nice_num bounds without forcing zero

RULE: if explicit domain is set by user:
    Respect user domain, do not force zero
```

### 4.6 Aspect Ratio: Banking to 45 Degrees — ❌ TODO (P3)

Cleveland's principle: choose aspect ratio so the average absolute slope of line
segments is 45 degrees, maximizing the discriminability of slope changes.

```rust
/// Compute optimal aspect ratio for a line chart.
/// Returns height/width ratio.
fn bank_to_45(xs: &[f64], ys: &[f64]) -> f64 {
    // Median absolute slope of consecutive segments
    let mut slopes: Vec<f64> = xs.windows(2).zip(ys.windows(2))
        .map(|(xw, yw)| ((yw[1] - yw[0]) / (xw[1] - xw[0])).abs())
        .collect();
    slopes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_slope = slopes[slopes.len() / 2];
    // aspect = 1/median_slope gives 45-degree average
    (1.0 / median_slope).clamp(0.2, 5.0)  // clamp to sane range
}
```

For the multi-scale banking variant (Heer & Agrawala 2006), decompose the signal
via FFT into frequency components and bank each independently, generating a separate
chart per informative frequency scale.

### 4.7 Log Scale Formatting — ⚠️ PARTIAL (P2: major ticks only, no minor ticks at 2x-9x)

> 📍 `scale.rs:327-331` — Only integer powers of base. No minor ticks per decade.

```
RULE: Major ticks at: 10^0, 10^1, 10^2, ... (integer powers of base)
RULE: Minor ticks at: 2*10^n, 3*10^n, ..., 9*10^n (8 minor ticks per decade for base 10)
RULE: Label format for major ticks:
    if range < 4 decades: show actual value (1, 10, 100, 1000)
    if range >= 4 decades: show as power (10^0, 10^1, ..., 10^6)
    if range >= 8 decades: show as scientific (1e0, 1e3, 1e6)
RULE: Minor tick labels: hidden by default; show if space permits and range < 2 decades
```

### 4.8 Gridline Rules — ⚠️ PARTIAL (P1: always both directions, no per-chart-type logic)

> 📍 `compile/axis_gen.rs:40-72` — Single `theme.show_grid` bool, always both axes.
> 📍 `new_theme.rs:46` — Grid color `(0.79, 0.79, 0.79)` ≈ #C9C9C9, too dark (spec: #E0E0E0).
> Bar charts wrongly show vertical gridlines. No band-axis suppression.

```
RULE: Bar charts: show horizontal gridlines only (value axis), no vertical
RULE: Line/scatter charts: show both horizontal and vertical gridlines
RULE: Pie/donut: no gridlines
RULE: Heatmaps: no gridlines

RULE: Grid color = light gray (e.g., #e0e0e0 or 15% opacity of text color)
RULE: Grid width = 1px (Vega default), or 0.5px for dense charts
RULE: Grid count = match tick count
RULE: Band axes (categorical): no gridlines (Vega: axisBand.grid = false)

RULE: Numeric axes: gridlines on
RULE: Time axes: tick marks only, no gridlines (ONS recommendation)
RULE: If data points are directly labeled: omit gridlines

Target: 6-10 gridlines for desktop, 3-6 for mobile/small views.
```

---

## 5. Color Rules

### 5.1 Default Categorical Palette — 🔶 ACCEPTABLE (Tableau 10 used, Okabe-Ito not defined)

> 📍 `esoc-color/src/palette.rs:56-69` — Uses Tableau 10 (D3 default). Spec prefers Okabe-Ito
> for colorblind safety but lists Tableau 10 as acceptable alternative.

Use the **Okabe-Ito** palette as the colorblind-safe default (8 colors):

| Index | Name | Hex | sRGB |
|---|---|---|---|
| 0 | Orange | `#E69F00` | (230, 159, 0) |
| 1 | Sky Blue | `#56B4E9` | (86, 180, 233) |
| 2 | Bluish Green | `#009E73` | (0, 158, 115) |
| 3 | Yellow | `#F0E442` | (240, 228, 66) |
| 4 | Blue | `#0072B2` | (0, 114, 178) |
| 5 | Vermillion | `#D55E00` | (213, 94, 0) |
| 6 | Reddish Purple | `#CC79A7` | (204, 121, 167) |
| 7 | Black | `#000000` | (0, 0, 0) |

Alternative: **Tableau 10** (Vega-Lite default for nominal fields):
`#4e79a7 #f28e2c #e15759 #76b7b2 #59a14f #edc949 #af7aa1 #ff9da7 #9c755f #bab0ab`

Vega's single-hue default mark color: `#4c78a8` (steel blue).

### 5.2 Palette Selection Rules — ❌ TODO (P2: no auto-selection by data type)

```
RULE: if data_type == Nominal (unordered categories):
    Use categorical palette (Okabe-Ito or Tableau 10)
    Max categories before degradation: 8 (Okabe-Ito) or 10 (Tableau 10)

RULE: if data_type == Ordinal (ordered categories):
    Use single-hue sequential palette (light to dark)
    e.g., Blues, Greens

RULE: if data_type == Quantitative (continuous):
    if has_meaningful_midpoint (e.g., 0, mean, threshold):
        Use diverging palette (e.g., Blue-White-Red, RdBu)
    else:
        Use sequential palette (e.g., Viridis, Blues)

RULE: if num_categories > 10:
    DO NOT use color alone
    Use faceting, or color + shape/pattern combinations
    Warn the user

RULE: if num_categories > palette_size:
    Recycle palette but warn (or extend with lighter/darker variants)
```

### 5.3 Opacity Rules — ❌ TODO (P1: no opacity control at all)

> No opacity field on marks or in theme. Area fill hardcodes 0.3 alpha at `mark_gen.rs:285`.
> No density-adaptive opacity for scatter overplotting.

```
RULE: if mark_type is POINT and data_count > 100:
    opacity = max(0.1, min(0.7, 100.0 / data_count.sqrt()))
    // sqrt scaling: 1000 points -> ~0.32, 10000 -> ~0.10

RULE: if mark_type is POINT and data_count <= 100:
    opacity = 0.7 (Vega default for non-aggregate)

RULE: if mark_type is BAR/AREA/RECT:
    opacity = 1.0 (filled marks should be opaque)

RULE: if multiple overlapping series (layered lines/areas):
    area opacity = 0.3-0.5 per layer
    line opacity = 1.0 (lines stay opaque)

RULE: aggregate marks (mean, sum, etc.):
    opacity = 1.0
```

### 5.4 Background Contrast — ⚠️ PARTIAL (defaults exist but not spec-aligned)

> 📍 `new_theme.rs:42-44` — bg=#FFFFFF ✅, fg=#1a1a1a (close to spec #333) ✅,
> but axis/grid colors don't derive from foreground opacity as spec recommends.

```
RULE: Default background = white (#FFFFFF)
RULE: Text color = near-black (#333333 or #1a1a1a), NOT pure black
RULE: Muted text (subtitles, secondary) = #666666
RULE: Axis domain/tick color = #888888 or 50% opacity of text
RULE: Grid color = #E0E0E0 or 15% opacity of text
RULE: All mark colors must have contrast ratio >= 3:1 against background
      (WCAG AA for non-text elements)
```

---

## 6. Mark Sizing Rules

### 6.1 Point / Circle Size — ⚠️ PARTIAL (P1: wrong unit + wrong value)

> 📍 `new_theme.rs:56` — `point_size: 8.0` treated as diameter (area ~50px²). Spec: area=30px².
> 📍 `scene_svg.rs:240` — `r = size * 0.5` confirms diameter interpretation.
> **Must change to area-based sizing**: `r = sqrt(area / PI)`. No density adaptation.

Vega uses **area** (px^2) not radius for point sizing. This gives perceptually
proportional scaling (doubling area = doubling perceived size).

```
DEFAULT: point area = 30 px^2 (radius ~ 3.1 px)  [Vega style.point]
         symbol area = 64 px^2 (radius ~ 4.5 px)  [Vega symbol mark]

RULE: Encode data to area, not radius
RULE: size_range for continuous size encoding = [9, 361] px^2 (radius 1.7 to 10.7)
      This is Vega-Lite's default: sqrt(9)=3px min radius, sqrt(361)=19px max radius

RULE: Data density adjustment (scatter):
    if points_per_pixel > 0.1:    reduce default size to 16 px^2 (r~2.3)
    if points_per_pixel > 0.5:    switch to 2D density / hexbin
    points_per_pixel = data_count / (plot_width * plot_height)
```

### 6.2 Line Width — ⚠️ PARTIAL (default 2px correct, no multi-series scaling)

> 📍 `new_theme.rs:48` — `line_width: 2.0` ✅. But no reduction for multi-series (1.5 for ≤5, 1.0 for >5).

```
DEFAULT: line strokeWidth = 2 px  [Vega]
RANGE: 1-4 px for multiple series (distinguish by color primarily, not width)
RULE: if num_series == 1: strokeWidth = 2 px
RULE: if num_series <= 5: strokeWidth = 1.5 px (slightly thinner to reduce clutter)
RULE: if num_series > 5:  strokeWidth = 1 px
RULE: Highlighted / focused line: 3 px (1.5x default)
RULE: Axis domain line: 1 px
RULE: Grid lines: 1 px (or 0.5 px for dense grids)
```

### 6.3 Bar Width & Gap — ⚠️ PARTIAL (P1: default 0.8 correct, histogram WRONG)

> 📍 `mark_gen.rs:206` — `* 0.8` = 20% gap ratio ✅ for bars.
> **But** histogram bars use same 0.8 multiplier → 20% gaps between bins. Spec: ~0 gap (1px spacing).
> No `bar_min_width`, no `bin_spacing` param, no grouped bar inner/outer gap.

```
RULE: Bar gap ratio (gap / (bar + gap)):
    Default: 0.2 (20% gap, 80% bar) — Datawrapper convention
    Range: 0.1-0.5 is acceptable
    Histogram: 0.0-0.05 (bars touch or nearly touch)
    Grouped bars: inner_gap = 0 to 0.1, outer_gap = 0.2

RULE: Bar width computation:
    band_width = axis_length / num_categories
    bar_width = band_width * (1.0 - gap_ratio)

RULE: Minimum bar width = 1 px (Vega minBandSize = 0.25, round up)
RULE: Maximum bar width: uncapped, but > 100px starts looking odd

RULE: For continuous bars (no band scale):
    bar_width = 5 px (Vega continuousBandSize)

RULE: Bin spacing (histogram): 1 px between bars (Vega binSpacing)
```

### 6.4 Stroke Width for Outlines — ⚠️ PARTIAL (defaults exist, not configurable)

```
DEFAULT: mark outline stroke = 0 px (filled marks have no outline by default)
RULE: Point marks (unfilled): strokeWidth = 2 px (Vega default)
RULE: Legend symbols: strokeWidth = 1.5 px (Vega legend.symbolStrokeWidth)
RULE: Selected/hovered marks: strokeWidth = 2 px, stroke = darker shade
```

---

## 7. Adaptive / Responsive Rules

### 7.1 Dimension-Based Scaling — ❌ TODO (P2: no responsive logic exists)

Define size classes with concrete thresholds:

| Class | Width | Base Font | Tick Target | Legend |
|---|---|---|---|---|
| Tiny | < 200 px | 9 px | 3-4 | Hidden |
| Small | 200-399 px | 10 px | 4-6 | Bottom, compact |
| Medium | 400-699 px | 11 px | 6-8 | Right or Bottom |
| Large | 700-999 px | 12 px | 8-10 | Right |
| XLarge | >= 1000 px | 13 px | 10-12 | Right |

```rust
fn responsive_config(width: f32, height: f32) -> Config {
    let base_font = if width < 200.0 { 9.0 }
                    else if width < 400.0 { 10.0 }
                    else if width < 700.0 { 11.0 }
                    else if width < 1000.0 { 12.0 }
                    else { 13.0 };

    let max_ticks_x = ((width / 80.0) as usize).clamp(3, 12);
    let max_ticks_y = ((height / 40.0) as usize).clamp(3, 12);

    Config { base_font, max_ticks_x, max_ticks_y, .. }
}
```

### 7.2 Small Multiples Simplification — ⚠️ PARTIAL (P2: facet gap=20px correct, no font/tick reduction)

> 📍 `compile/facet.rs:240` — `gap = 20.0` ✅. But no inner-panel axis hiding,
> no tick-only-on-edges, no font size reduction for facets.

```
RULE: if facet_count > 1:
    Hide axis titles on inner panels (share with outer)
    Show tick labels only on leftmost column (y) and bottom row (x)
    Reduce base_font by 1px
    Reduce tick count by 30%

RULE: if facet_count > 9:
    Additionally: hide grid lines
    Reduce base_font by 2px total
    Consider sparkline mode (no axes at all, just data + labels)

RULE: Facet spacing = 20px (Vega-Lite default)
RULE: All facets share the same scale domain by default (Vega-Lite "shared")
```

### 7.3 Data Density Adaptation — ❌ TODO (P2: no density-based rendering changes)

```
RULE: if data_count < 30:
    Show individual points, no aggregation
    Point size = default (30 px^2)
    Labels can be shown per-point if chart is large enough

RULE: if 30 <= data_count < 500:
    Show individual points
    Reduce opacity: alpha = max(0.2, 1.0 - data_count / 500.0)
    Point size = default

RULE: if 500 <= data_count < 5000:
    Reduce point size to 16 px^2
    Reduce opacity: alpha = max(0.1, 50.0 / data_count.sqrt())
    Consider: hexbin or 2D density as alternative

RULE: if data_count >= 5000:
    Switch to 2D density / hexbin / contour
    Or use canvas/GPU rasterized rendering with LTTB downsampling
    LTTB target: ~2 * pixel_width points for line charts
```

### 7.4 Label Collision Resolution — ❌ TODO (P0: see also Section 2.3)

> 📍 `compile/axis_gen.rs:87-105` — All labels placed horizontal, no collision check at all.

Priority cascade (same as Section 2.3, but generalized):

```
1. Try fitting all labels at full size, horizontal
2. If collision: try rotating to 45 degrees
3. If collision: try rotating to 90 degrees
4. If collision: skip every 2nd label (show every other)
5. If collision: skip every 3rd, 4th, ... Nth label
6. If collision at max skip: truncate labels to max_chars with ellipsis

Collision check: for adjacent labels A, B:
    gap = B.start - A.end  (along axis)
    collision = gap < MIN_LABEL_GAP (4px)
```

---

## 8. Algorithmic References

### 8.1 Heckbert's Nice Numbers (1990) — ✅ DONE

Source: Graphics Gems, `Label.c`

Produces axis labels by choosing a step that is a power of ten times 1, 2, or 5.

- **Input**: data min, data max, target tick count (~5)
- **Output**: graph min, graph max, step, decimal precision
- **Algorithm**: See Section 4.1 for complete pseudocode
- **Thresholds**: Round: {1.5, 3.0, 7.0}. Ceil: {1.0, 2.0, 5.0}
- **Time complexity**: O(1)

Reference: Heckbert, P.S. "Nice Numbers for Graph Labels." Graphics Gems, Academic Press, 1990.

### 8.2 Extended Wilkinson (Talbot, Lin, Hanrahan 2010) — ❌ TODO (P2)

Source: IEEE InfoVis 2010, R `labeling` package

Extends Heckbert with an optimization-based search over a large space of possible
labelings, scored on four criteria: simplicity, coverage, density, legibility.

- **Q sequence**: `[1, 5, 2, 2.5, 4, 3]` (nice numbers in preference order)
- **Weights**: `[0.25, 0.2, 0.5, 0.05]` (simplicity, coverage, density, legibility)
- **Algorithm**: See Section 4.3 for complete pseudocode
- **Search**: Three nested loops (j, q, k) with early termination via upper bounds
- **Time complexity**: O(n) in practice with early exit, where n ~ target tick count

Reference: Talbot, J., Lin, S., Hanrahan, P. "An Extension of Wilkinson's Algorithm
for Positioning Tick Labels on Axes." IEEE TVCG 16(6), 2010.

### 8.3 D3 Ticks (Bostock) — ❌ TODO (P2)

Source: `d3-array/src/ticks.js`

A simpler, faster approach than Wilkinson. Uses three threshold constants
(sqrt(50), sqrt(10), sqrt(2)) to select factors of {10, 5, 2, 1}.

- **Constants**: e10 = sqrt(50) ~ 7.07, e5 = sqrt(10) ~ 3.16, e2 = sqrt(2) ~ 1.41
- **Algorithm**: See Section 4.2 for complete pseudocode
- **Time complexity**: O(1) for step computation, O(k) for tick generation
- **Precision**: Uses integer arithmetic internally for IEEE 754 correctness

Reference: https://d3js.org/d3-array/ticks

### 8.4 R's pretty() Algorithm — ❌ TODO (P2)

Based on Heckbert, with bias parameters for refinement.

```
1. d = max - min
2. c = d / n  (target step size)
3. base = 10^floor(log10(c))
4. unit = select from {1, 2, 5, 10} * base based on c/base and bias
5. graph_min = floor(min / unit) * unit
6. graph_max = ceil(max / unit) * unit
```

Bias coefficients: `u_bias = [1.5 + 0.5/(n+1), 1+1.5/(n+1)]` (R default).

Reference: R Core, `base::pretty()`. Also Heckbert 1990.

### 8.5 Multi-Scale Banking to 45 Degrees (Heer & Agrawala 2006) — ❌ TODO (P3)

Extends Cleveland's banking to handle multi-frequency time series:

```
1. Compute FFT of the time series
2. Identify dominant frequency components via spectral analysis
3. For each informative frequency scale:
   a. Band-pass filter the signal
   b. Compute optimal aspect ratio via Cleveland's banking formula
   c. Generate a separate chart at that aspect ratio
4. Present as a vertical stack of charts, one per scale
```

For the simple (single-scale) case, banking to 45 degrees means:
```
aspect_ratio = median(|dy/dx|) for consecutive segments
```
where aspect_ratio = height/width of the plot area.

Reference: Heer, J., Agrawala, M. "Multi-Scale Banking to 45 Degrees."
IEEE TVCG 12(5), 2006.

### 8.6 LTTB (Largest Triangle Three Buckets) for Line Downsampling — ❌ TODO (P3)

For rendering line charts with far more data points than pixels:

```
1. Split data into N buckets (N = target point count, typically 2 * pixel_width)
2. Keep first and last points
3. For each bucket i (2..N-1):
   a. Compute the average point of bucket i+1
   b. For each point in bucket i, compute the area of the triangle formed by:
      - The selected point from bucket i-1
      - The candidate point
      - The average point of bucket i+1
   c. Select the point with the largest triangle area
4. Return selected points
```

Reference: Steinarsson, S. "Downsampling Time Series for Visual Representation." 2013.

---

## Summary: Recommended Defaults for esoc-chart — ❌ TODO (this struct replaces current theme/config)

> The `AutoFormatConfig` struct below should be the single source of truth.
> Current `ChartTheme` in `new_theme.rs` has ~15 fields. This needs ~40+ fields.

```rust
pub struct AutoFormatConfig {
    // Typography
    pub base_font_size: f32,           // 11.0 px
    pub title_font_ratio: f32,         // 1.2
    pub subtitle_font_ratio: f32,      // 1.1
    pub tick_label_font_ratio: f32,    // 0.9
    pub min_font_size: f32,            // 8.0 px
    pub title_font_weight: FontWeight, // Bold
    pub subtitle_color_opacity: f32,   // 0.6

    // Spacing (px)
    pub title_offset: f32,             // 4.0
    pub subtitle_padding: f32,         // 3.0
    pub axis_title_padding: f32,       // 4.0
    pub tick_label_padding: f32,       // 2.0
    pub legend_offset: f32,            // 18.0
    pub legend_column_padding: f32,    // 10.0
    pub legend_row_padding: f32,       // 2.0
    pub facet_spacing: f32,            // 20.0

    // Axes
    pub axis_tick_size: f32,           // 5.0 px
    pub axis_domain_width: f32,        // 1.0 px
    pub axis_grid_width: f32,          // 1.0 px
    pub axis_tick_width: f32,          // 1.0 px
    pub axis_label_limit: f32,         // 180.0 px
    pub axis_max_extent: f32,          // 200.0 px
    pub min_tick_spacing_x: f32,       // 80.0 px
    pub min_tick_spacing_y: f32,       // 40.0 px
    pub default_tick_count: usize,     // 10  (D3/Wilkinson default)

    // Marks
    pub point_size: f32,               // 30.0 px^2
    pub line_stroke_width: f32,        // 2.0 px
    pub bar_gap_ratio: f32,            // 0.2
    pub bar_bin_spacing: f32,          // 1.0 px
    pub bar_continuous_width: f32,     // 5.0 px
    pub bar_min_width: f32,            // 0.25 px
    pub default_opacity: f32,          // 0.7 (non-aggregate point/tick)
    pub aggregate_opacity: f32,        // 1.0
    pub default_mark_color: [f32; 4],  // #4c78a8 = steel blue

    // Colors
    pub categorical_palette: &'static [[f32; 4]], // Okabe-Ito (8 colors)
    pub max_categorical: usize,        // 10 before warning

    // Layout
    pub default_continuous_size: f32,  // 200.0 px (width and height)
    pub default_discrete_step: f32,    // 20.0 px per category
    pub plot_area_ratio_target: f32,   // 0.77 (77% of total)
    pub legend_max_height_ratio: f32,  // 0.30 (30% of chart)
    pub legend_max_width: f32,         // 220.0 px (right legend)
    pub min_label_gap: f32,            // 4.0 px

    // Grid
    pub grid_color_opacity: f32,       // 0.15
    pub grid_default_on_numeric: bool, // true
    pub grid_default_on_band: bool,    // false
    pub grid_default_on_time: bool,    // false
}
```

---

## Sources

### Academic Papers
- [Talbot, Lin, Hanrahan (2010) - Extended Wilkinson Algorithm](https://pubmed.ncbi.nlm.nih.gov/20975141/)
- [Heer & Agrawala (2006) - Multi-Scale Banking to 45 Degrees](https://idl.uw.edu/papers/banking)
- [Heckbert (1990) - Nice Numbers for Graph Labels (C source)](https://www.realtimerendering.com/resources/GraphicsGems/gems/Label.c)
- [Extended Wilkinson Observable Notebook](https://observablehq.com/@littleark/extended-wilkinson-algorithm)

### Library Documentation & Source Code
- [Vega Configuration Defaults](https://vega.github.io/vega/docs/config/)
- [Vega Parser config.js (source of truth)](https://github.com/vega/vega/blob/main/packages/vega-parser/src/config.js)
- [Vega-Lite Configuration](https://vega.github.io/vega-lite/docs/config.html)
- [Vega-Lite Mark Defaults (mark.ts)](https://github.com/vega/vega-lite/blob/master/src/mark.ts)
- [Vega-Lite Size Documentation](https://vega.github.io/vega-lite/docs/size.html)
- [D3 Ticks Algorithm](https://d3js.org/d3-array/ticks)
- [D3 Time Scales](https://d3js.org/d3-scale/time)
- [D3 Number Formatting](https://d3js.org/d3-format)
- [D3 Time Formatting](https://d3js.org/d3-time-format)
- [D3 Categorical Color Schemes](https://d3js.org/d3-scale-chromatic/categorical)
- [ggplot2 Themes](https://ggplot2-book.org/themes.html)
- [Matplotlib rcParams](https://matplotlib.org/stable/users/explain/customizing.html)
- [R pretty() Documentation](https://stat.ethz.ch/R-manual/R-devel/library/base/html/pretty.html)
- [R labeling Package](https://cran.r-project.org/web/packages/labeling/labeling.pdf)
- [Observable Plot Axis Mark](https://observablehq.com/plot/marks/axis)

### Design Guidelines
- [Bar Width Best Practices (Stephen Few)](https://www.perceptualedge.com/articles/visual_business_intelligence/bar_widths.pdf)
- [Gridlines Guidelines (Data Viz Standards)](https://xdgov.github.io/data-design-standards/components/grids)
- [Gridlines - ONS Service Manual](https://service-manual.ons.gov.uk/data-visualisation/guidance/axes-and-gridlines)
- [Zero Baseline Rule](https://www.storytellingwithdata.com/blog/2012/09/bar-charts-must-have-zero-baseline)
- [Okabe-Ito Colorblind Palette](https://siegal.bio.nyu.edu/color-palette/)
- [Diverging vs Sequential Color Scales (Datawrapper)](https://www.datawrapper.de/blog/diverging-vs-sequential-color-scales)
- [Legend Placement (Carbon Design)](https://carbondesignsystem.com/data-visualization/legends/)
- [amCharts Label Collision](https://www.amcharts.com/docs/v4/tutorials/wrapping-and-truncating-axis-labels/)
- [AG Charts Axis Labels](https://www.ag-grid.com/charts/javascript/axes-labels/)
- [Tufte's Principles of Data-Ink](https://jtr13.github.io/cc19/tuftes-principles-of-data-ink.html)
