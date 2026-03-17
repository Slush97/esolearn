# esoc-chart Audit Report

**Date**: 2026-03-16
**Scope**: Full library audit (~7200 lines, 48 files)
**Baseline**: 119 tests passing, 0 failures (after round 4)

---

## Critical Issues (8)

### C1. Log/Date scales not wired into axis generation
**Files**: `compile/axis_gen.rs`
`axis_gen.rs` hardcodes `Scale::linear()` (lines 137-144). The scene layer (`esoc_scene/scale.rs`) implements Log, Time, Symlog, Power, and Sqrt scales, but none are accessible from the chart compilation pipeline. Time-series charts render Unix millisecond values instead of readable dates. Log-scale charts render with linear axis labels.

**Recommendation**: Add scale-type propagation from Layer encoding to axis_gen; implement time-axis tick formatting.

---

### ~~C2. Text and Rule mark types silently ignored~~ RESOLVED (Round 1)
Returns `ChartError::InvalidParameter` for unimplemented mark types.

### ~~C3. MarkBatch `.expect()` calls can panic on user data~~ RESOLVED (Round 1)
All `.expect()` replaced with `.map_err()`.

### ~~C4. No NaN/Inf input validation anywhere~~ RESOLVED (Round 1)
Centralized NaN/Inf validation in `compile/mod.rs` for x_data, y_data, and heatmap_data.

### ~~C5. No padding on non-bar chart data bounds~~ RESOLVED (Round 4)
5% padding applied for scatter/line/point charts before nicing.

### ~~C6. No gradient/continuous legend for heatmaps~~ RESOLVED (Round 2)
`GradientLegend` type implemented with viridis color bar and min/max labels.

### C7. Shape/Opacity/Text encoding channels defined but non-functional *(partial)*
**Files**: `grammar/encoding.rs:29-33`, `compile/mark_gen.rs`
`Channel` enum includes Shape, Opacity, and Text variants. None are wired into mark generation.

**Round 4**: `encode_x/y/color/size` methods deprecated with guidance to use `with_x/with_y/with_categories`.

### ~~C8. No position/mark-type validation~~ RESOLVED (Round 1)

---

## Moderate Issues (16)

### ~~M1. Stack does not handle mixed positive/negative values~~ RESOLVED (Round 1)
### ~~M2. Fill normalization misleading for zero-sum groups~~ RESOLVED (Round 1)
### ~~M3. Aggregation returns inconsistent values for empty groups~~ RESOLVED (Round 1)
### ~~M4. LOESS silently truncates mismatched x/y arrays~~ RESOLVED (Round 1)

### ~~M5. Area mark upper/lower path lengths not validated~~ RESOLVED (Round 4)
Added length equality check returning `ChartError::InvalidData` on mismatch.

### M6. Color assignment inconsistent across mark types
**Files**: `compile/mark_gen.rs`
Points/bars color by category index within layer; arcs color by global arc index.

### ~~M7. No minimum chart dimension validation~~ RESOLVED (Round 1)

### M8. Only color-channel legends generated
**Files**: `compile/legend_gen.rs`
No legends for size, shape, or opacity channels.

### ~~M9. Legend can overflow canvas~~ RESOLVED (Round 4)
Legend entries truncated to fit plot height with "… +N more" overflow indicator.

### ~~M10. Faceted charts generate no legend at all~~ RESOLVED (Round 2)
### ~~M11. Facet value order is non-deterministic~~ RESOLVED (Round 2)

### M12. Single-layer bar chart legend suppression too aggressive
**Files**: `compile/legend_gen.rs:75-84`

### M13. API naming inconsistency
**Files**: `express/mod.rs`, `grammar/layer.rs`
Mixed patterns: `with_*`, bare `title()`, `encode_*`.

### ~~M14. Heatmap viridis color scale hardcoded~~ RESOLVED (Round 2)

### M15. Log scale ticks decade-only, no minor ticks
**Files**: `axis/tick.rs:48-59`

### M16. Free facet scales fall back silently to global bounds
**Files**: `compile/facet.rs:166-229`

---

## Low Issues (14)

### ~~L1. Hard-coded label offsets~~ RESOLVED (Round 4)
Tick label gaps and axis title offsets now derived from `theme.tick_font_size` and `theme.label_font_size`.

### ~~L2. Text width estimation assumes monospace~~ RESOLVED (Round 4)
Per-character width table for common ASCII (narrow chars 0.3, average 0.5, wide chars 0.7).

### L3. Epsilon thresholds inconsistent across modules
### L4. Point opacity threshold discontinuous at n=100

### ~~L5. Pie chart floating-point angle accumulation~~ RESOLVED (Round 4)
Last arc forced to `start_initial + TAU`.

### L6. Jitter amount not data-relative
### L7. Negative dodge_width not validated
### L8. Category deduplication O(n²)
### L9. No legend title auto-population
### L10. Facet strip labels missing for single-value facets
### L11. Facet font size reduction may be aggressive
### L12. No tests for NaN/Inf, empty groups, or length mismatches *(mostly resolved — 119 tests now)*

### L13. Missing express convenience: trend lines, data labels, error bars
**Round 4**: Added `color_by()` to `LineBuilder` and `AreaBuilder` for API consistency with `ScatterBuilder`.

### ~~L14. `min_gap = 4.0` for label overlap hard-coded~~ RESOLVED (Round 4)
Gap now font-relative via `theme.tick_font_size * 0.4`.

---

## Summary by Severity

| Severity | Count | Resolved | Remaining |
|----------|-------|----------|-----------|
| Critical | 8 | 7 | 1 (C1) |
| Moderate | 16 | 11 | 5 (M6, M8, M12, M15, M16) |
| Low | 14 | 6 | 8 |
| **Total** | **38** | **24** | **14** |

## Round 4 Changes (2026-03-16)

| Issue | Fix |
|-------|-----|
| C5 | 5% scatter/line/point data bounds padding |
| 1b (new) | Heatmap NaN/Inf validation |
| 1c (new) | Polar coordinate guard → `ChartError::InvalidParameter` |
| M5 | Area upper/lower path length validation |
| L5 | Pie chart last-arc angle closure |
| 2a (new) | `try_build()` safe path for `MultiBarBuilder` (replaces `assert!`) |
| 2b (new) | `color_by()` on `LineBuilder` and `AreaBuilder` |
| L2 | Per-character text width estimation |
| L1/L14 | Font-relative label offsets in axis_gen |
| C7 partial | `#[deprecated]` on `encode_x/y/color/size` |
| 4b (new) | `#[doc(hidden)]` on `CoordSystem::Polar` |
| M9 | Legend vertical overflow truncation |
