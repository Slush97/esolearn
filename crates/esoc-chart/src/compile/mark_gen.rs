// SPDX-License-Identifier: MIT OR Apache-2.0
//! Mark generation: turns ResolvedLayer data into scene marks.

use crate::compile::stat_boxplot::BoxPlotSummary;
use crate::compile::stat_transform::ResolvedLayer;
use crate::error::ChartError;
use crate::grammar::layer::MarkType;
use crate::new_theme::NewTheme;
use esoc_scene::bounds::{BoundingBox, DataBounds};
use esoc_scene::mark::{ArcMark, AreaMark, BatchAttr, Interpolation, LineMark, Mark, MarkBatch, RuleMark, TextMark, TextAnchor};
use esoc_scene::node::{Node, NodeId};
use esoc_scene::scale::Scale;
use esoc_scene::style::{FillStyle, FontStyle, MarkerShape, StrokeStyle};
use esoc_scene::SceneGraph;

/// Generate marks for a single resolved layer.
#[allow(clippy::too_many_arguments)]
pub fn generate_layer_marks(
    scene: &mut SceneGraph,
    plot_id: NodeId,
    layer: &ResolvedLayer,
    data_bounds: &DataBounds,
    plot_w: f32,
    plot_h: f32,
    theme: &NewTheme,
) -> Result<(), ChartError> {
    generate_layer_marks_inner(scene, plot_id, layer, data_bounds, plot_w, plot_h, theme, false)
}

/// Generate marks for a single resolved layer, with optional flip flag.
#[allow(clippy::too_many_arguments)]
pub fn generate_layer_marks_flipped(
    scene: &mut SceneGraph,
    plot_id: NodeId,
    layer: &ResolvedLayer,
    data_bounds: &DataBounds,
    plot_w: f32,
    plot_h: f32,
    theme: &NewTheme,
    is_flipped: bool,
) -> Result<(), ChartError> {
    generate_layer_marks_inner(scene, plot_id, layer, data_bounds, plot_w, plot_h, theme, is_flipped)
}

#[allow(clippy::too_many_arguments)]
fn generate_layer_marks_inner(
    scene: &mut SceneGraph,
    plot_id: NodeId,
    layer: &ResolvedLayer,
    data_bounds: &DataBounds,
    plot_w: f32,
    plot_h: f32,
    theme: &NewTheme,
    is_flipped: bool,
) -> Result<(), ChartError> {
    let x_scale = Scale::Linear {
        domain: (data_bounds.x_min, data_bounds.x_max),
        range: (0.0, plot_w),
    };
    let y_scale = Scale::Linear {
        domain: (data_bounds.y_min, data_bounds.y_max),
        range: (plot_h, 0.0),
    };

    let series_color = theme.palette.get(layer.layer_idx);

    // Special case: boxplot summaries
    if let Some(summaries) = &layer.boxplot {
        return generate_boxplot(scene, plot_id, summaries, &x_scale, &y_scale, plot_w, theme);
    }

    match layer.mark {
        MarkType::Point => {
            generate_points(scene, plot_id, layer, &x_scale, &y_scale, series_color, theme)?;
        }
        MarkType::Line => {
            generate_line(scene, plot_id, layer, &x_scale, &y_scale, series_color, theme)?;
        }
        MarkType::Bar => {
            generate_bars(
                scene,
                plot_id,
                layer,
                &x_scale,
                &y_scale,
                series_color,
                plot_h,
                theme,
                is_flipped,
            )?;
        }
        MarkType::Area => {
            generate_area(scene, plot_id, layer, &x_scale, &y_scale, series_color, theme)?;
        }
        MarkType::Arc => {
            generate_arcs(scene, plot_id, layer, plot_w, plot_h, theme)?;
        }
        MarkType::Heatmap => {
            generate_heatmap(scene, plot_id, layer, plot_w, plot_h, theme)?;
        }
        _ => {
            // Other mark types (Text, Rule) to be implemented in later sub-phases
        }
    }
    Ok(())
}

fn generate_points(
    scene: &mut SceneGraph,
    plot_id: NodeId,
    layer: &ResolvedLayer,
    x_scale: &Scale,
    y_scale: &Scale,
    color: esoc_color::Color,
    theme: &NewTheme,
) -> Result<(), ChartError> {
    if layer.x_data.is_empty() || layer.y_data.is_empty() {
        return Ok(());
    }

    // Convert area (px²) to diameter for rendering
    let diameter = (theme.point_size / std::f32::consts::PI).sqrt() * 2.0;

    // Density-adaptive opacity: reduce opacity for dense scatter plots
    let n = layer.x_data.len();
    let opacity = if n > 100 {
        (100.0 / (n as f32).sqrt()).clamp(0.1, 0.7)
    } else {
        0.7
    };

    if let Some(cats) = &layer.categories {
        let unique_cats: Vec<String> = {
            let mut seen = Vec::new();
            for c in cats {
                if !seen.contains(c) {
                    seen.push(c.clone());
                }
            }
            seen
        };

        let positions: Vec<[f32; 2]> = layer
            .x_data
            .iter()
            .zip(layer.y_data.iter())
            .map(|(&x, &y)| [x_scale.map(x), y_scale.map(y)])
            .collect();

        let fills: Vec<FillStyle> = cats
            .iter()
            .map(|c| {
                let idx = unique_cats.iter().position(|u| u == c).unwrap_or(0);
                FillStyle::Solid(theme.palette.get(idx).with_alpha(opacity))
            })
            .collect();

        let batch = MarkBatch::points(
            positions,
            BatchAttr::Uniform(diameter),
            BatchAttr::Varying(fills),
            MarkerShape::Circle,
            BatchAttr::Uniform(StrokeStyle {
                width: 0.0,
                ..Default::default()
            }),
        )
        .expect("batch validation failed");

        let node = Node::with_batch(batch).z_order(2);
        scene.insert_child(plot_id, node);
    } else {
        let positions: Vec<[f32; 2]> = layer
            .x_data
            .iter()
            .zip(layer.y_data.iter())
            .map(|(&x, &y)| [x_scale.map(x), y_scale.map(y)])
            .collect();

        let batch = MarkBatch::points(
            positions,
            BatchAttr::Uniform(diameter),
            BatchAttr::Uniform(FillStyle::Solid(color.with_alpha(opacity))),
            MarkerShape::Circle,
            BatchAttr::Uniform(StrokeStyle {
                width: 0.0,
                ..Default::default()
            }),
        )
        .expect("batch validation failed");

        let node = Node::with_batch(batch).z_order(2);
        scene.insert_child(plot_id, node);
    }
    Ok(())
}

fn generate_line(
    scene: &mut SceneGraph,
    plot_id: NodeId,
    layer: &ResolvedLayer,
    x_scale: &Scale,
    y_scale: &Scale,
    color: esoc_color::Color,
    theme: &NewTheme,
) -> Result<(), ChartError> {
    if layer.x_data.is_empty() || layer.y_data.is_empty() {
        return Ok(());
    }

    let points: Vec<[f32; 2]> = layer
        .x_data
        .iter()
        .zip(layer.y_data.iter())
        .map(|(&x, &y)| [x_scale.map(x), y_scale.map(y)])
        .collect();

    let mark = Mark::Line(LineMark {
        points,
        stroke: StrokeStyle::solid(color, theme.line_width),
        interpolation: Interpolation::Linear,
    });

    let node = Node::with_mark(mark).z_order(2);
    scene.insert_child(plot_id, node);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn generate_bars(
    scene: &mut SceneGraph,
    plot_id: NodeId,
    layer: &ResolvedLayer,
    x_scale: &Scale,
    y_scale: &Scale,
    color: esoc_color::Color,
    _plot_h: f32,
    theme: &NewTheme,
    is_flipped: bool,
) -> Result<(), ChartError> {
    let n = layer.x_data.len().min(layer.y_data.len());
    if n == 0 {
        return Ok(());
    }

    let mut rects = Vec::with_capacity(n);

    if is_flipped {
        // Horizontal bars: y_data has the category positions, x_data has the values.
        // After flip, x_data=values, y_data=positions were swapped in compile_chart.
        // Draw bars horizontally: position along y-axis, width along x-axis.
        let gap_factor = 0.8;
        let bar_height = if n > 1 {
            let y0 = y_scale.map(layer.y_data[0]);
            let y1 = y_scale.map(layer.y_data[1]);
            (y1 - y0).abs() * gap_factor
        } else {
            20.0
        };

        for i in 0..n {
            let cy = y_scale.map(layer.y_data[i]);
            let x_val = x_scale.map(layer.x_data[i]);
            let x_base = x_scale.map(0.0);

            let left = x_val.min(x_base);
            let w = (x_val - x_base).abs();
            let top = cy - bar_height * 0.5;
            rects.push(BoundingBox::new(left, top, w, bar_height));
        }
    } else {
        // Vertical bars: x_data has positions, y_data has values.
        let bar_width = if let Some(dw) = layer.dodge_width {
            let x0_px = x_scale.map(0.0);
            let x1_px = x_scale.map(dw);
            (x1_px - x0_px).abs()
        } else {
            let gap_factor = if layer.is_binned { 0.98 } else { 0.8 };
            if n > 1 {
                let x0 = x_scale.map(layer.x_data[0]);
                let x1 = x_scale.map(layer.x_data[1]);
                (x1 - x0).abs() * gap_factor
            } else {
                20.0
            }
        };

        for i in 0..n {
            let cx = x_scale.map(layer.x_data[i]);
            let y = y_scale.map(layer.y_data[i]);
            let x = cx - bar_width * 0.5;

            let base_y = if let Some(baseline) = &layer.y_baseline {
                y_scale.map(baseline.get(i).copied().unwrap_or(0.0))
            } else {
                y_scale.map(0.0)
            };

            let h = (base_y - y).abs();
            let top = y.min(base_y);
            rects.push(BoundingBox::new(x, top, bar_width, h));
        }
    }

    // Per-category coloring: each bar gets a color based on its category.
    // Skip per-category coloring for dodged (grouped) bars — those use
    // the layer's series color so each group (e.g. Precision/Recall/F1)
    // gets a distinct color instead of coloring by category.
    let fills = if layer.dodge_width.is_none() {
        if let Some(cats) = &layer.categories {
            let unique_cats: Vec<String> = {
                let mut seen = Vec::new();
                for c in cats {
                    if !seen.contains(c) {
                        seen.push(c.clone());
                    }
                }
                seen
            };
            BatchAttr::Varying(
                cats.iter()
                    .take(n)
                    .map(|c| {
                        let idx = unique_cats.iter().position(|u| u == c).unwrap_or(0);
                        FillStyle::Solid(theme.palette.get(idx))
                    })
                    .collect(),
            )
        } else {
            BatchAttr::Uniform(FillStyle::Solid(color))
        }
    } else {
        BatchAttr::Uniform(FillStyle::Solid(color))
    };

    // Histogram bars get a thin stroke to clearly separate bins
    let stroke = if layer.is_binned {
        StrokeStyle::solid(esoc_color::Color::WHITE, 0.5)
    } else {
        StrokeStyle {
            width: 0.0,
            ..Default::default()
        }
    };

    let batch = MarkBatch::rects(rects, fills, BatchAttr::Uniform(stroke), 0.0)
        .expect("batch validation failed");

    let node = Node::with_batch(batch).z_order(2);
    scene.insert_child(plot_id, node);
    Ok(())
}

/// Generate an area mark (filled region between data and baseline).
fn generate_area(
    scene: &mut SceneGraph,
    plot_id: NodeId,
    layer: &ResolvedLayer,
    x_scale: &Scale,
    y_scale: &Scale,
    color: esoc_color::Color,
    _theme: &NewTheme,
) -> Result<(), ChartError> {
    if layer.x_data.is_empty() || layer.y_data.is_empty() {
        return Ok(());
    }

    let upper: Vec<[f32; 2]> = layer
        .x_data
        .iter()
        .zip(layer.y_data.iter())
        .map(|(&x, &y)| [x_scale.map(x), y_scale.map(y)])
        .collect();

    let lower: Vec<[f32; 2]> = if let Some(baseline) = &layer.y_baseline {
        layer
            .x_data
            .iter()
            .zip(baseline.iter())
            .map(|(&x, &y)| [x_scale.map(x), y_scale.map(y)])
            .collect()
    } else {
        // Default baseline at y=0
        layer
            .x_data
            .iter()
            .map(|&x| [x_scale.map(x), y_scale.map(0.0)])
            .collect()
    };

    let mark = Mark::Area(AreaMark {
        upper,
        lower,
        fill: FillStyle::Solid(color.with_alpha(0.3)),
        stroke: StrokeStyle::solid(color, 1.5),
    });

    let node = Node::with_mark(mark).z_order(1);
    scene.insert_child(plot_id, node);
    Ok(())
}

/// Generate arc marks for pie/donut charts.
fn generate_arcs(
    scene: &mut SceneGraph,
    plot_id: NodeId,
    layer: &ResolvedLayer,
    plot_w: f32,
    plot_h: f32,
    theme: &NewTheme,
) -> Result<(), ChartError> {
    if layer.y_data.is_empty() {
        return Ok(());
    }

    let total: f64 = layer.y_data.iter().sum();
    if total <= 0.0 {
        return Ok(());
    }

    let center = [plot_w * 0.5, plot_h * 0.5];
    let outer_radius = plot_w.min(plot_h) * 0.4;
    let inner_radius = outer_radius * layer.inner_radius_fraction;

    let mut start_angle = -std::f32::consts::FRAC_PI_2; // Start from top (12 o'clock)

    for (i, &value) in layer.y_data.iter().enumerate() {
        let fraction = value / total;
        let sweep = (fraction * std::f64::consts::TAU) as f32;
        let end_angle = start_angle + sweep;

        let color = theme.palette.get(i);

        let arc = Node::with_mark(Mark::Arc(ArcMark {
            center,
            inner_radius,
            outer_radius,
            start_angle,
            end_angle,
            fill: FillStyle::Solid(color),
            stroke: StrokeStyle::solid(esoc_color::Color::WHITE, 1.5),
        }))
        .z_order(2);
        scene.insert_child(plot_id, arc);

        start_angle = end_angle;
    }

    Ok(())
}

/// Generate box plot marks: box (Q1→Q3), median line, whiskers, outlier points.
fn generate_boxplot(
    scene: &mut SceneGraph,
    plot_id: NodeId,
    summaries: &[BoxPlotSummary],
    x_scale: &Scale,
    y_scale: &Scale,
    plot_w: f32,
    theme: &NewTheme,
) -> Result<(), ChartError> {
    let n = summaries.len();
    if n == 0 {
        return Ok(());
    }

    // Compute box width based on available space
    let box_width = (plot_w / n as f32) * 0.6;

    for (i, s) in summaries.iter().enumerate() {
        let cx = x_scale.map(i as f64);
        let half_w = box_width * 0.5;

        let y_q1 = y_scale.map(s.q1);
        let y_q3 = y_scale.map(s.q3);
        let y_med = y_scale.map(s.median);
        let y_lo = y_scale.map(s.whisker_lo);
        let y_hi = y_scale.map(s.whisker_hi);

        let box_top = y_q3.min(y_q1);
        let box_h = (y_q1 - y_q3).abs();

        // Q1→Q3 box
        let color = theme.palette.get(i);
        let box_rect = Node::with_mark(Mark::Rect(esoc_scene::mark::RectMark {
            bounds: BoundingBox::new(cx - half_w, box_top, box_width, box_h),
            fill: FillStyle::Solid(color.with_alpha(0.6)),
            stroke: StrokeStyle::solid(color, 1.5),
            corner_radius: 0.0,
        }))
        .z_order(2);
        scene.insert_child(plot_id, box_rect);

        let rule_stroke = StrokeStyle::solid(theme.foreground, 1.5);
        let whisker_stroke = StrokeStyle::solid(theme.foreground, 1.0);

        // Median line
        let median_rule = Node::with_mark(Mark::Rule(RuleMark {
            segments: vec![([cx - half_w, y_med], [cx + half_w, y_med])],
            stroke: rule_stroke,
        }))
        .z_order(3);
        scene.insert_child(plot_id, median_rule);

        // Whiskers (vertical lines from box to whisker ends)
        let whisker_cap = half_w * 0.5;
        let whiskers = Node::with_mark(Mark::Rule(RuleMark {
            segments: vec![
                // Lower whisker
                ([cx, y_q1.max(y_q3)], [cx, y_lo]),
                ([cx - whisker_cap, y_lo], [cx + whisker_cap, y_lo]),
                // Upper whisker
                ([cx, y_q1.min(y_q3)], [cx, y_hi]),
                ([cx - whisker_cap, y_hi], [cx + whisker_cap, y_hi]),
            ],
            stroke: whisker_stroke,
        }))
        .z_order(2);
        scene.insert_child(plot_id, whiskers);

        // Outlier points
        if !s.outliers.is_empty() {
            let positions: Vec<[f32; 2]> = s
                .outliers
                .iter()
                .map(|&o| [cx, y_scale.map(o)])
                .collect();
            let batch = MarkBatch::points(
                positions,
                BatchAttr::Uniform((theme.point_size / std::f32::consts::PI).sqrt() * 2.0 * 0.6),
                BatchAttr::Uniform(FillStyle::Solid(theme.foreground)),
                MarkerShape::Circle,
                BatchAttr::Uniform(StrokeStyle {
                    width: 0.0,
                    ..Default::default()
                }),
            )
            .expect("batch validation failed");
            let node = Node::with_batch(batch).z_order(3);
            scene.insert_child(plot_id, node);
        }
    }

    Ok(())
}

/// Generate heatmap marks: colored rectangles for each cell, optional text annotations.
fn generate_heatmap(
    scene: &mut SceneGraph,
    plot_id: NodeId,
    layer: &ResolvedLayer,
    plot_w: f32,
    plot_h: f32,
    theme: &NewTheme,
) -> Result<(), ChartError> {
    let data = match &layer.heatmap_data {
        Some(d) if !d.is_empty() => d,
        _ => return Ok(()),
    };

    let rows = data.len();
    let cols = data.first().map_or(0, |r| r.len());
    if cols == 0 {
        return Ok(());
    }

    // Compute value range
    let mut v_min = f64::INFINITY;
    let mut v_max = f64::NEG_INFINITY;
    for row in data {
        for &v in row {
            if v < v_min { v_min = v; }
            if v > v_max { v_max = v; }
        }
    }
    let v_range = if (v_max - v_min).abs() < 1e-12 { 1.0 } else { v_max - v_min };

    let color_scale = esoc_color::ColorScale::viridis();

    let cell_w = plot_w / cols as f32;
    let cell_h = plot_h / rows as f32;

    let mut rects = Vec::with_capacity(rows * cols);
    let mut fills = Vec::with_capacity(rows * cols);

    for (r, row) in data.iter().enumerate() {
        for (c, &val) in row.iter().enumerate() {
            let t = ((val - v_min) / v_range) as f32;
            let color = color_scale.map(t);

            let x = c as f32 * cell_w;
            let y = r as f32 * cell_h; // row 0 at top
            rects.push(BoundingBox::new(x, y, cell_w, cell_h));
            fills.push(FillStyle::Solid(color));
        }
    }

    let batch = MarkBatch::rects(
        rects,
        BatchAttr::Varying(fills),
        BatchAttr::Uniform(StrokeStyle {
            width: 1.0,
            ..Default::default()
        }),
        0.0,
    )
    .expect("batch validation failed");

    let node = Node::with_batch(batch).z_order(2);
    scene.insert_child(plot_id, node);

    // Cell annotations
    if layer.annotate_cells {
        let font_size = (cell_h * 0.35).min(cell_w * 0.35).max(8.0);
        for (r, row) in data.iter().enumerate() {
            for (c, &val) in row.iter().enumerate() {
                let t = ((val - v_min) / v_range) as f32;
                let text_color = if t < 0.5 {
                    esoc_color::Color::WHITE
                } else {
                    esoc_color::Color::BLACK
                };

                let cx = (c as f32 + 0.5) * cell_w;
                let cy = (r as f32 + 0.5) * cell_h;

                // Format: integer if whole number, else 1 decimal
                let text = if (val - val.round()).abs() < 1e-9 {
                    format!("{}", val as i64)
                } else {
                    format!("{val:.1}")
                };

                let text_node = Node::with_mark(Mark::Text(TextMark {
                    position: [cx, cy],
                    text,
                    font: FontStyle {
                        family: theme.font_family.clone(),
                        size: font_size,
                        weight: 400,
                        italic: false,
                    },
                    fill: FillStyle::Solid(text_color),
                    angle: 0.0,
                    anchor: TextAnchor::Middle,
                }))
                .z_order(3);
                scene.insert_child(plot_id, text_node);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::stat_transform::ResolvedLayer;
    use crate::grammar::position::Position;
    use esoc_scene::bounds::DataBounds;

    fn make_resolved(mark: MarkType, x: Vec<f64>, y: Vec<f64>, idx: usize) -> ResolvedLayer {
        ResolvedLayer {
            mark,
            x_data: x,
            y_data: y,
            categories: None,
            y_baseline: None,
            boxplot: None,
            inner_radius_fraction: 0.0,
            position: Position::default(),
            is_binned: false,
            facet_values: None,
            layer_idx: idx,
            heatmap_data: None,
            row_labels: None,
            col_labels: None,
            annotate_cells: false,
            label: None,
            dodge_width: None,
        }
    }

    fn count_non_container(scene: &SceneGraph) -> usize {
        scene
            .iter()
            .filter(|(_, n)| !matches!(n.content, esoc_scene::node::NodeContent::Container))
            .count()
    }

    #[test]
    fn point_batch_generated() {
        let mut scene = SceneGraph::with_root();
        let root = scene.root().unwrap();
        let plot_id = scene.insert_child(root, Node::container());
        let layer = make_resolved(MarkType::Point, vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0], 0);
        let bounds = DataBounds::new(0.0, 2.0, 0.0, 5.0);
        let theme = NewTheme::default();
        generate_layer_marks(&mut scene, plot_id, &layer, &bounds, 400.0, 300.0, &theme).unwrap();
        assert!(count_non_container(&scene) >= 1);
    }

    #[test]
    fn bar_rects_generated() {
        let mut scene = SceneGraph::with_root();
        let root = scene.root().unwrap();
        let plot_id = scene.insert_child(root, Node::container());
        let layer = make_resolved(MarkType::Bar, vec![0.0, 1.0, 2.0], vec![10.0, 20.0, 30.0], 0);
        let bounds = DataBounds::new(0.0, 2.0, 0.0, 30.0);
        let theme = NewTheme::default();
        generate_layer_marks(&mut scene, plot_id, &layer, &bounds, 400.0, 300.0, &theme).unwrap();
        assert!(count_non_container(&scene) >= 1);
    }

    #[test]
    fn line_mark_generated() {
        let mut scene = SceneGraph::with_root();
        let root = scene.root().unwrap();
        let plot_id = scene.insert_child(root, Node::container());
        let layer = make_resolved(MarkType::Line, vec![0.0, 1.0, 2.0], vec![1.0, 2.0, 3.0], 0);
        let bounds = DataBounds::new(0.0, 2.0, 0.0, 3.0);
        let theme = NewTheme::default();
        generate_layer_marks(&mut scene, plot_id, &layer, &bounds, 400.0, 300.0, &theme).unwrap();
        assert!(count_non_container(&scene) >= 1);
    }

    #[test]
    fn arc_marks_generated() {
        let mut scene = SceneGraph::with_root();
        let root = scene.root().unwrap();
        let plot_id = scene.insert_child(root, Node::container());
        let mut layer = make_resolved(MarkType::Arc, vec![], vec![30.0, 50.0, 20.0], 0);
        layer.inner_radius_fraction = 0.0;
        let bounds = DataBounds::new(0.0, 1.0, 0.0, 1.0);
        let theme = NewTheme::default();
        generate_layer_marks(&mut scene, plot_id, &layer, &bounds, 400.0, 300.0, &theme).unwrap();
        // Should generate 3 arc marks
        assert_eq!(count_non_container(&scene), 3);
    }

    #[test]
    fn heatmap_rects_generated() {
        let mut scene = SceneGraph::with_root();
        let root = scene.root().unwrap();
        let plot_id = scene.insert_child(root, Node::container());
        let mut layer = make_resolved(MarkType::Heatmap, vec![], vec![], 0);
        layer.heatmap_data = Some(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        let bounds = DataBounds::new(-0.5, 2.5, -0.5, 1.5);
        let theme = NewTheme::default();
        generate_layer_marks(&mut scene, plot_id, &layer, &bounds, 300.0, 200.0, &theme).unwrap();
        assert!(count_non_container(&scene) >= 1);
    }

    #[test]
    fn heatmap_with_annotations() {
        let mut scene = SceneGraph::with_root();
        let root = scene.root().unwrap();
        let plot_id = scene.insert_child(root, Node::container());
        let mut layer = make_resolved(MarkType::Heatmap, vec![], vec![], 0);
        layer.heatmap_data = Some(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        layer.annotate_cells = true;
        let bounds = DataBounds::new(-0.5, 1.5, -0.5, 1.5);
        let theme = NewTheme::default();
        generate_layer_marks(&mut scene, plot_id, &layer, &bounds, 200.0, 200.0, &theme).unwrap();
        // Should have rects batch + 4 text marks
        assert!(count_non_container(&scene) >= 5);
    }

    #[test]
    fn area_mark_generated() {
        let mut scene = SceneGraph::with_root();
        let root = scene.root().unwrap();
        let plot_id = scene.insert_child(root, Node::container());
        let layer = make_resolved(MarkType::Area, vec![0.0, 1.0, 2.0], vec![1.0, 3.0, 2.0], 0);
        let bounds = DataBounds::new(0.0, 2.0, 0.0, 3.0);
        let theme = NewTheme::default();
        generate_layer_marks(&mut scene, plot_id, &layer, &bounds, 400.0, 300.0, &theme).unwrap();
        assert!(count_non_container(&scene) >= 1);
    }
}
