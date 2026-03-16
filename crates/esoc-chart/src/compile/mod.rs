// SPDX-License-Identifier: MIT OR Apache-2.0
//! Chart → SceneGraph compiler.

pub(crate) mod annotation;
mod axis_gen;
pub(crate) mod facet;
mod layout;
pub(crate) mod legend_gen;
mod mark_gen;
pub(crate) mod position;
pub(crate) mod stat_aggregate;
pub(crate) mod stat_bin;
pub(crate) mod stat_boxplot;
pub(crate) mod stat_smooth;
pub(crate) mod stat_transform;

use crate::error::{ChartError, Result};
use crate::grammar::chart::Chart;
use crate::grammar::facet::Facet;
use esoc_scene::bounds::DataBounds;
use esoc_scene::node::Node;
use esoc_scene::transform::Affine2D;
use esoc_scene::SceneGraph;
use stat_transform::ResolvedLayer;

/// Plot area margins.
pub(crate) struct Margins {
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub left: f32,
}

/// Compile a Chart definition into a SceneGraph.
pub fn compile_chart(chart: &Chart) -> Result<SceneGraph> {
    if chart.layers.is_empty() {
        return Err(ChartError::EmptyData);
    }

    // Validate dimensions before stat resolution
    for (i, layer) in chart.layers.iter().enumerate() {
        if let (Some(x), Some(y)) = (&layer.x_data, &layer.y_data) {
            if x.len() != y.len() {
                return Err(ChartError::DimensionMismatch {
                    layer: i,
                    x_len: x.len(),
                    y_len: y.len(),
                });
            }
        }
    }

    // Resolve stat transforms: Layer → ResolvedLayer
    let mut resolved: Vec<ResolvedLayer> = chart
        .layers
        .iter()
        .enumerate()
        .map(|(i, layer)| stat_transform::resolve_layer(layer, i))
        .collect::<Result<Vec<_>>>()?;

    // Apply position adjustments (stack, dodge, fill, jitter)
    position::apply_positions(&mut resolved)?;

    let mut scene = SceneGraph::with_root();
    let root = scene.root().unwrap();

    let theme = &chart.theme;

    // Compute global data bounds (before margins, since margins depend on tick labels)
    let mut data_bounds = compute_resolved_data_bounds(&resolved)?;

    // Nice the bounds so ticks align exactly with domain edges.
    // Build preliminary scales, nice them, then extract the niced domain back.
    {
        use esoc_scene::scale::Scale;
        let target_x = layout::target_tick_count(chart.width, 80.0);
        let target_y = layout::target_tick_count(chart.height, 40.0);
        let x_niced = Scale::Linear {
            domain: (data_bounds.x_min, data_bounds.x_max),
            range: (0.0, chart.width),
        }
        .nice(target_x);
        let y_niced = Scale::Linear {
            domain: (data_bounds.y_min, data_bounds.y_max),
            range: (chart.height, 0.0),
        }
        .nice(target_y);
        if let Scale::Linear { domain, .. } = &x_niced {
            data_bounds.x_min = domain.0;
            data_bounds.x_max = domain.1;
        }
        if let Scale::Linear { domain, .. } = &y_niced {
            data_bounds.y_min = domain.0;
            data_bounds.y_max = domain.1;
        }
    }

    // Bar/area charts: re-clamp y_min to 0 after nicing when the actual data minimum
    // is at or near zero.  nice() can round y_min to something like -0.05 even when
    // the smallest data value is -0.0004, creating a large gap that makes bars float
    // above the x-axis baseline.
    let has_bar_or_area = resolved.iter().any(|l| {
        matches!(
            l.mark,
            crate::grammar::layer::MarkType::Bar | crate::grammar::layer::MarkType::Area
        )
    });
    if has_bar_or_area && data_bounds.y_min < 0.0 {
        let actual_min: f64 = resolved
            .iter()
            .flat_map(|l| l.y_data.iter())
            .copied()
            .fold(f64::INFINITY, f64::min);
        let range = data_bounds.y_max - data_bounds.y_min;
        // Clamp if actual data min is non-negative, or if it's negligibly small
        // relative to the full range (< 2% of range).
        if actual_min >= 0.0 || (range > 0.0 && actual_min.abs() < 0.02 * range) {
            data_bounds.y_min = 0.0;
        }
    }

    // For Flipped coordinate system, swap x/y bounds
    let is_flipped = matches!(chart.coord, crate::grammar::coord::CoordSystem::Flipped);
    if is_flipped {
        data_bounds = esoc_scene::bounds::DataBounds::new(
            data_bounds.y_min,
            data_bounds.y_max,
            data_bounds.x_min,
            data_bounds.x_max,
        );
        for layer in &mut resolved {
            std::mem::swap(&mut layer.x_data, &mut layer.y_data);
        }
    }

    // Compute margins for axes/title (needs data_bounds for tick label measurement)
    let margins = layout::compute_margins(chart, &data_bounds);

    let plot_x = margins.left;
    let plot_y = margins.top;
    let plot_w = chart.width - margins.left - margins.right;
    let plot_h = chart.height - margins.top - margins.bottom;

    // Background
    let bg_node = Node::with_mark(esoc_scene::mark::Mark::Rect(
        esoc_scene::mark::RectMark {
            bounds: esoc_scene::bounds::BoundingBox::new(0.0, 0.0, chart.width, chart.height),
            fill: esoc_scene::style::FillStyle::Solid(theme.background),
            stroke: esoc_scene::style::StrokeStyle {
                width: 0.0,
                ..Default::default()
            },
            corner_radius: 0.0,
        },
    ))
    .z_order(-10);
    scene.insert_child(root, bg_node);

    // Check if faceting is needed
    let has_facets = !matches!(chart.facet, Facet::None)
        && resolved.iter().any(|l| l.facet_values.is_some());

    if has_facets {
        compile_faceted(
            chart, &mut scene, root, &resolved, &data_bounds, plot_x, plot_y, plot_w, plot_h,
        )?;
    } else {
        compile_single_panel(
            chart, &mut scene, root, &resolved, &data_bounds, plot_x, plot_y, plot_w, plot_h,
            is_flipped,
        )?;
    }

    // Title
    if let Some(title) = &chart.title {
        let title_node = Node::with_mark(esoc_scene::mark::Mark::Text(
            esoc_scene::mark::TextMark {
                position: [chart.width * 0.5, margins.top * 0.6],
                text: title.clone(),
                font: esoc_scene::style::FontStyle {
                    family: theme.font_family.clone(),
                    size: theme.title_font_size,
                    weight: 700,
                    italic: false,
                },
                fill: esoc_scene::style::FillStyle::Solid(theme.foreground),
                angle: 0.0,
                anchor: esoc_scene::mark::TextAnchor::Middle,
            },
        ))
        .z_order(10);
        scene.insert_child(root, title_node);
    }

    // Subtitle
    if let Some(subtitle) = &chart.subtitle {
        annotation::generate_subtitle(
            &mut scene, root, subtitle, chart.width, theme.title_font_size, theme,
        );
    }

    // Caption
    if let Some(caption) = &chart.caption {
        annotation::generate_caption(
            &mut scene, root, caption, chart.width, chart.height, theme,
        );
    }

    Ok(scene)
}

/// Generate axis labels and category tick labels for heatmap charts.
#[allow(clippy::too_many_arguments)]
fn generate_heatmap_axes(
    chart: &Chart,
    scene: &mut SceneGraph,
    root: esoc_scene::node::NodeId,
    _plot_id: esoc_scene::node::NodeId,
    resolved: &[ResolvedLayer],
    plot_x: f32,
    plot_y: f32,
    plot_w: f32,
    plot_h: f32,
) {
    use esoc_scene::mark::{Mark, TextMark, TextAnchor};
    use esoc_scene::style::{FillStyle, FontStyle};

    let theme = &chart.theme;
    let layer = resolved.first();

    // Column labels (below plot, at each column center)
    if let Some(col_labels) = layer.and_then(|l| l.col_labels.as_ref()) {
        let cols = col_labels.len();
        let cell_w = plot_w / cols as f32;
        for (c, label) in col_labels.iter().enumerate() {
            let x = plot_x + (c as f32 + 0.5) * cell_w;
            let y = plot_y + plot_h + theme.tick_font_size + 5.0;
            let text = Node::with_mark(Mark::Text(TextMark {
                position: [x, y],
                text: label.clone(),
                font: FontStyle {
                    family: theme.font_family.clone(),
                    size: theme.tick_font_size,
                    weight: 400,
                    italic: false,
                },
                fill: FillStyle::Solid(theme.foreground),
                angle: 0.0,
                anchor: TextAnchor::Middle,
            }))
            .z_order(5);
            scene.insert_child(root, text);
        }
    }

    // Row labels (left of plot, at each row center)
    if let Some(row_labels) = layer.and_then(|l| l.row_labels.as_ref()) {
        let rows = row_labels.len();
        let cell_h = plot_h / rows as f32;
        for (r, label) in row_labels.iter().enumerate() {
            let x = plot_x - 5.0;
            let y = plot_y + (r as f32 + 0.5) * cell_h;
            let text = Node::with_mark(Mark::Text(TextMark {
                position: [x, y],
                text: label.clone(),
                font: FontStyle {
                    family: theme.font_family.clone(),
                    size: theme.tick_font_size,
                    weight: 400,
                    italic: false,
                },
                fill: FillStyle::Solid(theme.foreground),
                angle: 0.0,
                anchor: TextAnchor::End,
            }))
            .z_order(5);
            scene.insert_child(root, text);
        }
    }

    // X axis label
    if let Some(label) = &chart.x_label {
        let text = Node::with_mark(Mark::Text(TextMark {
            position: [
                plot_x + plot_w * 0.5,
                plot_y + plot_h + theme.tick_font_size + theme.label_font_size + 15.0,
            ],
            text: label.clone(),
            font: FontStyle {
                family: theme.font_family.clone(),
                size: theme.label_font_size,
                weight: 400,
                italic: false,
            },
            fill: FillStyle::Solid(theme.foreground),
            angle: 0.0,
            anchor: TextAnchor::Middle,
        }))
        .z_order(5);
        scene.insert_child(root, text);
    }

    // Y axis label (rotated)
    if let Some(label) = &chart.y_label {
        let text = Node::with_mark(Mark::Text(TextMark {
            position: [
                plot_x - theme.tick_font_size * 3.5,
                plot_y + plot_h * 0.5,
            ],
            text: label.clone(),
            font: FontStyle {
                family: theme.font_family.clone(),
                size: theme.label_font_size,
                weight: 400,
                italic: false,
            },
            fill: FillStyle::Solid(theme.foreground),
            angle: -90.0,
            anchor: TextAnchor::Middle,
        }))
        .z_order(5);
        scene.insert_child(root, text);
    }
}

/// Compile a single (non-faceted) panel.
#[allow(clippy::too_many_arguments)]
fn compile_single_panel(
    chart: &Chart,
    scene: &mut SceneGraph,
    root: esoc_scene::node::NodeId,
    resolved: &[ResolvedLayer],
    data_bounds: &DataBounds,
    plot_x: f32,
    plot_y: f32,
    plot_w: f32,
    plot_h: f32,
    is_flipped: bool,
) -> Result<()> {
    let theme = &chart.theme;

    let plot_container = Node::container().transform(Affine2D::translate(plot_x, plot_y));
    let plot_id = scene.insert_child(root, plot_container);

    let is_pie = resolved
        .iter()
        .all(|l| matches!(l.mark, crate::grammar::layer::MarkType::Arc));
    let is_heatmap = resolved
        .iter()
        .all(|l| matches!(l.mark, crate::grammar::layer::MarkType::Heatmap));

    if is_heatmap {
        generate_heatmap_axes(
            chart, scene, root, plot_id, resolved, plot_x, plot_y, plot_w, plot_h,
        );
    } else if !is_pie {
        let (x_label, y_label) = if is_flipped {
            (chart.y_label.as_deref(), chart.x_label.as_deref())
        } else {
            (chart.x_label.as_deref(), chart.y_label.as_deref())
        };

        // Determine grid axes based on mark types
        let all_bar = resolved
            .iter()
            .all(|l| matches!(l.mark, crate::grammar::layer::MarkType::Bar));
        let grid_axes = if all_bar {
            axis_gen::GridAxes::HorizontalOnly
        } else {
            axis_gen::GridAxes::Both
        };

        // Extract category labels from bar layers for x-axis labeling
        let bar_categories: Option<Vec<String>> = if all_bar {
            resolved
                .iter()
                .find_map(|l| l.categories.clone())
        } else {
            None
        };

        axis_gen::generate_axes(
            scene, plot_id, root, data_bounds, plot_w, plot_h, plot_x, plot_y, theme,
            x_label, y_label, grid_axes,
            bar_categories.as_deref(),
        );
    }

    for resolved_layer in resolved {
        mark_gen::generate_layer_marks_flipped(
            scene, plot_id, resolved_layer, data_bounds, plot_w, plot_h, theme, is_flipped,
        )?;
    }

    // Legends
    let legends = legend_gen::collect_legends(resolved, theme);
    if !legends.is_empty() {
        legend_gen::generate_legends(scene, root, &legends, plot_x, plot_y, plot_w, plot_h, theme);
    }

    // Annotations
    if !chart.annotations.is_empty() && !is_pie {
        annotation::generate_annotations(
            scene, plot_id, root, &chart.annotations, data_bounds, plot_w, plot_h,
            plot_x, plot_y, theme,
        );
    }

    Ok(())
}

/// Compile a faceted chart (small multiples).
#[allow(clippy::too_many_arguments)]
fn compile_faceted(
    chart: &Chart,
    scene: &mut SceneGraph,
    root: esoc_scene::node::NodeId,
    resolved: &[ResolvedLayer],
    global_bounds: &DataBounds,
    plot_x: f32,
    plot_y: f32,
    plot_w: f32,
    plot_h: f32,
) -> Result<()> {
    let theme = &chart.theme;
    let panels = facet::compute_panels(&chart.facet, resolved);
    let ncol = match &chart.facet {
        Facet::Wrap { ncol } => *ncol,
        Facet::Grid { col_count, .. } => *col_count,
        Facet::None => 1,
    };

    let gap = 30.0_f32;
    let strip_h = theme.tick_font_size + 6.0;
    // Account for strip labels in available height
    let effective_h = plot_h - (strip_h * panels.len().div_ceil(ncol) as f32);
    let layout = facet::compute_facet_layout(panels.len(), ncol, plot_w, effective_h.max(100.0), gap);

    // Plot area container
    let plot_container = Node::container().transform(Affine2D::translate(plot_x, plot_y));
    let plot_area_id = scene.insert_child(root, plot_container);

    // Use smaller tick font for facet panels to prevent label overlap
    let mut facet_theme = theme.clone();
    facet_theme.tick_font_size = (theme.tick_font_size * 0.8).max(7.0);

    let nrow = panels.len().div_ceil(ncol);

    for (i, (panel, rect)) in panels.iter().zip(layout.iter()).enumerate() {
        let panel_bounds = facet::compute_panel_bounds(panel, chart.facet_scales, global_bounds);

        // Panel container offset within the plot area
        let panel_y_offset = rect.y + strip_h;
        let panel_container =
            Node::container().transform(Affine2D::translate(rect.x, panel_y_offset));
        let panel_id = scene.insert_child(plot_area_id, panel_container);

        let panel_row = i / ncol;
        let panel_col = i % ncol;
        let is_bottom_row = panel_row == nrow - 1;
        let is_left_col = panel_col == 0;

        // Only show x-labels on bottom row, y-labels on left column
        let show_x = is_bottom_row;
        let show_y = is_left_col;

        // Axes for this panel (use facet theme with smaller ticks)
        axis_gen::generate_axes(
            scene,
            panel_id,
            panel_id,
            &panel_bounds,
            rect.w,
            rect.h,
            0.0,
            0.0,
            &facet_theme,
            if show_x { chart.x_label.as_deref() } else { None },
            if show_y { chart.y_label.as_deref() } else { None },
            axis_gen::GridAxes::Both,
            None,
        );

        // Marks
        for layer in &panel.layers {
            mark_gen::generate_layer_marks(
                scene, panel_id, layer, &panel_bounds, rect.w, rect.h, theme,
            )?;
        }

        // Strip label
        facet::generate_strip_label(scene, panel_id, &panel.label, rect.w, theme);
    }

    Ok(())
}

fn compute_resolved_data_bounds(layers: &[ResolvedLayer]) -> Result<DataBounds> {
    // For Arc-only charts (pie/donut), scales are not used; return dummy bounds
    let all_arc = layers
        .iter()
        .all(|l| matches!(l.mark, crate::grammar::layer::MarkType::Arc));
    if all_arc {
        return Ok(DataBounds::new(0.0, 1.0, 0.0, 1.0));
    }

    // For Heatmap-only charts, bounds come from matrix dimensions
    let all_heatmap = layers
        .iter()
        .all(|l| matches!(l.mark, crate::grammar::layer::MarkType::Heatmap));
    if all_heatmap {
        if let Some(data) = layers.first().and_then(|l| l.heatmap_data.as_ref()) {
            let rows = data.len();
            let cols = data.first().map_or(0, |r| r.len());
            return Ok(DataBounds::new(-0.5, cols as f64 - 0.5, -0.5, rows as f64 - 0.5));
        }
        return Ok(DataBounds::new(0.0, 1.0, 0.0, 1.0));
    }

    let mut bounds =
        DataBounds::new(f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    let mut has_data = false;

    for layer in layers {
        for (&x, &y) in layer.x_data.iter().zip(layer.y_data.iter()) {
            bounds.include_point(x, y);
            has_data = true;
        }
        if let Some(summaries) = &layer.boxplot {
            for s in summaries {
                bounds.include_point(0.0, s.whisker_lo);
                bounds.include_point(0.0, s.whisker_hi);
                for &o in &s.outliers {
                    bounds.include_point(0.0, o);
                }
            }
        }
        if let Some(baseline) = &layer.y_baseline {
            for &y in baseline {
                bounds.include_point(0.0, y);
            }
        }
    }

    if !has_data {
        return Err(ChartError::EmptyData);
    }

    // Bar charts: pad x domain by 0.5 so edge bars don't overflow the plot area
    let has_bar = layers
        .iter()
        .any(|l| matches!(l.mark, crate::grammar::layer::MarkType::Bar));
    if has_bar {
        bounds.x_min -= 0.5;
        bounds.x_max += 0.5;
    }

    // Zero-inclusion: bar/area charts must include y=0
    let has_bar_or_area = layers.iter().any(|l| {
        matches!(
            l.mark,
            crate::grammar::layer::MarkType::Bar | crate::grammar::layer::MarkType::Area
        )
    });
    if has_bar_or_area {
        if bounds.y_min > 0.0 {
            bounds.y_min = 0.0;
        }
        if bounds.y_max < 0.0 {
            bounds.y_max = 0.0;
        }
    } else {
        // For line/point: include zero only if data is close to zero
        let y_range = bounds.y_max - bounds.y_min;
        if bounds.y_min > 0.0 && y_range > 0.0 && bounds.y_min < 0.25 * y_range {
            bounds.y_min = 0.0;
        }
    }

    Ok(bounds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::chart::Chart;
    use crate::grammar::coord::CoordSystem;
    use crate::grammar::layer::{Layer, MarkType};

    #[test]
    fn empty_chart_returns_error() {
        let chart = Chart::new(); // no layers
        let result = compile_chart(&chart);
        assert!(matches!(result, Err(ChartError::EmptyData)));
    }

    #[test]
    fn dimension_mismatch_returns_error() {
        let layer = Layer::new(MarkType::Point)
            .with_x(vec![1.0, 2.0, 3.0])
            .with_y(vec![1.0, 2.0]); // mismatched lengths
        let chart = Chart::new().layer(layer);
        let result = compile_chart(&chart);
        assert!(matches!(
            result,
            Err(ChartError::DimensionMismatch { layer: 0, x_len: 3, y_len: 2 })
        ));
    }

    #[test]
    fn bar_chart_zero_inclusion() {
        // Bar y-data all positive — bounds should include y=0
        let layer = Layer::new(MarkType::Bar)
            .with_x(vec![0.0, 1.0, 2.0])
            .with_y(vec![5.0, 10.0, 15.0]);
        let resolved = stat_transform::resolve_layer(&layer, 0).unwrap();
        let bounds = compute_resolved_data_bounds(&[resolved]).unwrap();
        assert!(bounds.y_min <= 0.0, "bar chart should include y=0, got y_min={}", bounds.y_min);
    }

    #[test]
    fn flipped_coords_swaps_data() {
        let layer = Layer::new(MarkType::Bar)
            .with_x(vec![0.0, 1.0, 2.0])
            .with_y(vec![10.0, 20.0, 30.0]);
        let chart = Chart::new().layer(layer).coord(CoordSystem::Flipped);
        let scene = compile_chart(&chart).unwrap();
        // Should succeed without error; scene should have nodes
        assert!(scene.root().is_some());
    }

    #[test]
    fn single_point_chart_compiles() {
        let layer = Layer::new(MarkType::Point)
            .with_x(vec![5.0])
            .with_y(vec![10.0]);
        let chart = Chart::new().layer(layer);
        let scene = compile_chart(&chart).unwrap();
        assert!(scene.root().is_some());
    }
}
