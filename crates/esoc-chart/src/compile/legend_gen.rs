// SPDX-License-Identifier: MIT OR Apache-2.0
//! Legend generation: collects legend entries from resolved layers and renders them.

use crate::compile::stat_transform::ResolvedLayer;
use crate::new_theme::NewTheme;
use esoc_color::Color;
use esoc_scene::bounds::BoundingBox;
use esoc_scene::mark::{Mark, RectMark, TextAnchor, TextMark};
use esoc_scene::node::{Node, NodeId};
use esoc_scene::style::{FillStyle, FontStyle, StrokeStyle};
use esoc_scene::SceneGraph;

/// A single legend entry.
pub struct LegendEntry {
    /// Display label.
    pub label: String,
    /// Swatch color.
    pub color: Color,
}

/// A complete legend specification.
pub struct LegendSpec {
    /// Legend title (optional).
    pub title: Option<String>,
    /// Entries in this legend.
    pub entries: Vec<LegendEntry>,
    /// Continuous gradient legend (for heatmaps).
    pub gradient: Option<GradientLegend>,
}

/// A continuous gradient legend for heatmaps.
pub struct GradientLegend {
    /// Minimum value.
    pub v_min: f64,
    /// Maximum value.
    pub v_max: f64,
}

/// Collect legend specs from resolved layers.
///
/// Scans layers for categorical data and generates legend entries.
/// Deduplicates categories across layers that share the same categorical mapping.
pub fn collect_legends(layers: &[ResolvedLayer], theme: &NewTheme) -> Vec<LegendSpec> {
    // Check for heatmap layers — generate gradient legend
    let is_heatmap = layers.iter().all(|l| {
        matches!(l.mark, crate::grammar::layer::MarkType::Heatmap)
    });
    if is_heatmap {
        if let Some(data) = layers.first().and_then(|l| l.heatmap_data.as_ref()) {
            let mut v_min = f64::INFINITY;
            let mut v_max = f64::NEG_INFINITY;
            for row in data {
                for &v in row {
                    if v < v_min { v_min = v; }
                    if v > v_max { v_max = v; }
                }
            }
            if v_min < v_max {
                return vec![LegendSpec {
                    title: None,
                    entries: vec![],
                    gradient: Some(GradientLegend { v_min, v_max }),
                }];
            }
        }
        return vec![];
    }

    // Collect unique categories from layers that have them
    let mut all_cats: Vec<String> = Vec::new();
    let mut has_categories = false;

    for layer in layers {
        if let Some(cats) = &layer.categories {
            has_categories = true;
            for c in cats {
                if !all_cats.contains(c) {
                    all_cats.push(c.clone());
                }
            }
        }
    }

    // Multi-layer series legend: when there are multiple layers with labels,
    // or multiple layers without categories, create one legend entry per layer.
    let has_labels = layers.iter().any(|l| l.label.is_some());
    if layers.len() > 1 && (has_labels || !has_categories) {
        let entries: Vec<LegendEntry> = layers
            .iter()
            .enumerate()
            .map(|(i, layer)| LegendEntry {
                label: layer.label.clone().unwrap_or_else(|| format!("Series {}", i + 1)),
                color: theme.palette.get(i),
            })
            .collect();
        return vec![LegendSpec {
            title: None,
            entries,
            gradient: None,
        }];
    }

    if !has_categories {
        return vec![];
    }

    if all_cats.is_empty() {
        return vec![];
    }

    // Single-layer bar charts: categories are already shown as x-axis labels,
    // so a legend would just duplicate them. Suppress it.
    if layers.len() == 1
        && matches!(
            layers[0].mark,
            crate::grammar::layer::MarkType::Bar
        )
    {
        return vec![];
    }

    let entries: Vec<LegendEntry> = all_cats
        .iter()
        .enumerate()
        .map(|(i, cat)| LegendEntry {
            label: cat.clone(),
            color: theme.palette.get(i),
        })
        .collect();

    vec![LegendSpec {
        title: None,
        entries,
        gradient: None,
    }]
}

/// Render legend marks into the scene graph.
///
/// Positioned to the right of the plot area.
#[allow(clippy::too_many_arguments)]
pub fn generate_legends(
    scene: &mut SceneGraph,
    root_id: NodeId,
    legends: &[LegendSpec],
    plot_x: f32,
    plot_y: f32,
    plot_w: f32,
    plot_h: f32,
    theme: &NewTheme,
) {
    let legend_x = plot_x + plot_w + 15.0;
    let mut y = plot_y + 5.0;
    let swatch_size = 12.0_f32;
    let line_height = theme.legend_font_size * 1.5;

    for legend in legends {
        // Gradient legend for heatmaps
        if let Some(grad) = &legend.gradient {
            let bar_w = 15.0_f32;
            let bar_h = plot_h * 0.85;
            let n_steps = 50_usize;
            let step_h = bar_h / n_steps as f32;
            let color_scale = theme.color_scale.clone().unwrap_or_else(esoc_color::ColorScale::viridis);

            for i in 0..n_steps {
                let t = 1.0 - i as f32 / n_steps as f32; // top = max
                let color = color_scale.map(t);
                let rect = Node::with_mark(Mark::Rect(RectMark {
                    bounds: BoundingBox::new(legend_x, y + i as f32 * step_h, bar_w, step_h + 0.5),
                    fill: FillStyle::Solid(color),
                    stroke: StrokeStyle { width: 0.0, ..Default::default() },
                    corner_radius: 0.0,
                })).z_order(10);
                scene.insert_child(root_id, rect);
            }

            // Min/max labels
            let fmt = |v: f64| -> String {
                if (v - v.round()).abs() < 1e-9 { format!("{}", v as i64) } else { format!("{v:.1}") }
            };
            let max_label = Node::with_mark(Mark::Text(TextMark {
                position: [legend_x + bar_w + 5.0, y + theme.legend_font_size * 0.8],
                text: fmt(grad.v_max),
                font: FontStyle {
                    family: theme.font_family.clone(),
                    size: theme.legend_font_size,
                    weight: 400, italic: false,
                },
                fill: FillStyle::Solid(theme.foreground),
                angle: 0.0, anchor: TextAnchor::Start,
            })).z_order(10);
            scene.insert_child(root_id, max_label);

            let min_label = Node::with_mark(Mark::Text(TextMark {
                position: [legend_x + bar_w + 5.0, y + bar_h],
                text: fmt(grad.v_min),
                font: FontStyle {
                    family: theme.font_family.clone(),
                    size: theme.legend_font_size,
                    weight: 400, italic: false,
                },
                fill: FillStyle::Solid(theme.foreground),
                angle: 0.0, anchor: TextAnchor::Start,
            })).z_order(10);
            scene.insert_child(root_id, min_label);

            y += bar_h + line_height;
            continue;
        }

        // Optional title
        if let Some(title) = &legend.title {
            let text = Node::with_mark(Mark::Text(TextMark {
                position: [legend_x, y + theme.legend_font_size * 0.8],
                text: title.clone(),
                font: FontStyle {
                    family: theme.font_family.clone(),
                    size: theme.legend_font_size,
                    weight: 700,
                    italic: false,
                },
                fill: FillStyle::Solid(theme.foreground),
                angle: 0.0,
                anchor: TextAnchor::Start,
            }))
            .z_order(10);
            scene.insert_child(root_id, text);
            y += line_height;
        }

        // M9: Compute max entries that fit vertically, truncate with "… +N more"
        let max_entries = ((plot_h - 10.0) / line_height).floor().max(1.0) as usize;
        let total_entries = legend.entries.len();
        let show_count = total_entries.min(max_entries);

        // Entries
        for entry in &legend.entries[..show_count] {
            // Color swatch
            let swatch = Node::with_mark(Mark::Rect(RectMark {
                bounds: BoundingBox::new(legend_x, y, swatch_size, swatch_size),
                fill: FillStyle::Solid(entry.color),
                stroke: StrokeStyle {
                    width: 0.0,
                    ..Default::default()
                },
                corner_radius: 2.0,
            }))
            .z_order(10);
            scene.insert_child(root_id, swatch);

            // Label
            let text = Node::with_mark(Mark::Text(TextMark {
                position: [legend_x + swatch_size + 5.0, y + swatch_size * 0.85],
                text: entry.label.clone(),
                font: FontStyle {
                    family: theme.font_family.clone(),
                    size: theme.legend_font_size,
                    weight: 400,
                    italic: false,
                },
                fill: FillStyle::Solid(theme.foreground),
                angle: 0.0,
                anchor: TextAnchor::Start,
            }))
            .z_order(10);
            scene.insert_child(root_id, text);

            y += line_height;
        }

        // Show overflow indicator if entries were truncated
        if total_entries > show_count {
            let remaining = total_entries - show_count;
            let overflow_text = Node::with_mark(Mark::Text(TextMark {
                position: [legend_x, y + theme.legend_font_size * 0.8],
                text: format!("\u{2026} +{remaining} more"),
                font: FontStyle {
                    family: theme.font_family.clone(),
                    size: theme.legend_font_size,
                    weight: 400,
                    italic: true,
                },
                fill: FillStyle::Solid(theme.foreground),
                angle: 0.0,
                anchor: TextAnchor::Start,
            }))
            .z_order(10);
            scene.insert_child(root_id, overflow_text);
            y += line_height;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::stat_transform::ResolvedLayer;
    use crate::grammar::layer::MarkType;
    use crate::grammar::position::Position;

    fn make_resolved(cats: Option<Vec<String>>, idx: usize) -> ResolvedLayer {
        ResolvedLayer {
            mark: MarkType::Point,
            x_data: vec![0.0, 1.0],
            y_data: vec![0.0, 1.0],
            categories: cats,
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

    #[test]
    fn no_cats_single_layer_no_legend() {
        let theme = NewTheme::default();
        let layers = vec![make_resolved(None, 0)];
        let legends = collect_legends(&layers, &theme);
        assert!(legends.is_empty());
    }

    #[test]
    fn cats_deduped_legend() {
        let theme = NewTheme::default();
        let cats = Some(vec!["A".into(), "B".into(), "A".into(), "C".into()]);
        let layers = vec![make_resolved(cats, 0)];
        let legends = collect_legends(&layers, &theme);
        assert_eq!(legends.len(), 1);
        let labels: Vec<&str> = legends[0].entries.iter().map(|e| e.label.as_str()).collect();
        assert_eq!(labels, vec!["A", "B", "C"]);
    }

    #[test]
    fn multi_layer_series_legend() {
        let theme = NewTheme::default();
        let layers = vec![
            make_resolved(None, 0),
            make_resolved(None, 1),
            make_resolved(None, 2),
        ];
        let legends = collect_legends(&layers, &theme);
        assert_eq!(legends.len(), 1);
        assert_eq!(legends[0].entries.len(), 3);
        assert_eq!(legends[0].entries[0].label, "Series 1");
        assert_eq!(legends[0].entries[2].label, "Series 3");
    }

    #[test]
    fn heatmap_generates_gradient_legend() {
        let theme = NewTheme::default();
        let mut layer = make_resolved(None, 0);
        layer.mark = MarkType::Heatmap;
        layer.heatmap_data = Some(vec![vec![1.0, 5.0], vec![3.0, 9.0]]);
        let legends = collect_legends(&[layer], &theme);
        assert_eq!(legends.len(), 1);
        assert!(legends[0].gradient.is_some());
        let g = legends[0].gradient.as_ref().unwrap();
        assert!((g.v_min - 1.0).abs() < 1e-10);
        assert!((g.v_max - 9.0).abs() < 1e-10);
    }

    #[test]
    fn single_bar_suppresses_legend() {
        let theme = NewTheme::default();
        let mut layer = make_resolved(Some(vec!["A".into(), "B".into()]), 0);
        layer.mark = MarkType::Bar;
        let legends = collect_legends(&[layer], &theme);
        assert!(legends.is_empty(), "single-layer bar should suppress legend");
    }

    #[test]
    fn multi_layer_uses_label_field() {
        let theme = NewTheme::default();
        let mut l0 = make_resolved(None, 0);
        l0.label = Some("Revenue".into());
        let mut l1 = make_resolved(None, 1);
        l1.label = Some("Expenses".into());
        let legends = collect_legends(&[l0, l1], &theme);
        assert_eq!(legends.len(), 1);
        assert_eq!(legends[0].entries[0].label, "Revenue");
        assert_eq!(legends[0].entries[1].label, "Expenses");
    }
}
