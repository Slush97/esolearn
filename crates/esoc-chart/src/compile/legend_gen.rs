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
}

/// Collect legend specs from resolved layers.
///
/// Scans layers for categorical data and generates legend entries.
/// Deduplicates categories across layers that share the same categorical mapping.
pub fn collect_legends(layers: &[ResolvedLayer], theme: &NewTheme) -> Vec<LegendSpec> {
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
        }];
    }

    if !has_categories {
        return vec![];
    }

    if all_cats.is_empty() {
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
    _plot_h: f32,
    theme: &NewTheme,
) {
    let legend_x = plot_x + plot_w + 15.0;
    let mut y = plot_y + 5.0;
    let swatch_size = 12.0_f32;
    let line_height = theme.legend_font_size * 1.5;

    for legend in legends {
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

        // Entries
        for entry in &legend.entries {
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
