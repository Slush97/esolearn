// SPDX-License-Identifier: MIT OR Apache-2.0
//! Faceting: split data into panels for small multiples.

use crate::compile::stat_transform::ResolvedLayer;
use crate::grammar::facet::{Facet, FacetScales};
use crate::new_theme::NewTheme;
use esoc_scene::bounds::DataBounds;
use esoc_scene::mark::{Mark, RectMark, TextAnchor, TextMark};
use esoc_scene::node::{Node, NodeId};
use esoc_scene::style::{FillStyle, FontStyle, StrokeStyle};
use esoc_scene::SceneGraph;

/// A single facet panel containing filtered data.
pub struct FacetPanel {
    /// Facet label.
    pub label: String,
    /// Row index in the grid (used by facet layout consumers).
    #[allow(dead_code)]
    pub row: usize,
    /// Column index in the grid (used by facet layout consumers).
    #[allow(dead_code)]
    pub col: usize,
    /// Filtered resolved layers for this panel.
    pub layers: Vec<ResolvedLayer>,
}

/// Rectangle for a panel's position and size.
pub struct PanelRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// Compute facet panels from chart layers.
///
/// Collects unique facet values, filters each layer's data per panel.
pub fn compute_panels(
    facet: &Facet,
    layers: &[ResolvedLayer],
) -> Vec<FacetPanel> {
    // Collect unique facet values from all layers
    let mut unique_facets: Vec<String> = Vec::new();
    for layer in layers {
        if let Some(facet_vals) = &layer.facet_values {
            for v in facet_vals {
                if !unique_facets.contains(v) {
                    unique_facets.push(v.clone());
                }
            }
        }
    }

    if unique_facets.is_empty() {
        // No faceting: single panel with all data
        return vec![FacetPanel {
            label: String::new(),
            row: 0,
            col: 0,
            layers: layers.to_vec(),
        }];
    }

    let ncol = match facet {
        Facet::Wrap { ncol } => *ncol,
        Facet::Grid { col_count, .. } => *col_count,
        Facet::None => unique_facets.len(), // shouldn't happen, but safe fallback
    };
    let ncol = ncol.max(1);

    unique_facets
        .iter()
        .enumerate()
        .map(|(i, facet_val)| {
            let row = i / ncol;
            let col = i % ncol;

            // Filter each layer's data to rows matching this facet value
            let filtered_layers: Vec<ResolvedLayer> = layers
                .iter()
                .map(|layer| filter_layer_for_facet(layer, facet_val))
                .collect();

            FacetPanel {
                label: facet_val.clone(),
                row,
                col,
                layers: filtered_layers,
            }
        })
        .collect()
}

/// Filter a resolved layer to only include rows matching a facet value.
fn filter_layer_for_facet(layer: &ResolvedLayer, facet_val: &str) -> ResolvedLayer {
    let Some(facet_values) = &layer.facet_values else {
        return layer.clone();
    };

    let n = facet_values.len();
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    let mut categories = layer.categories.as_ref().map(|_| Vec::new());

    for i in 0..n {
        if facet_values[i] == facet_val {
            if i < layer.x_data.len() {
                x_data.push(layer.x_data[i]);
            }
            if i < layer.y_data.len() {
                y_data.push(layer.y_data[i]);
            }
            if let Some(ref cats) = layer.categories {
                if i < cats.len() {
                    categories.as_mut().unwrap().push(cats[i].clone());
                }
            }
        }
    }

    ResolvedLayer {
        x_data,
        y_data,
        categories,
        facet_values: None, // Already filtered
        ..layer.clone()
    }
}

/// Compute layout rectangles for facet panels.
pub fn compute_facet_layout(
    n_panels: usize,
    ncol: usize,
    total_w: f32,
    total_h: f32,
    gap: f32,
) -> Vec<PanelRect> {
    let ncol = ncol.max(1);
    let nrow = n_panels.div_ceil(ncol).max(1);

    let cell_w = (total_w - gap * (ncol as f32 - 1.0)) / ncol as f32;
    let cell_h = (total_h - gap * (nrow as f32 - 1.0)) / nrow as f32;

    (0..n_panels)
        .map(|i| {
            let row = i / ncol;
            let col = i % ncol;
            PanelRect {
                x: col as f32 * (cell_w + gap),
                y: row as f32 * (cell_h + gap),
                w: cell_w,
                h: cell_h,
            }
        })
        .collect()
}

/// Compute data bounds for a panel, optionally using global bounds.
pub fn compute_panel_bounds(
    panel: &FacetPanel,
    facet_scales: FacetScales,
    global_bounds: &DataBounds,
) -> DataBounds {
    match facet_scales {
        FacetScales::Shared => *global_bounds,
        FacetScales::Free => {
            let mut bounds = DataBounds::new(
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
            );
            for layer in &panel.layers {
                for (&x, &y) in layer.x_data.iter().zip(layer.y_data.iter()) {
                    bounds.include_point(x, y);
                }
            }
            if bounds.x_min > bounds.x_max {
                *global_bounds
            } else {
                bounds.pad(0.05)
            }
        }
        FacetScales::FreeX => {
            let mut bounds = DataBounds::new(
                f64::INFINITY,
                f64::NEG_INFINITY,
                global_bounds.y_min,
                global_bounds.y_max,
            );
            for layer in &panel.layers {
                for &x in &layer.x_data {
                    if x < bounds.x_min {
                        bounds.x_min = x;
                    }
                    if x > bounds.x_max {
                        bounds.x_max = x;
                    }
                }
            }
            if bounds.x_min > bounds.x_max {
                *global_bounds
            } else {
                bounds.pad(0.05)
            }
        }
        FacetScales::FreeY => {
            let mut bounds = DataBounds::new(
                global_bounds.x_min,
                global_bounds.x_max,
                f64::INFINITY,
                f64::NEG_INFINITY,
            );
            for layer in &panel.layers {
                for &y in &layer.y_data {
                    if y < bounds.y_min {
                        bounds.y_min = y;
                    }
                    if y > bounds.y_max {
                        bounds.y_max = y;
                    }
                }
            }
            if bounds.y_min > bounds.y_max {
                *global_bounds
            } else {
                bounds.pad(0.05)
            }
        }
    }
}

/// Generate a strip label above a panel.
pub fn generate_strip_label(
    scene: &mut SceneGraph,
    parent_id: NodeId,
    label: &str,
    panel_w: f32,
    theme: &NewTheme,
) {
    if label.is_empty() {
        return;
    }

    // Strip background
    let strip_h = theme.tick_font_size + 6.0;
    let bg = Node::with_mark(Mark::Rect(RectMark {
        bounds: esoc_scene::bounds::BoundingBox::new(0.0, -strip_h, panel_w, strip_h),
        fill: FillStyle::Solid(theme.grid_color),
        stroke: StrokeStyle {
            width: 0.0,
            ..Default::default()
        },
        corner_radius: 0.0,
    }))
    .z_order(5);
    scene.insert_child(parent_id, bg);

    // Strip text
    let text = Node::with_mark(Mark::Text(TextMark {
        position: [panel_w * 0.5, -strip_h * 0.3],
        text: label.to_string(),
        font: FontStyle {
            family: theme.font_family.clone(),
            size: theme.tick_font_size,
            weight: 700,
            italic: false,
        },
        fill: FillStyle::Solid(theme.foreground),
        angle: 0.0,
        anchor: TextAnchor::Middle,
    }))
    .z_order(6);
    scene.insert_child(parent_id, text);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::facet::Facet;
    use crate::grammar::layer::MarkType;
    use crate::grammar::position::Position;

    fn make_layer_with_facets(facets: Vec<String>) -> ResolvedLayer {
        let n = facets.len();
        ResolvedLayer {
            mark: MarkType::Point,
            x_data: (0..n).map(|i| i as f64).collect(),
            y_data: (0..n).map(|i| i as f64 * 10.0).collect(),
            categories: None,
            y_baseline: None,
            boxplot: None,
            inner_radius_fraction: 0.0,
            position: Position::Identity,
            is_binned: false,
            facet_values: Some(facets),
            layer_idx: 0,
            heatmap_data: None,
            row_labels: None,
            col_labels: None,
            annotate_cells: false,
            label: None,
            dodge_width: None,
        }
    }

    #[test]
    fn facet_wrap_6_panels_3_cols() {
        let facets = vec!["A", "B", "C", "D", "E", "F"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let layer = make_layer_with_facets(
            (0..12)
                .map(|i| facets[i % 6].clone())
                .collect(),
        );

        let panels = compute_panels(&Facet::Wrap { ncol: 3 }, &[layer]);
        assert_eq!(panels.len(), 6);
        assert_eq!(panels[0].row, 0);
        assert_eq!(panels[0].col, 0);
        assert_eq!(panels[3].row, 1);
        assert_eq!(panels[3].col, 0);
    }

    #[test]
    fn each_panel_has_own_data() {
        let facets = vec!["A", "A", "B", "B"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let layer = make_layer_with_facets(facets);

        let panels = compute_panels(&Facet::Wrap { ncol: 2 }, &[layer]);
        assert_eq!(panels.len(), 2);
        assert_eq!(panels[0].layers[0].x_data.len(), 2);
        assert_eq!(panels[1].layers[0].x_data.len(), 2);
    }

    #[test]
    fn no_facets_single_panel() {
        let layer = ResolvedLayer {
            mark: MarkType::Point,
            x_data: vec![1.0, 2.0],
            y_data: vec![3.0, 4.0],
            categories: None,
            y_baseline: None,
            boxplot: None,
            inner_radius_fraction: 0.0,
            position: Position::Identity,
            is_binned: false,
            facet_values: None,
            layer_idx: 0,
            heatmap_data: None,
            row_labels: None,
            col_labels: None,
            annotate_cells: false,
            label: None,
            dodge_width: None,
        };

        let panels = compute_panels(&Facet::None, &[layer]);
        assert_eq!(panels.len(), 1);
    }

    #[test]
    fn layout_computation() {
        let rects = compute_facet_layout(6, 3, 600.0, 400.0, 10.0);
        assert_eq!(rects.len(), 6);
        // First panel at origin
        assert!((rects[0].x).abs() < 1e-3);
        assert!((rects[0].y).abs() < 1e-3);
        // 3 columns means 3rd panel has col=2
        assert!(rects[2].x > rects[1].x);
    }
}
