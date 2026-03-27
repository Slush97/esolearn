// SPDX-License-Identifier: MIT OR Apache-2.0
//! Axis, grid, and tick generation.

use crate::compile::layout;
use crate::new_theme::NewTheme;
use esoc_scene::bounds::DataBounds;
use esoc_scene::mark::{Mark, RuleMark, TextAnchor, TextMark};
use esoc_scene::node::{Node, NodeId};
use esoc_scene::scale::Scale;
use esoc_scene::style::{FillStyle, FontStyle, StrokeStyle};
use esoc_scene::SceneGraph;

/// Which grid axes to show.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridAxes {
    /// Show both horizontal and vertical gridlines.
    Both,
    /// Show only horizontal gridlines (value axis for bar charts).
    HorizontalOnly,
    /// Show only vertical gridlines (reserved for future use).
    #[allow(dead_code)]
    VerticalOnly,
    /// Show no gridlines (pie charts, etc.).
    None,
}

/// Label rendering strategy for X axis tick labels.
enum LabelStrategy {
    /// Horizontal labels, no rotation.
    Horizontal,
    /// Labels angled at 45 degrees.
    Angled45,
    /// Vertical labels (90 degrees).
    Vertical,
    /// Skip every N labels (with given angle in degrees).
    SkipN { n: usize, angle: f32 },
}

/// Compute the effective horizontal space a label occupies at a given angle.
fn effective_label_width(text: &str, font_size: f32, angle_deg: f32) -> f32 {
    let w = layout::estimate_text_width(text, font_size);
    let h = font_size;
    let rad = angle_deg.to_radians().abs();
    w * rad.cos() + h * rad.sin()
}

/// Choose the best X label strategy to avoid overlaps.
fn choose_x_label_strategy(
    ticks: &[f64],
    scale: &Scale,
    font_size: f32,
    min_gap: f32,
) -> LabelStrategy {
    if ticks.len() <= 1 {
        return LabelStrategy::Horizontal;
    }

    let labels: Vec<String> = ticks.iter().map(|&t| scale.format_tick(t)).collect();
    let positions: Vec<f32> = ticks.iter().map(|&t| scale.map(t)).collect();

    // Try horizontal
    if labels_fit(&labels, &positions, font_size, 0.0, min_gap, 1) {
        return LabelStrategy::Horizontal;
    }

    // Try 45°
    if labels_fit(&labels, &positions, font_size, 45.0, min_gap, 1) {
        return LabelStrategy::Angled45;
    }

    // Try 90°
    if labels_fit(&labels, &positions, font_size, 90.0, min_gap, 1) {
        return LabelStrategy::Vertical;
    }

    // Try skip-N at 45° for increasing N
    for n in 2..ticks.len() {
        if labels_fit(&labels, &positions, font_size, 45.0, min_gap, n) {
            return LabelStrategy::SkipN { n, angle: 45.0 };
        }
    }

    // Fallback: show first + last only
    LabelStrategy::SkipN {
        n: ticks.len().max(1),
        angle: 45.0,
    }
}

/// Check if labels fit with given angle and skip factor.
fn labels_fit(
    labels: &[String],
    positions: &[f32],
    font_size: f32,
    angle_deg: f32,
    min_gap: f32,
    skip_n: usize,
) -> bool {
    let visible: Vec<(f32, f32)> = labels
        .iter()
        .zip(positions.iter())
        .enumerate()
        .filter(|(i, _)| i % skip_n == 0)
        .map(|(_, (label, &pos))| {
            let w = effective_label_width(label, font_size, angle_deg);
            (pos, w)
        })
        .collect();

    for pair in visible.windows(2) {
        let (pos_a, w_a) = pair[0];
        let (pos_b, w_b) = pair[1];
        if pos_b - pos_a < (w_a + w_b) * 0.5 + min_gap {
            return false;
        }
    }
    true
}

/// Generate axes, grid lines, and tick labels.
#[allow(clippy::too_many_arguments)]
pub fn generate_axes(
    scene: &mut SceneGraph,
    plot_id: NodeId,
    root_id: NodeId,
    bounds: &DataBounds,
    plot_w: f32,
    plot_h: f32,
    plot_x: f32,
    plot_y: f32,
    theme: &NewTheme,
    x_label: Option<&str>,
    y_label: Option<&str>,
    grid_axes: GridAxes,
    x_categories: Option<&[String]>,
    y_categories: Option<&[String]>,
) {
    let x_scale = Scale::Linear {
        domain: (bounds.x_min, bounds.x_max),
        range: (0.0, plot_w),
    };
    let y_scale = Scale::Linear {
        domain: (bounds.y_min, bounds.y_max),
        range: (plot_h, 0.0), // Y axis is inverted in screen space
    };

    let x_tick_count = layout::target_tick_count(plot_w, 80.0);
    let y_tick_count = layout::target_tick_count(plot_h, 40.0);
    let x_ticks: Vec<f64> = x_scale
        .ticks(x_tick_count)
        .into_iter()
        .filter(|&t| t >= bounds.x_min - 1e-9 && t <= bounds.x_max + 1e-9)
        .collect();
    let y_ticks: Vec<f64> = y_scale
        .ticks(y_tick_count)
        .into_iter()
        .filter(|&t| t >= bounds.y_min - 1e-9 && t <= bounds.y_max + 1e-9)
        .collect();

    // Grid lines (conditionally per grid_axes)
    if theme.show_grid && grid_axes != GridAxes::None {
        let grid_stroke = StrokeStyle::solid(theme.grid_color, theme.grid_width);

        // Horizontal grid lines (from y ticks)
        if matches!(grid_axes, GridAxes::Both | GridAxes::HorizontalOnly) {
            let mut h_segments = Vec::new();
            for &tick in &y_ticks {
                let y = y_scale.map(tick);
                h_segments.push(([0.0, y], [plot_w, y]));
            }
            if !h_segments.is_empty() {
                let grid = Node::with_mark(Mark::Rule(RuleMark {
                    segments: h_segments,
                    stroke: grid_stroke.clone(),
                }))
                .z_order(-5);
                scene.insert_child(plot_id, grid);
            }
        }

        // Vertical grid lines (from x ticks)
        if matches!(grid_axes, GridAxes::Both | GridAxes::VerticalOnly) {
            let mut v_segments = Vec::new();
            for &tick in &x_ticks {
                let x = x_scale.map(tick);
                v_segments.push(([x, 0.0], [x, plot_h]));
            }
            if !v_segments.is_empty() {
                let grid = Node::with_mark(Mark::Rule(RuleMark {
                    segments: v_segments,
                    stroke: grid_stroke,
                }))
                .z_order(-5);
                scene.insert_child(plot_id, grid);
            }
        }
    }

    // Axis frame (left + bottom border)
    let axis_stroke = StrokeStyle::solid(theme.foreground, theme.axis_width);
    let frame = Node::with_mark(Mark::Rule(RuleMark {
        segments: vec![
            ([0.0, 0.0], [0.0, plot_h]),     // left
            ([0.0, plot_h], [plot_w, plot_h]), // bottom
        ],
        stroke: axis_stroke,
    }))
    .z_order(5);
    scene.insert_child(plot_id, frame);

    // ── X tick labels with collision detection ──
    // When category labels are provided (bar charts), render them at integer
    // positions instead of using the linear scale's numeric tick formatting.
    if let Some(cats) = x_categories {
        let cat_ticks: Vec<f64> = (0..cats.len()).map(|i| i as f64).collect();
        let cat_labels: Vec<String> = cats.to_vec();

        // Build a temporary scale just for label strategy (uses category text widths)
        let cat_strategy = choose_x_label_strategy_for_categories(
            &cat_labels,
            &cat_ticks,
            &x_scale,
            theme.tick_font_size,
            4.0,
        );
        let (x_angle, x_skip, x_anchor) = match &cat_strategy {
            LabelStrategy::Horizontal => (0.0, 1, TextAnchor::Middle),
            LabelStrategy::Angled45 => (-45.0, 1, TextAnchor::End),
            LabelStrategy::Vertical => (-90.0, 1, TextAnchor::End),
            LabelStrategy::SkipN { n, angle } => (-angle, *n, TextAnchor::End),
        };

        for (i, (label, &tick)) in cat_labels.iter().zip(cat_ticks.iter()).enumerate() {
            if x_skip > 1 && i % x_skip != 0 && i != cat_labels.len() - 1 {
                continue;
            }
            let x = x_scale.map(tick) + plot_x;
            let tick_gap = theme.tick_font_size * 0.4; // font-relative gap
            let y_offset = if x_angle.abs() > 0.01 {
                theme.tick_font_size + tick_gap * 2.0
            } else {
                theme.tick_font_size + tick_gap
            };
            let y = plot_y + plot_h + y_offset;
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
                angle: x_angle,
                anchor: x_anchor,
            }))
            .z_order(5);
            scene.insert_child(root_id, text);
        }
    } else {
        let strategy = choose_x_label_strategy(&x_ticks, &x_scale, theme.tick_font_size, 4.0);
        let (x_angle, x_skip, x_anchor) = match &strategy {
            LabelStrategy::Horizontal => (0.0, 1, TextAnchor::Middle),
            LabelStrategy::Angled45 => (-45.0, 1, TextAnchor::End),
            LabelStrategy::Vertical => (-90.0, 1, TextAnchor::End),
            LabelStrategy::SkipN { n, angle } => (-angle, *n, TextAnchor::End),
        };

        for (i, &tick) in x_ticks.iter().enumerate() {
            // Skip labels based on strategy (but always show first and last)
            if x_skip > 1 && i % x_skip != 0 && i != x_ticks.len() - 1 {
                continue;
            }
            let x = x_scale.map(tick) + plot_x;
            let tick_gap = theme.tick_font_size * 0.4; // font-relative gap
            let y_offset = if x_angle.abs() > 0.01 {
                theme.tick_font_size + tick_gap * 2.0
            } else {
                theme.tick_font_size + tick_gap
            };
            let y = plot_y + plot_h + y_offset;
            let label = x_scale.format_tick(tick);
            let text = Node::with_mark(Mark::Text(TextMark {
                position: [x, y],
                text: label,
                font: FontStyle {
                    family: theme.font_family.clone(),
                    size: theme.tick_font_size,
                    weight: 400,
                    italic: false,
                },
                fill: FillStyle::Solid(theme.foreground),
                angle: x_angle,
                anchor: x_anchor,
            }))
            .z_order(5);
            scene.insert_child(root_id, text);
        }
    }

    // Y tick labels (placed to the left of the plot in root coordinates)
    let y_tick_gap = theme.tick_font_size * 0.4; // font-relative gap
    if let Some(cats) = y_categories {
        // Render category text labels at integer y positions instead of numeric ticks
        for (i, label) in cats.iter().enumerate() {
            let x = plot_x - y_tick_gap;
            let y = y_scale.map(i as f64) + plot_y;
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
            scene.insert_child(root_id, text);
        }
    } else {
        for &tick in &y_ticks {
            let x = plot_x - y_tick_gap;
            let y = y_scale.map(tick) + plot_y;
            let label = y_scale.format_tick(tick);
            let text = Node::with_mark(Mark::Text(TextMark {
                position: [x, y],
                text: label,
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
            scene.insert_child(root_id, text);
        }
    }

    // X axis label
    if let Some(label) = x_label {
        let title_gap = theme.label_font_size * 1.2; // font-relative spacing
        // Push the axis title below rotated category labels when present
        let x_label_y_offset = if let Some(cats) = x_categories {
            let cat_ticks: Vec<f64> = (0..cats.len()).map(|i| i as f64).collect();
            let cat_labels: Vec<String> = cats.to_vec();
            let strat = choose_x_label_strategy_for_categories(
                &cat_labels, &cat_ticks, &x_scale, theme.tick_font_size, 4.0,
            );
            if matches!(strat, LabelStrategy::Horizontal) {
                theme.tick_font_size + theme.label_font_size + title_gap
            } else {
                let max_label_w = cats.iter()
                    .map(|c| layout::estimate_text_width(c, theme.tick_font_size))
                    .fold(0.0_f32, f32::max);
                let rotated_h = max_label_w * 0.71 * 1.3;
                rotated_h + theme.label_font_size + title_gap * 2.0
            }
        } else {
            theme.tick_font_size + theme.label_font_size + title_gap
        };
        let text = Node::with_mark(Mark::Text(TextMark {
            position: [
                plot_x + plot_w * 0.5,
                plot_y + plot_h + x_label_y_offset,
            ],
            text: label.to_string(),
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
        scene.insert_child(root_id, text);
    }

    // Y axis label (rotated)
    if let Some(label) = y_label {
        let y_label_x = if let Some(cats) = y_categories {
            let max_cat_w = cats.iter()
                .map(|c| layout::estimate_text_width(c, theme.tick_font_size))
                .fold(0.0_f32, f32::max);
            plot_x - max_cat_w - theme.label_font_size - 5.0
        } else {
            // Compute label offset from actual tick label widths to avoid overlap
            let max_tick_w = y_ticks.iter()
                .map(|&t| layout::estimate_text_width(&y_scale.format_tick(t), theme.tick_font_size))
                .fold(0.0_f32, f32::max);
            plot_x - max_tick_w - theme.label_font_size * 0.5 - y_tick_gap
        };
        let text = Node::with_mark(Mark::Text(TextMark {
            position: [
                y_label_x,
                plot_y + plot_h * 0.5,
            ],
            text: label.to_string(),
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
        scene.insert_child(root_id, text);
    }
}

/// Choose label strategy for category labels (bar charts).
fn choose_x_label_strategy_for_categories(
    labels: &[String],
    ticks: &[f64],
    scale: &Scale,
    font_size: f32,
    min_gap: f32,
) -> LabelStrategy {
    if labels.len() <= 1 {
        return LabelStrategy::Horizontal;
    }

    let positions: Vec<f32> = ticks.iter().map(|&t| scale.map(t)).collect();

    if labels_fit(labels, &positions, font_size, 0.0, min_gap, 1) {
        return LabelStrategy::Horizontal;
    }
    if labels_fit(labels, &positions, font_size, 45.0, min_gap, 1) {
        return LabelStrategy::Angled45;
    }
    if labels_fit(labels, &positions, font_size, 90.0, min_gap, 1) {
        return LabelStrategy::Vertical;
    }
    for n in 2..labels.len() {
        if labels_fit(labels, &positions, font_size, 45.0, min_gap, n) {
            return LabelStrategy::SkipN { n, angle: 45.0 };
        }
    }
    LabelStrategy::SkipN {
        n: labels.len().max(1),
        angle: 45.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizontal_label_strategy_for_few_ticks() {
        let scale = Scale::Linear {
            domain: (0.0, 10.0),
            range: (0.0, 800.0),
        };
        let ticks = vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0];
        let strategy = choose_x_label_strategy(&ticks, &scale, 11.0, 4.0);
        // With 800px and only 6 ticks, horizontal should fit
        assert!(matches!(strategy, LabelStrategy::Horizontal));
    }

    #[test]
    fn angled_fallback_for_many_ticks() {
        let scale = Scale::Linear {
            domain: (0.0, 100.0),
            range: (0.0, 200.0), // Very narrow
        };
        let ticks: Vec<f64> = (0..=20).map(|i| i as f64 * 5.0).collect();
        let strategy = choose_x_label_strategy(&ticks, &scale, 11.0, 4.0);
        // With 200px and 21 ticks, horizontal won't fit
        assert!(!matches!(strategy, LabelStrategy::Horizontal));
    }

    #[test]
    fn labels_fit_uses_both_widths() {
        // Two labels of very different widths next to each other
        let labels = vec!["X".to_string(), "WWWWWWWWWW".to_string()];
        let positions = vec![0.0, 50.0];
        // With only left width considered (old bug), this would pass
        // With both widths, it should fail because the right label is very wide
        let fits = labels_fit(&labels, &positions, 12.0, 0.0, 2.0, 1);
        // The combined half-widths + gap should be checked
        // "X" width ≈ 7.2, "WWWWWWWWWW" width ≈ 72 → (7.2+72)/2 + 2 = 41.6
        // gap = 50, so 50 > 41.6 → should fit
        assert!(fits);

        // Now test a case that would pass with old logic but fail with new
        let labels2 = vec!["X".to_string(), "WWWWWWWWWWWWWWWWWW".to_string()];
        let positions2 = vec![0.0, 50.0];
        let fits2 = labels_fit(&labels2, &positions2, 12.0, 0.0, 2.0, 1);
        // "X" ≈ 7.2, "WWW...W" (18 chars) ≈ 129.6 → (7.2+129.6)/2 + 2 = 70.4 > 50
        assert!(!fits2, "wide right label should cause overlap");
    }

    #[test]
    fn grid_horizontal_only_for_bars() {
        let mut scene = SceneGraph::with_root();
        let root = scene.root().unwrap();
        let plot_id = scene.insert_child(root, Node::container());
        let bounds = DataBounds::new(0.0, 5.0, 0.0, 100.0);
        let theme = NewTheme::default();

        let before = scene.len();
        generate_axes(
            &mut scene, plot_id, root, &bounds,
            400.0, 300.0, 50.0, 50.0, &theme,
            Some("X"), Some("Y"), GridAxes::HorizontalOnly,
            None, None,
        );
        // Should have added nodes (axes, ticks, labels, grid)
        assert!(scene.len() > before);
    }
}
