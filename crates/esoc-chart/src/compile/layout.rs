// SPDX-License-Identifier: MIT OR Apache-2.0
//! Layout computation for chart margins and spacing.

use crate::compile::Margins;
use crate::grammar::chart::Chart;
use esoc_scene::bounds::DataBounds;
use esoc_scene::scale::Scale;

/// Estimate rendered text width (sans-serif, ~0.6 char width factor).
pub fn estimate_text_width(text: &str, font_size: f32) -> f32 {
    text.len() as f32 * font_size * 0.6
}

/// Compute adaptive tick count from axis pixel length.
/// min_spacing: 80.0 for x-axis, 40.0 for y-axis.
pub fn target_tick_count(axis_length_px: f32, min_spacing: f32) -> usize {
    (axis_length_px / min_spacing).floor().clamp(2.0, 10.0) as usize
}

/// Compute margins based on chart properties and actual data bounds.
pub fn compute_margins(chart: &Chart, data_bounds: &DataBounds) -> Margins {
    let has_title = chart.title.is_some();
    let has_x_label = chart.x_label.is_some();
    let has_y_label = chart.y_label.is_some();
    let has_subtitle = chart.subtitle.is_some();
    let has_caption = chart.caption.is_some();

    // ── Top margin ──
    let top = if has_title && has_subtitle {
        chart.theme.title_font_size + chart.theme.subtitle_font_size + 35.0
    } else if has_title {
        chart.theme.title_font_size + 20.0
    } else {
        15.0
    };

    // ── Bottom margin ──
    // For bar charts with category labels, account for rotated label height.
    let has_bar = chart.layers.iter().any(|l| {
        matches!(l.mark, crate::grammar::layer::MarkType::Bar)
    });
    let rotated_label_extra = if has_bar {
        if let Some(cats) = chart.layers.iter().find_map(|l| l.categories.as_ref()) {
            // Estimate whether labels will need rotation by checking if they
            // fit horizontally in the available plot width
            let plot_w_approx = chart.width * 0.7; // rough estimate after margins
            let total_label_w: f32 = cats.iter()
                .map(|c| estimate_text_width(c, chart.theme.tick_font_size) + 4.0)
                .sum();
            if total_label_w > plot_w_approx {
                // Labels will be rotated — add extra bottom space
                let max_label_len = cats.iter().map(|c| c.len()).max().unwrap_or(0);
                let max_w = max_label_len as f32 * chart.theme.tick_font_size * 0.6;
                let rotated_h = max_w * 0.71;
                (rotated_h - chart.theme.tick_font_size).max(0.0)
            } else {
                0.0
            }
        } else {
            0.0
        }
    } else {
        0.0
    };
    let caption_extra = if has_caption { chart.theme.tick_font_size + 10.0 } else { 0.0 };
    let bottom = if has_x_label {
        chart.theme.tick_font_size + chart.theme.label_font_size + 25.0 + rotated_label_extra + caption_extra
    } else {
        chart.theme.tick_font_size + 20.0 + rotated_label_extra + caption_extra
    };

    // ── Left margin — measure actual Y tick labels ──
    let preliminary_plot_h = chart.height - top - 50.0; // rough bottom estimate
    let y_tick_count = target_tick_count(preliminary_plot_h.max(100.0), 40.0);
    let y_scale = Scale::Linear {
        domain: (data_bounds.y_min, data_bounds.y_max),
        range: (preliminary_plot_h.max(100.0), 0.0),
    }
    .nice(y_tick_count);
    let y_ticks = y_scale.ticks(y_tick_count);
    let max_y_label_width = y_ticks
        .iter()
        .map(|&t| estimate_text_width(&y_scale.format_tick(t), chart.theme.tick_font_size))
        .fold(0.0_f32, f32::max);

    let tick_mark_size = 5.0;
    let tick_label_pad = 2.0;
    let axis_title_pad = if has_y_label { 4.0 } else { 0.0 };
    let axis_title_height = if has_y_label { chart.theme.label_font_size } else { 0.0 };
    let left = tick_mark_size + tick_label_pad + max_y_label_width + axis_title_pad + axis_title_height + 5.0;

    // ── Right margin — measure legend labels ──
    let has_legend = chart.layers.iter().any(|l| l.categories.is_some())
        || chart.layers.len() > 1;
    let right = if has_legend {
        // Collect unique legend labels from categories and layer labels
        let mut all_labels: Vec<String> = Vec::new();
        let has_layer_labels = chart.layers.iter().any(|l| l.label.is_some());
        if has_layer_labels || chart.layers.len() > 1 {
            // Multi-layer: use layer labels (or "Series N" fallback)
            for (i, layer) in chart.layers.iter().enumerate() {
                let lbl = layer.label.clone()
                    .unwrap_or_else(|| format!("Series {}", i + 1));
                if !all_labels.contains(&lbl) {
                    all_labels.push(lbl);
                }
            }
        }
        // Also collect category labels
        for layer in &chart.layers {
            if let Some(cats) = &layer.categories {
                for c in cats {
                    if !all_labels.contains(c) {
                        all_labels.push(c.clone());
                    }
                }
            }
        }
        if all_labels.is_empty() {
            all_labels.push("Series 00".into());
        }
        let max_label_width = all_labels
            .iter()
            .map(|c| estimate_text_width(c, chart.theme.legend_font_size))
            .fold(0.0_f32, f32::max);
        let swatch = 12.0;
        let gaps = 20.0;
        (swatch + gaps + max_label_width).max(80.0)
    } else {
        15.0
    };

    Margins {
        top,
        right,
        bottom,
        left,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::chart::Chart;
    use crate::grammar::layer::{Layer, MarkType};

    fn simple_chart() -> Chart {
        Chart::new().layer(
            Layer::new(MarkType::Point)
                .with_x(vec![0.0, 10.0])
                .with_y(vec![0.0, 100.0]),
        )
    }

    #[test]
    fn title_increases_top_margin() {
        let bounds = DataBounds::new(0.0, 10.0, 0.0, 100.0);
        let no_title = compute_margins(&simple_chart(), &bounds);
        let with_title = compute_margins(&simple_chart().title("Test"), &bounds);
        assert!(with_title.top > no_title.top);
    }

    #[test]
    fn labels_increase_margins() {
        let bounds = DataBounds::new(0.0, 10.0, 0.0, 100.0);
        let no_labels = compute_margins(&simple_chart(), &bounds);
        let with_labels = compute_margins(
            &simple_chart().x_label("X axis").y_label("Y axis"),
            &bounds,
        );
        assert!(with_labels.bottom > no_labels.bottom);
    }

    #[test]
    fn tick_count_in_range() {
        // Very small axis
        assert!(target_tick_count(50.0, 80.0) >= 2);
        assert!(target_tick_count(50.0, 80.0) <= 10);
        // Very large axis
        assert!(target_tick_count(5000.0, 80.0) >= 2);
        assert!(target_tick_count(5000.0, 80.0) <= 10);
    }
}
