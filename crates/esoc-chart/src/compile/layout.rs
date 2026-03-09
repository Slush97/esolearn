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
    let caption_extra = if has_caption { chart.theme.tick_font_size + 10.0 } else { 0.0 };
    let bottom = if has_x_label {
        chart.theme.tick_font_size + chart.theme.label_font_size + 25.0 + caption_extra
    } else {
        chart.theme.tick_font_size + 20.0 + caption_extra
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
        // Collect unique category labels
        let mut all_cats: Vec<&str> = Vec::new();
        for layer in &chart.layers {
            if let Some(cats) = &layer.categories {
                for c in cats {
                    if !all_cats.contains(&c.as_str()) {
                        all_cats.push(c.as_str());
                    }
                }
            }
        }
        // If no explicit categories but multiple layers, use "Series N" labels
        if all_cats.is_empty() {
            for i in 0..chart.layers.len() {
                // Approximate "Series N" length
                all_cats.push("Series 00");
                let _ = i;
            }
        }
        let max_label_width = all_cats
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
