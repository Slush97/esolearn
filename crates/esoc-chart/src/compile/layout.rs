// SPDX-License-Identifier: MIT OR Apache-2.0
//! Layout computation for chart margins and spacing.

use crate::compile::Margins;
use crate::grammar::chart::Chart;
use esoc_scene::bounds::DataBounds;
use esoc_scene::scale::Scale;

/// Where the legend should be placed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LegendPlacement {
    /// No legend needed.
    None,
    /// Legend to the right of the plot.
    Right,
    /// Legend below the plot (horizontal layout).
    Bottom,
}

/// Per-character width factor for common ASCII glyphs (sans-serif approximation).
/// Narrow chars ~0.3, average ~0.5, wide chars ~0.7.
fn char_width_factor(c: char) -> f32 {
    match c {
        'i' | 'j' | 'l' | '!' | '|' | '.' | ',' | ':' | ';' | '\'' => 0.3,
        'f' | 'r' | 't' | '(' | ')' | '[' | ']' | '{' | '}' | ' ' | '1' => 0.35,
        'm' | 'w' | 'M' | 'W' => 0.7,
        'A'..='Z' => 0.6,
        _ => 0.5,
    }
}

/// Estimate rendered text width using per-character width factors (sans-serif).
pub fn estimate_text_width(text: &str, font_size: f32) -> f32 {
    text.chars().map(|c| char_width_factor(c) * font_size).sum()
}

/// Compute adaptive tick count from axis pixel length.
/// min_spacing: 80.0 for x-axis, 40.0 for y-axis.
pub fn target_tick_count(axis_length_px: f32, min_spacing: f32) -> usize {
    (axis_length_px / min_spacing).floor().clamp(2.0, 15.0) as usize
}

/// Compute margins based on chart properties and actual data bounds.
pub fn compute_margins(chart: &Chart, data_bounds: &DataBounds) -> Margins {
    // Treemap: minimal margins (no axes), only title/legend
    let is_treemap = chart
        .layers
        .iter()
        .all(|l| matches!(l.mark, crate::grammar::layer::MarkType::Treemap))
        && !chart.layers.is_empty();
    if is_treemap {
        return compute_treemap_margins(chart);
    }

    let has_title = chart.title.is_some();
    let has_x_label = chart.x_label.is_some();
    let has_y_label = chart.y_label.is_some();
    let has_subtitle = chart.subtitle.is_some();
    let has_caption = chart.caption.is_some();

    // ── Top margin ──
    // Title + gap so the title doesn't collide with the top tick label.
    let title_plot_gap = 12.0;
    let top = if has_title && has_subtitle {
        chart.theme.title_font_size + chart.theme.subtitle_font_size + 7.0 + title_plot_gap
    } else if has_title {
        chart.theme.title_font_size + 4.0 + title_plot_gap
    } else {
        5.0
    };

    // ── Bottom margin ──
    // For bar charts with category labels, account for rotated label height.
    let has_bar = chart
        .layers
        .iter()
        .any(|l| matches!(l.mark, crate::grammar::layer::MarkType::Bar));
    let rotated_label_extra = if has_bar {
        if let Some(cats) = chart.layers.iter().find_map(|l| l.categories.as_ref()) {
            // Estimate whether labels will need rotation by checking if they
            // fit horizontally in the available plot width
            let plot_w_approx = chart.width * 0.7; // rough estimate after margins
            let total_label_w: f32 = cats
                .iter()
                .map(|c| estimate_text_width(c, chart.theme.tick_font_size) + 4.0)
                .sum();
            if total_label_w > plot_w_approx {
                // Labels will be rotated — add extra bottom space
                let max_label_len = cats.iter().map(|c| c.len()).max().unwrap_or(0);
                let max_w = max_label_len as f32 * chart.theme.tick_font_size * 0.6;
                let rotated_h = max_w * 0.71 * 1.5; // 1.5× for descenders + baseline
                (rotated_h - chart.theme.tick_font_size).max(0.0) + 10.0
            } else {
                0.0
            }
        } else {
            0.0
        }
    } else {
        0.0
    };
    let caption_extra = if has_caption {
        chart.theme.tick_font_size + 10.0
    } else {
        0.0
    };
    let tick_size = 5.0;
    let tick_pad = 2.0;
    // Match the title_gap used in axis_gen for x-label placement (label_font_size * 1.2),
    // plus a descender allowance so the label text doesn't clip the chart edge.
    let title_pad = chart.theme.label_font_size * 1.2;
    let descender = chart.theme.label_font_size * 0.35;
    let bottom = if has_x_label {
        tick_size
            + tick_pad
            + chart.theme.tick_font_size
            + title_pad
            + chart.theme.label_font_size
            + descender
            + rotated_label_extra
            + caption_extra
    } else {
        tick_size + tick_pad + chart.theme.tick_font_size + rotated_label_extra + caption_extra
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
    let axis_title_height = if has_y_label {
        chart.theme.label_font_size
    } else {
        0.0
    };
    let label_extra = if has_y_label {
        chart.theme.label_font_size
    } else {
        0.0
    };
    let left = tick_mark_size
        + tick_label_pad
        + max_y_label_width
        + axis_title_pad
        + axis_title_height
        + label_extra
        + 5.0;

    // ── Right margin — measure legend labels ──
    // Match the suppression condition in legend_gen: single-layer bar charts suppress legends
    let has_legend = (chart.layers.iter().any(|l| l.categories.is_some())
        || chart.layers.len() > 1)
        && !(chart.layers.len() == 1
            && matches!(chart.layers[0].mark, crate::grammar::layer::MarkType::Bar));
    // Heatmap gradient legend needs right margin even when has_legend is false
    let is_heatmap = chart.layers.iter().all(|l| {
        matches!(l.mark, crate::grammar::layer::MarkType::Heatmap) && l.heatmap_data.is_some()
    }) && !chart.layers.is_empty();

    // Count legend entries for placement decision
    let legend_entry_count = if is_heatmap {
        0 // gradient legend always on right
    } else if has_legend {
        collect_legend_entry_count(chart)
    } else {
        0
    };

    // Determine legend placement: bottom when many entries or narrow chart
    let legend_placement = if is_heatmap {
        LegendPlacement::Right // gradient legend always right
    } else if !has_legend {
        LegendPlacement::None
    } else if legend_entry_count > 5 || chart.width < 500.0 {
        LegendPlacement::Bottom
    } else {
        LegendPlacement::Right
    };

    let (right, bottom_legend_extra) = if is_heatmap {
        // Colorbar: gap (10px) + bar (20px) + tick marks (4px) + gap (6px) + label width (~40px)
        (80.0, 0.0)
    } else if has_legend && legend_placement == LegendPlacement::Right {
        // Collect unique legend labels from categories and layer labels
        let all_labels = collect_legend_labels(chart);
        let max_label_width = all_labels
            .iter()
            .map(|c| estimate_text_width(c, chart.theme.legend_font_size))
            .fold(0.0_f32, f32::max);
        let swatch = 12.0;
        let gaps = 20.0;
        ((swatch + gaps + max_label_width).max(80.0), 0.0)
    } else if has_legend && legend_placement == LegendPlacement::Bottom {
        // Bottom legend: minimal right margin, add to bottom
        let all_labels = collect_legend_labels(chart);
        let line_height = chart.theme.legend_font_size * 1.5;
        let swatch = 12.0;
        let entry_gap = 16.0;
        let entry_widths: Vec<f32> = all_labels
            .iter()
            .map(|l| {
                swatch + 4.0 + estimate_text_width(l, chart.theme.legend_font_size) + entry_gap
            })
            .collect();
        let available_w = chart.width - left - 10.0;
        let mut rows = 1_usize;
        let mut row_w = 0.0_f32;
        for &w in &entry_widths {
            if row_w + w > available_w && row_w > 0.0 {
                rows += 1;
                row_w = w;
            } else {
                row_w += w;
            }
        }
        let legend_h = rows as f32 * line_height + 8.0; // 8px gap above legend
        (10.0, legend_h)
    } else {
        (10.0, 0.0)
    };

    Margins {
        top,
        right,
        bottom: bottom + bottom_legend_extra,
        left,
        legend_placement,
    }
}

/// Collect unique legend labels from a chart for margin measurement.
fn collect_legend_labels(chart: &Chart) -> Vec<String> {
    let mut all_labels: Vec<String> = Vec::new();
    let has_layer_labels = chart.layers.iter().any(|l| l.label.is_some());
    if has_layer_labels || chart.layers.len() > 1 {
        for (i, layer) in chart.layers.iter().enumerate() {
            let lbl = layer
                .label
                .clone()
                .unwrap_or_else(|| format!("Series {}", i + 1));
            if !all_labels.contains(&lbl) {
                all_labels.push(lbl);
            }
        }
    }
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
    all_labels
}

/// Count the total number of unique legend entries.
fn collect_legend_entry_count(chart: &Chart) -> usize {
    collect_legend_labels(chart).len()
}

/// Validate that the plot area occupies a reasonable fraction of the chart.
/// Clamps margins if the plot area would be squeezed below 65% of chart area.
pub fn validate_plot_ratio(margins: &mut super::Margins, chart_width: f32, chart_height: f32) {
    let plot_w = chart_width - margins.left - margins.right;
    let plot_h = chart_height - margins.top - margins.bottom;
    let plot_area = plot_w * plot_h;
    let chart_area = chart_width * chart_height;

    if chart_area > 0.0 && plot_area / chart_area < 0.65 {
        // Proportionally shrink all margins to bring plot area to ~65%
        let target_ratio = 0.65_f32;
        let total_h_margin = margins.left + margins.right;
        let total_v_margin = margins.top + margins.bottom;
        // Scale margins down uniformly
        let scale = {
            // We need: (W - h*s)(H - v*s) / (W*H) >= target
            // Binary search for the right scale factor
            let mut lo = 0.0_f32;
            let mut hi = 1.0_f32;
            for _ in 0..20 {
                let mid = (lo + hi) * 0.5;
                let pw = chart_width - total_h_margin * mid;
                let ph = chart_height - total_v_margin * mid;
                if pw * ph / chart_area >= target_ratio {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            hi
        };
        margins.left *= scale;
        margins.right *= scale;
        margins.top *= scale;
        margins.bottom *= scale;
    }
}

/// Wrap text at word boundaries, returning at most `max_lines` lines.
/// Adds ellipsis if text would require more lines than allowed.
pub fn wrap_text(text: &str, max_chars: usize, max_lines: usize) -> Vec<String> {
    if max_chars == 0 || max_lines == 0 {
        return vec![text.to_string()];
    }
    if text.len() <= max_chars {
        return vec![text.to_string()];
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![text.to_string()];
    }

    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();

    for word in &words {
        if current.is_empty() {
            current = word.to_string();
        } else if current.len() + 1 + word.len() <= max_chars {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(current);
            current = word.to_string();
            if lines.len() >= max_lines {
                break;
            }
        }
    }

    if lines.len() < max_lines {
        lines.push(current);
    } else {
        // Text overflows: add ellipsis to last line
        if let Some(last) = lines.last_mut() {
            if last.len() + 1 < max_chars {
                last.push('…');
            } else {
                let truncated: String = last.chars().take(max_chars.saturating_sub(1)).collect();
                *last = format!("{truncated}…");
            }
        }
    }

    lines
}

/// Compute margins for treemap charts (minimal: title + legend, no axes).
fn compute_treemap_margins(chart: &Chart) -> Margins {
    let top = if chart.title.is_some() && chart.subtitle.is_some() {
        chart.theme.title_font_size + chart.theme.subtitle_font_size + 35.0
    } else if chart.title.is_some() {
        chart.theme.title_font_size + 20.0
    } else {
        10.0
    };

    let bottom = if chart.caption.is_some() {
        chart.theme.tick_font_size + 15.0
    } else {
        10.0
    };

    // Right margin for legend (treemap always has categories → legend)
    let has_legend = chart.layers.iter().any(|l| l.categories.is_some());
    let right = if has_legend {
        let mut all_labels: Vec<String> = Vec::new();
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
            15.0
        } else {
            let max_label_width = all_labels
                .iter()
                .map(|c| estimate_text_width(c, chart.theme.legend_font_size))
                .fold(0.0_f32, f32::max);
            let swatch = 12.0;
            let gaps = 20.0;
            (swatch + gaps + max_label_width).max(80.0)
        }
    } else {
        15.0
    };

    Margins {
        top,
        right,
        bottom,
        left: 10.0,
        legend_placement: if has_legend {
            LegendPlacement::Right
        } else {
            LegendPlacement::None
        },
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
        let with_labels =
            compute_margins(&simple_chart().x_label("X axis").y_label("Y axis"), &bounds);
        assert!(with_labels.bottom > no_labels.bottom);
    }

    #[test]
    fn tick_count_in_range() {
        // Very small axis
        assert!(target_tick_count(50.0, 80.0) >= 2);
        assert!(target_tick_count(50.0, 80.0) <= 15);
        // Very large axis
        assert!(target_tick_count(5000.0, 80.0) >= 2);
        assert!(target_tick_count(5000.0, 80.0) <= 15);
    }
}
