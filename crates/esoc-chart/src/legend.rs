// SPDX-License-Identifier: MIT OR Apache-2.0
//! Legend rendering for chart series.

use esoc_gfx::canvas::Canvas;
use esoc_gfx::color::Color;
use esoc_gfx::element::{DrawElement, Element};
use esoc_gfx::geom::Rect;
use esoc_gfx::layer::Layer;
use esoc_gfx::style::{Fill, FontStyle, Stroke, TextAnchor};

use crate::theme::Theme;

/// Legend position relative to the plot area.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LegendPosition {
    /// Upper-right corner (default).
    #[default]
    UpperRight,
    /// Upper-left corner.
    UpperLeft,
    /// Lower-right corner.
    LowerRight,
    /// Lower-left corner.
    LowerLeft,
}

/// A single legend entry.
#[derive(Clone, Debug)]
pub struct LegendEntry {
    /// Series label.
    pub label: String,
    /// Series color.
    pub color: Color,
}

/// Render a legend onto a canvas.
pub fn render_legend(
    canvas: &mut Canvas,
    plot_area: Rect,
    entries: &[LegendEntry],
    position: LegendPosition,
    theme: &Theme,
) {
    if entries.is_empty() {
        return;
    }

    let row_height = theme.legend_font_size * 1.6;
    let swatch_size = theme.legend_font_size * 0.8;
    let padding = 8.0;
    let gap = 6.0;

    // Estimate legend box size
    let text_measurer = esoc_gfx::text::HeuristicTextMeasurer;
    let max_label_width = entries
        .iter()
        .map(|e| {
            esoc_gfx::text::TextMeasurer::measure_width(
                &text_measurer,
                &e.label,
                theme.legend_font_size,
            )
        })
        .fold(0.0_f64, f64::max);

    let box_width = padding * 2.0 + swatch_size + gap + max_label_width;
    let box_height = padding * 2.0 + entries.len() as f64 * row_height;

    // Position the legend box
    let (bx, by) = match position {
        LegendPosition::UpperRight => (
            plot_area.right() - box_width - 10.0,
            plot_area.y + 10.0,
        ),
        LegendPosition::UpperLeft => (plot_area.x + 10.0, plot_area.y + 10.0),
        LegendPosition::LowerRight => (
            plot_area.right() - box_width - 10.0,
            plot_area.bottom() - box_height - 10.0,
        ),
        LegendPosition::LowerLeft => (
            plot_area.x + 10.0,
            plot_area.bottom() - box_height - 10.0,
        ),
    };

    // Background box
    canvas.add(DrawElement::new(
        Element::Rect {
            rect: Rect::new(bx, by, box_width, box_height),
            fill: Fill::Solid(theme.background.with_alpha(0.9)),
            stroke: Some(Stroke::solid(theme.grid_color, 0.5)),
            rx: 3.0,
        },
        Layer::Legend,
    ));

    // Entries
    for (i, entry) in entries.iter().enumerate() {
        let ey = by + padding + i as f64 * row_height + row_height / 2.0;

        // Color swatch
        canvas.add(DrawElement::new(
            Element::Rect {
                rect: Rect::new(
                    bx + padding,
                    ey - swatch_size / 2.0,
                    swatch_size,
                    swatch_size,
                ),
                fill: Fill::Solid(entry.color),
                stroke: None,
                rx: 2.0,
            },
            Layer::Legend,
        ));

        // Label
        let font = FontStyle {
            family: theme.font_family.clone(),
            size: theme.legend_font_size,
            weight: 400,
            color: theme.foreground,
            anchor: TextAnchor::Start,
        };
        canvas.add(DrawElement::new(
            Element::text(
                bx + padding + swatch_size + gap,
                ey + theme.legend_font_size * 0.35,
                &entry.label,
                font,
            ),
            Layer::Legend,
        ));
    }
}
