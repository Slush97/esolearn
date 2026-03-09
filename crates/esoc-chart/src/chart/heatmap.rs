// SPDX-License-Identifier: MIT OR Apache-2.0
//! Heatmap series with color-mapped cells.

use esoc_gfx::canvas::Canvas;
use esoc_gfx::color::Color;
use esoc_gfx::element::{DrawElement, Element};
use esoc_gfx::geom::Rect;
use esoc_gfx::layer::Layer;
use esoc_gfx::palette::Palette;
use esoc_gfx::style::{Fill, FontStyle, TextAnchor};
use esoc_gfx::transform::CoordinateTransform;

use crate::series::{DataBounds, SeriesRenderer};
use crate::theme::Theme;

/// A heatmap series rendering a 2D matrix as colored cells.
#[derive(Clone, Debug)]
pub struct HeatmapSeries {
    /// 2D data: `data[row][col]`.
    pub data: Vec<Vec<f64>>,
    /// Optional series label.
    pub label: Option<String>,
    /// Whether to annotate cells with values.
    pub annotate: bool,
    /// Optional row labels.
    pub row_labels: Option<Vec<String>>,
    /// Optional column labels.
    pub col_labels: Option<Vec<String>>,
    /// Color palette for mapping values.
    pub palette: Option<Palette>,
}

impl HeatmapSeries {
    /// Create a new heatmap series.
    pub fn new(data: Vec<Vec<f64>>) -> Self {
        Self {
            data,
            label: None,
            annotate: false,
            row_labels: None,
            col_labels: None,
            palette: None,
        }
    }

    fn value_range(&self) -> (f64, f64) {
        let min = self
            .data
            .iter()
            .flat_map(|row| row.iter().copied())
            .fold(f64::INFINITY, f64::min);
        let max = self
            .data
            .iter()
            .flat_map(|row| row.iter().copied())
            .fold(f64::NEG_INFINITY, f64::max);
        (min, max)
    }
}

impl SeriesRenderer for HeatmapSeries {
    fn data_bounds(&self) -> DataBounds {
        let rows = self.data.len();
        let cols = self.data.first().map_or(0, Vec::len);
        DataBounds::new(-0.5, cols as f64 - 0.5, -0.5, rows as f64 - 0.5)
    }

    fn render(
        &self,
        canvas: &mut Canvas,
        transform: &CoordinateTransform,
        theme: &Theme,
        _series_index: usize,
    ) {
        let rows = self.data.len();
        if rows == 0 {
            return;
        }
        let _cols = self.data[0].len();
        let (vmin, vmax) = self.value_range();
        let palette = self.palette.clone().unwrap_or_else(Palette::viridis);

        for (r, row) in self.data.iter().enumerate() {
            for (c, &val) in row.iter().enumerate() {
                let t = if (vmax - vmin).abs() < 1e-15 {
                    0.5
                } else {
                    (val - vmin) / (vmax - vmin)
                };
                let color = palette.sample(t);

                // Cell rectangle — heatmap uses y-axis inverted from data row order
                let y = (rows - 1 - r) as f64;
                let p_tl = transform.to_pixel(c as f64 - 0.5, y + 0.5);
                let p_br = transform.to_pixel(c as f64 + 0.5, y - 0.5);
                let rx = p_tl.x.min(p_br.x);
                let ry = p_tl.y.min(p_br.y);
                let rw = (p_br.x - p_tl.x).abs();
                let rh = (p_br.y - p_tl.y).abs();

                canvas.add(DrawElement::new(
                    Element::Rect {
                        rect: Rect::new(rx, ry, rw, rh),
                        fill: Fill::Solid(color),
                        stroke: None,
                        rx: 0.0,
                    },
                    Layer::Data,
                ));

                // Annotation
                if self.annotate {
                    let center = transform.to_pixel(c as f64, y);
                    let text_color = if t > 0.5 {
                        Color::BLACK
                    } else {
                        Color::WHITE
                    };
                    let font = FontStyle {
                        family: theme.font_family.clone(),
                        size: theme.tick_font_size,
                        weight: 400,
                        color: text_color,
                        anchor: TextAnchor::Middle,
                    };
                    let text = if (val - val.round()).abs() < 1e-9 {
                        format!("{}", val as i64)
                    } else {
                        format!("{val:.2}")
                    };
                    canvas.add(DrawElement::new(
                        Element::text(center.x, center.y + theme.tick_font_size * 0.35, text, font),
                        Layer::Annotations,
                    ));
                }
            }
        }
    }

    fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}
