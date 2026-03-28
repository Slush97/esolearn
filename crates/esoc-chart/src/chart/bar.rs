// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bar series (vertical and horizontal).

use esoc_gfx::canvas::Canvas;
use esoc_gfx::color::Color;
use esoc_gfx::element::{DrawElement, Element};
use esoc_gfx::geom::Rect;
use esoc_gfx::layer::Layer;
use esoc_gfx::style::{Fill, Stroke};
use esoc_gfx::transform::CoordinateTransform;

use crate::series::{DataBounds, SeriesRenderer};
use crate::theme::Theme;

/// A bar chart series.
#[derive(Clone, Debug)]
pub struct BarSeries {
    /// X positions (bar centers).
    pub x: Vec<f64>,
    /// Bar heights (or lengths for horizontal).
    pub heights: Vec<f64>,
    /// Optional series label.
    pub label: Option<String>,
    /// Override color.
    pub color: Option<Color>,
    /// Bar width in data units.
    pub bar_width: f64,
    /// Whether to draw horizontal bars.
    pub horizontal: bool,
}

impl BarSeries {
    /// Create a new vertical bar series.
    pub fn new(x: &[f64], heights: &[f64]) -> Self {
        Self {
            x: x.to_vec(),
            heights: heights.to_vec(),
            label: None,
            color: None,
            bar_width: 0.8,
            horizontal: false,
        }
    }
}

impl SeriesRenderer for BarSeries {
    fn data_bounds(&self) -> DataBounds {
        if self.horizontal {
            let y_min = self.x.iter().copied().fold(f64::INFINITY, f64::min) - self.bar_width / 2.0;
            let y_max =
                self.x.iter().copied().fold(f64::NEG_INFINITY, f64::max) + self.bar_width / 2.0;
            let x_max = self.heights.iter().copied().fold(0.0_f64, f64::max);
            DataBounds::new(0.0, x_max, y_min, y_max)
        } else {
            let x_min = self.x.iter().copied().fold(f64::INFINITY, f64::min) - self.bar_width / 2.0;
            let x_max =
                self.x.iter().copied().fold(f64::NEG_INFINITY, f64::max) + self.bar_width / 2.0;
            let y_max = self.heights.iter().copied().fold(0.0_f64, f64::max);
            DataBounds::new(x_min, x_max, 0.0, y_max)
        }
    }

    fn render(
        &self,
        canvas: &mut Canvas,
        transform: &CoordinateTransform,
        theme: &Theme,
        series_index: usize,
    ) {
        let color = self
            .color
            .unwrap_or_else(|| theme.palette.get(series_index));

        for (&x, &h) in self.x.iter().zip(self.heights.iter()) {
            if self.horizontal {
                let p_start = transform.to_pixel(0.0, x - self.bar_width / 2.0);
                let p_end = transform.to_pixel(h, x + self.bar_width / 2.0);
                let rx = p_start.x.min(p_end.x);
                let ry = p_start.y.min(p_end.y);
                let rw = (p_end.x - p_start.x).abs();
                let rh = (p_end.y - p_start.y).abs();
                canvas.add(DrawElement::new(
                    Element::Rect {
                        rect: Rect::new(rx, ry, rw, rh),
                        fill: Fill::Solid(color),
                        stroke: Some(Stroke::solid(color.with_alpha(0.8), 0.5)),
                        rx: 0.0,
                    },
                    Layer::Data,
                ));
            } else {
                let p_top = transform.to_pixel(x - self.bar_width / 2.0, h);
                let p_bottom = transform.to_pixel(x + self.bar_width / 2.0, 0.0);
                let rx = p_top.x.min(p_bottom.x);
                let ry = p_top.y.min(p_bottom.y);
                let rw = (p_bottom.x - p_top.x).abs();
                let rh = (p_bottom.y - p_top.y).abs();
                canvas.add(DrawElement::new(
                    Element::Rect {
                        rect: Rect::new(rx, ry, rw, rh),
                        fill: Fill::Solid(color),
                        stroke: Some(Stroke::solid(color.with_alpha(0.8), 0.5)),
                        rx: 0.0,
                    },
                    Layer::Data,
                ));
            }
        }
    }

    fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}
