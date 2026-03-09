// SPDX-License-Identifier: MIT OR Apache-2.0
//! Scatter series.

use esoc_gfx::canvas::Canvas;
use esoc_gfx::color::Color;
use esoc_gfx::element::{DrawElement, Element};
use esoc_gfx::layer::Layer;
use esoc_gfx::style::Fill;
use esoc_gfx::transform::CoordinateTransform;

use crate::series::{DataBounds, SeriesRenderer};
use crate::theme::Theme;

/// A scatter series rendering individual data points as circles.
#[derive(Clone, Debug)]
pub struct ScatterSeries {
    /// X values.
    pub x: Vec<f64>,
    /// Y values.
    pub y: Vec<f64>,
    /// Optional series label.
    pub label: Option<String>,
    /// Override color.
    pub color: Option<Color>,
    /// Override point radius.
    pub radius: Option<f64>,
}

impl ScatterSeries {
    /// Create a new scatter series.
    pub fn new(x: &[f64], y: &[f64]) -> Self {
        Self {
            x: x.to_vec(),
            y: y.to_vec(),
            label: None,
            color: None,
            radius: None,
        }
    }
}

impl SeriesRenderer for ScatterSeries {
    fn data_bounds(&self) -> DataBounds {
        let x_min = self.x.iter().copied().fold(f64::INFINITY, f64::min);
        let x_max = self.x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let y_min = self.y.iter().copied().fold(f64::INFINITY, f64::min);
        let y_max = self.y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        DataBounds::new(x_min, x_max, y_min, y_max)
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
        let radius = self.radius.unwrap_or(theme.point_radius);

        for (&x, &y) in self.x.iter().zip(self.y.iter()) {
            let p = transform.to_pixel(x, y);
            canvas.add(DrawElement::new(
                Element::circle(p.x, p.y, radius, Fill::Solid(color)),
                Layer::Data,
            ));
        }
    }

    fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}
