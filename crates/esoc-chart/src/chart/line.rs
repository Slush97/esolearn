// SPDX-License-Identifier: MIT OR Apache-2.0
//! Line series.

use esoc_gfx::canvas::Canvas;
use esoc_gfx::color::Color;
use esoc_gfx::element::{DrawElement, Element};
use esoc_gfx::geom::Point;
use esoc_gfx::layer::Layer;
use esoc_gfx::style::{DashPattern, Stroke};
use esoc_gfx::transform::CoordinateTransform;

use crate::series::{DataBounds, SeriesRenderer};
use crate::theme::Theme;

/// A line series connecting data points with a polyline.
#[derive(Clone, Debug)]
pub struct LineSeries {
    /// X values.
    pub x: Vec<f64>,
    /// Y values.
    pub y: Vec<f64>,
    /// Optional series label.
    pub label: Option<String>,
    /// Override color.
    pub color: Option<Color>,
    /// Override line width.
    pub width: Option<f64>,
    /// Dash pattern.
    pub dash: Option<DashPattern>,
}

impl LineSeries {
    /// Create a new line series.
    pub fn new(x: &[f64], y: &[f64]) -> Self {
        Self {
            x: x.to_vec(),
            y: y.to_vec(),
            label: None,
            color: None,
            width: None,
            dash: None,
        }
    }

    /// Build a stroke from this series' config and the theme.
    pub fn build_stroke(&self, theme: &Theme, series_index: usize) -> Stroke {
        let color = self
            .color
            .unwrap_or_else(|| theme.palette.get(series_index));
        let width = self.width.unwrap_or(theme.line_width);
        match &self.dash {
            Some(dash) => Stroke::dashed(color, width, &dash.dashes),
            None => Stroke::solid(color, width),
        }
    }
}

impl SeriesRenderer for LineSeries {
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
        if self.x.is_empty() {
            return;
        }

        let points: Vec<Point> = self
            .x
            .iter()
            .zip(self.y.iter())
            .map(|(&x, &y)| transform.to_pixel(x, y))
            .collect();

        let stroke = self.build_stroke(theme, series_index);
        canvas.add(DrawElement::new(
            Element::polyline(points, stroke),
            Layer::Data,
        ));
    }

    fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}
