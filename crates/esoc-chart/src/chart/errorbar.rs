// SPDX-License-Identifier: MIT OR Apache-2.0
//! Error bar series (symmetric and asymmetric errors).

use esoc_gfx::canvas::Canvas;
use esoc_gfx::color::Color;
use esoc_gfx::element::DrawElement;
use esoc_gfx::layer::Layer;
use esoc_gfx::style::{Fill, Stroke};
use esoc_gfx::transform::CoordinateTransform;

use crate::series::{DataBounds, SeriesRenderer};
use crate::theme::Theme;

/// An error bar series showing data uncertainty.
#[derive(Clone, Debug)]
pub struct ErrorBarSeries {
    /// X values.
    pub x: Vec<f64>,
    /// Y values (centers).
    pub y: Vec<f64>,
    /// Error magnitudes (symmetric if `err_neg` is `None`).
    pub err: Vec<f64>,
    /// Optional negative error magnitudes (for asymmetric errors).
    pub err_neg: Option<Vec<f64>>,
    /// Optional series label.
    pub label: Option<String>,
    /// Override color.
    pub color: Option<Color>,
    /// Cap width in pixels.
    pub cap_width: f64,
}

impl ErrorBarSeries {
    /// Create a new symmetric error bar series.
    pub fn new(x: &[f64], y: &[f64], err: &[f64]) -> Self {
        Self {
            x: x.to_vec(),
            y: y.to_vec(),
            err: err.to_vec(),
            err_neg: None,
            label: None,
            color: None,
            cap_width: 6.0,
        }
    }

    /// Create asymmetric error bars.
    pub fn asymmetric(x: &[f64], y: &[f64], err_pos: &[f64], err_neg: &[f64]) -> Self {
        Self {
            x: x.to_vec(),
            y: y.to_vec(),
            err: err_pos.to_vec(),
            err_neg: Some(err_neg.to_vec()),
            label: None,
            color: None,
            cap_width: 6.0,
        }
    }
}

impl SeriesRenderer for ErrorBarSeries {
    fn data_bounds(&self) -> DataBounds {
        let x_min = self.x.iter().copied().fold(f64::INFINITY, f64::min);
        let x_max = self.x.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        for i in 0..self.y.len() {
            let lo = self.y[i] - self.err_neg.as_ref().map_or(self.err[i], |en| en[i]);
            let hi = self.y[i] + self.err[i];
            y_min = y_min.min(lo);
            y_max = y_max.max(hi);
        }

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
        let stroke = Stroke::solid(color, 1.5);

        for i in 0..self.x.len() {
            let x = self.x[i];
            let y = self.y[i];
            let err_pos = self.err[i];
            let err_neg = self.err_neg.as_ref().map_or(err_pos, |en| en[i]);

            let p_center = transform.to_pixel(x, y);
            let p_top = transform.to_pixel(x, y + err_pos);
            let p_bot = transform.to_pixel(x, y - err_neg);

            // Vertical error bar
            canvas.add(DrawElement::line(
                p_top.x,
                p_top.y,
                p_bot.x,
                p_bot.y,
                stroke.clone(),
                Layer::Data,
            ));

            // Top cap
            canvas.add(DrawElement::line(
                p_top.x - self.cap_width / 2.0,
                p_top.y,
                p_top.x + self.cap_width / 2.0,
                p_top.y,
                stroke.clone(),
                Layer::Data,
            ));

            // Bottom cap
            canvas.add(DrawElement::line(
                p_bot.x - self.cap_width / 2.0,
                p_bot.y,
                p_bot.x + self.cap_width / 2.0,
                p_bot.y,
                stroke.clone(),
                Layer::Data,
            ));

            // Center dot
            canvas.add(DrawElement::circle(
                p_center.x,
                p_center.y,
                3.0,
                Fill::Solid(color),
                Layer::Data,
            ));
        }
    }

    fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}
