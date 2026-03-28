// SPDX-License-Identifier: MIT OR Apache-2.0
//! Box plot series with quartile/whisker computation.

use esoc_gfx::canvas::Canvas;
use esoc_gfx::element::{DrawElement, Element};
use esoc_gfx::geom::Rect;
use esoc_gfx::layer::Layer;
use esoc_gfx::style::{Fill, Stroke};
use esoc_gfx::transform::CoordinateTransform;

use crate::series::{DataBounds, SeriesRenderer};
use crate::theme::Theme;

/// A box plot series showing distribution statistics.
#[derive(Clone, Debug)]
pub struct BoxPlotSeries {
    /// Datasets to plot (one box per dataset).
    pub datasets: Vec<Vec<f64>>,
    /// Optional series label.
    pub label: Option<String>,
    /// Optional category labels.
    pub labels: Option<Vec<String>>,
}

/// Computed statistics for a single box.
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct BoxStats {
    min: f64,
    q1: f64,
    median: f64,
    q3: f64,
    max: f64,
    whisker_lo: f64,
    whisker_hi: f64,
}

impl BoxPlotSeries {
    /// Create a new box plot series.
    pub fn new(datasets: Vec<Vec<f64>>) -> Self {
        Self {
            datasets,
            label: None,
            labels: None,
        }
    }

    fn compute_stats(data: &[f64]) -> Option<BoxStats> {
        if data.is_empty() {
            return None;
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let q1 = percentile(&sorted, 25.0);
        let median = percentile(&sorted, 50.0);
        let q3 = percentile(&sorted, 75.0);
        let iqr = q3 - q1;

        // Whiskers at 1.5 * IQR or data extent
        let whisker_lo = sorted
            .iter()
            .copied()
            .find(|&v| v >= q1 - 1.5 * iqr)
            .unwrap_or(min);
        let whisker_hi = sorted
            .iter()
            .rev()
            .copied()
            .find(|&v| v <= q3 + 1.5 * iqr)
            .unwrap_or(max);

        Some(BoxStats {
            min,
            q1,
            median,
            q3,
            max,
            whisker_lo,
            whisker_hi,
        })
    }
}

impl SeriesRenderer for BoxPlotSeries {
    fn data_bounds(&self) -> DataBounds {
        let n = self.datasets.len();
        let y_min = self
            .datasets
            .iter()
            .flat_map(|d| d.iter().copied())
            .fold(f64::INFINITY, f64::min);
        let y_max = self
            .datasets
            .iter()
            .flat_map(|d| d.iter().copied())
            .fold(f64::NEG_INFINITY, f64::max);

        DataBounds::new(-0.5, n as f64 - 0.5, y_min, y_max)
    }

    fn render(
        &self,
        canvas: &mut Canvas,
        transform: &CoordinateTransform,
        theme: &Theme,
        series_index: usize,
    ) {
        let box_width = 0.6;
        let color = theme.palette.get(series_index);

        for (i, dataset) in self.datasets.iter().enumerate() {
            let Some(stats) = Self::compute_stats(dataset) else {
                continue;
            };

            let x = i as f64;
            let half_w = box_width / 2.0;

            // Box (Q1 to Q3)
            let p_tl = transform.to_pixel(x - half_w, stats.q3);
            let p_br = transform.to_pixel(x + half_w, stats.q1);
            let rx = p_tl.x.min(p_br.x);
            let ry = p_tl.y.min(p_br.y);
            let rw = (p_br.x - p_tl.x).abs();
            let rh = (p_br.y - p_tl.y).abs();
            canvas.add(DrawElement::new(
                Element::Rect {
                    rect: Rect::new(rx, ry, rw, rh),
                    fill: Fill::Solid(color.with_alpha(0.3)),
                    stroke: Some(Stroke::solid(color, 1.5)),
                    rx: 0.0,
                },
                Layer::Data,
            ));

            // Median line
            let p_ml = transform.to_pixel(x - half_w, stats.median);
            let p_mr = transform.to_pixel(x + half_w, stats.median);
            canvas.add(DrawElement::line(
                p_ml.x,
                p_ml.y,
                p_mr.x,
                p_mr.y,
                Stroke::solid(color, 2.0),
                Layer::Data,
            ));

            // Whiskers
            let p_wl_top = transform.to_pixel(x, stats.whisker_hi);
            let p_wl_q3 = transform.to_pixel(x, stats.q3);
            canvas.add(DrawElement::line(
                p_wl_top.x,
                p_wl_top.y,
                p_wl_q3.x,
                p_wl_q3.y,
                Stroke::solid(color, 1.0),
                Layer::Data,
            ));

            let p_wl_bot = transform.to_pixel(x, stats.whisker_lo);
            let p_wl_q1 = transform.to_pixel(x, stats.q1);
            canvas.add(DrawElement::line(
                p_wl_bot.x,
                p_wl_bot.y,
                p_wl_q1.x,
                p_wl_q1.y,
                Stroke::solid(color, 1.0),
                Layer::Data,
            ));

            // Whisker caps
            let cap_w = half_w * 0.5;
            let p_cap_hi_l = transform.to_pixel(x - cap_w, stats.whisker_hi);
            let p_cap_hi_r = transform.to_pixel(x + cap_w, stats.whisker_hi);
            canvas.add(DrawElement::line(
                p_cap_hi_l.x,
                p_cap_hi_l.y,
                p_cap_hi_r.x,
                p_cap_hi_r.y,
                Stroke::solid(color, 1.0),
                Layer::Data,
            ));

            let p_cap_lo_l = transform.to_pixel(x - cap_w, stats.whisker_lo);
            let p_cap_lo_r = transform.to_pixel(x + cap_w, stats.whisker_lo);
            canvas.add(DrawElement::line(
                p_cap_lo_l.x,
                p_cap_lo_l.y,
                p_cap_lo_r.x,
                p_cap_lo_r.y,
                Stroke::solid(color, 1.0),
                Layer::Data,
            ));
        }
    }

    fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).clamp(0.0, (sorted.len() - 1) as f64);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}
