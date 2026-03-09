// SPDX-License-Identifier: MIT OR Apache-2.0
//! Histogram series with automatic binning.

use esoc_gfx::canvas::Canvas;
use esoc_gfx::color::Color;
use esoc_gfx::element::{DrawElement, Element};
use esoc_gfx::geom::Rect;
use esoc_gfx::layer::Layer;
use esoc_gfx::style::{Fill, Stroke};
use esoc_gfx::transform::CoordinateTransform;

use crate::series::{DataBounds, SeriesRenderer};
use crate::theme::Theme;

/// Binning strategy for histograms.
#[derive(Clone, Copy, Debug, Default)]
pub enum BinStrategy {
    /// Sturges' rule: `ceil(log2(n)) + 1` (default).
    #[default]
    Sturges,
    /// Scott's rule: `3.5 * std / n^(1/3)`.
    Scott,
    /// Freedman-Diaconis rule: `2 * IQR / n^(1/3)`.
    FreedmanDiaconis,
    /// Fixed number of bins.
    Fixed(usize),
}

/// A histogram series that bins data automatically.
#[derive(Clone, Debug)]
pub struct HistogramSeries {
    /// Raw data to bin.
    pub data: Vec<f64>,
    /// Optional series label.
    pub label: Option<String>,
    /// Override color.
    pub color: Option<Color>,
    /// Override bin count (if set, overrides strategy).
    pub bin_count: Option<usize>,
    /// Binning strategy.
    pub strategy: BinStrategy,
}

impl HistogramSeries {
    /// Create a new histogram series.
    pub fn new(data: &[f64]) -> Self {
        Self {
            data: data.to_vec(),
            label: None,
            color: None,
            bin_count: None,
            strategy: BinStrategy::Sturges,
        }
    }

    /// Compute bin edges and counts.
    fn compute_bins(&self) -> (Vec<f64>, Vec<usize>) {
        if self.data.is_empty() {
            return (vec![], vec![]);
        }

        let mut sorted = self.data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let n = sorted.len();

        let num_bins = self.bin_count.unwrap_or_else(|| match self.strategy {
            BinStrategy::Sturges => ((n as f64).log2().ceil() as usize + 1).max(1),
            BinStrategy::Scott => {
                let std_dev = std_deviation(&sorted);
                if std_dev < 1e-15 {
                    1
                } else {
                    let bin_width = 3.5 * std_dev / (n as f64).cbrt();
                    ((max - min) / bin_width).ceil() as usize
                }
                .max(1)
            }
            BinStrategy::FreedmanDiaconis => {
                let iqr = percentile(&sorted, 75.0) - percentile(&sorted, 25.0);
                if iqr < 1e-15 {
                    ((n as f64).log2().ceil() as usize + 1).max(1)
                } else {
                    let bin_width = 2.0 * iqr / (n as f64).cbrt();
                    ((max - min) / bin_width).ceil() as usize
                }
                .max(1)
            }
            BinStrategy::Fixed(k) => k.max(1),
        });

        let range = if (max - min).abs() < 1e-15 {
            1.0
        } else {
            max - min
        };
        let bin_width = range / num_bins as f64;

        let mut edges = Vec::with_capacity(num_bins + 1);
        for i in 0..=num_bins {
            edges.push(min + i as f64 * bin_width);
        }

        let mut counts = vec![0usize; num_bins];
        for &v in &sorted {
            let mut idx = ((v - min) / bin_width) as usize;
            if idx >= num_bins {
                idx = num_bins - 1;
            }
            counts[idx] += 1;
        }

        (edges, counts)
    }
}

impl SeriesRenderer for HistogramSeries {
    fn data_bounds(&self) -> DataBounds {
        let (edges, counts) = self.compute_bins();
        if edges.is_empty() {
            return DataBounds::new(0.0, 1.0, 0.0, 1.0);
        }
        let x_min = edges[0];
        let x_max = edges[edges.len() - 1];
        let y_max = counts.iter().copied().max().unwrap_or(1) as f64;
        DataBounds::new(x_min, x_max, 0.0, y_max)
    }

    fn render(
        &self,
        canvas: &mut Canvas,
        transform: &CoordinateTransform,
        theme: &Theme,
        series_index: usize,
    ) {
        let (edges, counts) = self.compute_bins();
        if edges.len() < 2 {
            return;
        }

        let color = self
            .color
            .unwrap_or_else(|| theme.palette.get(series_index));

        for i in 0..counts.len() {
            let x0 = edges[i];
            let x1 = edges[i + 1];
            let h = counts[i] as f64;

            let p_tl = transform.to_pixel(x0, h);
            let p_br = transform.to_pixel(x1, 0.0);
            let rx = p_tl.x.min(p_br.x);
            let ry = p_tl.y.min(p_br.y);
            let rw = (p_br.x - p_tl.x).abs();
            let rh = (p_br.y - p_tl.y).abs();

            canvas.add(DrawElement::new(
                Element::Rect {
                    rect: Rect::new(rx, ry, rw, rh),
                    fill: Fill::Solid(color.with_alpha(0.7)),
                    stroke: Some(Stroke::solid(color, 0.5)),
                    rx: 0.0,
                },
                Layer::Data,
            ));
        }
    }

    fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

fn std_deviation(sorted: &[f64]) -> f64 {
    let n = sorted.len() as f64;
    let mean = sorted.iter().sum::<f64>() / n;
    let variance = sorted.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_bins() {
        let h = HistogramSeries::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let (edges, counts) = h.compute_bins();
        assert!(edges.len() >= 2);
        assert_eq!(counts.iter().sum::<usize>(), 8);
    }

    #[test]
    fn test_histogram_empty() {
        let h = HistogramSeries::new(&[]);
        let (edges, counts) = h.compute_bins();
        assert!(edges.is_empty());
        assert!(counts.is_empty());
    }
}
