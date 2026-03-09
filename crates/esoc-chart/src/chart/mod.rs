// SPDX-License-Identifier: MIT OR Apache-2.0
//! Chart series types.

pub mod bar;
pub mod boxplot;
pub mod errorbar;
pub mod heatmap;
pub mod histogram;
pub mod line;
pub mod scatter;

pub use bar::BarSeries;
pub use boxplot::BoxPlotSeries;
pub use errorbar::ErrorBarSeries;
pub use heatmap::HeatmapSeries;
pub use histogram::HistogramSeries;
pub use line::LineSeries;
pub use scatter::ScatterSeries;
