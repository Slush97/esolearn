// SPDX-License-Identifier: MIT OR Apache-2.0
//! Axes: a single plot area with axis config, series, and legend.

use esoc_gfx::color::Color;
use esoc_gfx::style::DashPattern;

use crate::axis::AxisConfig;
use crate::chart::{
    BarSeries, BoxPlotSeries, ErrorBarSeries, HeatmapSeries, HistogramSeries, LineSeries,
    ScatterSeries,
};
use crate::legend::LegendPosition;
use crate::series::SeriesRenderer;

/// A single plot area containing axes, series, and legend configuration.
#[derive(Default)]
pub struct Axes {
    /// Title for this plot area.
    pub title: Option<String>,
    /// X-axis configuration.
    pub x_config: AxisConfig,
    /// Y-axis configuration.
    pub y_config: AxisConfig,
    /// Data series to render.
    pub(crate) series: Vec<Box<dyn SeriesRenderer>>,
    /// Legend position.
    pub legend_position: LegendPosition,
    /// Whether to show the legend.
    pub show_legend: bool,
}

impl Axes {
    /// Create a new axes.
    pub fn new() -> Self {
        Self {
            show_legend: true,
            ..Self::default()
        }
    }

    /// Set the plot title.
    pub fn title(&mut self, title: impl Into<String>) -> &mut Self {
        self.title = Some(title.into());
        self
    }

    /// Set the X-axis label.
    pub fn x_label(&mut self, label: impl Into<String>) -> &mut Self {
        self.x_config.label = Some(label.into());
        self
    }

    /// Set the Y-axis label.
    pub fn y_label(&mut self, label: impl Into<String>) -> &mut Self {
        self.y_config.label = Some(label.into());
        self
    }

    /// Set the X-axis range manually.
    pub fn x_range(&mut self, min: f64, max: f64) -> &mut Self {
        self.x_config.range = Some((min, max));
        self
    }

    /// Set the Y-axis range manually.
    pub fn y_range(&mut self, min: f64, max: f64) -> &mut Self {
        self.y_config.range = Some((min, max));
        self
    }

    /// Set X-axis config.
    pub fn x_axis(&mut self, config: AxisConfig) -> &mut Self {
        self.x_config = config;
        self
    }

    /// Set Y-axis config.
    pub fn y_axis(&mut self, config: AxisConfig) -> &mut Self {
        self.y_config = config;
        self
    }

    /// Set legend position.
    pub fn legend(&mut self, position: LegendPosition) -> &mut Self {
        self.legend_position = position;
        self
    }

    /// Add a line series and return a builder for customization.
    pub fn line(&mut self, x: &[f64], y: &[f64]) -> LineBuilder<'_> {
        LineBuilder {
            axes: self,
            series: Some(LineSeries::new(x, y)),
        }
    }

    /// Add a scatter series and return a builder for customization.
    pub fn scatter(&mut self, x: &[f64], y: &[f64]) -> ScatterBuilder<'_> {
        ScatterBuilder {
            axes: self,
            series: Some(ScatterSeries::new(x, y)),
        }
    }

    /// Add a bar series and return a builder for customization.
    pub fn bar(&mut self, x: &[f64], heights: &[f64]) -> BarBuilder<'_> {
        BarBuilder {
            axes: self,
            series: Some(BarSeries::new(x, heights)),
        }
    }

    /// Add a histogram series and return a builder for customization.
    pub fn histogram(&mut self, data: &[f64]) -> HistogramBuilder<'_> {
        HistogramBuilder {
            axes: self,
            series: Some(HistogramSeries::new(data)),
        }
    }

    /// Add a box plot series and return a builder for customization.
    pub fn boxplot(&mut self, datasets: Vec<Vec<f64>>) -> BoxPlotBuilder<'_> {
        BoxPlotBuilder {
            axes: self,
            series: Some(BoxPlotSeries::new(datasets)),
        }
    }

    /// Add a heatmap series and return a builder for customization.
    pub fn heatmap(&mut self, data: Vec<Vec<f64>>) -> HeatmapBuilder<'_> {
        HeatmapBuilder {
            axes: self,
            series: Some(HeatmapSeries::new(data)),
        }
    }

    /// Add an error bar series and return a builder for customization.
    pub fn errorbar(&mut self, x: &[f64], y: &[f64], err: &[f64]) -> ErrorBarBuilder<'_> {
        ErrorBarBuilder {
            axes: self,
            series: Some(ErrorBarSeries::new(x, y, err)),
        }
    }

    /// Add an arbitrary series.
    pub fn add_series(&mut self, series: Box<dyn SeriesRenderer>) -> &mut Self {
        self.series.push(series);
        self
    }
}

// --- Series builders with fluent API ---
// Each builder uses Option<Series> so Drop can take() the series
// and done() can also take() it, preventing double-add.

/// Builder for a line series.
pub struct LineBuilder<'a> {
    axes: &'a mut Axes,
    series: Option<LineSeries>,
}

impl<'a> LineBuilder<'a> {
    /// Set the series label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        if let Some(s) = &mut self.series {
            s.label = Some(label.into());
        }
        self
    }

    /// Set the line color.
    pub fn color(mut self, color: Color) -> Self {
        if let Some(s) = &mut self.series {
            s.color = Some(color);
        }
        self
    }

    /// Set the line width.
    pub fn width(mut self, width: f64) -> Self {
        if let Some(s) = &mut self.series {
            s.width = Some(width);
        }
        self
    }

    /// Set a dash pattern.
    pub fn dash(mut self, dashes: &[f64]) -> Self {
        if let Some(s) = &mut self.series {
            s.dash = Some(DashPattern::new(dashes));
        }
        self
    }

    /// Finish and add the series.
    pub fn done(mut self) -> &'a mut Axes {
        if let Some(s) = self.series.take() {
            self.axes.series.push(Box::new(s));
        }
        self.axes
    }
}


/// Builder for a scatter series.
pub struct ScatterBuilder<'a> {
    axes: &'a mut Axes,
    series: Option<ScatterSeries>,
}

impl<'a> ScatterBuilder<'a> {
    /// Set the series label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        if let Some(s) = &mut self.series {
            s.label = Some(label.into());
        }
        self
    }

    /// Set the point color.
    pub fn color(mut self, color: Color) -> Self {
        if let Some(s) = &mut self.series {
            s.color = Some(color);
        }
        self
    }

    /// Set the point radius.
    pub fn radius(mut self, r: f64) -> Self {
        if let Some(s) = &mut self.series {
            s.radius = Some(r);
        }
        self
    }

    /// Finish and add the series.
    pub fn done(mut self) -> &'a mut Axes {
        if let Some(s) = self.series.take() {
            self.axes.series.push(Box::new(s));
        }
        self.axes
    }
}


/// Builder for a bar series.
pub struct BarBuilder<'a> {
    axes: &'a mut Axes,
    series: Option<BarSeries>,
}

impl<'a> BarBuilder<'a> {
    /// Set the series label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        if let Some(s) = &mut self.series {
            s.label = Some(label.into());
        }
        self
    }

    /// Set bar color.
    pub fn color(mut self, color: Color) -> Self {
        if let Some(s) = &mut self.series {
            s.color = Some(color);
        }
        self
    }

    /// Set bar width.
    pub fn bar_width(mut self, width: f64) -> Self {
        if let Some(s) = &mut self.series {
            s.bar_width = width;
        }
        self
    }

    /// Use horizontal bars.
    pub fn horizontal(mut self) -> Self {
        if let Some(s) = &mut self.series {
            s.horizontal = true;
        }
        self
    }

    /// Finish and add the series.
    pub fn done(mut self) -> &'a mut Axes {
        if let Some(s) = self.series.take() {
            self.axes.series.push(Box::new(s));
        }
        self.axes
    }
}


/// Builder for a histogram series.
pub struct HistogramBuilder<'a> {
    axes: &'a mut Axes,
    series: Option<HistogramSeries>,
}

impl<'a> HistogramBuilder<'a> {
    /// Set the series label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        if let Some(s) = &mut self.series {
            s.label = Some(label.into());
        }
        self
    }

    /// Set bar color.
    pub fn color(mut self, color: Color) -> Self {
        if let Some(s) = &mut self.series {
            s.color = Some(color);
        }
        self
    }

    /// Set the number of bins.
    pub fn bins(mut self, bins: usize) -> Self {
        if let Some(s) = &mut self.series {
            s.bin_count = Some(bins);
        }
        self
    }

    /// Finish and add the series.
    pub fn done(mut self) -> &'a mut Axes {
        if let Some(s) = self.series.take() {
            self.axes.series.push(Box::new(s));
        }
        self.axes
    }
}


/// Builder for a box plot series.
pub struct BoxPlotBuilder<'a> {
    axes: &'a mut Axes,
    series: Option<BoxPlotSeries>,
}

impl<'a> BoxPlotBuilder<'a> {
    /// Set the series label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        if let Some(s) = &mut self.series {
            s.label = Some(label.into());
        }
        self
    }

    /// Set category labels.
    pub fn labels(mut self, labels: Vec<String>) -> Self {
        if let Some(s) = &mut self.series {
            s.labels = Some(labels);
        }
        self
    }

    /// Finish and add the series.
    pub fn done(mut self) -> &'a mut Axes {
        if let Some(s) = self.series.take() {
            self.axes.series.push(Box::new(s));
        }
        self.axes
    }
}


/// Builder for a heatmap series.
pub struct HeatmapBuilder<'a> {
    axes: &'a mut Axes,
    series: Option<HeatmapSeries>,
}

impl<'a> HeatmapBuilder<'a> {
    /// Set the series label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        if let Some(s) = &mut self.series {
            s.label = Some(label.into());
        }
        self
    }

    /// Enable cell value annotations.
    pub fn annotate(mut self) -> Self {
        if let Some(s) = &mut self.series {
            s.annotate = true;
        }
        self
    }

    /// Set row labels.
    pub fn row_labels(mut self, labels: Vec<String>) -> Self {
        if let Some(s) = &mut self.series {
            s.row_labels = Some(labels);
        }
        self
    }

    /// Set column labels.
    pub fn col_labels(mut self, labels: Vec<String>) -> Self {
        if let Some(s) = &mut self.series {
            s.col_labels = Some(labels);
        }
        self
    }

    /// Finish and add the series.
    pub fn done(mut self) -> &'a mut Axes {
        if let Some(s) = self.series.take() {
            self.axes.series.push(Box::new(s));
        }
        self.axes
    }
}


/// Builder for an error bar series.
pub struct ErrorBarBuilder<'a> {
    axes: &'a mut Axes,
    series: Option<ErrorBarSeries>,
}

impl<'a> ErrorBarBuilder<'a> {
    /// Set the series label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        if let Some(s) = &mut self.series {
            s.label = Some(label.into());
        }
        self
    }

    /// Set line color.
    pub fn color(mut self, color: Color) -> Self {
        if let Some(s) = &mut self.series {
            s.color = Some(color);
        }
        self
    }

    /// Finish and add the series.
    pub fn done(mut self) -> &'a mut Axes {
        if let Some(s) = self.series.take() {
            self.axes.series.push(Box::new(s));
        }
        self.axes
    }
}

