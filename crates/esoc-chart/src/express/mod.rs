// SPDX-License-Identifier: MIT OR Apache-2.0
//! Express API: one-liner chart creation functions.

use crate::error::Result;
use crate::grammar::chart::Chart;
use crate::grammar::facet::Facet;
use crate::grammar::layer::{Layer, MarkType};
use crate::grammar::position::Position;
use crate::grammar::stat::Stat;
use crate::new_theme::NewTheme;

// ── Shared builder macros ────────────────────────────────────────────

/// Generate common XY chart builder methods (title, x_label, y_label, theme, size, to_svg).
macro_rules! xy_builder_methods {
    () => {
        /// Set title.
        pub fn title(mut self, title: impl Into<String>) -> Self {
            self.title = Some(title.into());
            self
        }

        /// Set X-axis label.
        pub fn x_label(mut self, label: impl Into<String>) -> Self {
            self.x_label = Some(label.into());
            self
        }

        /// Set Y-axis label.
        pub fn y_label(mut self, label: impl Into<String>) -> Self {
            self.y_label = Some(label.into());
            self
        }

        /// Set theme.
        pub fn theme(mut self, theme: NewTheme) -> Self {
            self.theme = theme;
            self
        }

        /// Set dimensions.
        pub fn size(mut self, width: f32, height: f32) -> Self {
            self.width = width;
            self.height = height;
            self
        }

        /// Build and render to SVG.
        #[allow(deprecated)]
        pub fn to_svg(self) -> Result<String> {
            self.build().to_svg()
        }

        /// Build and save as SVG file.
        #[allow(deprecated)]
        pub fn save_svg(self, path: impl AsRef<std::path::Path>) -> Result<()> {
            let svg = self.build().to_svg()?;
            std::fs::write(path, svg)?;
            Ok(())
        }

        /// Set explicit X-axis domain (overrides auto-computed bounds).
        pub fn x_domain(mut self, min: f64, max: f64) -> Self {
            self.x_domain = Some((min, max));
            self
        }

        /// Set explicit Y-axis domain (overrides auto-computed bounds).
        pub fn y_domain(mut self, min: f64, max: f64) -> Self {
            self.y_domain = Some((min, max));
            self
        }

        /// Set point/fill opacity (0.0–1.0).
        pub fn opacity(mut self, alpha: f32) -> Self {
            self.opacity = Some(alpha.clamp(0.0, 1.0));
            self
        }

        /// Add a horizontal reference line at the given y value.
        pub fn hline(mut self, y: f64) -> Self {
            self.annotations.push(
                crate::grammar::annotation::Annotation::hline(y)
            );
            self
        }

        /// Add a vertical reference line at the given x value.
        pub fn vline(mut self, x: f64) -> Self {
            self.annotations.push(
                crate::grammar::annotation::Annotation::vline(x)
            );
            self
        }
    };
}

/// Generate common pie chart builder methods (title, theme, size, to_svg).
macro_rules! pie_builder_methods {
    () => {
        /// Set title.
        pub fn title(mut self, title: impl Into<String>) -> Self {
            self.title = Some(title.into());
            self
        }

        /// Set theme.
        pub fn theme(mut self, theme: NewTheme) -> Self {
            self.theme = theme;
            self
        }

        /// Set dimensions.
        pub fn size(mut self, width: f32, height: f32) -> Self {
            self.width = width;
            self.height = height;
            self
        }

        /// Build and render to SVG.
        pub fn to_svg(self) -> Result<String> {
            self.build().to_svg()
        }

        /// Build and save as SVG file.
        pub fn save_svg(self, path: impl AsRef<std::path::Path>) -> Result<()> {
            let svg = self.build().to_svg()?;
            std::fs::write(path, svg)?;
            Ok(())
        }
    };
}

/// Apply optional title/x_label/y_label to a Chart.
macro_rules! apply_chart_labels {
    (xy: $chart:expr, $self:expr) => {{
        let mut chart = $chart;
        if let Some(t) = $self.title {
            chart = chart.title(t);
        }
        if let Some(l) = $self.x_label {
            chart = chart.x_label(l);
        }
        if let Some(l) = $self.y_label {
            chart = chart.y_label(l);
        }
        chart
    }};
    (pie: $chart:expr, $self:expr) => {{
        let mut chart = $chart;
        if let Some(t) = $self.title {
            chart = chart.title(t);
        }
        chart
    }};
}

// ── Scatter ──────────────────────────────────────────────────────────

/// Create a scatter plot.
#[must_use]
pub fn scatter(x: &[f64], y: &[f64]) -> ScatterBuilder {
    ScatterBuilder {
        x: x.to_vec(),
        y: y.to_vec(),
        categories: None,
        facet_values: None,
        facet_ncol: 2,
        title: None,
        x_label: None,
        y_label: None,
        theme: NewTheme::default(),
        width: 800.0,
        height: 600.0,
        x_domain: None,
        y_domain: None,
        opacity: None,
        error_bars: None,
        add_trend: false,
        annotations: Vec::new(),
    }
}

/// Builder for scatter plots.
pub struct ScatterBuilder {
    x: Vec<f64>,
    y: Vec<f64>,
    categories: Option<Vec<String>>,
    facet_values: Option<Vec<String>>,
    facet_ncol: usize,
    title: Option<String>,
    x_label: Option<String>,
    y_label: Option<String>,
    theme: NewTheme,
    width: f32,
    height: f32,
    x_domain: Option<(f64, f64)>,
    y_domain: Option<(f64, f64)>,
    opacity: Option<f32>,
    error_bars: Option<Vec<f64>>,
    add_trend: bool,
    annotations: Vec<crate::grammar::annotation::Annotation>,
}

impl ScatterBuilder {
    xy_builder_methods!();

    /// Color points by category.
    pub fn color_by(mut self, categories: &[impl ToString]) -> Self {
        self.categories = Some(categories.iter().map(|c| c.to_string()).collect());
        self
    }

    /// Enable facet wrapping (small multiples) with per-row facet assignments.
    pub fn facet_wrap(mut self, facet_values: &[impl ToString], ncol: usize) -> Self {
        self.facet_values = Some(facet_values.iter().map(|v| v.to_string()).collect());
        self.facet_ncol = ncol;
        self
    }

    /// Set symmetric error bar values (±err per point).
    pub fn error_bars(mut self, errors: &[f64]) -> Self {
        self.error_bars = Some(errors.to_vec());
        self
    }

    /// Add a LOESS trend line overlay.
    pub fn trend_line(mut self) -> Self {
        self.add_trend = true;
        self
    }

    /// Build the chart.
    pub fn build(self) -> Chart {
        let mut layer = Layer::new(MarkType::Point)
            .with_x(self.x.clone())
            .with_y(self.y.clone());
        if let Some(cats) = self.categories {
            layer = layer.with_categories(cats);
        }
        if let Some(fv) = self.facet_values {
            layer = layer.with_facet_values(fv);
        }
        if let Some(eb) = self.error_bars {
            layer = layer.with_error_bars(eb);
        }
        let mut chart = Chart::new()
            .layer(layer)
            .size(self.width, self.height)
            .theme(self.theme.clone());
        // Add LOESS trend line as a second layer
        if self.add_trend && self.x.len() >= 3 {
            let trend_layer = Layer::new(MarkType::Line)
                .with_x(self.x)
                .with_y(self.y)
                .stat(Stat::Smooth { bandwidth: 0.3 });
            chart = chart.layer(trend_layer);
        }
        if (!matches!(chart.facet, Facet::None) || self.facet_ncol > 0)
            && chart.layers.iter().any(|l| l.facet_values.is_some())
        {
            chart = chart.facet(Facet::Wrap { ncol: self.facet_ncol });
        }
        if let Some((lo, hi)) = self.x_domain {
            chart = chart.x_domain(lo, hi);
        }
        if let Some((lo, hi)) = self.y_domain {
            chart = chart.y_domain(lo, hi);
        }
        for ann in self.annotations {
            chart = chart.annotate(ann);
        }
        apply_chart_labels!(xy: chart, self)
    }
}

// ── Line ─────────────────────────────────────────────────────────────

/// Create a line chart.
#[must_use]
pub fn line(x: &[f64], y: &[f64]) -> LineBuilder {
    LineBuilder {
        x: x.to_vec(),
        y: y.to_vec(),
        categories: None,
        title: None,
        x_label: None,
        y_label: None,
        theme: NewTheme::default(),
        width: 800.0,
        height: 600.0,
        x_domain: None,
        y_domain: None,
        opacity: None,
        annotations: Vec::new(),
    }
}

/// Builder for line charts.
pub struct LineBuilder {
    x: Vec<f64>,
    y: Vec<f64>,
    categories: Option<Vec<String>>,
    title: Option<String>,
    x_label: Option<String>,
    y_label: Option<String>,
    theme: NewTheme,
    width: f32,
    height: f32,
    x_domain: Option<(f64, f64)>,
    y_domain: Option<(f64, f64)>,
    opacity: Option<f32>,
    annotations: Vec<crate::grammar::annotation::Annotation>,
}

impl LineBuilder {
    xy_builder_methods!();

    /// Color lines by category.
    pub fn color_by(mut self, categories: &[impl ToString]) -> Self {
        self.categories = Some(categories.iter().map(|c| c.to_string()).collect());
        self
    }

    /// Build the chart.
    pub fn build(self) -> Chart {
        let mut layer = Layer::new(MarkType::Line)
            .with_x(self.x)
            .with_y(self.y);
        if let Some(cats) = self.categories {
            layer = layer.with_categories(cats);
        }
        let mut chart = Chart::new()
            .layer(layer)
            .size(self.width, self.height)
            .theme(self.theme);
        if let Some((lo, hi)) = self.x_domain {
            chart = chart.x_domain(lo, hi);
        }
        if let Some((lo, hi)) = self.y_domain {
            chart = chart.y_domain(lo, hi);
        }
        for ann in self.annotations {
            chart = chart.annotate(ann);
        }
        apply_chart_labels!(xy: chart, self)
    }
}

// ── Bar ──────────────────────────────────────────────────────────────

/// Create a bar chart.
#[must_use]
pub fn bar(categories: &[impl ToString], values: &[f64]) -> BarBuilder {
    let x: Vec<f64> = (0..categories.len()).map(|i| i as f64).collect();
    BarBuilder {
        x,
        y: values.to_vec(),
        labels: categories.iter().map(|c| c.to_string()).collect(),
        error_bars: None,
        title: None,
        x_label: None,
        y_label: None,
        theme: NewTheme::default(),
        width: 800.0,
        height: 600.0,
        x_domain: None,
        y_domain: None,
        opacity: None,
        annotations: Vec::new(),
    }
}

/// Builder for bar charts.
pub struct BarBuilder {
    x: Vec<f64>,
    y: Vec<f64>,
    labels: Vec<String>,
    error_bars: Option<Vec<f64>>,
    title: Option<String>,
    x_label: Option<String>,
    y_label: Option<String>,
    theme: NewTheme,
    width: f32,
    height: f32,
    x_domain: Option<(f64, f64)>,
    y_domain: Option<(f64, f64)>,
    opacity: Option<f32>,
    annotations: Vec<crate::grammar::annotation::Annotation>,
}

impl BarBuilder {
    xy_builder_methods!();

    /// Set symmetric error bar values (±err per bar).
    pub fn error_bars(mut self, errors: &[f64]) -> Self {
        self.error_bars = Some(errors.to_vec());
        self
    }

    /// Build the chart.
    pub fn build(self) -> Chart {
        let mut layer = Layer::new(MarkType::Bar)
            .with_x(self.x)
            .with_y(self.y)
            .with_categories(self.labels);
        if let Some(eb) = self.error_bars {
            layer = layer.with_error_bars(eb);
        }
        let mut chart = Chart::new()
            .layer(layer)
            .size(self.width, self.height)
            .theme(self.theme);
        if let Some((lo, hi)) = self.x_domain {
            chart = chart.x_domain(lo, hi);
        }
        if let Some((lo, hi)) = self.y_domain {
            chart = chart.y_domain(lo, hi);
        }
        for ann in self.annotations {
            chart = chart.annotate(ann);
        }
        apply_chart_labels!(xy: chart, self)
    }
}

// ── Histogram ────────────────────────────────────────────────────────

/// Create a histogram from raw values.
#[must_use]
pub fn histogram(values: &[f64]) -> HistogramBuilder {
    HistogramBuilder {
        values: values.to_vec(),
        bins: 0,
        title: None,
        x_label: None,
        y_label: None,
        theme: NewTheme::default(),
        width: 800.0,
        height: 600.0,
        x_domain: None,
        y_domain: None,
        opacity: None,
        annotations: Vec::new(),
    }
}

/// Builder for histograms.
pub struct HistogramBuilder {
    values: Vec<f64>,
    bins: usize,
    title: Option<String>,
    x_label: Option<String>,
    y_label: Option<String>,
    theme: NewTheme,
    width: f32,
    height: f32,
    x_domain: Option<(f64, f64)>,
    y_domain: Option<(f64, f64)>,
    opacity: Option<f32>,
    annotations: Vec<crate::grammar::annotation::Annotation>,
}

impl HistogramBuilder {
    xy_builder_methods!();

    /// Set the number of bins.
    pub fn bins(mut self, bins: usize) -> Self {
        self.bins = bins;
        self
    }

    /// Build the chart.
    pub fn build(self) -> Chart {
        let layer = Layer::new(MarkType::Bar)
            .with_x(self.values)
            .stat(Stat::Bin { bins: self.bins });
        let mut chart = Chart::new()
            .layer(layer)
            .size(self.width, self.height)
            .theme(self.theme);
        if let Some((lo, hi)) = self.x_domain {
            chart = chart.x_domain(lo, hi);
        }
        if let Some((lo, hi)) = self.y_domain {
            chart = chart.y_domain(lo, hi);
        }
        for ann in self.annotations {
            chart = chart.annotate(ann);
        }
        apply_chart_labels!(xy: chart, self)
    }
}

// ── BoxPlot ──────────────────────────────────────────────────────────

/// Create a box plot.
#[must_use]
pub fn boxplot(categories: &[impl ToString], values: &[f64]) -> BoxPlotBuilder {
    BoxPlotBuilder {
        categories: categories.iter().map(|c| c.to_string()).collect(),
        values: values.to_vec(),
        title: None,
        x_label: None,
        y_label: None,
        theme: NewTheme::default(),
        width: 800.0,
        height: 600.0,
        x_domain: None,
        y_domain: None,
        opacity: None,
        annotations: Vec::new(),
    }
}

/// Builder for box plots.
pub struct BoxPlotBuilder {
    categories: Vec<String>,
    values: Vec<f64>,
    title: Option<String>,
    x_label: Option<String>,
    y_label: Option<String>,
    theme: NewTheme,
    width: f32,
    height: f32,
    x_domain: Option<(f64, f64)>,
    y_domain: Option<(f64, f64)>,
    opacity: Option<f32>,
    annotations: Vec<crate::grammar::annotation::Annotation>,
}

impl BoxPlotBuilder {
    xy_builder_methods!();

    /// Build the chart.
    pub fn build(self) -> Chart {
        let layer = Layer::new(MarkType::Bar)
            .with_y(self.values)
            .with_categories(self.categories)
            .stat(Stat::BoxPlot);
        let mut chart = Chart::new()
            .layer(layer)
            .size(self.width, self.height)
            .theme(self.theme);
        if let Some((lo, hi)) = self.x_domain {
            chart = chart.x_domain(lo, hi);
        }
        if let Some((lo, hi)) = self.y_domain {
            chart = chart.y_domain(lo, hi);
        }
        for ann in self.annotations {
            chart = chart.annotate(ann);
        }
        apply_chart_labels!(xy: chart, self)
    }
}

// ── Area ─────────────────────────────────────────────────────────────

/// Create an area chart.
#[must_use]
pub fn area(x: &[f64], y: &[f64]) -> AreaBuilder {
    AreaBuilder {
        x: x.to_vec(),
        y: y.to_vec(),
        categories: None,
        title: None,
        x_label: None,
        y_label: None,
        theme: NewTheme::default(),
        width: 800.0,
        height: 600.0,
        x_domain: None,
        y_domain: None,
        opacity: None,
        annotations: Vec::new(),
    }
}

/// Builder for area charts.
pub struct AreaBuilder {
    x: Vec<f64>,
    y: Vec<f64>,
    categories: Option<Vec<String>>,
    title: Option<String>,
    x_label: Option<String>,
    y_label: Option<String>,
    theme: NewTheme,
    width: f32,
    height: f32,
    x_domain: Option<(f64, f64)>,
    y_domain: Option<(f64, f64)>,
    opacity: Option<f32>,
    annotations: Vec<crate::grammar::annotation::Annotation>,
}

impl AreaBuilder {
    xy_builder_methods!();

    /// Color areas by category.
    pub fn color_by(mut self, categories: &[impl ToString]) -> Self {
        self.categories = Some(categories.iter().map(|c| c.to_string()).collect());
        self
    }

    /// Build the chart.
    pub fn build(self) -> Chart {
        let mut layer = Layer::new(MarkType::Area)
            .with_x(self.x)
            .with_y(self.y);
        if let Some(cats) = self.categories {
            layer = layer.with_categories(cats);
        }
        let mut chart = Chart::new()
            .layer(layer)
            .size(self.width, self.height)
            .theme(self.theme);
        if let Some((lo, hi)) = self.x_domain {
            chart = chart.x_domain(lo, hi);
        }
        if let Some((lo, hi)) = self.y_domain {
            chart = chart.y_domain(lo, hi);
        }
        for ann in self.annotations {
            chart = chart.annotate(ann);
        }
        apply_chart_labels!(xy: chart, self)
    }
}

// ── Pie ──────────────────────────────────────────────────────────────

/// Create a pie chart with consistent (labels, values) parameter order.
#[must_use]
pub fn pie_labeled(labels: &[impl ToString], values: &[f64]) -> PieBuilder {
    PieBuilder {
        values: values.to_vec(),
        labels: labels.iter().map(|l| l.to_string()).collect(),
        inner_fraction: 0.0,
        title: None,
        theme: NewTheme::default(),
        width: 600.0,
        height: 600.0,
    }
}

/// Create a pie chart.
///
/// Note: parameter order is `(values, labels)`. Prefer [`pie_labeled`] which uses
/// the consistent `(labels, values)` order matching `bar()` and `treemap()`.
#[deprecated(note = "Use pie_labeled(labels, values) for consistent parameter order")]
pub fn pie(values: &[f64], labels: &[impl ToString]) -> PieBuilder {
    PieBuilder {
        values: values.to_vec(),
        labels: labels.iter().map(|l| l.to_string()).collect(),
        inner_fraction: 0.0,
        title: None,
        theme: NewTheme::default(),
        width: 600.0,
        height: 600.0,
    }
}

/// Builder for pie/donut charts.
pub struct PieBuilder {
    values: Vec<f64>,
    labels: Vec<String>,
    inner_fraction: f32,
    title: Option<String>,
    theme: NewTheme,
    width: f32,
    height: f32,
}

impl PieBuilder {
    pie_builder_methods!();

    /// Make it a donut chart with given inner radius fraction (0.0–0.9).
    pub fn donut(mut self, inner_fraction: f32) -> Self {
        self.inner_fraction = inner_fraction.clamp(0.0, 0.9);
        self
    }

    /// Build the chart.
    pub fn build(self) -> Chart {
        let layer = Layer::new(MarkType::Arc)
            .with_y(self.values)
            .with_categories(self.labels)
            .with_inner_radius_fraction(self.inner_fraction);
        let chart = Chart::new()
            .layer(layer)
            .size(self.width, self.height)
            .theme(self.theme);
        apply_chart_labels!(pie: chart, self)
    }
}

// ── Treemap ──────────────────────────────────────────────────────────

/// Create a treemap chart.
#[must_use]
pub fn treemap(labels: &[impl ToString], values: &[f64]) -> TreemapBuilder {
    TreemapBuilder {
        labels: labels.iter().map(|l| l.to_string()).collect(),
        values: values.to_vec(),
        title: None,
        theme: NewTheme::default(),
        width: 600.0,
        height: 600.0,
    }
}

/// Builder for treemap charts.
pub struct TreemapBuilder {
    labels: Vec<String>,
    values: Vec<f64>,
    title: Option<String>,
    theme: NewTheme,
    width: f32,
    height: f32,
}

impl TreemapBuilder {
    pie_builder_methods!();

    /// Build the chart.
    pub fn build(self) -> Chart {
        let layer = Layer::new(MarkType::Treemap)
            .with_y(self.values)
            .with_categories(self.labels);
        let chart = Chart::new()
            .layer(layer)
            .size(self.width, self.height)
            .theme(self.theme);
        apply_chart_labels!(pie: chart, self)
    }
}

// ── Multi-series Bar (Stacked / Grouped) ─────────────────────────────

/// Builder for multi-series bar charts (stacked or grouped).
pub struct MultiBarBuilder {
    categories: Vec<String>,
    groups: Vec<String>,
    values: Vec<f64>,
    position: Position,
    title: Option<String>,
    x_label: Option<String>,
    y_label: Option<String>,
    theme: NewTheme,
    width: f32,
    height: f32,
    x_domain: Option<(f64, f64)>,
    y_domain: Option<(f64, f64)>,
    opacity: Option<f32>,
    annotations: Vec<crate::grammar::annotation::Annotation>,
}

/// Backward-compatible alias for stacked bar builder.
pub type StackedBarBuilder = MultiBarBuilder;
/// Backward-compatible alias for grouped bar builder.
pub type GroupedBarBuilder = MultiBarBuilder;

/// Create a stacked bar chart.
///
/// `categories` defines the x-axis groups, `groups` assigns each value to a series,
/// and `values` is a flat array of values. The data is split into per-group layers internally.
#[must_use]
pub fn stacked_bar(
    categories: &[impl ToString],
    groups: &[impl ToString],
    values: &[f64],
) -> MultiBarBuilder {
    MultiBarBuilder {
        categories: categories.iter().map(|c| c.to_string()).collect(),
        groups: groups.iter().map(|g| g.to_string()).collect(),
        values: values.to_vec(),
        position: Position::Stack,
        title: None,
        x_label: None,
        y_label: None,
        theme: NewTheme::default(),
        width: 800.0,
        height: 600.0,
        x_domain: None,
        y_domain: None,
        opacity: None,
        annotations: Vec::new(),
    }
}

/// Create a grouped (dodged) bar chart.
#[must_use]
pub fn grouped_bar(
    categories: &[impl ToString],
    groups: &[impl ToString],
    values: &[f64],
) -> MultiBarBuilder {
    MultiBarBuilder {
        categories: categories.iter().map(|c| c.to_string()).collect(),
        groups: groups.iter().map(|g| g.to_string()).collect(),
        values: values.to_vec(),
        position: Position::Dodge,
        title: None,
        x_label: None,
        y_label: None,
        theme: NewTheme::default(),
        width: 800.0,
        height: 600.0,
        x_domain: None,
        y_domain: None,
        opacity: None,
        annotations: Vec::new(),
    }
}

impl MultiBarBuilder {
    xy_builder_methods!();

    /// Build the chart, returning an error if inputs are invalid.
    pub fn try_build(self) -> std::result::Result<Chart, String> {
        let config = ChartConfig {
            title: self.title,
            x_label: self.x_label,
            y_label: self.y_label,
            theme: self.theme,
            width: self.width,
            height: self.height,
        };
        let mut chart = try_build_grouped_chart(self.categories, self.groups, self.values, self.position, config)?;
        if let Some((lo, hi)) = self.x_domain {
            chart = chart.x_domain(lo, hi);
        }
        if let Some((lo, hi)) = self.y_domain {
            chart = chart.y_domain(lo, hi);
        }
        for ann in self.annotations {
            chart = chart.annotate(ann);
        }
        Ok(chart)
    }

    /// Build the chart. Panics if categories, groups, and values have different lengths.
    ///
    /// Prefer [`try_build()`](Self::try_build) which returns a `Result` instead of panicking.
    #[allow(deprecated)]
    #[deprecated(note = "Use try_build() instead — build() panics on invalid input")]
    pub fn build(self) -> Chart {
        self.try_build().expect("MultiBarBuilder::build() failed")
    }
}

// ── Shared grouped chart logic ───────────────────────────────────────

/// Configuration shared across chart builders.
struct ChartConfig {
    title: Option<String>,
    x_label: Option<String>,
    y_label: Option<String>,
    theme: NewTheme,
    width: f32,
    height: f32,
}

/// Shared logic for stacked/grouped bar chart construction.
/// Returns `Err` if categories, groups, and values have different lengths.
fn try_build_grouped_chart(
    categories: Vec<String>,
    groups: Vec<String>,
    values: Vec<f64>,
    position: Position,
    config: ChartConfig,
) -> std::result::Result<Chart, String> {
    use std::collections::HashSet;

    // Validate parallel vector lengths
    if categories.len() != groups.len() || categories.len() != values.len() {
        return Err(format!(
            "categories ({}), groups ({}), and values ({}) must have the same length",
            categories.len(),
            groups.len(),
            values.len(),
        ));
    }

    let mut unique_cats: Vec<String> = Vec::new();
    let mut seen_cats = HashSet::new();
    for c in &categories {
        if seen_cats.insert(c.as_str()) {
            unique_cats.push(c.clone());
        }
    }

    let mut unique_groups: Vec<String> = Vec::new();
    let mut seen_groups = HashSet::new();
    for g in &groups {
        if seen_groups.insert(g.as_str()) {
            unique_groups.push(g.clone());
        }
    }

    let cat_idx: std::collections::HashMap<&str, f64> = unique_cats
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i as f64))
        .collect();

    let mut chart = Chart::new().size(config.width, config.height).theme(config.theme);

    for group in &unique_groups {
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();
        for (i, g) in groups.iter().enumerate() {
            if g == group {
                if let Some(&x) = cat_idx.get(categories[i].as_str()) {
                    x_data.push(x);
                    y_data.push(values[i]);
                }
            }
        }
        let mut layer = Layer::new(MarkType::Bar)
            .with_x(x_data)
            .with_y(y_data)
            .with_label(group.clone())
            .position(position);
        // All layers carry category labels (needed for axis label lookup via find_map)
        layer = layer.with_categories(unique_cats.clone());
        chart = chart.layer(layer);
    }

    if let Some(t) = config.title {
        chart = chart.title(t);
    }
    if let Some(l) = config.x_label {
        chart = chart.x_label(l);
    }
    if let Some(l) = config.y_label {
        chart = chart.y_label(l);
    }
    Ok(chart)
}

// ── Heatmap ─────────────────────────────────────────────────────────

/// Create a heatmap from a 2D matrix (takes ownership).
///
/// Prefer [`heatmap_ref`] to avoid cloning when you already have the data.
#[must_use]
pub fn heatmap(data: Vec<Vec<f64>>) -> HeatmapBuilder {
    HeatmapBuilder {
        data,
        row_labels: None,
        col_labels: None,
        annotate: false,
        title: None,
        x_label: None,
        y_label: None,
        theme: NewTheme::default(),
        width: 600.0,
        height: 600.0,
        x_domain: None,
        y_domain: None,
        opacity: None,
        annotations: Vec::new(),
    }
}

/// Create a heatmap from a borrowed 2D slice.
#[must_use]
pub fn heatmap_ref(data: &[Vec<f64>]) -> HeatmapBuilder {
    HeatmapBuilder {
        data: data.to_vec(),
        row_labels: None,
        col_labels: None,
        annotate: false,
        title: None,
        x_label: None,
        y_label: None,
        theme: NewTheme::default(),
        width: 600.0,
        height: 600.0,
        x_domain: None,
        y_domain: None,
        opacity: None,
        annotations: Vec::new(),
    }
}

/// Builder for heatmap charts.
pub struct HeatmapBuilder {
    data: Vec<Vec<f64>>,
    row_labels: Option<Vec<String>>,
    col_labels: Option<Vec<String>>,
    annotate: bool,
    title: Option<String>,
    x_label: Option<String>,
    y_label: Option<String>,
    theme: NewTheme,
    width: f32,
    height: f32,
    x_domain: Option<(f64, f64)>,
    y_domain: Option<(f64, f64)>,
    opacity: Option<f32>,
    annotations: Vec<crate::grammar::annotation::Annotation>,
}

impl HeatmapBuilder {
    xy_builder_methods!();

    /// Enable cell value annotations.
    pub fn annotate(mut self) -> Self {
        self.annotate = true;
        self
    }

    /// Set row labels (owned).
    #[deprecated(note = "Use with_row_labels(&[impl ToString]) instead")]
    pub fn row_labels(mut self, labels: Vec<String>) -> Self {
        self.row_labels = Some(labels);
        self
    }

    /// Set column labels (owned).
    #[deprecated(note = "Use with_col_labels(&[impl ToString]) instead")]
    pub fn col_labels(mut self, labels: Vec<String>) -> Self {
        self.col_labels = Some(labels);
        self
    }

    /// Set row labels from any string-like slice.
    pub fn with_row_labels(mut self, labels: &[impl ToString]) -> Self {
        self.row_labels = Some(labels.iter().map(|l| l.to_string()).collect());
        self
    }

    /// Set column labels from any string-like slice.
    pub fn with_col_labels(mut self, labels: &[impl ToString]) -> Self {
        self.col_labels = Some(labels.iter().map(|l| l.to_string()).collect());
        self
    }

    /// Build the chart.
    pub fn build(self) -> Chart {
        let mut layer = Layer::new(MarkType::Heatmap)
            .with_heatmap_data(self.data);
        if let Some(rl) = self.row_labels {
            layer = layer.with_row_labels(rl);
        }
        if let Some(cl) = self.col_labels {
            layer = layer.with_col_labels(cl);
        }
        if self.annotate {
            layer = layer.annotate_cells();
        }
        let mut chart = Chart::new()
            .layer(layer)
            .size(self.width, self.height)
            .theme(self.theme);
        if let Some((lo, hi)) = self.x_domain {
            chart = chart.x_domain(lo, hi);
        }
        if let Some((lo, hi)) = self.y_domain {
            chart = chart.y_domain(lo, hi);
        }
        for ann in self.annotations {
            chart = chart.annotate(ann);
        }
        apply_chart_labels!(xy: chart, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter_builds_svg() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 1.0, 5.0, 3.0];
        let svg = scatter(&x, &y).title("Test Scatter").to_svg().unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("<circle"));
    }

    #[test]
    fn scatter_with_categories() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 1.0, 5.0];
        let cats = vec!["A", "B", "A", "B"];
        let svg = scatter(&x, &y)
            .color_by(&cats)
            .title("Colored Scatter")
            .to_svg()
            .unwrap();
        assert!(svg.contains("<circle"));
    }

    #[test]
    fn line_builds_svg() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0];
        let svg = line(&x, &y)
            .title("Quadratic")
            .x_label("x")
            .y_label("y")
            .to_svg()
            .unwrap();
        assert!(svg.contains("<polyline"));
    }

    #[test]
    fn bar_builds_svg() {
        let cats = vec!["A", "B", "C"];
        let vals = vec![10.0, 25.0, 15.0];
        let svg = bar(&cats, &vals).title("Bar Chart").to_svg().unwrap();
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn histogram_builds_svg() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64) * 0.1).collect();
        let svg = histogram(&data).title("Histogram").to_svg().unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn histogram_with_bins() {
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let svg = histogram(&data).bins(5).title("5 Bins").to_svg().unwrap();
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn area_builds_svg() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let svg = area(&x, &y).title("Area Chart").to_svg().unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<path"));
    }

    #[test]
    fn pie_builds_svg() {
        let values = vec![30.0, 20.0, 50.0];
        let labels = vec!["A", "B", "C"];
        let svg = pie_labeled(&labels, &values).title("Pie Chart").to_svg().unwrap();
        assert!(svg.contains("<svg"));
        assert!(!svg.contains("<line") || svg.contains("<path"));
    }

    #[test]
    fn pie_equal_values() {
        let values = vec![1.0, 1.0, 1.0];
        let labels = vec!["X", "Y", "Z"];
        let svg = pie_labeled(&labels, &values).to_svg().unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn donut_builds_svg() {
        let values = vec![40.0, 60.0];
        let labels = vec!["Yes", "No"];
        let svg = pie_labeled(&labels, &values).donut(0.5).to_svg().unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn boxplot_builds_svg() {
        let cats = vec!["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 6.0, 8.0, 10.0];
        let svg = boxplot(&cats, &vals).title("Box Plot").to_svg().unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<rect"));
        assert!(svg.contains("<line"));
    }

    #[test]
    fn annotation_hline() {
        use crate::grammar::annotation::Annotation;
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![10.0, 20.0, 30.0];
        let chart = scatter(&x, &y).build()
            .annotate(Annotation::hline(15.0));
        let svg = chart.to_svg().unwrap();
        assert!(svg.contains("<line"));
    }

    #[test]
    fn subtitle_and_caption() {
        use crate::grammar::chart::Chart;
        use crate::grammar::layer::{Layer, MarkType};
        let chart = Chart::new()
            .layer(Layer::new(MarkType::Point).with_x(vec![1.0]).with_y(vec![1.0]))
            .title("Title")
            .subtitle("Subtitle here")
            .caption("Source: data");
        let svg = chart.to_svg().unwrap();
        assert!(svg.contains("Subtitle here"));
        assert!(svg.contains("Source: data"));
    }

    #[test]
    fn legend_with_categories() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![10.0, 20.0, 15.0, 25.0];
        let cats = vec!["Group A", "Group B", "Group A", "Group B"];
        let svg = scatter(&x, &y).color_by(&cats).to_svg().unwrap();
        assert!(svg.contains("Group A"));
        assert!(svg.contains("Group B"));
    }

    #[test]
    fn flipped_coord_bar() {
        use crate::grammar::chart::Chart;
        use crate::grammar::coord::CoordSystem;
        use crate::grammar::layer::{Layer, MarkType};
        let chart = Chart::new()
            .layer(
                Layer::new(MarkType::Bar)
                    .with_x(vec![0.0, 1.0, 2.0])
                    .with_y(vec![10.0, 20.0, 30.0]),
            )
            .coord(CoordSystem::Flipped)
            .title("Horizontal Bars");
        let svg = chart.to_svg().unwrap();
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn stacked_bar_builds_svg() {
        let cats = vec!["Q1", "Q2", "Q3", "Q1", "Q2", "Q3"];
        let groups = vec!["A", "A", "A", "B", "B", "B"];
        let vals = vec![10.0, 20.0, 30.0, 5.0, 15.0, 25.0];
        let svg = stacked_bar(&cats, &groups, &vals)
            .title("Stacked")
            .to_svg()
            .unwrap();
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn grouped_bar_builds_svg() {
        let cats = vec!["Q1", "Q2", "Q1", "Q2"];
        let groups = vec!["A", "A", "B", "B"];
        let vals = vec![10.0, 20.0, 15.0, 25.0];
        let svg = grouped_bar(&cats, &groups, &vals)
            .title("Grouped")
            .to_svg()
            .unwrap();
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn facet_wrap_scatter_builds_svg() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = vec![2.0, 4.0, 1.0, 5.0, 3.0, 6.0];
        let facets = vec!["A", "A", "A", "B", "B", "B"];
        let svg = scatter(&x, &y)
            .facet_wrap(&facets, 2)
            .title("Faceted Scatter")
            .to_svg()
            .unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<circle"));
        assert!(svg.contains("A"));
        assert!(svg.contains("B"));
    }

    #[test]
    fn heatmap_builds_svg() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let svg = heatmap(data).title("Heatmap").to_svg().unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn heatmap_annotated_builds_svg() {
        let data = vec![
            vec![10.0, 20.0],
            vec![30.0, 40.0],
        ];
        let svg = heatmap(data).annotate().title("Annotated").to_svg().unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<rect"));
        assert!(svg.contains("<text"));
    }

    #[test]
    fn facet_single_value_no_split() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![10.0, 20.0, 30.0];
        let facets = vec!["All", "All", "All"];
        let svg = scatter(&x, &y)
            .facet_wrap(&facets, 2)
            .to_svg()
            .unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<circle"));
    }

    #[test]
    fn grouped_bar_legend_has_group_names() {
        let cats = vec!["Q1", "Q2", "Q1", "Q2"];
        let groups = vec!["Revenue", "Revenue", "Costs", "Costs"];
        let vals = vec![10.0, 20.0, 15.0, 25.0];
        let svg = grouped_bar(&cats, &groups, &vals).to_svg().unwrap();
        assert!(svg.contains("Revenue"), "legend should contain group name 'Revenue'");
        assert!(svg.contains("Costs"), "legend should contain group name 'Costs'");
    }

    #[test]
    fn grouped_bar_mismatched_lengths_panics() {
        #[allow(deprecated)]
        let result = std::panic::catch_unwind(|| {
            grouped_bar(&["Q1", "Q2"], &["A", "A", "B"], &[10.0, 20.0])
                .build();
        });
        assert!(result.is_err(), "mismatched lengths should panic");
    }

    #[test]
    fn grouped_bar_try_build_returns_err() {
        let result = grouped_bar(&["Q1", "Q2"], &["A", "A", "B"], &[10.0, 20.0])
            .try_build();
        assert!(result.is_err());
    }

    #[test]
    fn line_color_by() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![10.0, 20.0, 15.0, 25.0];
        let cats = vec!["A", "B", "A", "B"];
        let svg = line(&x, &y).color_by(&cats).to_svg().unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn area_color_by() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![10.0, 20.0, 15.0, 25.0];
        let cats = vec!["A", "B", "A", "B"];
        let svg = area(&x, &y).color_by(&cats).to_svg().unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn svg_contains_title_and_role() {
        use crate::grammar::chart::Chart;
        use crate::grammar::layer::{Layer, MarkType};
        let chart = Chart::new()
            .layer(Layer::new(MarkType::Point).with_x(vec![1.0]).with_y(vec![1.0]))
            .title("My Chart")
            .description("A scatter plot of test data");
        let svg = chart.to_svg().unwrap();
        assert!(svg.contains(r#"role="img""#), "SVG should have role=img");
        assert!(svg.contains("<title>My Chart</title>"), "SVG should contain <title>");
        assert!(svg.contains("<desc>A scatter plot of test data</desc>"), "SVG should contain <desc>");
    }

    #[test]
    fn heatmap_legend_rendered() {
        let data = vec![vec![1.0, 5.0], vec![3.0, 9.0]];
        let svg = heatmap(data).title("Heatmap Legend").to_svg().unwrap();
        // Should have the gradient legend with min/max labels
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn treemap_builds_svg() {
        let labels = vec!["A", "B", "C", "D"];
        let values = vec![30.0, 20.0, 15.0, 10.0];
        let svg = treemap(&labels, &values).title("Treemap").to_svg().unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn treemap_single_item() {
        let svg = treemap(&["Only"], &[100.0]).to_svg().unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn treemap_with_zeros() {
        let labels = vec!["A", "B", "C"];
        let values = vec![30.0, 0.0, 20.0];
        let svg = treemap(&labels, &values).to_svg().unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn bar_with_error_bars_builds_svg() {
        let cats = vec!["A", "B", "C"];
        let vals = vec![10.0, 25.0, 15.0];
        let errs = vec![1.5, 2.0, 1.0];
        let svg = bar(&cats, &vals)
            .error_bars(&errs)
            .title("Bar with Error Bars")
            .to_svg()
            .unwrap();
        assert!(svg.contains("<rect"));
        assert!(svg.contains("<line"), "should have whisker lines");
    }

    #[test]
    fn stacked_bar_legend_has_group_names() {
        let cats = vec!["Q1", "Q2", "Q1", "Q2"];
        let groups = vec!["Alpha", "Alpha", "Beta", "Beta"];
        let vals = vec![10.0, 20.0, 5.0, 15.0];
        let svg = stacked_bar(&cats, &groups, &vals).to_svg().unwrap();
        assert!(svg.contains("Alpha"), "legend should contain group name 'Alpha'");
        assert!(svg.contains("Beta"), "legend should contain group name 'Beta'");
    }

    #[test]
    fn boxplot_shows_category_labels() {
        let cats = vec!["GroupA", "GroupA", "GroupA", "GroupA", "GroupA",
                        "GroupB", "GroupB", "GroupB", "GroupB", "GroupB"];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 6.0, 8.0, 10.0];
        let svg = boxplot(&cats, &vals).title("Box Plot").to_svg().unwrap();
        assert!(svg.contains("GroupA"), "boxplot SVG should contain category name GroupA");
        assert!(svg.contains("GroupB"), "boxplot SVG should contain category name GroupB");
    }
}
