// SPDX-License-Identifier: MIT OR Apache-2.0
//! A chart layer: mark type + encodings + optional stat.

use crate::grammar::encoding::Encoding;
use crate::grammar::position::Position;
use crate::grammar::stat::Stat;

/// Mark type for a layer.
#[derive(Clone, Copy, Debug)]
pub enum MarkType {
    /// Points/scatter.
    Point,
    /// Lines.
    Line,
    /// Bars.
    Bar,
    /// Area fill.
    Area,
    /// Text labels.
    Text,
    /// Arc/pie.
    Arc,
    /// Rules (reference lines).
    Rule,
    /// Heatmap (2D matrix visualization).
    Heatmap,
    /// Treemap (hierarchical area chart).
    Treemap,
}

/// A single chart layer: a mark type with visual encodings.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Layer {
    /// The mark type.
    pub mark: MarkType,
    /// Encodings (visual channels mapped to data).
    pub encodings: Vec<Encoding>,
    /// Statistical transform.
    pub stat: Stat,
    /// X data (pre-resolved for express API).
    pub x_data: Option<Vec<f64>>,
    /// Y data (pre-resolved for express API).
    pub y_data: Option<Vec<f64>>,
    /// Category labels (for color encoding).
    pub categories: Option<Vec<String>>,
    /// Inner radius fraction for donut charts (0.0 = pie, >0 = donut).
    pub inner_radius_fraction: f32,
    /// Position adjustment.
    pub position: Position,
    /// Facet assignment per data row.
    pub facet_values: Option<Vec<String>>,
    /// Heatmap data (row-major 2D matrix).
    pub heatmap_data: Option<Vec<Vec<f64>>>,
    /// Row labels for heatmap.
    pub row_labels: Option<Vec<String>>,
    /// Column labels for heatmap.
    pub col_labels: Option<Vec<String>>,
    /// Whether to annotate heatmap cells with values.
    pub annotate_cells: bool,
    /// Human-readable label for this layer (used in legends).
    pub label: Option<String>,
    /// Symmetric error bar values (±err per data point).
    pub error_bars: Option<Vec<f64>>,
}

impl Layer {
    /// Create a new layer with a mark type.
    pub fn new(mark: MarkType) -> Self {
        Self {
            mark,
            encodings: Vec::new(),
            stat: Stat::Identity,
            x_data: None,
            y_data: None,
            categories: None,
            inner_radius_fraction: 0.0,
            position: Position::default(),
            facet_values: None,
            heatmap_data: None,
            row_labels: None,
            col_labels: None,
            annotate_cells: false,
            label: None,
            error_bars: None,
        }
    }

    /// Set a human-readable label for this layer (used in legends).
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Add an X encoding.
    #[deprecated(
        note = "Encoding-based API is not yet wired into the compile pipeline. Use with_x/with_y/with_categories instead."
    )]
    pub fn encode_x(mut self, mut enc: Encoding) -> Self {
        enc.channel = crate::grammar::encoding::Channel::X;
        self.encodings.push(enc);
        self
    }

    /// Add a Y encoding.
    #[deprecated(
        note = "Encoding-based API is not yet wired into the compile pipeline. Use with_x/with_y/with_categories instead."
    )]
    pub fn encode_y(mut self, mut enc: Encoding) -> Self {
        enc.channel = crate::grammar::encoding::Channel::Y;
        self.encodings.push(enc);
        self
    }

    /// Add a color encoding.
    #[deprecated(
        note = "Encoding-based API is not yet wired into the compile pipeline. Use with_x/with_y/with_categories instead."
    )]
    pub fn encode_color(mut self, mut enc: Encoding) -> Self {
        enc.channel = crate::grammar::encoding::Channel::Color;
        self.encodings.push(enc);
        self
    }

    /// Add a size encoding.
    #[deprecated(
        note = "Encoding-based API is not yet wired into the compile pipeline. Use with_x/with_y/with_categories instead."
    )]
    pub fn encode_size(mut self, mut enc: Encoding) -> Self {
        enc.channel = crate::grammar::encoding::Channel::Size;
        self.encodings.push(enc);
        self
    }

    /// Set the statistical transform.
    pub fn stat(mut self, stat: Stat) -> Self {
        self.stat = stat;
        self
    }

    /// Set pre-resolved X data.
    pub fn with_x(mut self, x: Vec<f64>) -> Self {
        self.x_data = Some(x);
        self
    }

    /// Set pre-resolved Y data.
    pub fn with_y(mut self, y: Vec<f64>) -> Self {
        self.y_data = Some(y);
        self
    }

    /// Set category labels.
    pub fn with_categories(mut self, cats: Vec<String>) -> Self {
        self.categories = Some(cats);
        self
    }

    /// Set inner radius fraction for donut charts.
    pub fn with_inner_radius_fraction(mut self, fraction: f32) -> Self {
        self.inner_radius_fraction = fraction;
        self
    }

    /// Set position adjustment.
    pub fn position(mut self, position: Position) -> Self {
        self.position = position;
        self
    }

    /// Set facet values (per-row facet assignment).
    pub fn with_facet_values(mut self, facet_values: Vec<String>) -> Self {
        self.facet_values = Some(facet_values);
        self
    }

    /// Set heatmap data (row-major 2D matrix).
    pub fn with_heatmap_data(mut self, data: Vec<Vec<f64>>) -> Self {
        self.heatmap_data = Some(data);
        self
    }

    /// Set row labels for heatmap.
    pub fn with_row_labels(mut self, labels: Vec<String>) -> Self {
        self.row_labels = Some(labels);
        self
    }

    /// Set column labels for heatmap.
    pub fn with_col_labels(mut self, labels: Vec<String>) -> Self {
        self.col_labels = Some(labels);
        self
    }

    /// Enable cell value annotations for heatmap.
    pub fn annotate_cells(mut self) -> Self {
        self.annotate_cells = true;
        self
    }

    /// Set symmetric error bar values (±err per data point).
    pub fn with_error_bars(mut self, errors: Vec<f64>) -> Self {
        self.error_bars = Some(errors);
        self
    }
}
