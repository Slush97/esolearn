// SPDX-License-Identifier: MIT OR Apache-2.0
//! Chart annotations: reference lines, bands, text labels.

use esoc_color::Color;

/// An annotation overlaid on the plot area.
#[derive(Clone, Debug)]
pub enum Annotation {
    /// Horizontal reference line at a given y value.
    HLine {
        /// Y data value.
        y: f64,
        /// Line color.
        color: Color,
        /// Line width.
        width: f32,
        /// Dash pattern (None = solid).
        dash: Option<Vec<f32>>,
        /// Optional label.
        label: Option<String>,
    },
    /// Vertical reference line at a given x value.
    VLine {
        /// X data value.
        x: f64,
        /// Line color.
        color: Color,
        /// Line width.
        width: f32,
        /// Dash pattern (None = solid).
        dash: Option<Vec<f32>>,
        /// Optional label.
        label: Option<String>,
    },
    /// Horizontal band between y_min and y_max.
    Band {
        /// Lower y value.
        y_min: f64,
        /// Upper y value.
        y_max: f64,
        /// Fill color (typically semi-transparent).
        color: Color,
    },
    /// Free-form text annotation.
    Text {
        /// X data value.
        x: f64,
        /// Y data value.
        y: f64,
        /// Text content.
        text: String,
        /// Text color.
        color: Color,
        /// Font size.
        font_size: f32,
    },
}

/// Convenience constructors.
impl Annotation {
    /// Create a horizontal reference line.
    pub fn hline(y: f64) -> Self {
        Self::HLine {
            y,
            color: Color::new(0.5, 0.5, 0.5, 1.0),
            width: 1.0,
            dash: Some(vec![4.0, 4.0]),
            label: None,
        }
    }

    /// Create a vertical reference line.
    pub fn vline(x: f64) -> Self {
        Self::VLine {
            x,
            color: Color::new(0.5, 0.5, 0.5, 1.0),
            width: 1.0,
            dash: Some(vec![4.0, 4.0]),
            label: None,
        }
    }

    /// Create a horizontal band.
    pub fn band(y_min: f64, y_max: f64) -> Self {
        Self::Band {
            y_min,
            y_max,
            color: Color::new(0.5, 0.5, 0.5, 0.15),
        }
    }

    /// Create a text annotation.
    pub fn text(x: f64, y: f64, text: impl Into<String>) -> Self {
        Self::Text {
            x,
            y,
            text: text.into(),
            color: Color::BLACK,
            font_size: 11.0,
        }
    }

    /// Set the color of this annotation.
    pub fn with_color(mut self, color: Color) -> Self {
        match &mut self {
            Self::HLine { color: c, .. }
            | Self::VLine { color: c, .. }
            | Self::Band { color: c, .. }
            | Self::Text { color: c, .. } => *c = color,
        }
        self
    }

    /// Set the label for reference lines.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        match &mut self {
            Self::HLine { label: l, .. } | Self::VLine { label: l, .. } => {
                *l = Some(label.into());
            }
            _ => {}
        }
        self
    }
}
