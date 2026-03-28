// SPDX-License-Identifier: MIT OR Apache-2.0
//! Stroke, fill, and text styling.

use crate::color::Color;

/// A dash pattern for stroked paths.
#[derive(Clone, Debug, PartialEq)]
pub struct DashPattern {
    /// Alternating dash/gap lengths.
    pub dashes: Vec<f64>,
    /// Offset into the pattern.
    pub offset: f64,
}

impl DashPattern {
    /// Create a new dash pattern.
    pub fn new(dashes: &[f64]) -> Self {
        Self {
            dashes: dashes.to_vec(),
            offset: 0.0,
        }
    }

    /// Format as an SVG `stroke-dasharray` attribute value.
    pub fn to_svg_string(&self) -> String {
        self.dashes
            .iter()
            .map(|d| format!("{d}"))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Stroke style for lines and shapes.
#[derive(Clone, Debug)]
pub struct Stroke {
    /// Stroke color.
    pub color: Color,
    /// Stroke width in pixels.
    pub width: f64,
    /// Optional dash pattern.
    pub dash: Option<DashPattern>,
    /// Line cap style.
    pub line_cap: LineCap,
    /// Line join style.
    pub line_join: LineJoin,
}

impl Stroke {
    /// Create a solid stroke.
    pub fn solid(color: Color, width: f64) -> Self {
        Self {
            color,
            width,
            dash: None,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
        }
    }

    /// Create a dashed stroke.
    pub fn dashed(color: Color, width: f64, dashes: &[f64]) -> Self {
        Self {
            color,
            width,
            dash: Some(DashPattern::new(dashes)),
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
        }
    }
}

impl Default for Stroke {
    fn default() -> Self {
        Self::solid(Color::BLACK, 1.0)
    }
}

/// Line cap style.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LineCap {
    /// Flat end (default).
    #[default]
    Butt,
    /// Rounded end.
    Round,
    /// Square end (extends past endpoint).
    Square,
}

impl LineCap {
    /// SVG attribute value.
    pub fn as_svg_str(self) -> &'static str {
        match self {
            Self::Butt => "butt",
            Self::Round => "round",
            Self::Square => "square",
        }
    }
}

/// Line join style.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LineJoin {
    /// Miter join (default).
    #[default]
    Miter,
    /// Rounded join.
    Round,
    /// Beveled join.
    Bevel,
}

impl LineJoin {
    /// SVG attribute value.
    pub fn as_svg_str(self) -> &'static str {
        match self {
            Self::Miter => "miter",
            Self::Round => "round",
            Self::Bevel => "bevel",
        }
    }
}

/// Fill style.
#[derive(Clone, Debug, Default)]
pub enum Fill {
    /// No fill.
    #[default]
    None,
    /// Solid color fill.
    Solid(Color),
    /// Reference a gradient by ID.
    Gradient(String),
}

impl Fill {
    /// Format as an SVG `fill` attribute value.
    pub fn to_svg_string(&self) -> String {
        match self {
            Self::None => "none".to_string(),
            Self::Solid(c) => c.to_svg_string(),
            Self::Gradient(id) => format!("url(#{id})"),
        }
    }
}

/// Text anchor position.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TextAnchor {
    /// Left-aligned (default).
    #[default]
    Start,
    /// Center-aligned.
    Middle,
    /// Right-aligned.
    End,
}

impl TextAnchor {
    /// SVG attribute value.
    pub fn as_svg_str(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Middle => "middle",
            Self::End => "end",
        }
    }
}

/// Font style for text elements.
#[derive(Clone, Debug)]
pub struct FontStyle {
    /// Font family.
    pub family: String,
    /// Font size in pixels.
    pub size: f64,
    /// Font weight (400 = normal, 700 = bold).
    pub weight: u16,
    /// Fill color.
    pub color: Color,
    /// Text anchor.
    pub anchor: TextAnchor,
}

impl FontStyle {
    /// Create a font style with sensible defaults.
    pub fn new(size: f64) -> Self {
        Self {
            family: "sans-serif".to_string(),
            size,
            weight: 400,
            color: Color::BLACK,
            anchor: TextAnchor::Start,
        }
    }
}

impl Default for FontStyle {
    fn default() -> Self {
        Self::new(12.0)
    }
}
