// SPDX-License-Identifier: MIT OR Apache-2.0
//! Visual styles for marks: stroke, fill, font, marker shapes.

use esoc_color::Color;

/// Line stroke style.
#[derive(Clone, Debug)]
pub struct StrokeStyle {
    /// Stroke color.
    pub color: Color,
    /// Line width in pixels.
    pub width: f32,
    /// Dash pattern (lengths of dash, gap, dash, gap, ...).
    pub dash: Vec<f32>,
    /// Dash offset.
    pub dash_offset: f32,
    /// Line cap style.
    pub line_cap: LineCap,
    /// Line join style.
    pub line_join: LineJoin,
}

impl Default for StrokeStyle {
    fn default() -> Self {
        Self {
            color: Color::BLACK,
            width: 1.0,
            dash: Vec::new(),
            dash_offset: 0.0,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
        }
    }
}

impl StrokeStyle {
    /// Create a solid stroke.
    pub fn solid(color: Color, width: f32) -> Self {
        Self {
            color,
            width,
            ..Default::default()
        }
    }

    /// Whether this stroke is invisible.
    pub fn is_none(&self) -> bool {
        self.width <= 0.0 || self.color.a <= 0.0
    }
}

/// Line cap style.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LineCap {
    /// Flat end at the endpoint.
    #[default]
    Butt,
    /// Rounded end.
    Round,
    /// Square end extending past the endpoint.
    Square,
}

/// Line join style.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LineJoin {
    /// Sharp corner.
    #[default]
    Miter,
    /// Rounded corner.
    Round,
    /// Beveled corner.
    Bevel,
}

/// Fill style.
#[derive(Clone, Debug, Default)]
pub enum FillStyle {
    /// No fill.
    #[default]
    None,
    /// Solid color fill.
    Solid(Color),
}

impl FillStyle {
    /// Get the color if solid.
    pub fn color(&self) -> Option<Color> {
        match self {
            Self::None => None,
            Self::Solid(c) => Some(*c),
        }
    }

    /// Whether this is no fill.
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

impl From<Color> for FillStyle {
    fn from(c: Color) -> Self {
        Self::Solid(c)
    }
}

/// Font style for text marks.
#[derive(Clone, Debug)]
pub struct FontStyle {
    /// Font family name.
    pub family: String,
    /// Font size in pixels.
    pub size: f32,
    /// Font weight (400 = normal, 700 = bold).
    pub weight: u16,
    /// Whether to use italic style.
    pub italic: bool,
}

impl Default for FontStyle {
    fn default() -> Self {
        Self {
            family: "sans-serif".to_string(),
            size: 12.0,
            weight: 400,
            italic: false,
        }
    }
}

impl FontStyle {
    /// Bold variant.
    pub fn bold(mut self) -> Self {
        self.weight = 700;
        self
    }

    /// Set size.
    pub fn with_size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }
}

/// Marker shapes for point marks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MarkerShape {
    /// Circle marker.
    #[default]
    Circle,
    /// Square marker.
    Square,
    /// Diamond (rotated square).
    Diamond,
    /// Upward triangle.
    TriangleUp,
    /// Downward triangle.
    TriangleDown,
    /// Cross (+).
    Cross,
    /// Star.
    Star,
    /// Plus (same as cross but thinner).
    Plus,
}

impl MarkerShape {
    /// Shape type index for GPU shader dispatch.
    pub fn type_index(self) -> u32 {
        match self {
            Self::Circle => 0,
            Self::Square => 1,
            Self::Diamond => 2,
            Self::TriangleUp => 3,
            Self::TriangleDown => 4,
            Self::Cross => 5,
            Self::Star => 6,
            Self::Plus => 7,
        }
    }
}
