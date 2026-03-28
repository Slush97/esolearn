// SPDX-License-Identifier: MIT OR Apache-2.0
//! SVG elements that make up a canvas.

use crate::color::Color;
use crate::geom::{Point, Rect};
use crate::layer::Layer;
use crate::path::PathData;
use crate::style::{DashPattern, Fill, FontStyle, Stroke};

/// A visual element on the canvas.
#[derive(Clone, Debug)]
pub struct DrawElement {
    /// The element kind.
    pub kind: Element,
    /// Which layer this element belongs to.
    pub layer: Layer,
}

/// Visual element types.
#[derive(Clone, Debug)]
pub enum Element {
    /// A straight line segment.
    Line {
        /// Start point.
        start: Point,
        /// End point.
        end: Point,
        /// Stroke style.
        stroke: Stroke,
    },

    /// A polyline (connected line segments).
    Polyline {
        /// Points along the polyline.
        points: Vec<Point>,
        /// Stroke style.
        stroke: Stroke,
        /// Optional fill.
        fill: Fill,
    },

    /// A rectangle.
    Rect {
        /// Rectangle bounds.
        rect: Rect,
        /// Fill style.
        fill: Fill,
        /// Optional stroke.
        stroke: Option<Stroke>,
        /// Corner radius for rounded rectangles.
        rx: f64,
    },

    /// A circle.
    Circle {
        /// Center point.
        center: Point,
        /// Radius.
        radius: f64,
        /// Fill style.
        fill: Fill,
        /// Optional stroke.
        stroke: Option<Stroke>,
    },

    /// An SVG path.
    Path {
        /// Path data.
        data: PathData,
        /// Fill style.
        fill: Fill,
        /// Optional stroke.
        stroke: Option<Stroke>,
    },

    /// A text label.
    Text {
        /// Anchor point (position depends on text-anchor).
        position: Point,
        /// Text content.
        content: String,
        /// Font style.
        font: FontStyle,
        /// Optional rotation in degrees (around the anchor point).
        rotation: Option<f64>,
    },

    /// A group of elements (used for clipping, transforms).
    Group {
        /// Child elements.
        children: Vec<DrawElement>,
        /// Optional clip rectangle.
        clip: Option<Rect>,
    },
}

/// A linear gradient definition.
#[derive(Clone, Debug)]
pub struct LinearGradient {
    /// Unique ID for referencing.
    pub id: String,
    /// Start X (0.0 to 1.0).
    pub x1: f64,
    /// Start Y.
    pub y1: f64,
    /// End X.
    pub x2: f64,
    /// End Y.
    pub y2: f64,
    /// Color stops: (offset 0.0–1.0, color).
    pub stops: Vec<(f64, Color)>,
}

/// A clipping rectangle definition.
#[derive(Clone, Debug)]
pub struct ClipDef {
    /// Unique ID for referencing.
    pub id: String,
    /// Clip bounds.
    pub rect: Rect,
}

impl DrawElement {
    /// Create a new element on the given layer.
    pub fn new(kind: Element, layer: Layer) -> Self {
        Self { kind, layer }
    }
}

// --- Convenience constructors ---

impl Element {
    /// Create a line element.
    pub fn line(x1: f64, y1: f64, x2: f64, y2: f64, stroke: Stroke) -> Self {
        Self::Line {
            start: Point::new(x1, y1),
            end: Point::new(x2, y2),
            stroke,
        }
    }

    /// Create a filled rectangle.
    pub fn filled_rect(rect: Rect, fill: Fill) -> Self {
        Self::Rect {
            rect,
            fill,
            stroke: None,
            rx: 0.0,
        }
    }

    /// Create a circle.
    pub fn circle(cx: f64, cy: f64, r: f64, fill: Fill) -> Self {
        Self::Circle {
            center: Point::new(cx, cy),
            radius: r,
            fill,
            stroke: None,
        }
    }

    /// Create a text element.
    pub fn text(x: f64, y: f64, content: impl Into<String>, font: FontStyle) -> Self {
        Self::Text {
            position: Point::new(x, y),
            content: content.into(),
            font,
            rotation: None,
        }
    }

    /// Create a polyline.
    pub fn polyline(points: Vec<Point>, stroke: Stroke) -> Self {
        Self::Polyline {
            points,
            stroke,
            fill: Fill::None,
        }
    }
}

impl DrawElement {
    /// Shorthand: line on a given layer.
    pub fn line(x1: f64, y1: f64, x2: f64, y2: f64, stroke: Stroke, layer: Layer) -> Self {
        Self::new(Element::line(x1, y1, x2, y2, stroke), layer)
    }

    /// Shorthand: filled rect on a given layer.
    pub fn filled_rect(rect: Rect, fill: Fill, layer: Layer) -> Self {
        Self::new(Element::filled_rect(rect, fill), layer)
    }

    /// Shorthand: circle on a given layer.
    pub fn circle(cx: f64, cy: f64, r: f64, fill: Fill, layer: Layer) -> Self {
        Self::new(Element::circle(cx, cy, r, fill), layer)
    }

    /// Shorthand: text on a given layer.
    pub fn text(x: f64, y: f64, content: impl Into<String>, font: FontStyle, layer: Layer) -> Self {
        Self::new(Element::text(x, y, content, font), layer)
    }
}

/// Helper to escape XML special characters in text content.
pub fn escape_xml(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

/// Helper to format a dash pattern to SVG attributes.
pub fn stroke_dash_attrs(dash: &Option<DashPattern>) -> String {
    match dash {
        Some(dp) => format!(" stroke-dasharray=\"{}\"", dp.to_svg_string()),
        None => String::new(),
    }
}
