// SPDX-License-Identifier: MIT OR Apache-2.0
//! Geometric primitives: Point, Rect, Size.

/// A 2D point.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
}

impl Point {
    /// Create a new point.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// A 2D size (width, height).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Size {
    /// Width.
    pub width: f64,
    /// Height.
    pub height: f64,
}

impl Size {
    /// Create a new size.
    pub fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }
}

/// An axis-aligned rectangle.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
    /// Left edge X coordinate.
    pub x: f64,
    /// Top edge Y coordinate.
    pub y: f64,
    /// Width.
    pub width: f64,
    /// Height.
    pub height: f64,
}

impl Rect {
    /// Create a new rectangle.
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Right edge X coordinate.
    pub fn right(&self) -> f64 {
        self.x + self.width
    }

    /// Bottom edge Y coordinate.
    pub fn bottom(&self) -> f64 {
        self.y + self.height
    }

    /// Center point.
    pub fn center(&self) -> Point {
        Point::new(self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    /// Whether this rect contains a point.
    pub fn contains(&self, p: Point) -> bool {
        p.x >= self.x && p.x <= self.right() && p.y >= self.y && p.y <= self.bottom()
    }
}
