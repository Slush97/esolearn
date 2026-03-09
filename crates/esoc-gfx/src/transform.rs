// SPDX-License-Identifier: MIT OR Apache-2.0
//! Coordinate transforms: data space → normalized → pixel space.

use crate::geom::{Point, Rect};

/// Axis transform type.
#[derive(Clone, Debug)]
pub enum AxisTransform {
    /// Linear mapping from `[min, max]`.
    Linear {
        /// Minimum value.
        min: f64,
        /// Maximum value.
        max: f64,
    },
    /// Logarithmic mapping from `[min, max]` (base 10).
    Log {
        /// Minimum value.
        min: f64,
        /// Maximum value.
        max: f64,
    },
    /// Categorical: maps index → evenly-spaced position.
    Categorical {
        /// Number of categories.
        count: usize,
    },
}

impl AxisTransform {
    /// Map a data value to normalized `[0, 1]` space.
    pub fn normalize(&self, value: f64) -> f64 {
        match self {
            Self::Linear { min, max } => {
                if (max - min).abs() < 1e-15 {
                    0.5
                } else {
                    (value - min) / (max - min)
                }
            }
            Self::Log { min, max } => {
                let log_min = min.max(1e-15).ln();
                let log_max = max.max(1e-15).ln();
                if (log_max - log_min).abs() < 1e-15 {
                    0.5
                } else {
                    (value.max(1e-15).ln() - log_min) / (log_max - log_min)
                }
            }
            Self::Categorical { count } => {
                if *count <= 1 {
                    0.5
                } else {
                    value / (*count - 1) as f64
                }
            }
        }
    }

    /// Map from normalized `[0, 1]` back to data space.
    pub fn denormalize(&self, t: f64) -> f64 {
        match self {
            Self::Linear { min, max } => min + t * (max - min),
            Self::Log { min, max } => {
                let log_min = min.max(1e-15).ln();
                let log_max = max.max(1e-15).ln();
                (log_min + t * (log_max - log_min)).exp()
            }
            Self::Categorical { count } => {
                if *count <= 1 {
                    0.0
                } else {
                    t * (*count - 1) as f64
                }
            }
        }
    }
}

/// Maps normalized `[0, 1]` space to pixel space within a viewport rectangle.
///
/// Y-axis is inverted for SVG (0 = top).
#[derive(Clone, Debug)]
pub struct ViewportTransform {
    /// The pixel-space rectangle for this plot area.
    pub viewport: Rect,
}

impl ViewportTransform {
    /// Create a viewport transform for the given rectangle.
    pub fn new(viewport: Rect) -> Self {
        Self { viewport }
    }

    /// Map normalized (nx, ny) in `[0,1]` to pixel coordinates.
    /// Y is inverted: ny=0 → bottom, ny=1 → top.
    pub fn to_pixel(&self, nx: f64, ny: f64) -> Point {
        Point::new(
            self.viewport.x + nx * self.viewport.width,
            self.viewport.y + self.viewport.height - ny * self.viewport.height,
        )
    }

    /// Map pixel coordinates back to normalized `[0,1]`.
    pub fn from_pixel(&self, px: f64, py: f64) -> (f64, f64) {
        let nx = (px - self.viewport.x) / self.viewport.width;
        let ny = 1.0 - (py - self.viewport.y) / self.viewport.height;
        (nx, ny)
    }
}

/// Full coordinate transform pipeline: data space → pixel space.
#[derive(Clone, Debug)]
pub struct CoordinateTransform {
    /// X-axis transform.
    pub x_transform: AxisTransform,
    /// Y-axis transform.
    pub y_transform: AxisTransform,
    /// Viewport mapping.
    pub viewport: ViewportTransform,
}

impl CoordinateTransform {
    /// Create a coordinate transform.
    pub fn new(
        x_transform: AxisTransform,
        y_transform: AxisTransform,
        viewport: ViewportTransform,
    ) -> Self {
        Self {
            x_transform,
            y_transform,
            viewport,
        }
    }

    /// Map a data-space point to pixel coordinates.
    pub fn to_pixel(&self, x: f64, y: f64) -> Point {
        let nx = self.x_transform.normalize(x);
        let ny = self.y_transform.normalize(y);
        self.viewport.to_pixel(nx, ny)
    }

    /// Map pixel coordinates back to data space.
    pub fn from_pixel(&self, px: f64, py: f64) -> (f64, f64) {
        let (nx, ny) = self.viewport.from_pixel(px, py);
        let x = self.x_transform.denormalize(nx);
        let y = self.y_transform.denormalize(ny);
        (x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_normalize() {
        let t = AxisTransform::Linear {
            min: 0.0,
            max: 100.0,
        };
        assert!((t.normalize(0.0)).abs() < 1e-10);
        assert!((t.normalize(50.0) - 0.5).abs() < 1e-10);
        assert!((t.normalize(100.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_viewport_y_inversion() {
        let vt = ViewportTransform::new(Rect::new(0.0, 0.0, 100.0, 100.0));
        // ny=0 → bottom of viewport (y=100)
        let p = vt.to_pixel(0.0, 0.0);
        assert!((p.y - 100.0).abs() < 1e-10);
        // ny=1 → top of viewport (y=0)
        let p = vt.to_pixel(0.0, 1.0);
        assert!(p.y.abs() < 1e-10);
    }

    #[test]
    fn test_coordinate_roundtrip() {
        let ct = CoordinateTransform::new(
            AxisTransform::Linear {
                min: 0.0,
                max: 10.0,
            },
            AxisTransform::Linear {
                min: 0.0,
                max: 100.0,
            },
            ViewportTransform::new(Rect::new(50.0, 50.0, 400.0, 300.0)),
        );
        let p = ct.to_pixel(5.0, 50.0);
        let (x, y) = ct.from_pixel(p.x, p.y);
        assert!((x - 5.0).abs() < 1e-6);
        assert!((y - 50.0).abs() < 1e-6);
    }
}
