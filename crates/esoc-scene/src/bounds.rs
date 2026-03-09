// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bounding boxes and data bounds.

/// An axis-aligned bounding box in visual space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BoundingBox {
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width.
    pub w: f32,
    /// Height.
    pub h: f32,
}

impl BoundingBox {
    /// Create a bounding box.
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    /// Right edge.
    pub fn right(&self) -> f32 {
        self.x + self.w
    }

    /// Bottom edge.
    pub fn bottom(&self) -> f32 {
        self.y + self.h
    }

    /// Center point.
    pub fn center(&self) -> [f32; 2] {
        [self.x + self.w * 0.5, self.y + self.h * 0.5]
    }

    /// Whether a point is inside.
    pub fn contains(&self, p: [f32; 2]) -> bool {
        p[0] >= self.x && p[0] <= self.right() && p[1] >= self.y && p[1] <= self.bottom()
    }

    /// Union of two bounding boxes.
    pub fn union(self, other: Self) -> Self {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let r = self.right().max(other.right());
        let b = self.bottom().max(other.bottom());
        Self {
            x,
            y,
            w: r - x,
            h: b - y,
        }
    }
}

/// Data-space bounds (f64 for scientific precision).
#[derive(Clone, Copy, Debug, Default)]
pub struct DataBounds {
    /// Minimum X value.
    pub x_min: f64,
    /// Maximum X value.
    pub x_max: f64,
    /// Minimum Y value.
    pub y_min: f64,
    /// Maximum Y value.
    pub y_max: f64,
}

impl DataBounds {
    /// Create from x/y ranges.
    pub fn new(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Self {
        Self {
            x_min,
            x_max,
            y_min,
            y_max,
        }
    }

    /// Union of two data bounds.
    pub fn union(self, other: Self) -> Self {
        Self {
            x_min: self.x_min.min(other.x_min),
            x_max: self.x_max.max(other.x_max),
            y_min: self.y_min.min(other.y_min),
            y_max: self.y_max.max(other.y_max),
        }
    }

    /// Expand to include a point.
    pub fn include_point(&mut self, x: f64, y: f64) {
        self.x_min = self.x_min.min(x);
        self.x_max = self.x_max.max(x);
        self.y_min = self.y_min.min(y);
        self.y_max = self.y_max.max(y);
    }

    /// Add padding (fraction of range).
    pub fn pad(self, fraction: f64) -> Self {
        let dx = (self.x_max - self.x_min) * fraction;
        let dy = (self.y_max - self.y_min) * fraction;
        Self {
            x_min: self.x_min - dx,
            x_max: self.x_max + dx,
            y_min: self.y_min - dy,
            y_max: self.y_max + dy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbox_contains() {
        let bb = BoundingBox::new(10.0, 20.0, 100.0, 50.0);
        assert!(bb.contains([50.0, 40.0]));
        assert!(!bb.contains([0.0, 0.0]));
    }

    #[test]
    fn bbox_union() {
        let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BoundingBox::new(5.0, 5.0, 10.0, 10.0);
        let u = a.union(b);
        assert!((u.x).abs() < 1e-6);
        assert!((u.y).abs() < 1e-6);
        assert!((u.w - 15.0).abs() < 1e-6);
        assert!((u.h - 15.0).abs() < 1e-6);
    }

    #[test]
    fn data_bounds_pad() {
        let b = DataBounds::new(0.0, 100.0, 0.0, 50.0);
        let padded = b.pad(0.1);
        assert!((padded.x_min - (-10.0)).abs() < 1e-6);
        assert!((padded.x_max - 110.0).abs() < 1e-6);
    }
}
