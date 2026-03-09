// SPDX-License-Identifier: MIT OR Apache-2.0
//! Series trait and data bounds for chart rendering.

use esoc_gfx::canvas::Canvas;
use esoc_gfx::transform::CoordinateTransform;

use crate::theme::Theme;

/// Axis-aligned bounding box for data in a series.
#[derive(Clone, Copy, Debug)]
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
    /// Create new data bounds.
    pub fn new(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Self {
        Self {
            x_min,
            x_max,
            y_min,
            y_max,
        }
    }

    /// Merge with another bounds, taking the union.
    pub fn union(self, other: Self) -> Self {
        Self {
            x_min: self.x_min.min(other.x_min),
            x_max: self.x_max.max(other.x_max),
            y_min: self.y_min.min(other.y_min),
            y_max: self.y_max.max(other.y_max),
        }
    }

    /// Add padding as a fraction of the range.
    pub fn pad(self, fraction: f64) -> Self {
        let x_pad = (self.x_max - self.x_min) * fraction;
        let y_pad = (self.y_max - self.y_min) * fraction;
        Self {
            x_min: self.x_min - x_pad,
            x_max: self.x_max + x_pad,
            y_min: self.y_min - y_pad,
            y_max: self.y_max + y_pad,
        }
    }
}

/// Trait implemented by all chart series types.
pub trait SeriesRenderer {
    /// Compute the data bounds for this series.
    fn data_bounds(&self) -> DataBounds;

    /// Render this series onto a canvas using the given coordinate transform.
    fn render(
        &self,
        canvas: &mut Canvas,
        transform: &CoordinateTransform,
        theme: &Theme,
        series_index: usize,
    );

    /// Optional label for the legend.
    fn label(&self) -> Option<&str>;
}
