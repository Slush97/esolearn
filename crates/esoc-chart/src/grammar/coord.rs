// SPDX-License-Identifier: MIT OR Apache-2.0
//! Coordinate system types.

/// Coordinate system for the chart.
#[derive(Clone, Debug, Default)]
pub enum CoordSystem {
    /// Standard Cartesian (x right, y up).
    #[default]
    Cartesian,
    /// Flipped: x and y axes swapped (horizontal bars).
    Flipped,
    /// Polar coordinates (r, θ) — **not yet implemented**.
    /// Selecting this will produce a compile-time error.
    #[doc(hidden)]
    Polar,
}
