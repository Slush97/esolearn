// SPDX-License-Identifier: MIT OR Apache-2.0
//! Rendering layers for z-ordering elements.

/// Drawing layers, rendered bottom-to-top.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Layer {
    /// Background fills and decorations.
    Background = 0,
    /// Grid lines and axis frames.
    Grid = 1,
    /// Data series (lines, bars, scatter points, etc.).
    #[default]
    Data = 2,
    /// Annotations and labels.
    Annotations = 3,
    /// Legend overlay.
    Legend = 4,
}
