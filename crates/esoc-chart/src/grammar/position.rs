// SPDX-License-Identifier: MIT OR Apache-2.0
//! Position adjustments for grouped/stacked visualizations.

/// How multiple layers at the same x position are arranged.
#[derive(Clone, Copy, Debug, Default)]
pub enum Position {
    /// No adjustment (overlay).
    #[default]
    Identity,
    /// Stack layers vertically.
    Stack,
    /// Place layers side by side.
    Dodge,
    /// Stack + normalize to 100%.
    Fill,
    /// Add random displacement.
    Jitter {
        /// X displacement amount.
        x_amount: f64,
        /// Y displacement amount.
        y_amount: f64,
    },
}
