// SPDX-License-Identifier: MIT OR Apache-2.0
//! Faceting types for small multiples.

/// Faceting mode.
#[derive(Clone, Debug, Default)]
pub enum Facet {
    /// No faceting (single panel).
    #[default]
    None,
    /// Wrap into rows of `ncol` panels.
    Wrap {
        /// Number of columns.
        ncol: usize,
    },
    /// Grid layout with explicit row/col counts.
    Grid {
        /// Number of rows.
        row_count: usize,
        /// Number of columns.
        col_count: usize,
    },
}

/// How axes are shared across facet panels.
#[derive(Clone, Copy, Debug, Default)]
pub enum FacetScales {
    /// All panels share the same axis ranges.
    #[default]
    Shared,
    /// Each panel has its own axis ranges.
    Free,
    /// Free X axis, shared Y.
    FreeX,
    /// Free Y axis, shared X.
    FreeY,
}
