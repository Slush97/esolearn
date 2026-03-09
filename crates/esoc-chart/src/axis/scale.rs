// SPDX-License-Identifier: MIT OR Apache-2.0
//! Axis scale types.

/// Scale type for an axis.
#[derive(Clone, Debug, Default)]
pub enum Scale {
    /// Linear scale (default).
    #[default]
    Linear,
    /// Logarithmic scale (base 10).
    Log,
    /// Categorical scale with string labels.
    Categorical(Vec<String>),
}
