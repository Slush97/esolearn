// SPDX-License-Identifier: MIT OR Apache-2.0
//! Error types for esoc-chart.

use std::fmt;

/// Errors produced by esoc-chart operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum ChartError {
    /// No data provided.
    EmptyData,

    /// Mismatched array lengths.
    LengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        got: usize,
    },

    /// Layer contains invalid data (NaN, Inf, etc.).
    InvalidData {
        /// Layer index.
        layer: usize,
        /// Description of the problem.
        detail: String,
    },

    /// X and Y data have different lengths in a layer.
    DimensionMismatch {
        /// Layer index.
        layer: usize,
        /// X data length.
        x_len: usize,
        /// Y data length.
        y_len: usize,
    },

    /// Invalid chart parameter.
    InvalidParameter(String),

    /// Underlying graphics error.
    Gfx(esoc_gfx::error::GfxError),

    /// I/O error.
    Io(std::io::Error),
}

impl fmt::Display for ChartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyData => f.write_str("no data provided"),
            Self::LengthMismatch { expected, got } => {
                write!(f, "length mismatch: expected {expected}, got {got}")
            }
            Self::InvalidData { layer, detail } => {
                write!(f, "layer {layer}: {detail}")
            }
            Self::DimensionMismatch {
                layer,
                x_len,
                y_len,
            } => {
                write!(f, "layer {layer}: x has {x_len} elements but y has {y_len}")
            }
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::Gfx(err) => write!(f, "graphics error: {err}"),
            Self::Io(err) => write!(f, "I/O error: {err}"),
        }
    }
}

impl std::error::Error for ChartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gfx(err) => Some(err),
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<esoc_gfx::error::GfxError> for ChartError {
    fn from(err: esoc_gfx::error::GfxError) -> Self {
        Self::Gfx(err)
    }
}

impl From<std::io::Error> for ChartError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

/// Convenience type alias.
pub type Result<T> = std::result::Result<T, ChartError>;
