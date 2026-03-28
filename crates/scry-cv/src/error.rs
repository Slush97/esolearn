// SPDX-License-Identifier: MIT OR Apache-2.0
//! Crate-specific error types for scry-cv.

use thiserror::Error;

/// All errors produced by scry-cv operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ScryVisionError {
    /// Image dimensions are invalid (zero width/height, or overflow).
    #[error("invalid dimensions: {0}")]
    InvalidDimensions(String),

    /// Data buffer size does not match expected dimensions.
    #[error("buffer size mismatch: expected {expected}, got {got}")]
    BufferSizeMismatch {
        /// Expected buffer length.
        expected: usize,
        /// Actual buffer length.
        got: usize,
    },

    /// Region of interest is out of bounds.
    #[error("region out of bounds: {0}")]
    OutOfBounds(String),

    /// An algorithm parameter has an invalid value.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// Algorithm did not converge within the iteration limit.
    #[error("convergence failure after {iterations} iterations (tolerance: {tolerance})")]
    ConvergenceFailure {
        /// Number of iterations executed.
        iterations: usize,
        /// Target tolerance that was not reached.
        tolerance: f64,
    },

    /// Not enough input data (e.g. too few keypoints for homography).
    #[error("insufficient data: {0}")]
    InsufficientData(String),

    /// Channel layout mismatch during conversion.
    #[error("channel mismatch: expected {expected}, got {got}")]
    ChannelMismatch {
        /// Expected channel layout name.
        expected: &'static str,
        /// Actual channel layout name.
        got: &'static str,
    },

    /// Filesystem I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenience alias for `Result<T, ScryVisionError>`.
pub type Result<T> = std::result::Result<T, ScryVisionError>;
