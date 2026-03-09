// SPDX-License-Identifier: MIT OR Apache-2.0
//! Error types for scry-learn.

use std::fmt;

/// Errors produced by scry-learn operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum ScryLearnError {
    /// The dataset contains no samples.
    EmptyDataset,

    /// The input data contains NaN, Inf, or other non-finite values.
    InvalidData(String),

    /// A referenced column does not exist.
    InvalidColumn(String),

    /// Feature matrix dimensions do not match what the model expects.
    ShapeMismatch {
        /// Expected number of features.
        expected: usize,
        /// Actual number of features provided.
        got: usize,
    },

    /// `predict()` or `transform()` called before `fit()`.
    NotFitted,

    /// An invalid hyperparameter was supplied.
    InvalidParameter(String),

    /// No convergence within the iteration limit.
    ConvergenceFailure {
        /// Number of iterations attempted.
        iterations: usize,
        /// Convergence tolerance.
        tolerance: f64,
    },

    /// I/O error during file operations.
    Io(std::io::Error),

    /// CSV parsing error.
    Csv(String),

    /// A referenced feature index is out of bounds.
    InvalidFeatureIndex(usize),
}

impl fmt::Display for ScryLearnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDataset => f.write_str("dataset is empty — no samples to train on"),
            Self::InvalidData(msg) => write!(f, "invalid data: {msg}"),
            Self::InvalidColumn(col) => write!(f, "column not found: {col}"),
            Self::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected} features, got {got}")
            }
            Self::NotFitted => f.write_str("model has not been fitted — call .fit() first"),
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::ConvergenceFailure {
                iterations,
                tolerance,
            } => {
                write!(
                    f,
                    "failed to converge after {iterations} iterations (tolerance: {tolerance})"
                )
            }
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Csv(msg) => write!(f, "CSV error: {msg}"),
            Self::InvalidFeatureIndex(idx) => write!(f, "invalid feature index: {idx}"),
        }
    }
}

impl std::error::Error for ScryLearnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ScryLearnError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

/// Convenience type alias.
pub type Result<T> = std::result::Result<T, ScryLearnError>;
