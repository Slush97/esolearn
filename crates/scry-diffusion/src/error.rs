// SPDX-License-Identifier: MIT OR Apache-2.0
//! Error types for `scry-diffusion`.

use thiserror::Error;

/// Errors produced by `scry-diffusion`.
#[derive(Debug, Error)]
pub enum Error {
    /// Underlying scry-llm operation failed.
    #[error("scry-llm: {0}")]
    Llm(String),

    /// Tokenizer construction or encoding failure.
    #[error("tokenizer: {0}")]
    Tokenizer(String),

    /// Weight loading from safetensors failed (missing key, shape mismatch,
    /// or unsupported dtype).
    #[error("weights: {0}")]
    Weights(String),

    /// Scheduler step or configuration is invalid.
    #[error("scheduler: {0}")]
    Scheduler(String),

    /// I/O error while reading weights or writing images.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Result alias for `scry-diffusion`.
pub type Result<T, E = Error> = std::result::Result<T, E>;
