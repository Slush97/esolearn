// SPDX-License-Identifier: MIT OR Apache-2.0
//! Error types for scry-vision.

/// Errors that can occur during vision operations.
#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    #[error("image decode failed: {0}")]
    Decode(String),

    #[error("invalid image dimensions: {width}x{height}x{channels}")]
    InvalidDimensions {
        width: u32,
        height: u32,
        channels: u8,
    },

    #[error("transform failed: {0}")]
    Transform(String),

    #[error("model inference failed: {0}")]
    Inference(String),

    #[error("model loading failed: {0}")]
    ModelLoad(String),

    #[error("missing weight: {0}")]
    MissingWeight(String),

    #[error("shape mismatch for '{name}': expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, VisionError>;
