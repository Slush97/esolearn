// SPDX-License-Identifier: MIT OR Apache-2.0
//! Error types for esoc-gpu.

use std::fmt;

/// Errors produced by GPU rendering operations.
#[derive(Debug)]
pub enum GpuError {
    /// Buffer readback from GPU failed.
    ReadbackFailed(String),
    /// Viewport has zero or invalid dimensions.
    InvalidViewport {
        /// Viewport width.
        width: u32,
        /// Viewport height.
        height: u32,
    },
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReadbackFailed(msg) => write!(f, "GPU readback failed: {msg}"),
            Self::InvalidViewport { width, height } => {
                write!(f, "invalid viewport: {width}x{height}")
            }
        }
    }
}

impl std::error::Error for GpuError {}
