// SPDX-License-Identifier: MIT OR Apache-2.0
//! Error types for esoc-gfx.

use std::fmt;

/// Errors produced by esoc-gfx operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum GfxError {
    /// Invalid hex color string.
    InvalidColor(String),

    /// I/O error during file write.
    Io(std::io::Error),

    /// Format error during SVG generation.
    Fmt(fmt::Error),

    /// PNG rendering failed (requires `png` feature).
    Render(String),

    /// Invalid parameter.
    InvalidParameter(String),
}

impl fmt::Display for GfxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidColor(s) => write!(f, "invalid color: {s}"),
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Fmt(err) => write!(f, "format error: {err}"),
            Self::Render(msg) => write!(f, "render error: {msg}"),
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
        }
    }
}

impl std::error::Error for GfxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Fmt(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for GfxError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<fmt::Error> for GfxError {
    fn from(err: fmt::Error) -> Self {
        Self::Fmt(err)
    }
}

/// Convenience type alias.
pub type Result<T> = std::result::Result<T, GfxError>;
