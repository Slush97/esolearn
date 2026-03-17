// SPDX-License-Identifier: MIT OR Apache-2.0
//! Error types for esoc-geo.

use std::fmt;

/// Errors produced by esoc-geo operations.
#[derive(Debug)]
pub enum GeoError {
    /// Invalid geometry (e.g., ring with < 3 points).
    InvalidGeometry(String),
    /// Parse error when reading `GeoJSON`.
    ParseError(String),
    /// I/O error.
    Io(std::io::Error),
}

impl fmt::Display for GeoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGeometry(msg) => write!(f, "invalid geometry: {msg}"),
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
            Self::Io(err) => write!(f, "I/O error: {err}"),
        }
    }
}

impl std::error::Error for GeoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for GeoError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

/// Convenience type alias.
pub type Result<T> = std::result::Result<T, GeoError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let err = GeoError::InvalidGeometry("bad ring".into());
        assert!(err.to_string().contains("bad ring"));
    }

    #[test]
    fn error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let err: GeoError = io_err.into();
        assert!(err.to_string().contains("I/O error"));
    }
}
