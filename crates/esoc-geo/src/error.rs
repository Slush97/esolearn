// SPDX-License-Identifier: MIT OR Apache-2.0
//! Error types for esoc-geo.

use std::fmt;

/// Errors produced by esoc-geo operations.
#[non_exhaustive]
#[derive(Debug)]
pub enum GeoError {
    /// Invalid geometry (e.g., ring with < 3 points).
    InvalidGeometry(String),
    /// I/O error.
    Io(std::io::Error),
    /// `GeoJSON` JSON-syntax failure (message from `serde_json`).
    GeoJsonSyntax(String),
    /// A required `GeoJSON` field was missing.
    ///
    /// `object` names the surrounding object kind ("Feature", "Polygon", ...)
    /// and `field` names the missing field ("geometry", "coordinates", ...).
    MissingField {
        /// The kind of `GeoJSON` object the field was expected on.
        object: &'static str,
        /// The name of the missing field.
        field: &'static str,
    },
    /// A `GeoJSON` `type` value we do not recognise or support.
    UnknownGeometryType(String),
    /// Coordinates did not match the expected shape (e.g., point with < 2 numbers).
    InvalidCoordinates(&'static str),
    /// A `GeoJSON` polygon had no rings.
    EmptyPolygon,
    /// A `GeoJSON` `GeometryCollection` had no geometries.
    EmptyGeometryCollection,
    /// A `GeoJSON` `Feature` had no `geometry` field.
    FeatureMissingGeometry,
}

impl fmt::Display for GeoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGeometry(msg) => write!(f, "invalid geometry: {msg}"),
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::GeoJsonSyntax(msg) => write!(f, "GeoJSON syntax error: {msg}"),
            Self::MissingField { object, field } => {
                write!(f, "GeoJSON {object} is missing required field `{field}`")
            }
            Self::UnknownGeometryType(t) => write!(f, "unknown GeoJSON geometry type: {t}"),
            Self::InvalidCoordinates(msg) => write!(f, "invalid coordinates: {msg}"),
            Self::EmptyPolygon => write!(f, "polygon has no rings"),
            Self::EmptyGeometryCollection => write!(f, "GeometryCollection is empty"),
            Self::FeatureMissingGeometry => write!(f, "feature has no geometry"),
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

    #[test]
    fn missing_field_display_names_both_parts() {
        let err = GeoError::MissingField {
            object: "Polygon",
            field: "coordinates",
        };
        let s = err.to_string();
        assert!(s.contains("Polygon"));
        assert!(s.contains("coordinates"));
    }

    #[test]
    fn unknown_geometry_type_round_trips_payload() {
        let err = GeoError::UnknownGeometryType("Hexagon".into());
        assert!(err.to_string().contains("Hexagon"));
    }
}
