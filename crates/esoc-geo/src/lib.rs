// SPDX-License-Identifier: MIT OR Apache-2.0
//! Geographic types, map projections, and spatial utilities.
//!
//! `esoc-geo` provides core geographic primitives (`GeoPoint`, `GeoPolygon`,
//! `GeoFeature`, etc.), spatial operations (centroid, bounds, point-in-polygon,
//! area), polygon simplification (Ramer-Douglas-Peucker), and several map
//! projections (Equirectangular, Mercator, Equal Earth, Natural Earth,
//! Albers USA composite).
//!
//! # Features
//!
//! - **`geojson`** — `GeoJSON` parsing via serde
//! - **`bundled`** — Pre-bundled world countries and US states data (implies `geojson`)

pub mod error;
pub mod geometry;
pub mod projection;
pub mod properties;
pub mod simplify;
pub mod spatial;

#[cfg(feature = "geojson")]
pub mod geojson;

#[cfg(feature = "bundled")]
pub mod bundled;

pub use error::{GeoError, Result};
pub use geometry::{
    GeoBounds, GeoCollection, GeoFeature, GeoGeometry, GeoLineString, GeoMultiPolygon, GeoPoint,
    GeoPolygon, Ring,
};
pub use projection::Projection;
pub use properties::{Properties, PropertyValue};
pub use simplify::simplify;
pub use spatial::{area, bounds, centroid, point_in_polygon};
