// SPDX-License-Identifier: MIT OR Apache-2.0
//! Core geographic types.

use crate::properties::Properties;

/// A geographic point (longitude, latitude) in degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoPoint {
    /// Longitude in degrees (−180 to 180).
    pub lon: f64,
    /// Latitude in degrees (−90 to 90).
    pub lat: f64,
}

impl GeoPoint {
    /// Create a new point.
    pub fn new(lon: f64, lat: f64) -> Self {
        Self { lon, lat }
    }
}

/// A closed ring of points (exterior or hole boundary).
pub type Ring = Vec<GeoPoint>;

/// A polygon with an exterior ring and optional interior holes.
#[derive(Clone, Debug)]
pub struct GeoPolygon {
    /// Exterior ring (first ring).
    pub exterior: Ring,
    /// Interior rings (holes).
    pub holes: Vec<Ring>,
}

/// A collection of polygons forming a single feature.
#[derive(Clone, Debug)]
pub struct GeoMultiPolygon {
    /// Individual polygons.
    pub polygons: Vec<GeoPolygon>,
}

/// A line string (open sequence of points).
#[derive(Clone, Debug)]
pub struct GeoLineString {
    /// Points along the line.
    pub points: Vec<GeoPoint>,
}

/// A geographic geometry (one of several types).
#[derive(Clone, Debug)]
pub enum GeoGeometry {
    /// Single point.
    Point(GeoPoint),
    /// Line string.
    LineString(GeoLineString),
    /// Single polygon.
    Polygon(GeoPolygon),
    /// Multiple polygons.
    MultiPolygon(GeoMultiPolygon),
}

/// A geographic feature: geometry + properties.
#[derive(Clone, Debug)]
pub struct GeoFeature {
    /// The geometry.
    pub geometry: GeoGeometry,
    /// Key-value properties.
    pub properties: Properties,
}

/// A collection of features.
#[derive(Clone, Debug)]
pub struct GeoCollection {
    /// Features in this collection.
    pub features: Vec<GeoFeature>,
}

/// Axis-aligned bounding box in geographic coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoBounds {
    /// Minimum longitude.
    pub min_lon: f64,
    /// Minimum latitude.
    pub min_lat: f64,
    /// Maximum longitude.
    pub max_lon: f64,
    /// Maximum latitude.
    pub max_lat: f64,
}

impl GeoBounds {
    /// Create a new bounding box.
    pub fn new(min_lon: f64, min_lat: f64, max_lon: f64, max_lat: f64) -> Self {
        Self {
            min_lon,
            min_lat,
            max_lon,
            max_lat,
        }
    }

    /// Create an empty (inverted) bounding box for incremental expansion.
    pub fn empty() -> Self {
        Self {
            min_lon: f64::INFINITY,
            min_lat: f64::INFINITY,
            max_lon: f64::NEG_INFINITY,
            max_lat: f64::NEG_INFINITY,
        }
    }

    /// Expand this bounding box to include a point.
    pub fn include(&mut self, point: GeoPoint) {
        self.min_lon = self.min_lon.min(point.lon);
        self.min_lat = self.min_lat.min(point.lat);
        self.max_lon = self.max_lon.max(point.lon);
        self.max_lat = self.max_lat.max(point.lat);
    }

    /// Width in degrees.
    pub fn width(&self) -> f64 {
        self.max_lon - self.min_lon
    }

    /// Height in degrees.
    pub fn height(&self) -> f64 {
        self.max_lat - self.min_lat
    }

    /// Center point.
    pub fn center(&self) -> GeoPoint {
        GeoPoint::new(
            (self.min_lon + self.max_lon) * 0.5,
            (self.min_lat + self.max_lat) * 0.5,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_creation() {
        let p = GeoPoint::new(-73.9857, 40.7484);
        assert!((p.lon - (-73.9857)).abs() < 1e-10);
        assert!((p.lat - 40.7484).abs() < 1e-10);
    }

    #[test]
    fn bounds_include() {
        let mut b = GeoBounds::empty();
        b.include(GeoPoint::new(-10.0, -5.0));
        b.include(GeoPoint::new(10.0, 5.0));
        assert!((b.min_lon - (-10.0)).abs() < 1e-10);
        assert!((b.max_lat - 5.0).abs() < 1e-10);
        assert!((b.width() - 20.0).abs() < 1e-10);
        assert!((b.height() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn bounds_center() {
        let b = GeoBounds::new(-10.0, -5.0, 10.0, 5.0);
        let c = b.center();
        assert!((c.lon).abs() < 1e-10);
        assert!((c.lat).abs() < 1e-10);
    }

    #[test]
    fn polygon_structure() {
        let exterior = vec![
            GeoPoint::new(0.0, 0.0),
            GeoPoint::new(1.0, 0.0),
            GeoPoint::new(1.0, 1.0),
            GeoPoint::new(0.0, 0.0),
        ];
        let poly = GeoPolygon {
            exterior,
            holes: vec![],
        };
        assert_eq!(poly.exterior.len(), 4);
        assert!(poly.holes.is_empty());
    }
}
