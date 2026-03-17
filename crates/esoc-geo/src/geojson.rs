// SPDX-License-Identifier: MIT OR Apache-2.0
//! `GeoJSON` parser (feature-gated: `geojson`).
//!
//! Parses `GeoJSON` strings into our types. Nested properties are flattened
//! to JSON strings.

use crate::error::{GeoError, Result};
use crate::geometry::{
    GeoCollection, GeoFeature, GeoGeometry, GeoLineString, GeoMultiPolygon, GeoPoint, GeoPolygon,
    Ring,
};
use crate::properties::{Properties, PropertyValue};

use serde::Deserialize;

// ── Serde intermediate types ────────────────────────────────────────

#[derive(Deserialize)]
struct GeoJsonRoot {
    #[serde(rename = "type")]
    type_: String,
    features: Option<Vec<GeoJsonFeature>>,
    // For single geometry or geometry collection at root level
    geometry: Option<GeoJsonGeometry>,
    geometries: Option<Vec<GeoJsonGeometry>>,
    coordinates: Option<serde_json::Value>,
    // For bare Feature at root level
    properties: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct GeoJsonFeature {
    geometry: Option<GeoJsonGeometry>,
    properties: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct GeoJsonGeometry {
    #[serde(rename = "type")]
    type_: String,
    coordinates: Option<serde_json::Value>,
    geometries: Option<Vec<Self>>,
}

// ── Public API ──────────────────────────────────────────────────────

/// Parse a `GeoJSON` string into a `GeoCollection`.
pub fn parse(input: &str) -> Result<GeoCollection> {
    let root: GeoJsonRoot =
        serde_json::from_str(input).map_err(|e| GeoError::ParseError(e.to_string()))?;

    match root.type_.as_str() {
        "FeatureCollection" => {
            let features = root.features.unwrap_or_default();
            let parsed: Result<Vec<GeoFeature>> = features.iter().map(parse_feature).collect();
            Ok(GeoCollection {
                features: parsed?,
            })
        }
        "Feature" => {
            let feature = GeoJsonFeature {
                geometry: root.geometry,
                properties: root.properties,
            };
            Ok(GeoCollection {
                features: vec![parse_feature(&feature)?],
            })
        }
        _ => {
            // Try as a bare geometry
            let geom = GeoJsonGeometry {
                type_: root.type_,
                coordinates: root.coordinates,
                geometries: root.geometries,
            };
            let geometry = parse_geometry(&geom)?;
            Ok(GeoCollection {
                features: vec![GeoFeature {
                    geometry,
                    properties: Properties::new(),
                }],
            })
        }
    }
}

fn parse_feature(feature: &GeoJsonFeature) -> Result<GeoFeature> {
    let geometry = match &feature.geometry {
        Some(g) => parse_geometry(g)?,
        None => {
            return Err(GeoError::ParseError(
                "feature has no geometry".into(),
            ));
        }
    };

    let properties = match &feature.properties {
        Some(serde_json::Value::Object(map)) => parse_properties(map),
        _ => Properties::new(),
    };

    Ok(GeoFeature {
        geometry,
        properties,
    })
}

fn parse_properties(map: &serde_json::Map<String, serde_json::Value>) -> Properties {
    let mut props = Properties::new();
    for (key, value) in map {
        let pv = match value {
            serde_json::Value::String(s) => PropertyValue::String(s.clone()),
            serde_json::Value::Number(n) => {
                PropertyValue::Number(n.as_f64().unwrap_or(0.0))
            }
            serde_json::Value::Bool(b) => PropertyValue::Bool(*b),
            serde_json::Value::Null => PropertyValue::Null,
            // Nested objects/arrays → serialize back to JSON string
            other => PropertyValue::String(other.to_string()),
        };
        props.insert(key.clone(), pv);
    }
    props
}

fn parse_geometry(geom: &GeoJsonGeometry) -> Result<GeoGeometry> {
    match geom.type_.as_str() {
        "Point" => {
            let coords = geom
                .coordinates
                .as_ref()
                .ok_or_else(|| GeoError::ParseError("Point missing coordinates".into()))?;
            let point = parse_point(coords)?;
            Ok(GeoGeometry::Point(point))
        }
        "LineString" => {
            let coords = geom
                .coordinates
                .as_ref()
                .ok_or_else(|| GeoError::ParseError("LineString missing coordinates".into()))?;
            let points = parse_line_coords(coords)?;
            Ok(GeoGeometry::LineString(GeoLineString { points }))
        }
        "Polygon" => {
            let coords = geom
                .coordinates
                .as_ref()
                .ok_or_else(|| GeoError::ParseError("Polygon missing coordinates".into()))?;
            let polygon = parse_polygon_coords(coords)?;
            Ok(GeoGeometry::Polygon(polygon))
        }
        "MultiPolygon" => {
            let coords = geom
                .coordinates
                .as_ref()
                .ok_or_else(|| GeoError::ParseError("MultiPolygon missing coordinates".into()))?;
            let polys = parse_multi_polygon_coords(coords)?;
            Ok(GeoGeometry::MultiPolygon(GeoMultiPolygon {
                polygons: polys,
            }))
        }
        "GeometryCollection" => {
            let geometries = geom.geometries.as_ref().ok_or_else(|| {
                GeoError::ParseError("GeometryCollection missing geometries".into())
            })?;
            // Return the first geometry, or error
            if let Some(first) = geometries.first() {
                parse_geometry(first)
            } else {
                Err(GeoError::ParseError(
                    "GeometryCollection is empty".into(),
                ))
            }
        }
        other => Err(GeoError::ParseError(format!(
            "unknown geometry type: {other}"
        ))),
    }
}

fn parse_point(value: &serde_json::Value) -> Result<GeoPoint> {
    let arr = value
        .as_array()
        .ok_or_else(|| GeoError::ParseError("expected array for point".into()))?;
    if arr.len() < 2 {
        return Err(GeoError::ParseError("point needs at least 2 coordinates".into()));
    }
    let lon = arr[0]
        .as_f64()
        .ok_or_else(|| GeoError::ParseError("invalid lon".into()))?;
    let lat = arr[1]
        .as_f64()
        .ok_or_else(|| GeoError::ParseError("invalid lat".into()))?;
    Ok(GeoPoint::new(lon, lat))
}

fn parse_line_coords(value: &serde_json::Value) -> Result<Vec<GeoPoint>> {
    let arr = value
        .as_array()
        .ok_or_else(|| GeoError::ParseError("expected array for line".into()))?;
    arr.iter().map(parse_point).collect()
}

fn parse_ring(value: &serde_json::Value) -> Result<Ring> {
    parse_line_coords(value)
}

fn parse_polygon_coords(value: &serde_json::Value) -> Result<GeoPolygon> {
    let rings = value
        .as_array()
        .ok_or_else(|| GeoError::ParseError("expected array of rings".into()))?;
    if rings.is_empty() {
        return Err(GeoError::ParseError("polygon has no rings".into()));
    }
    let exterior = parse_ring(&rings[0])?;
    let holes: Result<Vec<Ring>> = rings[1..].iter().map(parse_ring).collect();
    Ok(GeoPolygon {
        exterior,
        holes: holes?,
    })
}

fn parse_multi_polygon_coords(value: &serde_json::Value) -> Result<Vec<GeoPolygon>> {
    let polys = value
        .as_array()
        .ok_or_else(|| GeoError::ParseError("expected array of polygons".into()))?;
    polys.iter().map(parse_polygon_coords).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_feature_collection() {
        let input = r#"{
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [-73.9857, 40.7484]
                    },
                    "properties": {
                        "name": "Empire State Building",
                        "height": 443.2,
                        "open": true
                    }
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0,0],[1,0],[1,1],[0,1],[0,0]]]
                    },
                    "properties": {
                        "name": "Unit Square"
                    }
                }
            ]
        }"#;

        let coll = parse(input).unwrap();
        assert_eq!(coll.features.len(), 2);

        // Check first feature (point)
        let f0 = &coll.features[0];
        match &f0.geometry {
            GeoGeometry::Point(p) => {
                assert!((p.lon - (-73.9857)).abs() < 1e-4);
                assert!((p.lat - 40.7484).abs() < 1e-4);
            }
            _ => panic!("expected Point"),
        }
        assert_eq!(
            f0.properties.get("name").unwrap().as_str(),
            Some("Empire State Building")
        );
        assert_eq!(f0.properties.get("height").unwrap().as_f64(), Some(443.2));
        assert_eq!(f0.properties.get("open").unwrap().as_bool(), Some(true));

        // Check second feature (polygon)
        let f1 = &coll.features[1];
        match &f1.geometry {
            GeoGeometry::Polygon(p) => {
                assert_eq!(p.exterior.len(), 5);
            }
            _ => panic!("expected Polygon"),
        }
    }

    #[test]
    fn parse_multipolygon() {
        let input = r#"{
            "type": "Feature",
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [[[0,0],[1,0],[1,1],[0,1],[0,0]]],
                    [[[2,2],[3,2],[3,3],[2,3],[2,2]]]
                ]
            },
            "properties": null
        }"#;

        let coll = parse(input).unwrap();
        assert_eq!(coll.features.len(), 1);
        match &coll.features[0].geometry {
            GeoGeometry::MultiPolygon(mp) => {
                assert_eq!(mp.polygons.len(), 2);
            }
            _ => panic!("expected MultiPolygon"),
        }
    }

    #[test]
    fn parse_null_properties() {
        let input = r#"{
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [0, 0]
            },
            "properties": {"val": null}
        }"#;
        let coll = parse(input).unwrap();
        assert!(coll.features[0].properties.get("val").unwrap().is_null());
    }

    #[test]
    fn parse_nested_property_flattened() {
        let input = r#"{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [0, 0]},
            "properties": {"nested": {"a": 1, "b": 2}}
        }"#;
        let coll = parse(input).unwrap();
        let val = coll.features[0].properties.get("nested").unwrap();
        // Nested object should be serialized to a JSON string
        assert!(val.as_str().is_some());
        let s = val.as_str().unwrap();
        assert!(s.contains("\"a\""));
    }

    #[test]
    fn parse_error_missing_geometry() {
        let input = r#"{
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {}}]
        }"#;
        assert!(parse(input).is_err());
    }
}
