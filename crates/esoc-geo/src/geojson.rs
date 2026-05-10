// SPDX-License-Identifier: MIT OR Apache-2.0
//! `GeoJSON` parser (feature-gated: `geojson`).
//!
//! Parses `GeoJSON` strings into our types. Nested properties are flattened
//! to JSON strings.
//!
//! Error handling: every fallible step returns a structured [`GeoError`]
//! variant ([`GeoError::MissingField`], [`GeoError::UnknownGeometryType`],
//! [`GeoError::InvalidCoordinates`], etc.) — never panics on malformed
//! input. Tests assert against the specific variant they expect.

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
        serde_json::from_str(input).map_err(|e| GeoError::GeoJsonSyntax(e.to_string()))?;

    match root.type_.as_str() {
        "FeatureCollection" => {
            let features = root.features.unwrap_or_default();
            let parsed: Result<Vec<GeoFeature>> = features.iter().map(parse_feature).collect();
            Ok(GeoCollection { features: parsed? })
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
        None => return Err(GeoError::FeatureMissingGeometry),
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
            serde_json::Value::Number(n) => PropertyValue::Number(n.as_f64().unwrap_or(0.0)),
            serde_json::Value::Bool(b) => PropertyValue::Bool(*b),
            serde_json::Value::Null => PropertyValue::Null,
            // Nested objects/arrays → serialize back to JSON string
            other => PropertyValue::String(other.to_string()),
        };
        props.insert(key.clone(), pv);
    }
    props
}

fn coords_of<'a>(geom: &'a GeoJsonGeometry, object: &'static str) -> Result<&'a serde_json::Value> {
    geom.coordinates.as_ref().ok_or(GeoError::MissingField {
        object,
        field: "coordinates",
    })
}

fn parse_geometry(geom: &GeoJsonGeometry) -> Result<GeoGeometry> {
    match geom.type_.as_str() {
        "Point" => {
            let point = parse_point(coords_of(geom, "Point")?)?;
            Ok(GeoGeometry::Point(point))
        }
        "LineString" => {
            let points = parse_line_coords(coords_of(geom, "LineString")?)?;
            Ok(GeoGeometry::LineString(GeoLineString { points }))
        }
        "Polygon" => {
            let polygon = parse_polygon_coords(coords_of(geom, "Polygon")?)?;
            Ok(GeoGeometry::Polygon(polygon))
        }
        "MultiPolygon" => {
            let polys = parse_multi_polygon_coords(coords_of(geom, "MultiPolygon")?)?;
            Ok(GeoGeometry::MultiPolygon(GeoMultiPolygon {
                polygons: polys,
            }))
        }
        "GeometryCollection" => {
            let geometries = geom.geometries.as_ref().ok_or(GeoError::MissingField {
                object: "GeometryCollection",
                field: "geometries",
            })?;
            // Lossy: returns the first geometry. esoc-geo's GeoGeometry has
            // no Collection variant, and this is the historical behavior.
            geometries
                .first()
                .map_or(Err(GeoError::EmptyGeometryCollection), parse_geometry)
        }
        other => Err(GeoError::UnknownGeometryType(other.to_string())),
    }
}

fn parse_point(value: &serde_json::Value) -> Result<GeoPoint> {
    let arr = value
        .as_array()
        .ok_or(GeoError::InvalidCoordinates("expected array for point"))?;
    if arr.len() < 2 {
        return Err(GeoError::InvalidCoordinates(
            "point needs at least 2 coordinates",
        ));
    }
    let lon = arr[0]
        .as_f64()
        .ok_or(GeoError::InvalidCoordinates("longitude is not a number"))?;
    let lat = arr[1]
        .as_f64()
        .ok_or(GeoError::InvalidCoordinates("latitude is not a number"))?;
    Ok(GeoPoint::new(lon, lat))
}

fn parse_line_coords(value: &serde_json::Value) -> Result<Vec<GeoPoint>> {
    let arr = value
        .as_array()
        .ok_or(GeoError::InvalidCoordinates("expected array for line"))?;
    arr.iter().map(parse_point).collect()
}

fn parse_ring(value: &serde_json::Value) -> Result<Ring> {
    parse_line_coords(value)
}

fn parse_polygon_coords(value: &serde_json::Value) -> Result<GeoPolygon> {
    let rings = value
        .as_array()
        .ok_or(GeoError::InvalidCoordinates("expected array of rings"))?;
    if rings.is_empty() {
        return Err(GeoError::EmptyPolygon);
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
        .ok_or(GeoError::InvalidCoordinates("expected array of polygons"))?;
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

        let f0 = &coll.features[0];
        let GeoGeometry::Point(p) = &f0.geometry else {
            unreachable!("first feature parsed as {:?}, not Point", f0.geometry)
        };
        assert!((p.lon - (-73.9857)).abs() < 1e-4);
        assert!((p.lat - 40.7484).abs() < 1e-4);
        assert_eq!(
            f0.properties.get("name").unwrap().as_str(),
            Some("Empire State Building")
        );
        assert_eq!(f0.properties.get("height").unwrap().as_f64(), Some(443.2));
        assert_eq!(f0.properties.get("open").unwrap().as_bool(), Some(true));

        let f1 = &coll.features[1];
        let GeoGeometry::Polygon(poly) = &f1.geometry else {
            unreachable!("second feature parsed as {:?}, not Polygon", f1.geometry)
        };
        assert_eq!(poly.exterior.len(), 5);
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
        let GeoGeometry::MultiPolygon(mp) = &coll.features[0].geometry else {
            unreachable!("expected MultiPolygon, got {:?}", coll.features[0].geometry)
        };
        assert_eq!(mp.polygons.len(), 2);
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

    // ── Error-path coverage ────────────────────────────────────────

    #[test]
    fn err_feature_missing_geometry() {
        let input = r#"{
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {}}]
        }"#;
        assert!(matches!(
            parse(input),
            Err(GeoError::FeatureMissingGeometry)
        ));
    }

    #[test]
    fn err_unknown_geometry_type() {
        let input = r#"{
            "type": "Hexagon",
            "coordinates": [0, 0]
        }"#;
        let Err(GeoError::UnknownGeometryType(name)) = parse(input) else {
            panic!("expected UnknownGeometryType");
        };
        assert_eq!(name, "Hexagon");
    }

    #[test]
    fn err_point_missing_coordinates() {
        let input = r#"{"type": "Point"}"#;
        assert!(matches!(
            parse(input),
            Err(GeoError::MissingField {
                object: "Point",
                field: "coordinates"
            })
        ));
    }

    #[test]
    fn err_point_too_few_coords() {
        let input = r#"{"type": "Point", "coordinates": [1.0]}"#;
        assert!(matches!(parse(input), Err(GeoError::InvalidCoordinates(_))));
    }

    #[test]
    fn err_polygon_no_rings() {
        let input = r#"{"type": "Polygon", "coordinates": []}"#;
        assert!(matches!(parse(input), Err(GeoError::EmptyPolygon)));
    }

    #[test]
    fn err_geometry_collection_empty() {
        let input = r#"{"type": "GeometryCollection", "geometries": []}"#;
        assert!(matches!(
            parse(input),
            Err(GeoError::EmptyGeometryCollection)
        ));
    }

    #[test]
    fn err_invalid_json_syntax() {
        let input = "not json";
        assert!(matches!(parse(input), Err(GeoError::GeoJsonSyntax(_))));
    }
}
