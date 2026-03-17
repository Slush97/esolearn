// SPDX-License-Identifier: MIT OR Apache-2.0
//! Spatial operations: centroid, bounds, point-in-polygon, area.

use crate::geometry::{GeoBounds, GeoGeometry, GeoPoint, GeoPolygon};

/// Compute the centroid of a ring (simple average of vertices, excluding closing duplicate).
fn ring_centroid(ring: &[GeoPoint]) -> GeoPoint {
    if ring.is_empty() {
        return GeoPoint::new(0.0, 0.0);
    }
    // Use the signed-area weighted centroid for accuracy
    let n = if ring.len() > 1 && ring.first() == ring.last() {
        ring.len() - 1
    } else {
        ring.len()
    };
    if n == 0 {
        return GeoPoint::new(0.0, 0.0);
    }

    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut signed_area = 0.0;

    for i in 0..n {
        let j = (i + 1) % n;
        let xi = ring[i].lon;
        let yi = ring[i].lat;
        let xj = ring[j].lon;
        let yj = ring[j].lat;
        let cross = xi * yj - xj * yi;
        signed_area += cross;
        cx += (xi + xj) * cross;
        cy += (yi + yj) * cross;
    }

    if signed_area.abs() < 1e-20 {
        // Degenerate: fall back to simple average
        let sum_lon: f64 = ring[..n].iter().map(|p| p.lon).sum();
        let sum_lat: f64 = ring[..n].iter().map(|p| p.lat).sum();
        return GeoPoint::new(sum_lon / n as f64, sum_lat / n as f64);
    }

    signed_area *= 0.5;
    cx /= 6.0 * signed_area;
    cy /= 6.0 * signed_area;

    GeoPoint::new(cx, cy)
}

/// Compute the centroid of a geometry.
///
/// For points, returns the point itself. For polygons, returns the
/// area-weighted centroid of the exterior ring. For multi-polygons,
/// returns the area-weighted average of polygon centroids.
pub fn centroid(geom: &GeoGeometry) -> GeoPoint {
    match geom {
        GeoGeometry::Point(p) => *p,
        GeoGeometry::LineString(ls) => {
            if ls.points.is_empty() {
                return GeoPoint::new(0.0, 0.0);
            }
            let n = ls.points.len() as f64;
            let sum_lon: f64 = ls.points.iter().map(|p| p.lon).sum();
            let sum_lat: f64 = ls.points.iter().map(|p| p.lat).sum();
            GeoPoint::new(sum_lon / n, sum_lat / n)
        }
        GeoGeometry::Polygon(poly) => ring_centroid(&poly.exterior),
        GeoGeometry::MultiPolygon(mp) => {
            if mp.polygons.is_empty() {
                return GeoPoint::new(0.0, 0.0);
            }
            let mut total_area = 0.0;
            let mut cx = 0.0;
            let mut cy = 0.0;
            for poly in &mp.polygons {
                let a = ring_area(&poly.exterior).abs();
                let c = ring_centroid(&poly.exterior);
                total_area += a;
                cx += c.lon * a;
                cy += c.lat * a;
            }
            if total_area.abs() < 1e-20 {
                ring_centroid(&mp.polygons[0].exterior)
            } else {
                GeoPoint::new(cx / total_area, cy / total_area)
            }
        }
    }
}

/// Compute the bounding box of a geometry.
pub fn bounds(geom: &GeoGeometry) -> GeoBounds {
    let mut b = GeoBounds::empty();
    match geom {
        GeoGeometry::Point(p) => b.include(*p),
        GeoGeometry::LineString(ls) => {
            for p in &ls.points {
                b.include(*p);
            }
        }
        GeoGeometry::Polygon(poly) => {
            for p in &poly.exterior {
                b.include(*p);
            }
        }
        GeoGeometry::MultiPolygon(mp) => {
            for poly in &mp.polygons {
                for p in &poly.exterior {
                    b.include(*p);
                }
            }
        }
    }
    b
}

/// Test if a point is inside a polygon using the ray-casting algorithm.
///
/// Tests against the exterior ring and subtracts holes.
pub fn point_in_polygon(point: GeoPoint, polygon: &GeoPolygon) -> bool {
    if !point_in_ring(point, &polygon.exterior) {
        return false;
    }
    // Inside exterior, check holes
    for hole in &polygon.holes {
        if point_in_ring(point, hole) {
            return false;
        }
    }
    true
}

/// Ray-casting point-in-ring test.
fn point_in_ring(point: GeoPoint, ring: &[GeoPoint]) -> bool {
    let n = ring.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let px = point.lon;
    let py = point.lat;
    let mut j = n - 1;
    for i in 0..n {
        let yi = ring[i].lat;
        let yj = ring[j].lat;
        let xi = ring[i].lon;
        let xj = ring[j].lon;
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Compute the signed area of a ring using the shoelace formula.
///
/// Positive for counter-clockwise, negative for clockwise. Units are
/// square degrees (not meaningful for real-world area; use a projection first).
fn ring_area(ring: &[GeoPoint]) -> f64 {
    let n = ring.len();
    if n < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut j = n - 1;
    for i in 0..n {
        sum += (ring[j].lon - ring[i].lon) * (ring[j].lat + ring[i].lat);
        j = i;
    }
    sum * 0.5
}

/// Compute the unsigned area of a geometry in square degrees.
///
/// For real-world area in square meters, project coordinates first.
pub fn area(geom: &GeoGeometry) -> f64 {
    match geom {
        GeoGeometry::Point(_) | GeoGeometry::LineString(_) => 0.0,
        GeoGeometry::Polygon(poly) => {
            let mut a = ring_area(&poly.exterior).abs();
            for hole in &poly.holes {
                a -= ring_area(hole).abs();
            }
            a.max(0.0)
        }
        GeoGeometry::MultiPolygon(mp) => {
            mp.polygons
                .iter()
                .map(|poly| {
                    let mut a = ring_area(&poly.exterior).abs();
                    for hole in &poly.holes {
                        a -= ring_area(hole).abs();
                    }
                    a.max(0.0)
                })
                .sum()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{GeoMultiPolygon, GeoLineString};

    fn unit_square() -> GeoPolygon {
        GeoPolygon {
            exterior: vec![
                GeoPoint::new(0.0, 0.0),
                GeoPoint::new(1.0, 0.0),
                GeoPoint::new(1.0, 1.0),
                GeoPoint::new(0.0, 1.0),
                GeoPoint::new(0.0, 0.0),
            ],
            holes: vec![],
        }
    }

    fn triangle() -> GeoPolygon {
        GeoPolygon {
            exterior: vec![
                GeoPoint::new(0.0, 0.0),
                GeoPoint::new(4.0, 0.0),
                GeoPoint::new(2.0, 3.0),
                GeoPoint::new(0.0, 0.0),
            ],
            holes: vec![],
        }
    }

    #[test]
    fn centroid_of_point() {
        let p = GeoPoint::new(10.0, 20.0);
        let c = centroid(&GeoGeometry::Point(p));
        assert!((c.lon - 10.0).abs() < 1e-10);
        assert!((c.lat - 20.0).abs() < 1e-10);
    }

    #[test]
    fn centroid_of_polygon() {
        let c = centroid(&GeoGeometry::Polygon(unit_square()));
        assert!((c.lon - 0.5).abs() < 1e-6);
        assert!((c.lat - 0.5).abs() < 1e-6);
    }

    #[test]
    fn centroid_of_multipolygon() {
        let mp = GeoMultiPolygon {
            polygons: vec![unit_square()],
        };
        let c = centroid(&GeoGeometry::MultiPolygon(mp));
        assert!((c.lon - 0.5).abs() < 1e-6);
        assert!((c.lat - 0.5).abs() < 1e-6);
    }

    #[test]
    fn centroid_of_linestring() {
        let ls = GeoLineString {
            points: vec![GeoPoint::new(0.0, 0.0), GeoPoint::new(2.0, 4.0)],
        };
        let c = centroid(&GeoGeometry::LineString(ls));
        assert!((c.lon - 1.0).abs() < 1e-10);
        assert!((c.lat - 2.0).abs() < 1e-10);
    }

    #[test]
    fn bounds_of_polygon() {
        let b = bounds(&GeoGeometry::Polygon(unit_square()));
        assert!((b.min_lon).abs() < 1e-10);
        assert!((b.min_lat).abs() < 1e-10);
        assert!((b.max_lon - 1.0).abs() < 1e-10);
        assert!((b.max_lat - 1.0).abs() < 1e-10);
    }

    #[test]
    fn point_in_polygon_inside() {
        let poly = triangle();
        assert!(point_in_polygon(GeoPoint::new(2.0, 1.0), &poly));
    }

    #[test]
    fn point_in_polygon_outside() {
        let poly = triangle();
        assert!(!point_in_polygon(GeoPoint::new(5.0, 5.0), &poly));
    }

    #[test]
    fn point_in_polygon_with_hole() {
        let mut poly = unit_square();
        // Small hole in the center
        poly.holes.push(vec![
            GeoPoint::new(0.25, 0.25),
            GeoPoint::new(0.75, 0.25),
            GeoPoint::new(0.75, 0.75),
            GeoPoint::new(0.25, 0.75),
            GeoPoint::new(0.25, 0.25),
        ]);
        // Point in the hole — should be outside
        assert!(!point_in_polygon(GeoPoint::new(0.5, 0.5), &poly));
        // Point outside the hole but inside polygon
        assert!(point_in_polygon(GeoPoint::new(0.1, 0.1), &poly));
    }

    #[test]
    fn area_of_unit_square() {
        let a = area(&GeoGeometry::Polygon(unit_square()));
        assert!((a - 1.0).abs() < 1e-10);
    }

    #[test]
    fn area_of_triangle() {
        // Triangle: base=4, height=3, area=6
        let a = area(&GeoGeometry::Polygon(triangle()));
        assert!((a - 6.0).abs() < 1e-10);
    }

    #[test]
    fn area_of_point_is_zero() {
        let a = area(&GeoGeometry::Point(GeoPoint::new(0.0, 0.0)));
        assert!(a.abs() < 1e-20);
    }

    #[test]
    fn area_with_hole() {
        let mut poly = unit_square();
        // Hole is a 0.5×0.5 square: area = 0.25
        poly.holes.push(vec![
            GeoPoint::new(0.25, 0.25),
            GeoPoint::new(0.75, 0.25),
            GeoPoint::new(0.75, 0.75),
            GeoPoint::new(0.25, 0.75),
            GeoPoint::new(0.25, 0.25),
        ]);
        let a = area(&GeoGeometry::Polygon(poly));
        assert!((a - 0.75).abs() < 1e-10);
    }
}
