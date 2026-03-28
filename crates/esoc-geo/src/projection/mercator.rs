// SPDX-License-Identifier: MIT OR Apache-2.0
//! Web Mercator projection with latitude clamping at ±85.0511°.

use super::Projection;

/// Maximum latitude for Web Mercator (atan(sinh(π)) in degrees).
const MAX_LAT: f64 = 85.051_129;

/// Web Mercator projection.
///
/// Clamps latitude to ±85.0511° to prevent infinite y values at the poles.
/// Zero-sized type (no state).
pub struct Mercator;

impl Projection for Mercator {
    fn project(&self, lon: f64, lat: f64) -> (f64, f64) {
        let lat_clamped = lat.clamp(-MAX_LAT, MAX_LAT);
        let x = lon.to_radians();
        let y = (lat_clamped.to_radians() * 0.5 + std::f64::consts::FRAC_PI_4)
            .tan()
            .ln();
        (x, y)
    }

    fn invert(&self, x: f64, y: f64) -> (f64, f64) {
        let lon = x.to_degrees();
        let lat = (2.0 * y.exp().atan() - std::f64::consts::FRAC_PI_2).to_degrees();
        (lon, lat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin() {
        let proj = Mercator;
        let (x, y) = proj.project(0.0, 0.0);
        assert!(x.abs() < 1e-10);
        assert!(y.abs() < 1e-10);
    }

    #[test]
    fn roundtrip() {
        let proj = Mercator;
        for &(lon, lat) in &[(45.0, 30.0), (-120.0, -45.0), (0.0, 85.0), (180.0, 0.0)] {
            let (x, y) = proj.project(lon, lat);
            let (lon2, lat2) = proj.invert(x, y);
            assert!(
                (lon2 - lon).abs() < 1e-6,
                "lon roundtrip failed for ({lon}, {lat})"
            );
            assert!(
                (lat2 - lat).abs() < 1e-4,
                "lat roundtrip failed for ({lon}, {lat})"
            );
        }
    }

    #[test]
    fn latitude_clamped() {
        let proj = Mercator;
        let (_, y90) = proj.project(0.0, 90.0);
        let (_, y85) = proj.project(0.0, MAX_LAT);
        assert!(
            (y90 - y85).abs() < 1e-6,
            "90° should be clamped to 85.0511°"
        );
    }

    #[test]
    fn symmetric() {
        let proj = Mercator;
        let (x1, y1) = proj.project(45.0, 30.0);
        let (x2, y2) = proj.project(-45.0, -30.0);
        assert!((x1 + x2).abs() < 1e-10);
        assert!((y1 + y2).abs() < 1e-10);
    }
}
