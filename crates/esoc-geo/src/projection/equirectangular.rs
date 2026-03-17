// SPDX-License-Identifier: MIT OR Apache-2.0
//! Equirectangular (Plate Carrée) projection — identity transform.

use super::Projection;

/// Equirectangular projection: lon → x, lat → y (identity in radians).
///
/// This is the simplest projection, mapping longitude directly to x and
/// latitude directly to y. Zero-sized type (no state).
pub struct Equirectangular;

impl Projection for Equirectangular {
    fn project(&self, lon: f64, lat: f64) -> (f64, f64) {
        let x = lon.to_radians();
        let y = lat.to_radians();
        (x, y)
    }

    fn invert(&self, x: f64, y: f64) -> (f64, f64) {
        let lon = x.to_degrees();
        let lat = y.to_degrees();
        (lon, lat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin() {
        let proj = Equirectangular;
        let (x, y) = proj.project(0.0, 0.0);
        assert!(x.abs() < 1e-10);
        assert!(y.abs() < 1e-10);
    }

    #[test]
    fn roundtrip() {
        let proj = Equirectangular;
        let (lon, lat) = (45.0, 30.0);
        let (x, y) = proj.project(lon, lat);
        let (lon2, lat2) = proj.invert(x, y);
        assert!((lon2 - lon).abs() < 1e-10);
        assert!((lat2 - lat).abs() < 1e-10);
    }

    #[test]
    fn known_values() {
        let proj = Equirectangular;
        let (x, y) = proj.project(180.0, 90.0);
        assert!((x - std::f64::consts::PI).abs() < 1e-10);
        assert!((y - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }
}
