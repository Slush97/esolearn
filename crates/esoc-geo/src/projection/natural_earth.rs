// SPDX-License-Identifier: MIT OR Apache-2.0
//! Natural Earth I projection (polynomial pseudocylindrical).

use super::Projection;

// Polynomial coefficients for Natural Earth I (Šavrič et al.)
const A0: f64 = 0.870_700;
const A1: f64 = -0.131_979;
const A2: f64 = -0.013_791;
const A3: f64 = 0.003_971;
const A4: f64 = -0.001_529;

const B0: f64 = 1.007_226;
const B1: f64 = 0.015_085;
const B2: f64 = -0.044_475;
const B3: f64 = 0.028_874;
const B4: f64 = -0.005_916;

/// Natural Earth I projection.
///
/// A visually appealing pseudocylindrical projection using polynomial
/// approximations. Zero-sized type.
pub struct NaturalEarth1;

impl Projection for NaturalEarth1 {
    fn project(&self, lon: f64, lat: f64) -> (f64, f64) {
        let lam = lon.to_radians();
        let phi = lat.to_radians();
        let phi2 = phi * phi;

        let x = lam * (A0 + phi2 * (A1 + phi2 * (A2 + phi2 * (A3 + phi2 * A4))));
        let y = phi * (B0 + phi2 * (B1 + phi2 * (B2 + phi2 * (B3 + phi2 * B4))));
        (x, y)
    }

    fn invert(&self, x: f64, y: f64) -> (f64, f64) {
        // Newton iteration to find φ from y
        let mut phi = y;
        for _ in 0..12 {
            let phi2 = phi * phi;
            let f = phi * (B0 + phi2 * (B1 + phi2 * (B2 + phi2 * (B3 + phi2 * B4)))) - y;
            let df =
                B0 + phi2 * (3.0 * B1 + phi2 * (5.0 * B2 + phi2 * (7.0 * B3 + 9.0 * phi2 * B4)));
            if df.abs() < 1e-20 {
                break;
            }
            let dphi = f / df;
            phi -= dphi;
            if dphi.abs() < 1e-12 {
                break;
            }
        }

        let phi2 = phi * phi;
        let denom = A0 + phi2 * (A1 + phi2 * (A2 + phi2 * (A3 + phi2 * A4)));
        let lam = if denom.abs() < 1e-20 { 0.0 } else { x / denom };

        (lam.to_degrees(), phi.to_degrees())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin() {
        let proj = NaturalEarth1;
        let (x, y) = proj.project(0.0, 0.0);
        assert!(x.abs() < 1e-10);
        assert!(y.abs() < 1e-10);
    }

    #[test]
    fn roundtrip() {
        let proj = NaturalEarth1;
        for &(lon, lat) in &[
            (0.0, 0.0),
            (45.0, 30.0),
            (-120.0, -60.0),
            (180.0, 0.0),
            (0.0, 80.0),
        ] {
            let (x, y) = proj.project(lon, lat);
            let (lon2, lat2) = proj.invert(x, y);
            assert!(
                (lon2 - lon).abs() < 1e-6,
                "lon roundtrip failed for ({lon}, {lat}): got {lon2}"
            );
            assert!(
                (lat2 - lat).abs() < 1e-6,
                "lat roundtrip failed for ({lon}, {lat}): got {lat2}"
            );
        }
    }

    #[test]
    fn symmetric() {
        let proj = NaturalEarth1;
        let (x1, y1) = proj.project(0.0, 45.0);
        let (x2, y2) = proj.project(0.0, -45.0);
        assert!((x1 - x2).abs() < 1e-10);
        assert!((y1 + y2).abs() < 1e-10);
    }
}
