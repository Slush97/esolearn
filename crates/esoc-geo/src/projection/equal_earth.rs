// SPDX-License-Identifier: MIT OR Apache-2.0
//! Equal Earth projection (Šavrič, Patterson, Jenny 2018).

use super::Projection;

// Polynomial coefficients for Equal Earth
const A1: f64 = 1.340_264;
const A2: f64 = -0.081_106;
const A3: f64 = 0.000_893;
const A4: f64 = 0.003_796;

/// The parametric latitude sqrt(3)/2 factor.
const M: f64 = 0.866_025_403_784_438_6; // sqrt(3)/2

/// Equal Earth projection.
///
/// An equal-area pseudocylindrical projection designed as a visually
/// pleasing alternative to Gall-Peters. Zero-sized type.
pub struct EqualEarth;

/// Evaluate the parametric latitude θ from geographic latitude φ.
/// θ = asin(sqrt(3)/2 * sin(φ))
fn theta(lat_rad: f64) -> f64 {
    (M * lat_rad.sin()).asin()
}

impl Projection for EqualEarth {
    fn project(&self, lon: f64, lat: f64) -> (f64, f64) {
        let lam = lon.to_radians();
        let phi = lat.to_radians();
        let t = theta(phi);
        let t2 = t * t;
        let t6 = t2 * t2 * t2;

        let x = lam * t.cos() / (M * (A1 + 3.0 * A2 * t2 + t6 * (7.0 * A3 + 9.0 * A4 * t2)));
        let y = t * (A1 + A2 * t2 + t6 * (A3 + A4 * t2));
        (x, y)
    }

    fn invert(&self, x: f64, y: f64) -> (f64, f64) {
        // Newton iteration to find θ from y
        let mut t = y; // initial guess
        for _ in 0..12 {
            let t2 = t * t;
            let t6 = t2 * t2 * t2;
            let f = t * (A1 + A2 * t2 + t6 * (A3 + A4 * t2)) - y;
            let df = A1 + 3.0 * A2 * t2 + t6 * (7.0 * A3 + 9.0 * A4 * t2);
            if df.abs() < 1e-20 {
                break;
            }
            let dt = f / df;
            t -= dt;
            if dt.abs() < 1e-12 {
                break;
            }
        }

        let t2 = t * t;
        let t6 = t2 * t2 * t2;
        let denom = t.cos() / (M * (A1 + 3.0 * A2 * t2 + t6 * (7.0 * A3 + 9.0 * A4 * t2)));
        let lam = if denom.abs() < 1e-20 { 0.0 } else { x / denom };
        // φ = asin(sin(θ) / (sqrt(3)/2))
        let sin_t = t.sin();
        let sin_phi = (sin_t / M).clamp(-1.0, 1.0);
        let phi = sin_phi.asin();

        (lam.to_degrees(), phi.to_degrees())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin() {
        let proj = EqualEarth;
        let (x, y) = proj.project(0.0, 0.0);
        assert!(x.abs() < 1e-10);
        assert!(y.abs() < 1e-10);
    }

    #[test]
    fn roundtrip() {
        let proj = EqualEarth;
        for &(lon, lat) in &[
            (0.0, 0.0),
            (45.0, 30.0),
            (-120.0, -60.0),
            (180.0, 0.0),
            (0.0, 89.0),
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
    fn symmetric_about_equator() {
        let proj = EqualEarth;
        let (x1, y1) = proj.project(0.0, 45.0);
        let (x2, y2) = proj.project(0.0, -45.0);
        assert!((x1 - x2).abs() < 1e-10);
        assert!((y1 + y2).abs() < 1e-10);
    }
}
