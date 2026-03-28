// SPDX-License-Identifier: MIT OR Apache-2.0
//! Albers USA composite projection: lower 48 + Alaska + Hawaii insets.

use super::Projection;

/// Albers equal-area conic projection (private helper).
struct AlbersConic {
    /// Central meridian (radians).
    lam0: f64,
    // Derived constants
    n: f64,
    c: f64,
    rho0: f64,
}

impl AlbersConic {
    fn new(phi1_deg: f64, phi2_deg: f64, lam0_deg: f64, phi0_deg: f64) -> Self {
        let phi1 = phi1_deg.to_radians();
        let phi2 = phi2_deg.to_radians();
        let lam0 = lam0_deg.to_radians();
        let phi0 = phi0_deg.to_radians();

        let n = (phi1.sin() + phi2.sin()) * 0.5;
        let c = phi1.cos() * phi1.cos() + 2.0 * n * phi1.sin();
        let rho0 = (c - 2.0 * n * phi0.sin()).abs().sqrt() / n;

        Self { lam0, n, c, rho0 }
    }

    fn project(&self, lon: f64, lat: f64) -> (f64, f64) {
        let lam = lon.to_radians();
        let phi = lat.to_radians();
        let theta = self.n * (lam - self.lam0);
        let rho_val = (self.c - 2.0 * self.n * phi.sin()).abs().sqrt() / self.n;
        let x = rho_val * theta.sin();
        let y = self.rho0 - rho_val * theta.cos();
        (x, y)
    }

    fn invert(&self, x: f64, y: f64) -> (f64, f64) {
        let rho0_minus_y = self.rho0 - y;
        let rho = x.hypot(rho0_minus_y) * self.n.signum();
        let theta = (x).atan2(rho0_minus_y);

        let sin_phi = ((self.c - rho * rho * self.n * self.n) / (2.0 * self.n)).clamp(-1.0, 1.0);
        let phi = sin_phi.asin();
        let lam = theta / self.n + self.lam0;

        (lam.to_degrees(), phi.to_degrees())
    }
}

/// Albers USA composite projection.
///
/// Routes points through one of three Albers equal-area conic sub-projections
/// based on longitude range:
/// - **Lower 48**: standard parallels 29.5°/45.5°, center −98°, origin 38°
/// - **Alaska**: standard parallels 55°/65°, center −154°, origin 50°, scaled 0.35× and offset
/// - **Hawaii**: standard parallels 8°/18°, center −157°, origin 13°, scaled 0.35× and offset
pub struct AlbersUsa {
    lower48: AlbersConic,
    alaska: AlbersConic,
    hawaii: AlbersConic,
}

impl Default for AlbersUsa {
    fn default() -> Self {
        Self::new()
    }
}

impl AlbersUsa {
    /// Create with default inset positions.
    pub fn new() -> Self {
        Self {
            lower48: AlbersConic::new(29.5, 45.5, -98.0, 38.0),
            alaska: AlbersConic::new(55.0, 65.0, -154.0, 50.0),
            hawaii: AlbersConic::new(8.0, 18.0, -157.0, 13.0),
        }
    }

    /// Determine which sub-projection handles a given (lon, lat).
    #[allow(clippy::unused_self)]
    fn route(&self, lon: f64, lat: f64) -> Region {
        // Hawaii: lat < 25 and lon between -180 and -154
        if lat < 25.0 && lon > -180.0 && lon < -154.0 {
            return Region::Hawaii;
        }
        // Alaska: lat > 50 and (lon < -130 or lon > 170)
        // The Aleutian Islands cross the antimeridian
        if lat > 50.0 && !(-130.0..=170.0).contains(&lon) {
            return Region::Alaska;
        }
        Region::Lower48
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Region {
    Lower48,
    Alaska,
    Hawaii,
}

// Inset scale and offset for Alaska and Hawaii
const ALASKA_SCALE: f64 = 0.35;
const ALASKA_DX: f64 = -0.25;
const ALASKA_DY: f64 = -0.37;

const HAWAII_SCALE: f64 = 0.35;
const HAWAII_DX: f64 = -0.07;
const HAWAII_DY: f64 = -0.35;

impl Projection for AlbersUsa {
    fn project(&self, lon: f64, lat: f64) -> (f64, f64) {
        match self.route(lon, lat) {
            Region::Lower48 => self.lower48.project(lon, lat),
            Region::Alaska => {
                let (x, y) = self.alaska.project(lon, lat);
                (x * ALASKA_SCALE + ALASKA_DX, y * ALASKA_SCALE + ALASKA_DY)
            }
            Region::Hawaii => {
                let (x, y) = self.hawaii.project(lon, lat);
                (x * HAWAII_SCALE + HAWAII_DX, y * HAWAII_SCALE + HAWAII_DY)
            }
        }
    }

    fn invert(&self, x: f64, y: f64) -> (f64, f64) {
        // Try lower 48 first (most common), then check if the result makes sense
        let (lon, lat) = self.lower48.invert(x, y);
        if (24.5..=50.0).contains(&lat) && (-125.0..=-66.0).contains(&lon) {
            return (lon, lat);
        }

        // Try Alaska
        let ax = (x - ALASKA_DX) / ALASKA_SCALE;
        let ay = (y - ALASKA_DY) / ALASKA_SCALE;
        let (alon, alat) = self.alaska.invert(ax, ay);
        if alat > 50.0 && !(-130.0..=170.0).contains(&alon) {
            return (alon, alat);
        }

        // Try Hawaii
        let hx = (x - HAWAII_DX) / HAWAII_SCALE;
        let hy = (y - HAWAII_DY) / HAWAII_SCALE;
        let (hlon, hlat) = self.hawaii.invert(hx, hy);
        if hlat < 25.0 && hlon > -180.0 && hlon < -154.0 {
            return (hlon, hlat);
        }

        // Fallback to lower 48
        (lon, lat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower48_roundtrip() {
        let proj = AlbersUsa::new();
        let (lon, lat) = (-98.0, 38.0); // center of lower 48
        let (x, y) = proj.project(lon, lat);
        let (lon2, lat2) = proj.invert(x, y);
        assert!(
            (lon2 - lon).abs() < 1e-4,
            "lon roundtrip: expected {lon}, got {lon2}"
        );
        assert!(
            (lat2 - lat).abs() < 1e-4,
            "lat roundtrip: expected {lat}, got {lat2}"
        );
    }

    #[test]
    fn alaska_routes_correctly() {
        let proj = AlbersUsa::new();
        let region = proj.route(-150.0, 64.0);
        assert_eq!(region, Region::Alaska);
    }

    #[test]
    fn hawaii_routes_correctly() {
        let proj = AlbersUsa::new();
        let region = proj.route(-157.0, 21.0);
        assert_eq!(region, Region::Hawaii);
    }

    #[test]
    fn lower48_routes_correctly() {
        let proj = AlbersUsa::new();
        let region = proj.route(-98.0, 38.0);
        assert_eq!(region, Region::Lower48);
    }

    #[test]
    fn alaska_aleutian_antimeridian() {
        let proj = AlbersUsa::new();
        // Attu Island (westernmost Aleutian), crosses antimeridian
        let region = proj.route(172.0, 52.9);
        assert_eq!(region, Region::Alaska);
    }

    #[test]
    fn new_york() {
        let proj = AlbersUsa::new();
        let (x, y) = proj.project(-74.006, 40.7128);
        // Should produce a reasonable projected coordinate
        assert!(x.is_finite());
        assert!(y.is_finite());
        let (lon2, lat2) = proj.invert(x, y);
        assert!((lon2 - (-74.006)).abs() < 0.5);
        assert!((lat2 - 40.7128).abs() < 0.5);
    }
}
