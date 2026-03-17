// SPDX-License-Identifier: MIT OR Apache-2.0
//! Map projections: geographic (lon, lat) → projected (x, y).

pub mod albers_usa;
pub mod equal_earth;
pub mod equirectangular;
pub mod mercator;
pub mod natural_earth;

pub use albers_usa::AlbersUsa;
pub use equal_earth::EqualEarth;
pub use equirectangular::Equirectangular;
pub use mercator::Mercator;
pub use natural_earth::NaturalEarth1;

/// A map projection that converts between geographic and projected coordinates.
pub trait Projection {
    /// Project geographic coordinates (lon, lat) in degrees to (x, y).
    fn project(&self, lon: f64, lat: f64) -> (f64, f64);

    /// Inverse: projected (x, y) back to (lon, lat) in degrees.
    fn invert(&self, x: f64, y: f64) -> (f64, f64);
}
