// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pre-bundled geographic datasets (feature-gated: `bundled`).
//!
//! Provides `world_countries()` and `us_states()` which decompress and parse
//! embedded `GeoJSON` data on first access, then cache the result forever via
//! `OnceLock`.

use crate::geometry::GeoCollection;
use std::sync::OnceLock;

static WORLD_COUNTRIES: OnceLock<GeoCollection> = OnceLock::new();
static US_STATES: OnceLock<GeoCollection> = OnceLock::new();

/// Natural Earth 110m world countries.
///
/// Returns a reference to a lazily decompressed and parsed `GeoCollection`.
/// Panics if the embedded data is corrupt (should not happen with valid builds).
pub fn world_countries() -> &'static GeoCollection {
    WORLD_COUNTRIES.get_or_init(|| {
        let compressed = include_bytes!("../data/world_110m.geojson.zst");
        let json = decompress(compressed);
        crate::geojson::parse(&json).expect("failed to parse bundled world_110m.geojson")
    })
}

/// US Census 20m state boundaries.
///
/// Returns a reference to a lazily decompressed and parsed `GeoCollection`.
/// Panics if the embedded data is corrupt (should not happen with valid builds).
pub fn us_states() -> &'static GeoCollection {
    US_STATES.get_or_init(|| {
        let compressed = include_bytes!("../data/us_states_20m.geojson.zst");
        let json = decompress(compressed);
        crate::geojson::parse(&json).expect("failed to parse bundled us_states_20m.geojson")
    })
}

fn decompress(data: &[u8]) -> String {
    let decoded = zstd::decode_all(data).expect("failed to decompress zstd data");
    String::from_utf8(decoded).expect("decompressed data is not valid UTF-8")
}

// Tests require the actual data files to be present.
// They are run via `cargo test -p esoc-geo --all-features`.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_countries_loads() {
        let coll = world_countries();
        // Natural Earth 110m has approximately 177 countries
        assert!(
            coll.features.len() > 150,
            "expected 150+ countries, got {}",
            coll.features.len()
        );
    }

    #[test]
    fn us_states_loads() {
        let coll = us_states();
        // 50 states + DC + territories
        assert!(
            coll.features.len() >= 50,
            "expected 50+ states, got {}",
            coll.features.len()
        );
    }
}
