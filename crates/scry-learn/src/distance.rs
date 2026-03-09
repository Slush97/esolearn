// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared distance functions.

/// Squared Euclidean distance between two slices.
///
/// Avoids the `sqrt` — monotonic, so it preserves ordering for
/// nearest-neighbor and centroid comparisons.
#[inline]
pub(crate) fn euclidean_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}
