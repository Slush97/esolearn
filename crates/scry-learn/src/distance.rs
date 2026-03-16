// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared distance functions.

use crate::sparse::SparseRow;

// ─── Dense distance functions ───────────────────────────────────────────────

/// Squared Euclidean distance between two slices.
///
/// Avoids the `sqrt` — monotonic, so it preserves ordering for
/// nearest-neighbor and centroid comparisons.
#[inline]
pub(crate) fn euclidean_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Manhattan (L1) distance between two slices.
#[inline]
pub(crate) fn manhattan(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Cosine distance: `1 − cos(θ)`, range `[0, 2]`.
///
/// Returns `1.0` when either vector has zero norm (treat as orthogonal).
#[inline]
pub(crate) fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f64::EPSILON {
        return 1.0; // One or both vectors are zero — treat as orthogonal.
    }
    1.0 - (dot / denom)
}

// ─── Sparse distance functions (merge-join on sorted index arrays) ──────────

/// Sparse dot product via two-pointer merge on sorted indices.
pub(crate) fn sparse_dot(a: &SparseRow<'_>, b: &SparseRow<'_>) -> f64 {
    let (a_idx, a_val) = (a.indices(), a.values());
    let (b_idx, b_val) = (b.indices(), b.values());
    let (mut i, mut j) = (0, 0);
    let mut dot = 0.0;
    while i < a_idx.len() && j < b_idx.len() {
        match a_idx[i].cmp(&b_idx[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                dot += a_val[i] * b_val[j];
                i += 1;
                j += 1;
            }
        }
    }
    dot
}

/// Squared L2 norm of a sparse row: `||a||² = Σ a_i²`.
#[inline]
pub(crate) fn sparse_norm_sq(a: &SparseRow<'_>) -> f64 {
    a.values().iter().map(|v| v * v).sum()
}

/// Sparse squared Euclidean distance: `d²(a,b) = ||a||² + ||b||² - 2·a·b`.
#[inline]
pub(crate) fn sparse_euclidean_sq(a: &SparseRow<'_>, b: &SparseRow<'_>) -> f64 {
    let d2 = sparse_norm_sq(a) + sparse_norm_sq(b) - 2.0 * sparse_dot(a, b);
    d2.max(0.0) // Guard against floating-point rounding
}

/// Sparse Manhattan distance via merge-join.
pub(crate) fn sparse_manhattan(a: &SparseRow<'_>, b: &SparseRow<'_>) -> f64 {
    let (a_idx, a_val) = (a.indices(), a.values());
    let (b_idx, b_val) = (b.indices(), b.values());
    let (mut i, mut j) = (0, 0);
    let mut dist = 0.0;
    while i < a_idx.len() && j < b_idx.len() {
        match a_idx[i].cmp(&b_idx[j]) {
            std::cmp::Ordering::Less => {
                dist += a_val[i].abs();
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                dist += b_val[j].abs();
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                dist += (a_val[i] - b_val[j]).abs();
                i += 1;
                j += 1;
            }
        }
    }
    while i < a_idx.len() {
        dist += a_val[i].abs();
        i += 1;
    }
    while j < b_idx.len() {
        dist += b_val[j].abs();
        j += 1;
    }
    dist
}

/// Sparse cosine distance: `1 − cos(θ)`.
#[inline]
pub(crate) fn sparse_cosine(a: &SparseRow<'_>, b: &SparseRow<'_>) -> f64 {
    let dot = sparse_dot(a, b);
    let norm_a = sparse_norm_sq(a).sqrt();
    let norm_b = sparse_norm_sq(b).sqrt();
    let denom = norm_a * norm_b;
    if denom < f64::EPSILON {
        return 1.0;
    }
    1.0 - (dot / denom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CsrMatrix;

    #[test]
    fn test_euclidean_sq() {
        let d = euclidean_sq(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!((d - 27.0).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan() {
        let d = manhattan(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!((d - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_distance_same_direction() {
        let d = cosine_distance(&[1.0, 0.0], &[100.0, 0.0]);
        assert!(d < 1e-9, "same direction should be ~0, got {d}");
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let d = cosine_distance(&[1.0, 0.0], &[0.0, 1.0]);
        assert!((d - 1.0).abs() < 1e-9, "orthogonal should be ~1, got {d}");
    }

    #[test]
    fn test_cosine_distance_zero_vector() {
        let d = cosine_distance(&[0.0, 0.0], &[1.0, 2.0]);
        assert!((d - 1.0).abs() < 1e-9, "zero vector should give 1.0");
    }

    #[test]
    fn test_sparse_dot() {
        let a = CsrMatrix::from_dense(&[vec![1.0, 0.0, 3.0]]);
        let b = CsrMatrix::from_dense(&[vec![2.0, 5.0, 4.0]]);
        let d = sparse_dot(&a.row(0), &b.row(0));
        // 1*2 + 0*5 + 3*4 = 14
        assert!((d - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_norm_sq() {
        let a = CsrMatrix::from_dense(&[vec![3.0, 0.0, 4.0]]);
        let n = sparse_norm_sq(&a.row(0));
        // 9 + 16 = 25
        assert!((n - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_euclidean_sq() {
        let a = CsrMatrix::from_dense(&[vec![1.0, 0.0, 3.0]]);
        let b = CsrMatrix::from_dense(&[vec![0.0, 2.0, 3.0]]);
        let d2 = sparse_euclidean_sq(&a.row(0), &b.row(0));
        assert!((d2 - 5.0).abs() < 1e-10, "Expected 5.0, got {d2}");
    }

    #[test]
    fn test_sparse_manhattan() {
        let a = CsrMatrix::from_dense(&[vec![1.0, 0.0, 3.0]]);
        let b = CsrMatrix::from_dense(&[vec![0.0, 2.0, 3.0]]);
        let d = sparse_manhattan(&a.row(0), &b.row(0));
        assert!((d - 3.0).abs() < 1e-10, "Expected 3.0, got {d}");
    }

    #[test]
    fn test_sparse_cosine() {
        let a = CsrMatrix::from_dense(&[vec![1.0, 0.0]]);
        let b = CsrMatrix::from_dense(&[vec![100.0, 0.0]]);
        let d = sparse_cosine(&a.row(0), &b.row(0));
        assert!(d < 1e-9, "Same direction should be ~0, got {d}");

        let c = CsrMatrix::from_dense(&[vec![0.0, 1.0]]);
        let d_orth = sparse_cosine(&a.row(0), &c.row(0));
        assert!(
            (d_orth - 1.0).abs() < 1e-9,
            "Orthogonal should be ~1, got {d_orth}"
        );
    }
}
