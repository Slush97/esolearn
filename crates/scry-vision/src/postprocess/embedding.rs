// SPDX-License-Identifier: MIT OR Apache-2.0
//! L2 normalization for embedding vectors.

/// L2-normalize an embedding vector in place.
///
/// After normalization, the vector has unit L2 norm. If the input norm is
/// zero (or very close), the vector is left unchanged.
pub fn l2_normalize(embedding: &mut [f32]) {
    let norm = l2_norm(embedding);
    if norm > f32::EPSILON {
        let inv = 1.0 / norm;
        for v in embedding.iter_mut() {
            *v *= inv;
        }
    }
}

/// Return a new L2-normalized copy of the embedding.
pub fn l2_normalized(embedding: &[f32]) -> Vec<f32> {
    let mut out = embedding.to_vec();
    l2_normalize(&mut out);
    out
}

/// Compute the L2 norm of a vector.
#[must_use]
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Cosine similarity between two vectors.
///
/// Returns `a . b / (|a| * |b|)`. If either vector has zero norm, returns 0.0.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    let denom = norm_a * norm_b;
    if denom < f32::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_normalize_unit_vector() {
        let mut v = vec![1.0, 0.0, 0.0];
        l2_normalize(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1]).abs() < 1e-6);
        assert!((v[2]).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_general() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        let norm = l2_norm(&v);
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut v);
        // Should remain zero, not NaN
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn l2_normalized_returns_copy() {
        let v = vec![3.0, 4.0];
        let n = l2_normalized(&v);
        // Original unchanged
        assert!((v[0] - 3.0).abs() < 1e-6);
        // Normalized copy
        assert!((l2_norm(&n) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_opposite() {
        let a = vec![1.0, 2.0];
        let b = vec![-1.0, -2.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-6);
    }
}
