// SPDX-License-Identifier: MIT OR Apache-2.0
//! Homography estimation via DLT with Hartley normalization + RANSAC.

use crate::error::{Result, ScryVisionError};
use crate::registration::ransac::{ransac, RansacConfig, RansacModel, RansacResult};

/// A 3x3 homography matrix (stored in row-major order).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Homography {
    /// Row-major 3x3 matrix elements.
    pub data: [f64; 9],
}

/// A point correspondence (source, destination).
#[derive(Clone, Debug)]
pub struct PointPair {
    /// Source point.
    pub src: (f64, f64),
    /// Destination point.
    pub dst: (f64, f64),
}

/// Result of homography estimation with RANSAC.
#[derive(Clone, Debug)]
pub struct HomographyResult {
    /// The estimated homography.
    pub h: Homography,
    /// Inlier mask.
    pub inlier_mask: Vec<bool>,
    /// Number of inliers.
    pub n_inliers: usize,
}

impl Homography {
    /// Apply this homography to a point, returning the transformed point.
    pub fn transform_point(&self, x: f64, y: f64) -> (f64, f64) {
        let d = &self.data;
        let w = d[6] * x + d[7] * y + d[8];
        if w.abs() < 1e-12 {
            return (f64::NAN, f64::NAN);
        }
        let tx = (d[0] * x + d[1] * y + d[2]) / w;
        let ty = (d[3] * x + d[4] * y + d[5]) / w;
        (tx, ty)
    }
}

impl RansacModel for Homography {
    type Point = PointPair;
    const MIN_SAMPLES: usize = 4;

    fn estimate(points: &[Self::Point]) -> Option<Self> {
        estimate_dlt(points)
    }

    fn error(&self, point: &Self::Point) -> f64 {
        let (tx, ty) = self.transform_point(point.src.0, point.src.1);
        let dx = tx - point.dst.0;
        let dy = ty - point.dst.1;
        dx.hypot(dy)
    }
}

/// Estimate a homography from 4+ point correspondences using Direct Linear Transform.
///
/// Includes Hartley normalization for numerical stability.
pub fn estimate_dlt(pairs: &[PointPair]) -> Option<Homography> {
    if pairs.len() < 4 {
        return None;
    }

    // Hartley normalization
    let (src_norm, t_src) = normalize_points(pairs.iter().map(|p| p.src));
    let (dst_norm, t_dst) = normalize_points(pairs.iter().map(|p| p.dst));

    let n = pairs.len();
    // Build the 2n x 9 matrix A
    let mut ata = [0.0f64; 81]; // 9x9 A^T * A (normal equations)

    for i in 0..n {
        let (x, y) = src_norm[i];
        let (u, v) = dst_norm[i];

        // Row 1: [x, y, 1, 0, 0, 0, -u*x, -u*y, -u]
        let r1 = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u];
        // Row 2: [0, 0, 0, x, y, 1, -v*x, -v*y, -v]
        let r2 = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v];

        // Accumulate A^T * A
        for j in 0..9 {
            for k in 0..9 {
                ata[j * 9 + k] += r1[j] * r1[k] + r2[j] * r2[k];
            }
        }
    }

    // Find the eigenvector of A^T*A with smallest eigenvalue via inverse power iteration
    let h_norm = smallest_eigenvector_9x9(&ata)?;

    // Denormalize: H = T_dst^{-1} * H_norm * T_src
    let h_normalized = [
        h_norm[0], h_norm[1], h_norm[2],
        h_norm[3], h_norm[4], h_norm[5],
        h_norm[6], h_norm[7], h_norm[8],
    ];

    let h = denormalize_homography(&h_normalized, &t_src, &t_dst);

    // Normalize so h[8] = 1
    if h[8].abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / h[8];
    let data = [
        h[0] * inv, h[1] * inv, h[2] * inv,
        h[3] * inv, h[4] * inv, h[5] * inv,
        h[6] * inv, h[7] * inv, 1.0,
    ];

    Some(Homography { data })
}

/// Estimate homography with RANSAC for outlier rejection.
pub fn find_homography(
    pairs: &[PointPair],
    config: &RansacConfig,
) -> Result<HomographyResult> {
    if pairs.len() < 4 {
        return Err(ScryVisionError::InsufficientData(
            "need at least 4 point correspondences for homography".into(),
        ));
    }

    let result: RansacResult<Homography> = ransac(pairs, config).ok_or(
        ScryVisionError::ConvergenceFailure {
            iterations: config.max_iterations as usize,
            tolerance: config.threshold,
        },
    )?;

    Ok(HomographyResult {
        h: result.model,
        inlier_mask: result.inlier_mask,
        n_inliers: result.n_inliers,
    })
}

// ── Internal helpers ──

/// Hartley normalization: translate centroid to origin, scale avg distance to sqrt(2).
fn normalize_points(
    points: impl Iterator<Item = (f64, f64)> + Clone,
) -> (Vec<(f64, f64)>, [f64; 9]) {
    let pts: Vec<(f64, f64)> = points.collect();
    let n = pts.len() as f64;

    let cx: f64 = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let cy: f64 = pts.iter().map(|p| p.1).sum::<f64>() / n;

    let avg_dist: f64 = pts
        .iter()
        .map(|p| (p.0 - cx).hypot(p.1 - cy))
        .sum::<f64>()
        / n;

    let s = if avg_dist > 1e-10 {
        std::f64::consts::SQRT_2 / avg_dist
    } else {
        1.0
    };

    let normalized: Vec<(f64, f64)> = pts
        .iter()
        .map(|p| ((p.0 - cx) * s, (p.1 - cy) * s))
        .collect();

    // Normalization matrix T: [s, 0, -s*cx; 0, s, -s*cy; 0, 0, 1]
    let t = [s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0];

    (normalized, t)
}

/// Denormalize: H = `T_dst_inv` * `H_norm` * `T_src`
fn denormalize_homography(h: &[f64; 9], t_src: &[f64; 9], t_dst: &[f64; 9]) -> [f64; 9] {
    // T_dst_inv
    let s = t_dst[0];
    let tx = t_dst[2];
    let ty = t_dst[5];
    let inv_s = if s.abs() > 1e-12 { 1.0 / s } else { 1.0 };
    let t_dst_inv = [
        inv_s, 0.0, -tx * inv_s,
        0.0, inv_s, -ty * inv_s,
        0.0, 0.0, 1.0,
    ];

    let tmp = mat3_mul(h, t_src);
    mat3_mul(&t_dst_inv, &tmp)
}

fn mat3_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut c = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            c[i * 3 + j] =
                a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
        }
    }
    c
}

/// Find the eigenvector corresponding to the smallest eigenvalue of a 9x9 symmetric matrix.
/// Uses inverse power iteration with shift.
fn smallest_eigenvector_9x9(ata: &[f64; 81]) -> Option<[f64; 9]> {
    // Start with a random unit vector
    let mut v = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0f64];
    let norm = vec_norm_9(&v);
    for x in &mut v {
        *x /= norm;
    }

    // Power iteration on (A^T A)^{-1} to find smallest eigenvector
    // Since direct inversion of 9x9 is complex, use iterative approach:
    // Solve (A^T A) x = v via Gauss-Seidel iteration
    for _ in 0..100 {
        let new_v = gauss_seidel_solve_9x9(ata, &v, 50);
        let n = vec_norm_9(&new_v);
        if n < 1e-15 {
            return None;
        }
        for i in 0..9 {
            v[i] = new_v[i] / n;
        }
    }

    Some(v)
}

fn gauss_seidel_solve_9x9(a: &[f64; 81], b: &[f64; 9], iters: usize) -> [f64; 9] {
    let mut x = *b;
    for _ in 0..iters {
        for i in 0..9 {
            let mut sum = b[i];
            for j in 0..9 {
                if j != i {
                    sum -= a[i * 9 + j] * x[j];
                }
            }
            let diag = a[i * 9 + i];
            if diag.abs() > 1e-15 {
                x[i] = sum / diag;
            }
        }
    }
    x
}

fn vec_norm_9(v: &[f64; 9]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_homography() {
        let pairs: Vec<PointPair> = vec![
            PointPair { src: (0.0, 0.0), dst: (0.0, 0.0) },
            PointPair { src: (100.0, 0.0), dst: (100.0, 0.0) },
            PointPair { src: (100.0, 100.0), dst: (100.0, 100.0) },
            PointPair { src: (0.0, 100.0), dst: (0.0, 100.0) },
        ];
        let h = estimate_dlt(&pairs).unwrap();

        // Should be close to identity
        for p in &pairs {
            let (tx, ty) = h.transform_point(p.src.0, p.src.1);
            assert!(
                (tx - p.dst.0).abs() < 1.0 && (ty - p.dst.1).abs() < 1.0,
                "identity H should map ({},{}) to ({},{}), got ({tx},{ty})",
                p.src.0, p.src.1, p.dst.0, p.dst.1
            );
        }
    }

    #[test]
    fn translation_homography() {
        let tx_off = 10.0;
        let ty_off = 20.0;
        let pairs: Vec<PointPair> = vec![
            PointPair { src: (0.0, 0.0), dst: (tx_off, ty_off) },
            PointPair { src: (100.0, 0.0), dst: (100.0 + tx_off, ty_off) },
            PointPair { src: (100.0, 100.0), dst: (100.0 + tx_off, 100.0 + ty_off) },
            PointPair { src: (0.0, 100.0), dst: (tx_off, 100.0 + ty_off) },
        ];
        let h = estimate_dlt(&pairs).unwrap();

        let (rx, ry) = h.transform_point(50.0, 50.0);
        assert!(
            (rx - 60.0).abs() < 1.0 && (ry - 70.0).abs() < 1.0,
            "translation: expected (60,70), got ({rx},{ry})"
        );
    }

    #[test]
    fn ransac_homography_with_outliers() {
        // True translation: (dx=5, dy=10)
        let mut pairs: Vec<PointPair> = (0..20)
            .map(|i| {
                let x = (i * 10) as f64;
                let y = (i * 7) as f64;
                PointPair {
                    src: (x, y),
                    dst: (x + 5.0, y + 10.0),
                }
            })
            .collect();
        // Add outliers
        pairs.push(PointPair { src: (50.0, 50.0), dst: (200.0, 300.0) });
        pairs.push(PointPair { src: (80.0, 80.0), dst: (-100.0, -200.0) });

        let config = RansacConfig {
            max_iterations: 500,
            threshold: 5.0,
            confidence: 0.999,
            seed: 42,
        };

        let result = find_homography(&pairs, &config).unwrap();
        assert!(result.n_inliers >= 8, "should have >= 8 inliers, got {}", result.n_inliers);

        // The RANSAC should at least reject the obvious outliers
        assert!(!result.inlier_mask[20], "outlier at index 20 should be rejected");
        assert!(!result.inlier_mask[21], "outlier at index 21 should be rejected");
    }
}
