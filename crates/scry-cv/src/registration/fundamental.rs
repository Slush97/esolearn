// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fundamental matrix estimation (8-point, 7-point) with RANSAC.
//!
//! The fundamental matrix **F** encodes the epipolar geometry between two views:
//! for a point correspondence `(x, x')`, the constraint `x'^T F x = 0` holds.
//!
//! # Algorithms
//!
//! - **8-point** (Hartley-normalized): requires ≥8 correspondences, returns a
//!   single F with rank-2 enforced via SVD.
//! - **7-point**: uses exactly 7 correspondences and the `det(F)=0` cubic
//!   constraint, returning 1 or 3 real solutions.
//! - **RANSAC wrapper** ([`find_fundamental`]) for robust estimation with
//!   outlier rejection. Uses Sampson distance as the error metric.

use crate::error::{Result, ScryVisionError};
use crate::registration::ransac::{ransac, RansacConfig, RansacModel, RansacResult};

/// A 3×3 fundamental matrix (row-major).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FundamentalMatrix {
    /// Row-major 3×3 matrix elements.
    pub data: [f64; 9],
}

/// Result of fundamental matrix estimation with RANSAC.
#[derive(Clone, Debug)]
pub struct FundamentalResult {
    /// The estimated fundamental matrix.
    pub f: FundamentalMatrix,
    /// Inlier mask.
    pub inlier_mask: Vec<bool>,
    /// Number of inliers.
    pub n_inliers: usize,
}

/// A point correspondence for epipolar geometry.
#[derive(Clone, Debug)]
pub struct EpipolarPair {
    /// Point in the first image.
    pub p1: (f64, f64),
    /// Point in the second image.
    pub p2: (f64, f64),
}

impl FundamentalMatrix {
    /// Compute the Sampson distance for a point correspondence.
    ///
    /// This is a first-order approximation to the geometric (reprojection) error
    /// and is the recommended error metric for RANSAC on F.
    pub fn sampson_distance(&self, p1: (f64, f64), p2: (f64, f64)) -> f64 {
        let f = &self.data;
        let (x1, y1) = p1;
        let (x2, y2) = p2;

        // Epipolar constraint: p2^T F p1
        let pfp = x2 * (f[0] * x1 + f[1] * y1 + f[2])
            + y2 * (f[3] * x1 + f[4] * y1 + f[5])
            + (f[6] * x1 + f[7] * y1 + f[8]);

        // Fp1 = F * p1
        let fp1_0 = f[0] * x1 + f[1] * y1 + f[2];
        let fp1_1 = f[3] * x1 + f[4] * y1 + f[5];

        // Ft_p2 = F^T * p2
        let ft_p2_0 = f[0] * x2 + f[3] * y2 + f[6];
        let ft_p2_1 = f[1] * x2 + f[4] * y2 + f[7];

        let denom = fp1_0 * fp1_0 + fp1_1 * fp1_1 + ft_p2_0 * ft_p2_0 + ft_p2_1 * ft_p2_1;
        if denom < 1e-30 {
            return f64::MAX;
        }

        (pfp * pfp) / denom
    }

    /// Compute the epipolar line `l = F * p` in the second image for a point in
    /// the first image. Returns `(a, b, c)` such that `ax + by + c = 0`.
    pub fn epipolar_line(&self, p: (f64, f64)) -> (f64, f64, f64) {
        let f = &self.data;
        let (x, y) = p;
        (
            f[0] * x + f[1] * y + f[2],
            f[3] * x + f[4] * y + f[5],
            f[6] * x + f[7] * y + f[8],
        )
    }
}

// ── RANSAC integration ──

impl RansacModel for FundamentalMatrix {
    type Point = EpipolarPair;
    const MIN_SAMPLES: usize = 8;

    fn estimate(points: &[Self::Point]) -> Option<Self> {
        estimate_8point(points)
    }

    fn error(&self, point: &Self::Point) -> f64 {
        self.sampson_distance(point.p1, point.p2)
    }
}

/// Estimate fundamental matrix with RANSAC for outlier rejection.
///
/// Requires at least 8 point correspondences. The error metric is Sampson
/// distance (squared, in pixels²) — set `config.threshold` accordingly
/// (e.g. 3.84 for a χ² test at 95% confidence with 1 DOF).
pub fn find_fundamental(
    pairs: &[EpipolarPair],
    config: &RansacConfig,
) -> Result<FundamentalResult> {
    if pairs.len() < 8 {
        return Err(ScryVisionError::InsufficientData(
            "need at least 8 point correspondences for fundamental matrix".into(),
        ));
    }

    let result: RansacResult<FundamentalMatrix> =
        ransac(pairs, config).ok_or(ScryVisionError::ConvergenceFailure {
            iterations: config.max_iterations as usize,
            tolerance: config.threshold,
        })?;

    Ok(FundamentalResult {
        f: result.model,
        inlier_mask: result.inlier_mask,
        n_inliers: result.n_inliers,
    })
}

// ── 8-point algorithm ──

/// Estimate F from ≥8 correspondences using the normalized 8-point algorithm.
///
/// Applies Hartley normalization for numerical stability and enforces rank-2
/// via SVD.
pub fn estimate_8point(pairs: &[EpipolarPair]) -> Option<FundamentalMatrix> {
    if pairs.len() < 8 {
        return None;
    }

    // Hartley normalization
    let (p1_norm, t1) = normalize_points(pairs.iter().map(|p| p.p1));
    let (p2_norm, t2) = normalize_points(pairs.iter().map(|p| p.p2));

    let n = pairs.len();

    // Build A^T A (9×9) from the epipolar constraint equations.
    // Each pair gives one row of A: [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    let mut ata = [0.0f64; 81];
    for i in 0..n {
        let (x1, y1) = p1_norm[i];
        let (x2, y2) = p2_norm[i];
        let row = [
            x2 * x1,
            x2 * y1,
            x2,
            y2 * x1,
            y2 * y1,
            y2,
            x1,
            y1,
            1.0,
        ];
        for j in 0..9 {
            for k in 0..9 {
                ata[j * 9 + k] += row[j] * row[k];
            }
        }
    }

    // Smallest eigenvector of A^T A → vectorized F
    let f_vec = smallest_eigenvector_9x9(&ata)?;

    // Enforce rank-2 constraint via SVD
    let f_rank2 = enforce_rank2(&f_vec);

    // Denormalize: F = T2^T * F_norm * T1
    let f_denorm = denormalize_fundamental(&f_rank2, &t1, &t2);

    // Normalize so Frobenius norm = 1
    let f = normalize_matrix(&f_denorm);

    Some(FundamentalMatrix { data: f })
}

// ── 7-point algorithm ──

/// Estimate F from exactly 7 correspondences using the 7-point algorithm.
///
/// Returns 1 or 3 real fundamental matrices (from the cubic `det(F)=0`
/// constraint).
pub fn estimate_7point(pairs: &[EpipolarPair]) -> Vec<FundamentalMatrix> {
    if pairs.len() != 7 {
        return Vec::new();
    }

    // Hartley normalization
    let (p1_norm, t1) = normalize_points(pairs.iter().map(|p| p.p1));
    let (p2_norm, t2) = normalize_points(pairs.iter().map(|p| p.p2));

    // Build A (7×9) and find the two-dimensional null space
    // A^T A has rank 7, so the null space is 2D.
    let mut ata = [0.0f64; 81];
    for i in 0..7 {
        let (x1, y1) = p1_norm[i];
        let (x2, y2) = p2_norm[i];
        let row = [
            x2 * x1,
            x2 * y1,
            x2,
            y2 * x1,
            y2 * y1,
            y2,
            x1,
            y1,
            1.0,
        ];
        for j in 0..9 {
            for k in 0..9 {
                ata[j * 9 + k] += row[j] * row[k];
            }
        }
    }

    // Find two smallest eigenvectors
    let Some((f1_vec, f2_vec)) = two_smallest_eigenvectors_9x9(&ata) else {
        return Vec::new();
    };

    // F = α*F1 + (1-α)*F2, find α such that det(F) = 0
    // This gives a cubic in α.
    let alphas = solve_det_cubic(&f1_vec, &f2_vec);

    let mut results = Vec::with_capacity(3);
    for alpha in alphas {
        let mut f_vec = [0.0f64; 9];
        for i in 0..9 {
            f_vec[i] = alpha * f1_vec[i] + (1.0 - alpha) * f2_vec[i];
        }

        let f_denorm = denormalize_fundamental(&f_vec, &t1, &t2);
        let f = normalize_matrix(&f_denorm);
        results.push(FundamentalMatrix { data: f });
    }

    results
}

// ── Internal: normalization ──

/// Hartley normalization: translate centroid to origin, scale avg distance to √2.
fn normalize_points(
    points: impl Iterator<Item = (f64, f64)>,
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

    let t = [s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0];
    (normalized, t)
}

/// Denormalize: `F = T2^T * F_norm * T1`.
fn denormalize_fundamental(f: &[f64; 9], t1: &[f64; 9], t2: &[f64; 9]) -> [f64; 9] {
    let t2t = transpose3(t2);
    let tmp = mat3_mul(&t2t, f);
    mat3_mul(&tmp, t1)
}

// ── Internal: SVD rank-2 enforcement ──

/// Enforce rank-2 on a 3×3 matrix by zeroing the smallest singular value.
///
/// Uses a compact Jacobi SVD for 3×3 matrices.
fn enforce_rank2(f: &[f64; 9]) -> [f64; 9] {
    // Compute F^T F
    let ft = transpose3(f);
    let ftf = mat3_mul(&ft, f);

    // Eigendecompose F^T F → V, eigenvalues (σ²)
    let (eigenvalues, v) = symmetric_eigen_3x3(&ftf);

    // σ_i = sqrt(eigenvalue_i), sorted descending
    let mut sv: [(f64, usize); 3] = [
        (eigenvalues[0].max(0.0).sqrt(), 0),
        (eigenvalues[1].max(0.0).sqrt(), 1),
        (eigenvalues[2].max(0.0).sqrt(), 2),
    ];
    sv.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // V columns reordered by descending singular value
    let mut v_sorted = [0.0f64; 9];
    for (col_out, &(_, col_in)) in sv.iter().enumerate() {
        for row in 0..3 {
            v_sorted[row * 3 + col_out] = v[row * 3 + col_in];
        }
    }

    // U = F * V * Σ^{-1}
    let fv = mat3_mul(f, &v_sorted);
    let mut u = [0.0f64; 9];
    for col in 0..3 {
        let sigma = sv[col].0;
        if sigma > 1e-14 {
            let inv_s = 1.0 / sigma;
            for row in 0..3 {
                u[row * 3 + col] = fv[row * 3 + col] * inv_s;
            }
        }
        // else: column stays zero
    }

    // Reconstruct with smallest singular value set to zero
    // F' = U * diag(σ0, σ1, 0) * V^T
    let mut result = [0.0f64; 9];
    let vt = transpose3(&v_sorted);
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            // Only use the first two singular values
            for k in 0..2 {
                sum += u[i * 3 + k] * sv[k].0 * vt[k * 3 + j];
            }
            result[i * 3 + j] = sum;
        }
    }

    result
}

/// Eigendecomposition of a 3×3 symmetric matrix via Jacobi iteration.
/// Returns `(eigenvalues, eigenvectors_row_major)`.
fn symmetric_eigen_3x3(a: &[f64; 9]) -> ([f64; 3], [f64; 9]) {
    let mut m = *a;
    // V starts as identity
    let mut v = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0f64];

    for _ in 0..100 {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..3 {
            for j in (i + 1)..3 {
                let val = m[i * 3 + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-15 {
            break;
        }

        // Jacobi rotation
        let app = m[p * 3 + p];
        let aqq = m[q * 3 + q];
        let apq = m[p * 3 + q];

        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau.abs() < 1e-30 {
            1.0
        } else {
            let sign = if tau > 0.0 { 1.0 } else { -1.0 };
            sign / (tau.abs() + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        // Update matrix
        let mut new_m = m;
        new_m[p * 3 + p] = app - t * apq;
        new_m[q * 3 + q] = aqq + t * apq;
        new_m[p * 3 + q] = 0.0;
        new_m[q * 3 + p] = 0.0;

        for r in 0..3 {
            if r != p && r != q {
                let rp = m[r * 3 + p];
                let rq = m[r * 3 + q];
                new_m[r * 3 + p] = c * rp - s * rq;
                new_m[p * 3 + r] = new_m[r * 3 + p];
                new_m[r * 3 + q] = s * rp + c * rq;
                new_m[q * 3 + r] = new_m[r * 3 + q];
            }
        }
        m = new_m;

        // Update eigenvector matrix V
        for r in 0..3 {
            let vp = v[r * 3 + p];
            let vq = v[r * 3 + q];
            v[r * 3 + p] = c * vp - s * vq;
            v[r * 3 + q] = s * vp + c * vq;
        }
    }

    ([m[0], m[4], m[8]], v)
}

// ── Internal: 9×9 eigenvector solvers ──

/// Find the eigenvector corresponding to the smallest eigenvalue of a 9×9
/// symmetric positive-(semi)definite matrix, via inverse power iteration.
fn smallest_eigenvector_9x9(ata: &[f64; 81]) -> Option<[f64; 9]> {
    let mut v = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0f64];
    let norm = vec_norm_9(&v);
    for x in &mut v {
        *x /= norm;
    }

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

/// Find the two eigenvectors corresponding to the two smallest eigenvalues
/// of a 9×9 symmetric matrix. Uses deflation after finding the first.
fn two_smallest_eigenvectors_9x9(ata: &[f64; 81]) -> Option<([f64; 9], [f64; 9])> {
    // First smallest eigenvector
    let v1 = smallest_eigenvector_9x9(ata)?;

    // Estimate eigenvalue: λ = v^T A v
    let lambda1 = quadratic_form_9x9(ata, &v1);

    // Deflate: A' = A - λ1 * v1 * v1^T
    let mut ata_deflated = *ata;
    for i in 0..9 {
        for j in 0..9 {
            ata_deflated[i * 9 + j] -= lambda1 * v1[i] * v1[j];
        }
    }

    // Second smallest eigenvector from deflated matrix
    let v2 = smallest_eigenvector_9x9(&ata_deflated)?;

    Some((v1, v2))
}

fn quadratic_form_9x9(a: &[f64; 81], v: &[f64; 9]) -> f64 {
    let mut result = 0.0;
    for i in 0..9 {
        for j in 0..9 {
            result += v[i] * a[i * 9 + j] * v[j];
        }
    }
    result
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

// ── Internal: 7-point cubic solver ──

/// Solve `det(α F1 + (1-α) F2) = 0` for α.
///
/// Expands the determinant as a cubic `c3 α³ + c2 α² + c1 α + c0 = 0` and
/// finds real roots.
fn solve_det_cubic(f1: &[f64; 9], f2: &[f64; 9]) -> Vec<f64> {
    // det(α F1 + (1-α) F2) = det((α-1) F2 + α F1)
    // Let G(α) = α F1 + (1-α) F2, expand det as polynomial in α.
    // Evaluate det at 4 values of α, then fit the cubic.

    let det_at = |alpha: f64| -> f64 {
        let mut g = [0.0f64; 9];
        for i in 0..9 {
            g[i] = alpha * f1[i] + (1.0 - alpha) * f2[i];
        }
        det3(&g)
    };

    // Evaluate at α = 0, 1, -1, 2
    let d0 = det_at(0.0);
    let d1 = det_at(1.0);
    let dm1 = det_at(-1.0);
    let d2 = det_at(2.0);

    // Fit cubic c3 α³ + c2 α² + c1 α + c0 using Lagrange interpolation
    // at α = 0, 1, -1, 2:
    //   c0 = d0
    //   From system of equations:
    //     d1  = c3 + c2 + c1 + c0
    //     dm1 = -c3 + c2 - c1 + c0
    //     d2  = 8c3 + 4c2 + 2c1 + c0
    let c0 = d0;
    // c1 + c2 + c3 = d1 - c0
    // -c1 + c2 - c3 = dm1 - c0
    // 2c1 + 4c2 + 8c3 = d2 - c0
    let s1 = d1 - c0;
    let s2 = dm1 - c0;
    let s3 = d2 - c0;

    // s1 + s2 = 2 c2  =>  c2 = (s1 + s2) / 2
    let c2 = (s1 + s2) / 2.0;
    // s1 - s2 = 2(c1 + c3)  => c1 + c3 = (s1 - s2) / 2
    let c1_plus_c3 = (s1 - s2) / 2.0;
    // s3 = 2 c1 + 4 c2 + 8 c3  => c1 + 4 c3 = (s3 - 4 c2) / 2
    let c1_plus_4c3 = (s3 - 4.0 * c2) / 2.0;
    // 3 c3 = c1_plus_4c3 - c1_plus_c3
    let c3 = (c1_plus_4c3 - c1_plus_c3) / 3.0;
    let c1 = c1_plus_c3 - c3;

    solve_cubic(c3, c2, c1, c0)
}

/// Find real roots of `a x³ + b x² + c x + d = 0`.
fn solve_cubic(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    if a.abs() < 1e-14 {
        // Degenerate to quadratic
        return solve_quadratic(b, c, d);
    }

    // Normalize: x³ + px² + qx + r = 0
    let p = b / a;
    let q = c / a;
    let r = d / a;

    // Depressed cubic via substitution t = x - p/3
    // t³ + pt² + qt + r = 0  →  t³ + Qt + R = 0
    let q_dep = q - p * p / 3.0;
    let r_dep = r - p * q / 3.0 + 2.0 * p * p * p / 27.0;

    let discriminant = -4.0 * q_dep * q_dep * q_dep - 27.0 * r_dep * r_dep;

    let shift = p / 3.0;

    if discriminant < -1e-12 {
        // One real root (Cardano's formula)
        let sq = (r_dep * r_dep / 4.0 + q_dep * q_dep * q_dep / 27.0).max(0.0).sqrt();
        let u = -r_dep / 2.0 + sq;
        let v = -r_dep / 2.0 - sq;
        let t = cbrt(u) + cbrt(v);
        vec![t - shift]
    } else {
        // Three real roots (trigonometric method)
        let m = (-q_dep / 3.0).max(0.0).sqrt();
        if m < 1e-15 {
            return vec![-shift];
        }
        let theta = (-r_dep / (2.0 * m * m * m)).clamp(-1.0, 1.0).acos() / 3.0;
        let two_m = 2.0 * m;
        vec![
            two_m * theta.cos() - shift,
            two_m * (theta + 2.0 * std::f64::consts::FRAC_PI_3).cos() - shift,
            two_m * (theta + 4.0 * std::f64::consts::FRAC_PI_3).cos() - shift,
        ]
    }
}

fn solve_quadratic(a: f64, b: f64, c: f64) -> Vec<f64> {
    if a.abs() < 1e-14 {
        if b.abs() < 1e-14 {
            return Vec::new();
        }
        return vec![-c / b];
    }
    let disc = b * b - 4.0 * a * c;
    if disc < -1e-12 {
        Vec::new()
    } else if disc.abs() < 1e-12 {
        vec![-b / (2.0 * a)]
    } else {
        let sq = disc.max(0.0).sqrt();
        vec![(-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a)]
    }
}

fn cbrt(x: f64) -> f64 {
    if x >= 0.0 {
        x.cbrt()
    } else {
        -(-x).cbrt()
    }
}

// ── Internal: 3×3 matrix utilities ──

fn det3(m: &[f64; 9]) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7])
        - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

fn transpose3(m: &[f64; 9]) -> [f64; 9] {
    [
        m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8],
    ]
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

fn normalize_matrix(m: &[f64; 9]) -> [f64; 9] {
    let norm = m.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-15 {
        return *m;
    }
    let inv = 1.0 / norm;
    let mut out = *m;
    for x in &mut out {
        *x *= inv;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a known fundamental matrix from two camera matrices and verify
    /// the epipolar constraint holds.
    fn make_test_pairs() -> (Vec<EpipolarPair>, FundamentalMatrix) {
        // Camera 1 = [I | 0], Camera 2 = [I | t] with t = (1, 0, 0)
        // F = [t]_x = [[0, 0, 0], [0, 0, -1], [0, 1, 0]]
        let f_true = FundamentalMatrix {
            data: [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        };

        // Generate 3D points and project them
        let points_3d: Vec<(f64, f64, f64)> = vec![
            (1.0, 2.0, 5.0),
            (3.0, -1.0, 4.0),
            (-2.0, 3.0, 6.0),
            (0.5, 0.5, 3.0),
            (2.0, 2.0, 7.0),
            (-1.0, -1.0, 5.0),
            (4.0, 1.0, 8.0),
            (0.0, 3.0, 4.0),
            (1.5, -2.0, 6.0),
            (-3.0, 0.0, 5.0),
            (2.5, 1.5, 3.5),
            (1.0, -1.0, 4.5),
        ];

        let pairs: Vec<EpipolarPair> = points_3d
            .iter()
            .map(|&(x, y, z)| {
                // Camera 1 projection: (x/z, y/z)
                let p1 = (x / z, y / z);
                // Camera 2 projection: (x-1)/z, y/z  (translated by (1,0,0))
                let p2 = ((x - 1.0) / z, y / z);
                EpipolarPair { p1, p2 }
            })
            .collect();

        (pairs, f_true)
    }

    #[test]
    fn epipolar_constraint_holds_for_true_f() {
        let (pairs, f) = make_test_pairs();
        for pair in &pairs {
            let d = f.sampson_distance(pair.p1, pair.p2);
            assert!(d < 1e-10, "Sampson distance should be ~0 for true F, got {d}");
        }
    }

    #[test]
    fn eight_point_recovers_epipolar_geometry() {
        let (pairs, _f_true) = make_test_pairs();
        let f_est = estimate_8point(&pairs).expect("8-point should succeed");

        // Verify epipolar constraint: p2^T F p1 ≈ 0 for all pairs
        for pair in &pairs {
            let d = f_est.sampson_distance(pair.p1, pair.p2);
            assert!(
                d < 1e-3,
                "Sampson distance should be small, got {d}"
            );
        }
    }

    #[test]
    fn eight_point_rank_is_two() {
        let (pairs, _) = make_test_pairs();
        let f = estimate_8point(&pairs).expect("8-point should succeed");
        let det = det3(&f.data);
        assert!(
            det.abs() < 1e-6,
            "F should have rank 2 (det ≈ 0), got det = {det}"
        );
    }

    #[test]
    fn seven_point_returns_solutions() {
        let (pairs, _) = make_test_pairs();
        let seven = &pairs[..7];
        let solutions = estimate_7point(seven);
        assert!(
            !solutions.is_empty(),
            "7-point should return at least 1 solution"
        );

        // At least one solution should satisfy epipolar constraint
        let best = solutions
            .iter()
            .min_by(|a, b| {
                let ea: f64 = pairs.iter().map(|p| a.sampson_distance(p.p1, p.p2)).sum();
                let eb: f64 = pairs.iter().map(|p| b.sampson_distance(p.p1, p.p2)).sum();
                ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        for pair in &pairs {
            let d = best.sampson_distance(pair.p1, pair.p2);
            assert!(d < 1e-2, "Best 7-point solution Sampson error = {d}");
        }
    }

    #[test]
    fn seven_point_wrong_count_returns_empty() {
        let (pairs, _) = make_test_pairs();
        assert!(estimate_7point(&pairs[..6]).is_empty());
        assert!(estimate_7point(&pairs).is_empty());
    }

    #[test]
    fn ransac_fundamental_with_outliers() {
        let (mut pairs, _) = make_test_pairs();

        // Add outliers
        pairs.push(EpipolarPair {
            p1: (0.0, 0.0),
            p2: (100.0, 100.0),
        });
        pairs.push(EpipolarPair {
            p1: (1.0, 1.0),
            p2: (-50.0, -50.0),
        });

        let config = RansacConfig {
            max_iterations: 500,
            threshold: 0.01,
            confidence: 0.999,
            seed: 42,
        };

        let result = find_fundamental(&pairs, &config).expect("RANSAC should succeed");
        assert!(
            result.n_inliers >= 10,
            "should find >= 10 inliers, got {}",
            result.n_inliers
        );

        // Outliers should be rejected
        let n = pairs.len();
        assert!(
            !result.inlier_mask[n - 1] || !result.inlier_mask[n - 2],
            "at least one outlier should be rejected"
        );

        // Inlier pairs should satisfy epipolar constraint
        for (i, pair) in pairs.iter().enumerate() {
            if result.inlier_mask[i] {
                let d = result.f.sampson_distance(pair.p1, pair.p2);
                assert!(d < 0.01, "inlier Sampson distance = {d}");
            }
        }
    }

    #[test]
    fn sampson_distance_symmetric() {
        let f = FundamentalMatrix {
            data: [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        };
        let ft = FundamentalMatrix {
            data: transpose3(&f.data),
        };

        let p1 = (0.5, 0.3);
        let p2 = (0.7, 0.1);

        let d1 = f.sampson_distance(p1, p2);
        let d2 = ft.sampson_distance(p2, p1);
        assert!(
            (d1 - d2).abs() < 1e-10,
            "Sampson distance should be symmetric: {d1} vs {d2}"
        );
    }

    #[test]
    fn epipolar_line_orthogonal_to_match() {
        let (pairs, f_true) = make_test_pairs();
        for pair in &pairs {
            let (a, b, c) = f_true.epipolar_line(pair.p1);
            // p2 should lie on this line: a*x2 + b*y2 + c ≈ 0
            let residual = a * pair.p2.0 + b * pair.p2.1 + c;
            let line_norm = a.hypot(b);
            let dist = if line_norm > 1e-15 {
                residual.abs() / line_norm
            } else {
                residual.abs()
            };
            assert!(
                dist < 1e-10,
                "p2 should lie on epipolar line, dist = {dist}"
            );
        }
    }

    #[test]
    fn insufficient_points_returns_error() {
        let pairs: Vec<EpipolarPair> = (0..5)
            .map(|i| EpipolarPair {
                p1: (i as f64, 0.0),
                p2: (i as f64, 0.0),
            })
            .collect();
        let config = RansacConfig::default();
        let result = find_fundamental(&pairs, &config);
        assert!(result.is_err());
    }

    #[test]
    fn cubic_solver_known_roots() {
        // (x - 1)(x - 2)(x - 3) = x³ - 6x² + 11x - 6
        let roots = solve_cubic(1.0, -6.0, 11.0, -6.0);
        assert_eq!(roots.len(), 3);
        let mut sorted = roots;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 1.0).abs() < 1e-8, "root 0 = {}", sorted[0]);
        assert!((sorted[1] - 2.0).abs() < 1e-8, "root 1 = {}", sorted[1]);
        assert!((sorted[2] - 3.0).abs() < 1e-8, "root 2 = {}", sorted[2]);
    }

    #[test]
    fn cubic_solver_one_real_root() {
        // x³ + x + 1 = 0 has one real root ≈ -0.6824
        let roots = solve_cubic(1.0, 0.0, 1.0, 1.0);
        assert_eq!(roots.len(), 1);
        assert!((roots[0] - (-0.682_327_803_828_019_3)).abs() < 1e-6);
    }
}
