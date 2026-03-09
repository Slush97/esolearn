// SPDX-License-Identifier: MIT OR Apache-2.0
//! SVD least-squares solver via one-sided Jacobi rotations.

use crate::error::{Result, ScryLearnError};

/// SVD least-squares solver result.
#[non_exhaustive]
#[allow(dead_code)]
pub(crate) struct SvdResult {
    pub coefficients: Vec<f64>,
    pub singular_values: Vec<f64>,
    pub condition_number: f64,
    pub rank: usize,
}

/// Solve the least-squares problem min ||Xb - y||² via SVD.
///
/// `x` is column-major contiguous: `x[col * n_rows + row]`.
/// Returns coefficients `b` such that `Xb ≈ y`.
pub(crate) fn svd_solve(x: &[f64], y: &[f64], n_rows: usize, n_cols: usize) -> Result<SvdResult> {
    if n_rows == 0 || n_cols == 0 {
        return Err(ScryLearnError::EmptyDataset);
    }
    if x.len() != n_rows * n_cols {
        return Err(ScryLearnError::InvalidParameter(format!(
            "x length {} != n_rows({}) * n_cols({})",
            x.len(),
            n_rows,
            n_cols
        )));
    }
    if y.len() != n_rows {
        return Err(ScryLearnError::InvalidParameter(format!(
            "y length {} != n_rows({})",
            y.len(),
            n_rows
        )));
    }

    let m = n_rows;
    let n = n_cols;

    // One-sided Jacobi SVD:
    // We work on a copy of X (column-major). We apply Jacobi rotations to pairs
    // of columns until X^T X is diagonal. Then column norms are singular values,
    // normalized columns are U, and accumulated rotations are V.

    // W = copy of X (m x n column-major)
    let mut w = x.to_vec();
    // V = identity (n x n column-major)
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_sweeps = 100;
    let eps = f64::EPSILON;

    for _sweep in 0..max_sweeps {
        let mut converged = true;

        for p in 0..n {
            for q in (p + 1)..n {
                // Compute 2x2 Gram submatrix elements:
                // a = w_p^T w_p, b = w_p^T w_q, c = w_q^T w_q
                let mut a = 0.0;
                let mut b = 0.0;
                let mut c = 0.0;
                for i in 0..m {
                    let wp = w[p * m + i];
                    let wq = w[q * m + i];
                    a += wp * wp;
                    b += wp * wq;
                    c += wq * wq;
                }

                // Check if off-diagonal element is negligible.
                if b.abs() <= eps * (a * c).sqrt() {
                    continue;
                }
                converged = false;

                // Jacobi rotation angle for 2x2 symmetric [[a, b], [b, c]].
                let tau = (c - a) / (2.0 * b);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let cs = 1.0 / (1.0 + t * t).sqrt();
                let sn = t * cs;

                // Rotate columns p and q of W.
                for i in 0..m {
                    let wp = w[p * m + i];
                    let wq = w[q * m + i];
                    w[p * m + i] = cs * wp - sn * wq;
                    w[q * m + i] = sn * wp + cs * wq;
                }

                // Rotate columns p and q of V.
                for i in 0..n {
                    let vp = v[p * n + i];
                    let vq = v[q * n + i];
                    v[p * n + i] = cs * vp - sn * vq;
                    v[q * n + i] = sn * vp + cs * vq;
                }
            }
        }

        if converged {
            break;
        }
    }

    // Singular values = column norms of W. U columns = W columns / sigma.
    let mut singular_values = vec![0.0; n];
    // We don't need to store full U explicitly. We need U^T y which is:
    // (U^T y)_j = (w_j / sigma_j)^T y = (w_j^T y) / sigma_j

    let eps_threshold = eps * (m.max(n) as f64);
    let mut sigma_max = 0.0f64;

    for j in 0..n {
        let mut norm_sq = 0.0;
        for i in 0..m {
            norm_sq += w[j * m + i] * w[j * m + i];
        }
        singular_values[j] = norm_sq.sqrt();
        if singular_values[j] > sigma_max {
            sigma_max = singular_values[j];
        }
    }

    let threshold = eps_threshold * sigma_max;
    let mut rank = 0usize;
    let mut sigma_min_nonzero = f64::INFINITY;

    // Compute coefficients = V * Sigma^+ * U^T * y
    // = V * Sigma^+ * [(w_j^T y) / sigma_j for each j where sigma_j > threshold]
    // = sum_j [ (w_j^T y / sigma_j^2) * v_j ]  for non-zero sigma_j
    let mut coefficients = vec![0.0; n];

    for j in 0..n {
        if singular_values[j] > threshold {
            rank += 1;
            if singular_values[j] < sigma_min_nonzero {
                sigma_min_nonzero = singular_values[j];
            }
            // w_j^T y
            let mut wty = 0.0;
            for i in 0..m {
                wty += w[j * m + i] * y[i];
            }
            // coefficient contribution: (wty / sigma_j^2) * v_j
            let factor = wty / (singular_values[j] * singular_values[j]);
            for k in 0..n {
                coefficients[k] += factor * v[j * n + k];
            }
        }
    }

    let condition_number = if sigma_min_nonzero.is_finite() && sigma_min_nonzero > 0.0 {
        sigma_max / sigma_min_nonzero
    } else {
        f64::INFINITY
    };

    // Sort singular values descending.
    singular_values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    Ok(SvdResult {
        coefficients,
        singular_values,
        condition_number,
        rank,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build column-major flat array from row-major 2D data.
    fn to_col_major(rows: &[&[f64]], n_rows: usize, n_cols: usize) -> Vec<f64> {
        let mut out = vec![0.0; n_rows * n_cols];
        for r in 0..n_rows {
            for c in 0..n_cols {
                out[c * n_rows + r] = rows[r][c];
            }
        }
        out
    }

    #[test]
    fn test_svd_simple_2x2() {
        let x = to_col_major(&[&[1.0, 0.0], &[0.0, 1.0]], 2, 2);
        let y = vec![3.0, 7.0];
        let result = svd_solve(&x, &y, 2, 2).unwrap();
        assert!((result.coefficients[0] - 3.0).abs() < 1e-10);
        assert!((result.coefficients[1] - 7.0).abs() < 1e-10);
        assert!((result.condition_number - 1.0).abs() < 1e-10);
        assert_eq!(result.rank, 2);
    }

    #[test]
    fn test_svd_overdetermined_5x2() {
        let rows: &[&[f64]] = &[
            &[1.0, 1.0],
            &[2.0, 1.0],
            &[3.0, 2.0],
            &[4.0, 2.0],
            &[5.0, 3.0],
        ];
        let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] + 3.0 * r[1]).collect();
        let x = to_col_major(rows, 5, 2);
        let result = svd_solve(&x, &y, 5, 2).unwrap();
        assert!(
            (result.coefficients[0] - 2.0).abs() < 1e-8,
            "got {}",
            result.coefficients[0]
        );
        assert!(
            (result.coefficients[1] - 3.0).abs() < 1e-8,
            "got {}",
            result.coefficients[1]
        );
        assert_eq!(result.rank, 2);
    }

    #[test]
    fn test_svd_hilbert_5x5() {
        let n = 5;
        let mut rows = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                rows[i][j] = 1.0 / (i + j + 1) as f64;
            }
        }
        let true_beta = vec![1.0; n];
        let y: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| rows[i][j] * true_beta[j]).sum())
            .collect();

        let row_refs: Vec<&[f64]> = rows.iter().map(std::vec::Vec::as_slice).collect();
        let x = to_col_major(&row_refs, n, n);
        let result = svd_solve(&x, &y, n, n).unwrap();

        for (i, &c) in result.coefficients.iter().enumerate() {
            assert!(
                (c - 1.0).abs() < 1e-4,
                "Hilbert coefficient[{}] = {}, expected ~1.0",
                i,
                c
            );
        }
        assert!(
            result.condition_number > 1e4,
            "Hilbert should be ill-conditioned"
        );
    }

    #[test]
    fn test_svd_wide_matrix() {
        let rows: &[&[f64]] = &[
            &[1.0, 0.0, 0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0, 1.0, 0.0, 0.0],
        ];
        let y = vec![1.0, 2.0, 3.0];
        let x = to_col_major(rows, 3, 5);
        let result = svd_solve(&x, &y, 3, 5).unwrap();
        assert!((result.coefficients[0] - 1.0).abs() < 1e-10);
        assert!((result.coefficients[1] - 2.0).abs() < 1e-10);
        assert!((result.coefficients[2] - 3.0).abs() < 1e-10);
        assert!(result.coefficients[3].abs() < 1e-10);
        assert!(result.coefficients[4].abs() < 1e-10);
        assert_eq!(result.rank, 3);
    }

    #[test]
    fn test_svd_identity() {
        let n = 3;
        let mut x = vec![0.0; n * n];
        for i in 0..n {
            x[i * n + i] = 1.0;
        }
        let y = vec![1.0, 2.0, 3.0];
        let result = svd_solve(&x, &y, n, n).unwrap();
        for sv in &result.singular_values {
            assert!(
                (sv - 1.0).abs() < 1e-10,
                "singular value {} should be 1.0",
                sv
            );
        }
        assert!((result.condition_number - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_svd_rank_deficient() {
        // Column 2 = Column 1, so rank should be 1.
        let rows: &[&[f64]] = &[&[1.0, 1.0], &[2.0, 2.0], &[3.0, 3.0]];
        let y = vec![1.0, 2.0, 3.0];
        let x = to_col_major(rows, 3, 2);
        let result = svd_solve(&x, &y, 3, 2).unwrap();
        assert_eq!(result.rank, 1, "duplicate columns -> rank 1");
        let err: f64 = (0..3)
            .map(|i| {
                let pred =
                    result.coefficients[0] * rows[i][0] + result.coefficients[1] * rows[i][1];
                (pred - y[i]).powi(2)
            })
            .sum();
        assert!(
            err < 1e-10,
            "reconstruction error should be tiny, got {}",
            err
        );
    }
}
