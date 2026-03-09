// SPDX-License-Identifier: MIT OR Apache-2.0
//! QR decomposition least-squares solver via Householder reflections.

use crate::error::{Result, ScryLearnError};

/// Solve min ||Xb - y||² via QR decomposition.
///
/// `x` is column-major contiguous: `x[col * n_rows + row]`.
/// Requires `n_rows >= n_cols` (overdetermined or square).
pub(crate) fn qr_solve(x: &[f64], y: &[f64], n_rows: usize, n_cols: usize) -> Result<Vec<f64>> {
    if n_rows == 0 || n_cols == 0 {
        return Err(ScryLearnError::EmptyDataset);
    }
    if n_rows < n_cols {
        return Err(ScryLearnError::InvalidParameter(
            "QR solver requires n_rows >= n_cols (overdetermined system)".into(),
        ));
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

    // Work on copies.
    let mut a = x.to_vec(); // column-major
    let mut qty = y.to_vec(); // will become Q^T * y

    // Householder QR: for each column, compute reflector and apply.
    for k in 0..n {
        // Compute the norm of column k below the diagonal.
        let mut norm_sq = 0.0;
        for i in k..m {
            norm_sq += a[k * m + i] * a[k * m + i];
        }
        let norm = norm_sq.sqrt();

        if norm < 1e-300 {
            return Err(ScryLearnError::InvalidParameter(
                "QR: rank-deficient matrix (zero column norm)".into(),
            ));
        }

        // Choose sign to avoid cancellation.
        let sign = if a[k * m + k] >= 0.0 { 1.0 } else { -1.0 };
        let alpha = -sign * norm;

        // Householder vector v (stored in-place below diagonal of A).
        let vk = a[k * m + k] - alpha;
        a[k * m + k] = alpha; // R[k,k]

        // Compute beta = 2 / (v^T v)
        let mut vtv = vk * vk;
        for i in (k + 1)..m {
            vtv += a[k * m + i] * a[k * m + i];
        }

        if vtv < 1e-300 {
            continue;
        }
        let beta = 2.0 / vtv;

        // Apply reflector to remaining columns of A.
        for j in (k + 1)..n {
            let mut dot = vk * a[j * m + k];
            for i in (k + 1)..m {
                dot += a[k * m + i] * a[j * m + i];
            }
            let factor = beta * dot;
            a[j * m + k] -= factor * vk;
            for i in (k + 1)..m {
                a[j * m + i] -= factor * a[k * m + i];
            }
        }

        // Apply reflector to qty.
        {
            let mut dot = vk * qty[k];
            for i in (k + 1)..m {
                dot += a[k * m + i] * qty[i];
            }
            let factor = beta * dot;
            qty[k] -= factor * vk;
            for i in (k + 1)..m {
                qty[i] -= factor * a[k * m + i];
            }
        }
    }

    // Back-substitution: R * b = (Q^T y)[0..n]
    let mut b = vec![0.0; n];
    for k in (0..n).rev() {
        let rkk = a[k * m + k];
        if rkk.abs() < 1e-300 {
            return Err(ScryLearnError::InvalidParameter(
                "QR: singular R matrix during back-substitution".into(),
            ));
        }
        let mut sum = qty[k];
        for j in (k + 1)..n {
            sum -= a[j * m + k] * b[j]; // R[k,j] = a[j*m + k] (column-major upper triangle)
        }
        b[k] = sum / rkk;
    }

    Ok(b)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_qr_simple_overdetermined() {
        // y = 2*x + 3, overdetermined (5 eqns, 2 unknowns: [1, x])
        let rows: &[&[f64]] = &[
            &[1.0, 1.0],
            &[1.0, 2.0],
            &[1.0, 3.0],
            &[1.0, 4.0],
            &[1.0, 5.0],
        ];
        let y: Vec<f64> = rows.iter().map(|r| 3.0 + 2.0 * r[1]).collect();
        let x = to_col_major(rows, 5, 2);
        let b = qr_solve(&x, &y, 5, 2).unwrap();
        assert!(
            (b[0] - 3.0).abs() < 1e-10,
            "intercept should be ~3.0, got {}",
            b[0]
        );
        assert!(
            (b[1] - 2.0).abs() < 1e-10,
            "slope should be ~2.0, got {}",
            b[1]
        );
    }

    #[test]
    fn test_qr_hilbert_5x5() {
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
        let b = qr_solve(&x, &y, n, n).unwrap();
        for (i, &c) in b.iter().enumerate() {
            assert!(
                (c - 1.0).abs() < 1e-4,
                "Hilbert QR coefficient[{}] = {}, expected ~1.0",
                i,
                c
            );
        }
    }

    #[test]
    fn test_qr_matches_svd() {
        use super::super::svd::svd_solve;
        let rows: &[&[f64]] = &[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0], &[7.0, 8.0]];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let x = to_col_major(rows, 4, 2);
        let qr_b = qr_solve(&x, &y, 4, 2).unwrap();
        let svd_result = svd_solve(&x, &y, 4, 2).unwrap();
        for i in 0..2 {
            assert!(
                (qr_b[i] - svd_result.coefficients[i]).abs() < 1e-8,
                "QR[{}]={} vs SVD[{}]={}",
                i,
                qr_b[i],
                i,
                svd_result.coefficients[i]
            );
        }
    }

    #[test]
    fn test_qr_underdetermined_rejected() {
        // 2 rows, 3 cols — should fail.
        let x = vec![1.0; 6];
        let y = vec![1.0, 2.0];
        let result = qr_solve(&x, &y, 2, 3);
        assert!(result.is_err());
    }
}
