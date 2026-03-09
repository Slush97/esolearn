// SPDX-License-Identifier: MIT OR Apache-2.0
//! CPU compute backend — pure Rust, always available.

use super::ComputeBackend;

/// CPU compute backend using pure Rust.
///
/// This is the default fallback backend. All operations are
/// single-threaded (Rayon parallelism is handled at a higher level
/// by the model training loop, not here).
#[non_exhaustive]
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn matmul(&self, a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), k * n);

        let mut c = vec![0.0; m * n];
        // Iterate k in the outer loop for cache-friendly access to B rows.
        for i in 0..m {
            for p in 0..k {
                let a_ip = a[i * k + p];
                for j in 0..n {
                    c[i * n + j] += a_ip * b[p * n + j];
                }
            }
        }
        c
    }

    fn xtx_xty(&self, features: &[Vec<f64>], target: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n_samples = target.len();
        let n_features = features.len();
        let dim = n_features + 1; // +1 for intercept

        let mut xtx = vec![0.0; dim * dim];
        let mut xty = vec![0.0; dim];

        for i in 0..n_samples {
            let y = target[i];

            // Intercept terms
            xtx[0] += 1.0;
            xty[0] += y;

            for j in 0..n_features {
                let xj = features[j][i];
                xtx[(j + 1) * dim] += xj; // first column
                xtx[j + 1] += xj; // first row
                xty[j + 1] += xj * y;

                for k in 0..n_features {
                    let xk = features[k][i];
                    xtx[(j + 1) * dim + (k + 1)] += xj * xk;
                }
            }
        }

        (xtx, xty)
    }

    fn xtx_xty_contiguous(
        &self,
        data: &[f64],
        target: &[f64],
        n_samples: usize,
        n_features: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let dim = n_features + 1;
        let mut xtx = vec![0.0; dim * dim];
        let mut xty = vec![0.0; dim];

        for i in 0..n_samples {
            let y = target[i];

            // Intercept terms
            xtx[0] += 1.0;
            xty[0] += y;

            for j in 0..n_features {
                let xj = data[j * n_samples + i];
                xtx[(j + 1) * dim] += xj;
                xtx[j + 1] += xj;
                xty[j + 1] += xj * y;

                for k in 0..n_features {
                    let xk = data[k * n_samples + i];
                    xtx[(j + 1) * dim + (k + 1)] += xj * xk;
                }
            }
        }

        (xtx, xty)
    }

    fn pairwise_distances_squared(
        &self,
        queries: &[f64],
        train: &[f64],
        n_q: usize,
        n_t: usize,
        dim: usize,
    ) -> Vec<f64> {
        debug_assert_eq!(queries.len(), n_q * dim);
        debug_assert_eq!(train.len(), n_t * dim);

        let mut dists = vec![0.0; n_q * n_t];
        for i in 0..n_q {
            let q = &queries[i * dim..(i + 1) * dim];
            for j in 0..n_t {
                let t = &train[j * dim..(j + 1) * dim];
                let mut d = 0.0;
                for f in 0..dim {
                    let diff = q[f] - t[f];
                    d += diff * diff;
                }
                dists[i * n_t + j] = d;
            }
        }
        dists
    }

    fn name(&self) -> &'static str {
        "cpu"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_identity() {
        let backend = CpuBackend;
        // 2x2 identity × [1,2; 3,4]
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = backend.matmul(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn matmul_2x3_times_3x2() {
        let backend = CpuBackend;
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3×2
        let c = backend.matmul(&a, &b, 2, 3, 2);
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn pairwise_distances_simple() {
        let backend = CpuBackend;
        let queries = vec![0.0, 0.0, 1.0, 0.0]; // 2 queries, 2D
        let train = vec![0.0, 0.0, 3.0, 4.0]; // 2 train points, 2D
        let d = backend.pairwise_distances_squared(&queries, &train, 2, 2, 2);
        assert!((d[0] - 0.0).abs() < 1e-12); // (0,0) → (0,0) = 0
        assert!((d[1] - 25.0).abs() < 1e-12); // (0,0) → (3,4) = 25
        assert!((d[2] - 1.0).abs() < 1e-12); // (1,0) → (0,0) = 1
        assert!((d[3] - 20.0).abs() < 1e-12); // (1,0) → (3,4) = 20
    }

    #[test]
    fn xtx_xty_simple() {
        let backend = CpuBackend;
        // 3 samples, 1 feature, target = 2*x + 1
        let features = vec![vec![1.0, 2.0, 3.0]];
        let target = vec![3.0, 5.0, 7.0];
        let (xtx, xty) = backend.xtx_xty(&features, &target);
        // dim = 2 (intercept + 1 feature)
        assert_eq!(xtx.len(), 4);
        assert!((xtx[0] - 3.0).abs() < 1e-12); // sum of 1s = n
        assert!((xtx[1] - 6.0).abs() < 1e-12); // sum of x = 6
        assert!((xtx[2] - 6.0).abs() < 1e-12); // sum of x = 6
        assert!((xtx[3] - 14.0).abs() < 1e-12); // sum of x^2 = 14
        assert!((xty[0] - 15.0).abs() < 1e-12); // sum of y = 15
        assert!((xty[1] - 34.0).abs() < 1e-12); // sum of x*y = 34
    }
}
