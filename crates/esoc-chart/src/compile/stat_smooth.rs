// SPDX-License-Identifier: MIT OR Apache-2.0
//! LOESS (locally estimated scatterplot smoothing) transform.

/// Compute LOESS smoothed values.
///
/// For each x_i, performs weighted local linear regression using a tricube
/// kernel. The `bandwidth` parameter controls the fraction of data used
/// in each local neighborhood (0 < bandwidth <= 1).
///
/// Returns an error if `x_data` and `y_data` have different lengths.
pub fn compute_loess(x_data: &[f64], y_data: &[f64], bandwidth: f64) -> crate::error::Result<(Vec<f64>, Vec<f64>)> {
    if x_data.is_empty() || y_data.is_empty() {
        return Ok((vec![], vec![]));
    }

    if x_data.len() != y_data.len() {
        return Err(crate::error::ChartError::LengthMismatch {
            expected: x_data.len(),
            got: y_data.len(),
        });
    }

    let n = x_data.len();
    let bandwidth = bandwidth.clamp(0.05, 1.0);
    let neighbors = ((n as f64 * bandwidth).ceil() as usize).max(2).min(n);

    // Sort by x for output
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| x_data[a].partial_cmp(&x_data[b]).unwrap_or(std::cmp::Ordering::Equal));

    let mut x_out = Vec::with_capacity(n);
    let mut y_out = Vec::with_capacity(n);

    for &i in &indices {
        let xi = x_data[i];

        // Find k nearest neighbors by x distance
        let mut dists: Vec<(usize, f64)> = (0..n)
            .map(|j| (j, (x_data[j] - xi).abs()))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let dists = &dists[..neighbors];

        let max_dist = dists.last().map_or(1.0, |d| d.1);
        let max_dist = if max_dist < 1e-15 { 1.0 } else { max_dist };

        // Tricube kernel weights
        let weights: Vec<f64> = dists
            .iter()
            .map(|&(_, d)| {
                let u = (d / max_dist).min(1.0);
                let t = 1.0 - u * u * u;
                t * t * t
            })
            .collect();

        // Weighted linear regression: y = a + b*x
        let sum_w: f64 = weights.iter().sum();
        if sum_w < 1e-15 {
            x_out.push(xi);
            y_out.push(y_data[i]);
            continue;
        }

        let sum_wx: f64 = dists.iter().zip(weights.iter()).map(|(&(j, _), &w)| w * x_data[j]).sum();
        let sum_wy: f64 = dists.iter().zip(weights.iter()).map(|(&(j, _), &w)| w * y_data[j]).sum();
        let sum_wxx: f64 = dists.iter().zip(weights.iter()).map(|(&(j, _), &w)| w * x_data[j] * x_data[j]).sum();
        let sum_wxy: f64 = dists.iter().zip(weights.iter()).map(|(&(j, _), &w)| w * x_data[j] * y_data[j]).sum();

        let mean_x = sum_wx / sum_w;
        let mean_y = sum_wy / sum_w;
        #[allow(clippy::suspicious_operation_groupings)] // E[X²] - E[X]² is correct
        let var_x = (sum_wxx / sum_w) - (mean_x * mean_x);

        let yi = if var_x.abs() < 1e-15 {
            mean_y
        } else {
            let cov_xy = sum_wxy / sum_w - mean_x * mean_y;
            let b = cov_xy / var_x;
            let a = mean_y - b * mean_x;
            a + b * xi
        };

        x_out.push(xi);
        y_out.push(yi);
    }

    Ok((x_out, y_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_data_near_linear_output() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let (_x_out, y_out) = compute_loess(&x, &y, 0.5).unwrap();
        // For perfectly linear data, LOESS should produce near-linear output
        for (i, &yi) in y_out.iter().enumerate() {
            let expected = 2.0 * _x_out[i] + 1.0;
            assert!(
                (yi - expected).abs() < 0.5,
                "At x={}, expected ~{}, got {}",
                _x_out[i],
                expected,
                yi
            );
        }
    }

    #[test]
    fn empty_data() {
        let (x, y) = compute_loess(&[], &[], 0.5).unwrap();
        assert!(x.is_empty());
        assert!(y.is_empty());
    }

    #[test]
    fn length_mismatch_returns_error() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        let result = compute_loess(&x, &y, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn single_data_point() {
        let (x_out, y_out) = compute_loess(&[5.0], &[10.0], 0.5).unwrap();
        assert_eq!(x_out.len(), 1);
        assert!((y_out[0] - 10.0).abs() < 1e-10);
    }
}
