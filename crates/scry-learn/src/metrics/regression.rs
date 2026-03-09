// SPDX-License-Identifier: MIT OR Apache-2.0
//! Regression metrics: MSE, RMSE, MAE, R².

/// Mean Squared Error.
pub fn mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f64>()
        / y_true.len() as f64
}

/// Root Mean Squared Error.
pub fn root_mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    mean_squared_error(y_true, y_pred).sqrt()
}

/// Mean Absolute Error.
pub fn mean_absolute_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).abs())
        .sum::<f64>()
        / y_true.len() as f64
}

/// R² (coefficient of determination).
///
/// Returns 1.0 for perfect predictions, 0.0 for predicting the mean,
/// and negative values for worse-than-mean predictions.
pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();
    let ss_tot: f64 = y_true.iter().map(|t| (t - mean).powi(2)).sum();

    if ss_tot < 1e-12 {
        return if ss_res < 1e-12 { 1.0 } else { 0.0 };
    }
    1.0 - ss_res / ss_tot
}

/// Explained variance score.
///
/// Computes `1 - Var(y_true - y_pred) / Var(y_true)`.
/// Unlike R², this uses the variance of the residuals rather than the
/// sum of squared residuals, so it does not account for bias.
pub fn explained_variance_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let n = y_true.len() as f64;

    // Residuals
    let residuals: Vec<f64> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| t - p)
        .collect();
    let res_mean = residuals.iter().sum::<f64>() / n;
    let var_res = residuals
        .iter()
        .map(|r| (r - res_mean).powi(2))
        .sum::<f64>()
        / n;

    let y_mean = y_true.iter().sum::<f64>() / n;
    let var_y = y_true.iter().map(|t| (t - y_mean).powi(2)).sum::<f64>() / n;

    if var_y < 1e-15 {
        return if var_res < 1e-15 { 1.0 } else { 0.0 };
    }

    1.0 - var_res / var_y
}

/// Mean Absolute Percentage Error (MAPE).
///
/// Computes `mean(|y_true - y_pred| / |y_true|)`.
/// Samples where `y_true` is zero are skipped to avoid division by zero.
pub fn mean_absolute_percentage_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        if t.abs() < 1e-15 {
            continue;
        }
        total += ((t - p) / t).abs();
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_perfect() {
        assert!((mean_squared_error(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0])).abs() < 1e-10);
    }

    #[test]
    fn test_mse_known() {
        // MSE = ((1-2)² + (2-3)²) / 2 = 1.0
        assert!((mean_squared_error(&[1.0, 2.0], &[2.0, 3.0]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_r2_perfect() {
        assert!((r2_score(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_r2_mean_prediction() {
        // Predicting the mean should give R² ≈ 0
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean_pred = vec![3.0; 5];
        assert!((r2_score(&y, &mean_pred)).abs() < 1e-10);
    }

    #[test]
    fn test_mae() {
        assert!((mean_absolute_error(&[1.0, 2.0], &[3.0, 4.0]) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_explained_variance_perfect() {
        let ev = explained_variance_score(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!((ev - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_explained_variance_with_bias() {
        // y_pred = y_true + 1 (constant bias)
        // Residuals are all 1.0 → Var(residual) = 0 → EV = 1.0
        // (This differs from R² which would be < 1 due to bias)
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![2.0, 3.0, 4.0, 5.0];
        let ev = explained_variance_score(&y_true, &y_pred);
        assert!((ev - 1.0).abs() < 1e-10, "constant bias → EV=1, got {ev}");
    }

    #[test]
    fn test_mape_known() {
        // y_true = [1, 2], y_pred = [1.1, 2.2]
        // MAPE = (|0.1/1| + |0.2/2|) / 2 = (0.1 + 0.1) / 2 = 0.1
        let mape = mean_absolute_percentage_error(&[1.0, 2.0], &[1.1, 2.2]);
        assert!((mape - 0.1).abs() < 1e-10, "expected MAPE=0.1, got {mape}");
    }

    #[test]
    fn test_mape_skips_zeros() {
        // y_true = [0, 1, 2], y_pred = [0.5, 1.5, 2.5]
        // Skips y_true=0, so MAPE = (|0.5/1| + |0.5/2|) / 2 = (0.5 + 0.25) / 2 = 0.375
        let mape = mean_absolute_percentage_error(&[0.0, 1.0, 2.0], &[0.5, 1.5, 2.5]);
        assert!(
            (mape - 0.375).abs() < 1e-10,
            "expected MAPE=0.375, got {mape}"
        );
    }
}
