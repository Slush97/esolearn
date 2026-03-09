// SPDX-License-Identifier: MIT OR Apache-2.0
//! Model-agnostic permutation importance (Breiman 2001).
//!
//! Measures the decrease in a scoring function when each feature is
//! randomly permuted, breaking the association between the feature and
//! the target.

use crate::rng::FastRng;

/// Result of permutation importance analysis.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PermutationImportance {
    /// Mean score decrease per feature (higher = more important).
    pub importances_mean: Vec<f64>,
    /// Standard deviation of score decrease per feature.
    pub importances_std: Vec<f64>,
    /// Raw score decreases: `importances_raw[feature][repeat]`.
    pub importances_raw: Vec<Vec<f64>>,
}

/// Compute permutation importance for any model.
///
/// # Arguments
///
/// * `features` - Column-major feature matrix: `features[feature_idx][sample_idx]`.
/// * `target` - Target values, one per sample.
/// * `predict` - Prediction function: given column-major features, returns predictions.
/// * `scorer` - Scoring function: `scorer(y_true, y_pred) -> score`.
///   Higher is better (e.g. accuracy, R2). The importance is
///   `baseline_score - permuted_score`.
/// * `n_repeats` - Number of times to permute each feature (default: 5).
/// * `seed` - RNG seed for reproducibility.
///
/// # Returns
///
/// A `PermutationImportance` with mean, std, and raw importances per feature.
///
/// # Panics
///
/// Panics if `features` is empty or if feature columns have different lengths.
pub fn permutation_importance(
    features: &[Vec<f64>],
    target: &[f64],
    predict: &dyn Fn(&[Vec<f64>]) -> Vec<f64>,
    scorer: fn(&[f64], &[f64]) -> f64,
    n_repeats: usize,
    seed: u64,
) -> PermutationImportance {
    assert!(!features.is_empty(), "features must not be empty");
    let n_features = features.len();
    let n_samples = features[0].len();
    assert_eq!(
        target.len(),
        n_samples,
        "target length must match number of samples"
    );

    // Compute baseline score on unperturbed data.
    let baseline_preds = predict(features);
    let baseline_score = scorer(target, &baseline_preds);

    let mut rng = FastRng::new(seed);
    let mut importances_raw = vec![Vec::with_capacity(n_repeats); n_features];

    // Pre-allocate a mutable copy of the features for permutation.
    let mut permuted = features.to_vec();

    for feat_idx in 0..n_features {
        for _ in 0..n_repeats {
            // Save the original column.
            let original_col = features[feat_idx].clone();

            // Create a shuffled index array and apply it.
            let mut indices: Vec<usize> = (0..n_samples).collect();
            rng.shuffle(&mut indices);

            for (i, &idx) in indices.iter().enumerate() {
                permuted[feat_idx][i] = original_col[idx];
            }

            // Score with permuted feature.
            let permuted_preds = predict(&permuted);
            let permuted_score = scorer(target, &permuted_preds);

            importances_raw[feat_idx].push(baseline_score - permuted_score);

            // Restore original column.
            permuted[feat_idx].clone_from(&features[feat_idx]);
        }
    }

    let importances_mean: Vec<f64> = importances_raw
        .iter()
        .map(|raw| raw.iter().sum::<f64>() / raw.len() as f64)
        .collect();

    let importances_std: Vec<f64> = importances_raw
        .iter()
        .zip(importances_mean.iter())
        .map(|(raw, &mean)| {
            if raw.len() <= 1 {
                return 0.0;
            }
            let variance =
                raw.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (raw.len() - 1) as f64;
            variance.sqrt()
        })
        .collect();

    PermutationImportance {
        importances_mean,
        importances_std,
        importances_raw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutation_importance_basic() {
        // Feature 0 is predictive (y = x0), feature 1 is noise.
        let n = 100;
        let mut rng = FastRng::new(42);
        let f0: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let f1: Vec<f64> = (0..n).map(|_| rng.f64() * 100.0).collect();
        let target: Vec<f64> = f0.clone();
        let features = vec![f0, f1];

        let predict = |feats: &[Vec<f64>]| -> Vec<f64> {
            // Simple model: predict = feature 0.
            feats[0].clone()
        };

        let scorer = |y_true: &[f64], y_pred: &[f64]| -> f64 {
            // Negative MSE (higher is better).
            let mse = y_true
                .iter()
                .zip(y_pred.iter())
                .map(|(t, p)| (t - p).powi(2))
                .sum::<f64>()
                / y_true.len() as f64;
            -mse
        };

        let result = permutation_importance(&features, &target, &predict, scorer, 5, 42);

        assert_eq!(result.importances_mean.len(), 2);
        // Feature 0 should have high importance (positive score decrease).
        assert!(
            result.importances_mean[0] > 0.0,
            "Feature 0 should be important: {}",
            result.importances_mean[0]
        );
        // Feature 1 should have near-zero importance.
        assert!(
            result.importances_mean[1].abs() < result.importances_mean[0].abs() * 0.1,
            "Feature 1 should be less important: {} vs {}",
            result.importances_mean[1],
            result.importances_mean[0]
        );
    }
}
