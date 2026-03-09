// SPDX-License-Identifier: MIT OR Apache-2.0
//! Class weighting for imbalanced datasets.
//!
//! Provides the [`ClassWeight`] enum and [`compute_sample_weights`] function
//! to generate per-sample weights that compensate for class imbalance.
//!
//! # Example
//!
//! ```
//! use scry_learn::weights::{ClassWeight, compute_sample_weights};
//!
//! let targets = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
//! let weights = compute_sample_weights(&targets, &ClassWeight::Balanced);
//!
//! // Minority class (1) gets higher weight to compensate for imbalance.
//! assert!(weights[9] > weights[0]);
//! ```

use std::collections::HashMap;

/// Strategy for weighting classes during training.
///
/// Used by classifiers to handle imbalanced datasets. When set to `Balanced`,
/// minority classes receive higher weight, making the model pay more attention
/// to underrepresented classes.
///
/// # Example
///
/// ```
/// use scry_learn::weights::ClassWeight;
/// use scry_learn::tree::DecisionTreeClassifier;
///
/// let dt = DecisionTreeClassifier::new()
///     .class_weight(ClassWeight::Balanced);
/// ```
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ClassWeight {
    /// All classes weighted equally (weight = 1.0). This is the default.
    #[default]
    Uniform,
    /// Automatically adjust weights inversely proportional to class frequencies.
    ///
    /// Uses the sklearn formula: `weight_c = n_samples / (n_classes × n_c)`.
    Balanced,
    /// User-specified per-class weights (class label → weight).
    Custom(HashMap<usize, f64>),
}

/// Compute per-sample weights from target labels and a class weighting strategy.
///
/// Returns a vector with one weight per sample. For `Uniform`, all weights are 1.0.
/// For `Balanced`, uses the sklearn formula:
/// `weight_c = n_samples / (n_classes × count_c)`.
///
/// # Example
///
/// ```
/// use scry_learn::weights::{ClassWeight, compute_sample_weights};
///
/// // 8 samples of class 0, 2 samples of class 1
/// let targets = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
/// let weights = compute_sample_weights(&targets, &ClassWeight::Balanced);
///
/// // n_samples=10, n_classes=2
/// // weight_0 = 10 / (2 × 8) = 0.625
/// // weight_1 = 10 / (2 × 2) = 2.5
/// assert!((weights[0] - 0.625).abs() < 1e-6);
/// assert!((weights[8] - 2.5).abs() < 1e-6);
/// ```
pub fn compute_sample_weights(targets: &[f64], class_weight: &ClassWeight) -> Vec<f64> {
    let n = targets.len();
    match class_weight {
        ClassWeight::Uniform => vec![1.0; n],
        ClassWeight::Balanced => {
            // Count samples per class.
            let mut counts: HashMap<usize, usize> = HashMap::new();
            for &t in targets {
                *counts.entry(t as usize).or_insert(0) += 1;
            }
            let n_classes = counts.len();
            let n_f = n as f64;

            // weight_c = n_samples / (n_classes × count_c)
            let class_weights: HashMap<usize, f64> = counts
                .iter()
                .map(|(&cls, &count)| {
                    let w = n_f / (n_classes as f64 * count as f64);
                    (cls, w)
                })
                .collect();

            targets
                .iter()
                .map(|&t| class_weights.get(&(t as usize)).copied().unwrap_or(1.0))
                .collect()
        }
        ClassWeight::Custom(map) => targets
            .iter()
            .map(|&t| map.get(&(t as usize)).copied().unwrap_or(1.0))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_weights() {
        let targets = vec![0.0, 0.0, 1.0, 1.0, 2.0];
        let weights = compute_sample_weights(&targets, &ClassWeight::Uniform);
        assert_eq!(weights.len(), 5);
        assert!(weights.iter().all(|&w| (w - 1.0).abs() < 1e-12));
    }

    #[test]
    fn test_balanced_weights_equal_classes() {
        // 5 samples of each class → balanced already → all weights ≈ 1.0
        let targets = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let weights = compute_sample_weights(&targets, &ClassWeight::Balanced);
        // n=10, n_classes=2, count_0=5, count_1=5
        // weight = 10/(2*5) = 1.0
        for &w in &weights {
            assert!((w - 1.0).abs() < 1e-6, "expected 1.0, got {w}");
        }
    }

    #[test]
    fn test_balanced_weights_imbalanced() {
        // 90% class 0, 10% class 1
        let mut targets = vec![0.0; 90];
        targets.extend(vec![1.0; 10]);
        let weights = compute_sample_weights(&targets, &ClassWeight::Balanced);

        // weight_0 = 100/(2*90) = 0.5556
        // weight_1 = 100/(2*10) = 5.0
        let w0 = weights[0];
        let w1 = weights[90];
        assert!(
            (w0 - 100.0 / 180.0).abs() < 1e-6,
            "majority weight: expected {}, got {w0}",
            100.0 / 180.0
        );
        assert!(
            (w1 - 5.0).abs() < 1e-6,
            "minority weight: expected 5.0, got {w1}"
        );
        // Minority weight should be much higher.
        assert!(w1 > w0 * 5.0);
    }

    #[test]
    fn test_custom_weights() {
        let mut map = HashMap::new();
        map.insert(0, 1.0);
        map.insert(1, 10.0);
        let targets = vec![0.0, 0.0, 1.0, 1.0];
        let weights = compute_sample_weights(&targets, &ClassWeight::Custom(map));
        assert!((weights[0] - 1.0).abs() < 1e-12);
        assert!((weights[2] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_custom_weights_missing_class_defaults_to_one() {
        let mut map = HashMap::new();
        map.insert(1, 5.0);
        // Class 0 not in map → defaults to 1.0
        let targets = vec![0.0, 1.0];
        let weights = compute_sample_weights(&targets, &ClassWeight::Custom(map));
        assert!((weights[0] - 1.0).abs() < 1e-12);
        assert!((weights[1] - 5.0).abs() < 1e-12);
    }
}
