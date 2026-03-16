// SPDX-License-Identifier: MIT OR Apache-2.0
//! CART (Classification And Regression Trees) implementation.
//!
//! Implements the full CART algorithm with Gini impurity, entropy,
//! and MSE split criteria. Supports feature bagging for Random Forest.
//!
//! Trees are built recursively using `TreeNode`, then flattened into a
//! contiguous `FlatTree` (`Vec<FlatNode>`) for cache-optimal prediction.

mod builder;
mod flat;
mod node;

pub(crate) use builder::presort_indices;
pub use builder::{DecisionTreeClassifier, DecisionTreeRegressor};
pub use flat::FlatTree;
pub use node::TreeNode;

// ---------------------------------------------------------------------------
// Split criterion
// ---------------------------------------------------------------------------

/// Split quality criterion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum SplitCriterion {
    /// Gini impurity: `1 - Σ(pᵢ²)`.
    Gini,
    /// Information entropy: `-Σ(pᵢ log₂ pᵢ)`.
    Entropy,
    /// Mean squared error (for regression).
    Mse,
}

/// Leaf sentinel — stored in `FlatNode::right` to indicate a leaf node.
pub(crate) const LEAF_SENTINEL: u32 = u32::MAX;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub(super) struct BestSplit {
    pub(super) feature_idx: usize,
    pub(super) threshold: f64,
    pub(super) impurity_decrease: f64,
}

pub(super) fn compute_impurity(counts: &[usize], n: usize, criterion: SplitCriterion) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    match criterion {
        SplitCriterion::Gini => {
            let sum_sq: f64 = counts
                .iter()
                .map(|&c| {
                    let p = c as f64 / n_f;
                    p * p
                })
                .sum();
            1.0 - sum_sq
        }
        SplitCriterion::Entropy => {
            let mut entropy = 0.0;
            for &c in counts {
                if c > 0 {
                    let p = c as f64 / n_f;
                    entropy -= p * p.log2();
                }
            }
            entropy
        }
        SplitCriterion::Mse => {
            // MSE is not applicable for class counts — used only in regressor.
            0.0
        }
    }
}

pub(super) fn majority_class(counts: &[usize]) -> f64 {
    counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, &count)| count)
        .map_or(0.0, |(idx, _)| idx as f64)
}

// ---------------------------------------------------------------------------
// Weighted impurity helpers (for class_weight support)
// ---------------------------------------------------------------------------

pub(super) fn compute_impurity_weighted(
    counts: &[f64],
    total: f64,
    criterion: SplitCriterion,
) -> f64 {
    if total < 1e-12 {
        return 0.0;
    }
    match criterion {
        SplitCriterion::Gini => {
            let sum_sq: f64 = counts
                .iter()
                .map(|&c| {
                    let p = c / total;
                    p * p
                })
                .sum();
            1.0 - sum_sq
        }
        SplitCriterion::Entropy => {
            let mut entropy = 0.0;
            for &c in counts {
                if c > 1e-12 {
                    let p = c / total;
                    entropy -= p * p.log2();
                }
            }
            entropy
        }
        SplitCriterion::Mse => 0.0,
    }
}

pub(super) fn weighted_majority_class(counts: &[f64]) -> f64 {
    counts
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0.0, |(idx, _)| idx as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;

    fn make_linearly_separable() -> Dataset {
        // Class 0: x < 5, Class 1: x >= 5
        let features = vec![(0..20).map(|i| i as f64).collect()];
        let target: Vec<f64> = (0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }).collect();
        Dataset::new(features, target, vec!["x".into()], "class")
    }

    #[test]
    fn test_decision_tree_perfect_split() {
        let data = make_linearly_separable();
        let mut dt = DecisionTreeClassifier::new();
        dt.fit(&data).unwrap();

        let matrix = data.feature_matrix();
        let preds = dt.predict(&matrix).unwrap();
        let acc = preds
            .iter()
            .zip(data.target.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-6)
            .count() as f64
            / data.n_samples() as f64;

        assert!(
            acc >= 0.95,
            "expected ≥95% accuracy on linearly separable data, got {:.1}%",
            acc * 100.0
        );
    }

    #[test]
    fn test_feature_importance_sums_to_one() {
        let data = make_linearly_separable();
        let mut dt = DecisionTreeClassifier::new();
        dt.fit(&data).unwrap();

        let importances = dt.feature_importances().unwrap();
        let total: f64 = importances.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "feature importances should sum to 1.0, got {total}"
        );
    }

    #[test]
    fn test_max_depth() {
        let data = make_linearly_separable();
        let mut dt = DecisionTreeClassifier::new().max_depth(2);
        dt.fit(&data).unwrap();
        assert!(dt.depth() <= 2 + 1); // depth includes leaf
    }

    #[test]
    fn test_predict_proba() {
        let data = make_linearly_separable();
        let mut dt = DecisionTreeClassifier::new();
        dt.fit(&data).unwrap();

        let sample_class0 = vec![2.0]; // clearly class 0
        let proba = dt.predict_proba(&[sample_class0]).unwrap();
        assert!(proba[0][0] > 0.5, "should predict class 0 with >50%");
    }

    #[test]
    fn test_regressor_basic() {
        // y = x
        let features = vec![(0..50).map(|i| i as f64).collect()];
        let target: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut dt = DecisionTreeRegressor::new().max_depth(10);
        dt.fit(&data).unwrap();

        let matrix = data.feature_matrix();
        let preds = dt.predict(&matrix).unwrap();

        // Should get low MSE on training data.
        let mse: f64 = preds
            .iter()
            .zip(data.target.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / data.n_samples() as f64;

        assert!(mse < 5.0, "MSE on training data should be low, got {mse}");
    }

    #[test]
    fn test_not_fitted_error() {
        let dt = DecisionTreeClassifier::new();
        assert!(dt.predict(&[vec![1.0]]).is_err());
    }

    // -------------------------------------------------------------------
    // Cost-complexity pruning tests
    // -------------------------------------------------------------------

    fn make_iris_like() -> Dataset {
        // A small 3-class dataset with enough samples to build a deep tree.
        let mut rng = crate::rng::FastRng::new(42);
        let n = 150;
        let mut f1 = Vec::with_capacity(n);
        let mut f2 = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for _ in 0..50 {
            f1.push(rng.f64() * 2.0);
            f2.push(rng.f64() * 2.0);
            target.push(0.0);
        }
        for _ in 0..50 {
            f1.push(rng.f64() * 2.0 + 3.0);
            f2.push(rng.f64() * 2.0 + 3.0);
            target.push(1.0);
        }
        for _ in 0..50 {
            f1.push(rng.f64() * 2.0 + 6.0);
            f2.push(rng.f64() * 2.0);
            target.push(2.0);
        }
        Dataset::new(
            vec![f1, f2],
            target,
            vec!["f1".into(), "f2".into()],
            "class",
        )
    }

    #[test]
    fn test_ccp_alpha_reduces_depth() {
        let data = make_iris_like();

        let mut dt_full = DecisionTreeClassifier::new();
        dt_full.fit(&data).unwrap();
        let depth_full = dt_full.depth();
        let leaves_full = dt_full.n_leaves();

        let mut dt_pruned = DecisionTreeClassifier::new().ccp_alpha(0.02);
        dt_pruned.fit(&data).unwrap();
        let depth_pruned = dt_pruned.depth();
        let leaves_pruned = dt_pruned.n_leaves();

        eprintln!("Full tree: depth={depth_full}, leaves={leaves_full}");
        eprintln!("Pruned tree: depth={depth_pruned}, leaves={leaves_pruned}");

        assert!(
            leaves_pruned <= leaves_full,
            "Pruned tree should have ≤ leaves than full: {leaves_pruned} vs {leaves_full}"
        );
    }

    #[test]
    fn test_ccp_alpha_zero_no_change() {
        let data = make_iris_like();

        let mut dt_zero = DecisionTreeClassifier::new().ccp_alpha(0.0);
        dt_zero.fit(&data).unwrap();
        let mut dt_default = DecisionTreeClassifier::new();
        dt_default.fit(&data).unwrap();

        assert_eq!(
            dt_zero.n_leaves(),
            dt_default.n_leaves(),
            "ccp_alpha=0.0 should not change the tree"
        );
    }

    #[test]
    fn test_ccp_alpha_large_collapses_to_root() {
        let data = make_iris_like();
        let mut dt = DecisionTreeClassifier::new().ccp_alpha(1000.0);
        dt.fit(&data).unwrap();
        assert_eq!(
            dt.n_leaves(),
            1,
            "Very large ccp_alpha should collapse to a single leaf"
        );
    }

    #[test]
    fn test_regressor_ccp_alpha() {
        let features = vec![(0..100).map(|i| i as f64).collect()];
        let target: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut dt_full = DecisionTreeRegressor::new();
        dt_full.fit(&data).unwrap();

        let mut dt_pruned = DecisionTreeRegressor::new().ccp_alpha(0.01);
        dt_pruned.fit(&data).unwrap();

        let full_leaves = dt_full.flat_tree().unwrap().n_leaves();
        let pruned_leaves = dt_pruned.flat_tree().unwrap().n_leaves();

        eprintln!("Regressor: full={full_leaves} leaves, pruned={pruned_leaves} leaves");
        assert!(
            pruned_leaves <= full_leaves,
            "Pruned regressor should have ≤ leaves: {pruned_leaves} vs {full_leaves}"
        );
    }

    #[test]
    fn test_pruning_path_monotonic() {
        let data = make_iris_like();
        let mut dt = DecisionTreeClassifier::new();
        dt.fit(&data).unwrap();

        let (alphas, impurities) = dt.cost_complexity_pruning_path(&data).unwrap();

        assert!(alphas.len() >= 2, "Should have at least 2 pruning steps");
        // Alphas should be monotonically non-decreasing.
        for w in alphas.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-12,
                "Alphas should be monotonically non-decreasing: {} -> {}",
                w[0],
                w[1]
            );
        }
        // Impurities should be monotonically non-decreasing.
        for w in impurities.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-12,
                "Impurities should be non-decreasing: {} -> {}",
                w[0],
                w[1]
            );
        }
        eprintln!("Pruning path: {} steps", alphas.len());
        eprintln!("Alphas: {:?}", &alphas[..alphas.len().min(5)]);
    }
}
