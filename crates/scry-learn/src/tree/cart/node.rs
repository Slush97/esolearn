// SPDX-License-Identifier: MIT OR Apache-2.0
//! Recursive tree node representation used during construction.
//!
//! `TreeNode` is the recursive enum used to build the tree, which is then
//! flattened into a `FlatTree` for cache-optimal prediction.

/// A node in the decision tree (recursive representation).
///
/// Used during tree construction, then flattened into a `FlatTree` for
/// cache-optimal prediction. Exposed publicly for visualization.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum TreeNode {
    /// A leaf node — produces a prediction.
    Leaf {
        /// Predicted class (classification) or value (regression).
        prediction: f64,
        /// Number of training samples that reached this node.
        n_samples: usize,
        /// Class distribution at this node (classification only).
        class_counts: Vec<usize>,
        /// Impurity at this node.
        impurity: f64,
    },
    /// An internal split node.
    Split {
        /// Index of the feature used for the split.
        feature_idx: usize,
        /// Threshold value: left if ≤ threshold, right if > threshold.
        threshold: f64,
        /// Left child (≤ threshold).
        left: Box<TreeNode>,
        /// Right child (> threshold).
        right: Box<TreeNode>,
        /// Number of training samples that reached this node.
        n_samples: usize,
        /// Impurity at this node (before split).
        impurity: f64,
        /// Class distribution at this node.
        class_counts: Vec<usize>,
        /// Majority class prediction at this node.
        prediction: f64,
    },
}

impl TreeNode {
    /// Predict for a single sample by walking the tree.
    pub fn predict(&self, sample: &[f64]) -> f64 {
        match self {
            TreeNode::Leaf { prediction, .. } => *prediction,
            TreeNode::Split {
                feature_idx,
                threshold,
                left,
                right,
                ..
            } => {
                if sample[*feature_idx] <= *threshold {
                    left.predict(sample)
                } else {
                    right.predict(sample)
                }
            }
        }
    }

    /// Get class probabilities for a single sample.
    pub fn predict_proba(&self, sample: &[f64], n_classes: usize) -> Vec<f64> {
        match self {
            TreeNode::Leaf {
                class_counts,
                n_samples,
                ..
            } => {
                let mut proba = vec![0.0; n_classes];
                let total = *n_samples as f64;
                for (i, &count) in class_counts.iter().enumerate() {
                    if i < n_classes {
                        proba[i] = count as f64 / total;
                    }
                }
                proba
            }
            TreeNode::Split {
                feature_idx,
                threshold,
                left,
                right,
                ..
            } => {
                if sample[*feature_idx] <= *threshold {
                    left.predict_proba(sample, n_classes)
                } else {
                    right.predict_proba(sample, n_classes)
                }
            }
        }
    }

    /// Depth of this subtree.
    pub fn depth(&self) -> usize {
        match self {
            TreeNode::Leaf { .. } => 1,
            TreeNode::Split { left, right, .. } => 1 + left.depth().max(right.depth()),
        }
    }

    /// Number of leaf nodes in this subtree.
    pub fn n_leaves(&self) -> usize {
        match self {
            TreeNode::Leaf { .. } => 1,
            TreeNode::Split { left, right, .. } => left.n_leaves() + right.n_leaves(),
        }
    }

    /// Number of samples at this node.
    pub fn n_samples(&self) -> usize {
        match self {
            TreeNode::Leaf { n_samples, .. } | TreeNode::Split { n_samples, .. } => *n_samples,
        }
    }

    /// Sum of weighted leaf impurities: Σ(impurity_leaf × n_samples_leaf).
    ///
    /// This is R(T_t) in the cost-complexity pruning literature.
    pub fn total_leaf_impurity(&self) -> f64 {
        match self {
            TreeNode::Leaf {
                impurity,
                n_samples,
                ..
            } => *impurity * (*n_samples as f64),
            TreeNode::Split { left, right, .. } => {
                left.total_leaf_impurity() + right.total_leaf_impurity()
            }
        }
    }

    /// Minimal cost-complexity pruning (MCCP).
    ///
    /// Recursively prunes subtrees whose effective alpha is ≤ `ccp_alpha`.
    /// Effective alpha = (R(t) - R(T_t)) / (|T_t| - 1), where R(t) is the
    /// re-substitution error if this node were a leaf and R(T_t) is the
    /// total leaf impurity of the subtree.
    ///
    /// This matches sklearn's `ccp_alpha` parameter behavior.
    pub fn prune_ccp(self, ccp_alpha: f64) -> TreeNode {
        match self {
            TreeNode::Leaf { .. } => self,
            TreeNode::Split {
                feature_idx,
                threshold,
                left,
                right,
                n_samples,
                impurity,
                class_counts,
                prediction,
            } => {
                // Recursively prune children first (bottom-up).
                let pruned_left = left.prune_ccp(ccp_alpha);
                let pruned_right = right.prune_ccp(ccp_alpha);

                // Build the pruned split node to compute its subtree stats.
                let subtree = TreeNode::Split {
                    feature_idx,
                    threshold,
                    left: Box::new(pruned_left),
                    right: Box::new(pruned_right),
                    n_samples,
                    impurity,
                    class_counts: class_counts.clone(),
                    prediction,
                };

                let n_leaves = subtree.n_leaves();
                if n_leaves <= 1 {
                    return subtree;
                }

                // R(t) = impurity if this node were a leaf.
                let r_node = impurity * (n_samples as f64);
                // R(T_t) = total leaf impurity of subtree.
                let r_subtree = subtree.total_leaf_impurity();

                let effective_alpha = (r_node - r_subtree) / (n_leaves as f64 - 1.0);

                if effective_alpha <= ccp_alpha {
                    // Collapse to leaf.
                    TreeNode::Leaf {
                        prediction,
                        n_samples,
                        class_counts,
                        impurity,
                    }
                } else {
                    subtree
                }
            }
        }
    }

    /// Compute the cost-complexity pruning path.
    ///
    /// Returns `(ccp_alphas, total_impurities)` — a sequence of effective
    /// alpha values and the corresponding total tree impurity at each
    /// pruning step. Useful for elbow-method selection of `ccp_alpha`.
    pub fn cost_complexity_pruning_path(&self) -> (Vec<f64>, Vec<f64>) {
        let mut alphas = vec![0.0];
        let mut impurities = vec![self.total_leaf_impurity()];

        let mut current = self.clone();
        loop {
            // Find the minimum effective alpha across all internal nodes.
            let min_alpha = Self::min_effective_alpha(&current);
            match min_alpha {
                None => break, // no more internal nodes
                Some(alpha) => {
                    current = current.prune_ccp(alpha);
                    alphas.push(alpha);
                    impurities.push(current.total_leaf_impurity());
                }
            }
        }
        (alphas, impurities)
    }

    /// Find the minimum effective alpha among all internal nodes.
    fn min_effective_alpha(node: &TreeNode) -> Option<f64> {
        match node {
            TreeNode::Leaf { .. } => None,
            TreeNode::Split {
                left,
                right,
                n_samples,
                impurity,
                ..
            } => {
                let n_leaves = node.n_leaves();
                let r_node = impurity * (*n_samples as f64);
                let r_subtree = node.total_leaf_impurity();
                let my_alpha = if n_leaves > 1 {
                    Some((r_node - r_subtree) / (n_leaves as f64 - 1.0))
                } else {
                    None
                };

                let left_alpha = Self::min_effective_alpha(left);
                let right_alpha = Self::min_effective_alpha(right);

                [my_alpha, left_alpha, right_alpha]
                    .iter()
                    .filter_map(|a| *a)
                    .reduce(f64::min)
            }
        }
    }
}
