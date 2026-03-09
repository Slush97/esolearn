// SPDX-License-Identifier: MIT OR Apache-2.0
//! Decision tree classifier and regressor implementations.
//!
//! Contains the CART tree-building algorithm with pre-sorted indices,
//! feature bagging, class weighting, and cost-complexity pruning.

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::weights::{compute_sample_weights, ClassWeight};

use super::{
    compute_impurity, compute_impurity_weighted, majority_class, weighted_majority_class,
    BestSplit, FlatTree, SplitCriterion, TreeNode,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Pre-sort sample indices by each feature value. O(n·log n) per feature.
pub(crate) fn presort_indices(data: &Dataset, indices: &[usize]) -> Vec<Vec<usize>> {
    let n_features = data.n_features();
    let mut sorted_by_feature = Vec::with_capacity(n_features);
    for feat_idx in 0..n_features {
        let col = &data.features[feat_idx];
        let mut sorted = indices.to_vec();
        sorted.sort_unstable_by(|&a, &b| {
            col[a]
                .partial_cmp(&col[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted_by_feature.push(sorted);
    }
    sorted_by_feature
}

/// Filter global sorted arrays to only include member indices.
fn filter_sorted(global_sorted: &[Vec<usize>], membership: &[bool]) -> Vec<Vec<usize>> {
    global_sorted
        .iter()
        .map(|gs| gs.iter().copied().filter(|&idx| membership[idx]).collect())
        .collect()
}

/// Partition sorted arrays into left/right based on a split decision.
/// Preserves sort order within each partition.
///
/// Reuses the parent's Vec allocations for the left child (in-place
/// stable partition), only allocating new Vecs for the right child.
/// This halves the number of heap allocations during tree building.
fn partition_sorted(
    mut sorted_by_feature: Vec<Vec<usize>>,
    split_col: &[f64],
    threshold: f64,
    _left_count: usize,
    right_count: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n_feat = sorted_by_feature.len();
    let mut right_sorted = Vec::with_capacity(n_feat);
    for feat_sorted in &mut sorted_by_feature {
        let mut right = Vec::with_capacity(right_count);
        let mut write = 0;
        for read in 0..feat_sorted.len() {
            let idx = feat_sorted[read];
            if split_col[idx] <= threshold {
                feat_sorted[write] = idx;
                write += 1;
            } else {
                right.push(idx);
            }
        }
        feat_sorted.truncate(write);
        right_sorted.push(right);
    }
    (sorted_by_feature, right_sorted)
}

/// Populate the feature buffer with indices, optionally shuffled for feature bagging.
///
/// When `max_features` is set, uses `rng` to select a random subset via
/// partial Fisher-Yates shuffle. The caller must supply a mutable RNG whose
/// state advances between calls so that each split considers a *different*
/// random feature subset (critical for Random Forest decorrelation).
fn fill_feature_buf(
    feature_buf: &mut Vec<usize>,
    n_features: usize,
    max_features: Option<usize>,
    rng: &mut crate::rng::FastRng,
) {
    feature_buf.clear();
    feature_buf.extend(0..n_features);
    if let Some(max_f) = max_features {
        let m = max_f.min(n_features);
        for i in 0..m {
            let j = rng.usize(i..n_features);
            feature_buf.swap(i, j);
        }
        feature_buf.truncate(m);
    }
}

// ---------------------------------------------------------------------------
// Decision Tree Classifier
// ---------------------------------------------------------------------------

/// CART decision tree for classification.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DecisionTreeClassifier {
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: Option<usize>,
    criterion: SplitCriterion,
    ccp_alpha: f64,
    /// Class weighting strategy for imbalanced datasets.
    pub(crate) class_weight: ClassWeight,
    /// Per-sample weights computed from `class_weight` during fit.
    pub(crate) sample_weights: Option<Vec<f64>>,
    /// Flattened tree for cache-optimal prediction.
    pub(crate) flat_tree: Option<FlatTree>,
    n_classes: usize,
    n_features: usize,
    pub(crate) feature_importances_: Vec<f64>,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl DecisionTreeClassifier {
    /// Create a new classifier with default parameters.
    pub fn new() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            criterion: SplitCriterion::Gini,
            ccp_alpha: 0.0,
            class_weight: ClassWeight::Uniform,
            sample_weights: None,
            flat_tree: None,
            n_classes: 0,
            n_features: 0,
            feature_importances_: Vec::new(),
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set maximum tree depth.
    pub fn max_depth(mut self, d: usize) -> Self {
        self.max_depth = Some(d);
        self
    }

    /// Set minimum samples required to split an internal node.
    pub fn min_samples_split(mut self, n: usize) -> Self {
        self.min_samples_split = n;
        self
    }

    /// Set minimum samples required in a leaf node.
    pub fn min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }

    /// Set maximum features to consider per split (for random forest).
    pub fn max_features(mut self, n: usize) -> Self {
        self.max_features = Some(n);
        self
    }

    /// Set the split criterion.
    pub fn criterion(mut self, c: SplitCriterion) -> Self {
        self.criterion = c;
        self
    }

    /// Set class weighting strategy for imbalanced datasets.
    ///
    /// When set to [`ClassWeight::Balanced`], minority classes receive
    /// higher weight in impurity calculations, improving their recall.
    ///
    /// # Example
    /// ```
    /// use scry_learn::tree::DecisionTreeClassifier;
    /// use scry_learn::weights::ClassWeight;
    ///
    /// let dt = DecisionTreeClassifier::new()
    ///     .class_weight(ClassWeight::Balanced);
    /// ```
    pub fn class_weight(mut self, cw: ClassWeight) -> Self {
        self.class_weight = cw;
        self
    }

    /// Set cost-complexity pruning parameter.
    ///
    /// Subtrees with effective alpha ≤ `ccp_alpha` are pruned after
    /// tree construction. A value of 0.0 (default) disables pruning.
    /// Larger values produce smaller, more regularized trees.
    ///
    /// # Example
    /// ```
    /// use scry_learn::tree::DecisionTreeClassifier;
    ///
    /// let dt = DecisionTreeClassifier::new()
    ///     .ccp_alpha(0.01);
    /// ```
    pub fn ccp_alpha(mut self, alpha: f64) -> Self {
        self.ccp_alpha = alpha;
        self
    }

    /// Train the decision tree on a dataset.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        let indices: Vec<usize> = (0..data.n_samples()).collect();
        self.fit_on_indices(data, &indices)
    }

    /// Train the decision tree on a dataset using a subset of sample indices.
    ///
    /// This is the production path used by Random Forest — avoids copying
    /// the dataset by training directly on indices into the original data.
    ///
    /// Internally uses pre-sorted indices: sorts once at the root (O(n·log n)),
    /// then partitions at each node (O(n) per feature per node) — matching
    /// scikit-learn's optimized CART implementation.
    pub(crate) fn fit_on_indices(
        &mut self,
        data: &Dataset,
        sample_indices: &[usize],
    ) -> Result<()> {
        let sorted_by_feature = presort_indices(data, sample_indices);
        self.fit_with_sorted(data, sample_indices, sorted_by_feature)
    }

    /// Train using pre-sorted indices shared across trees (RF memory optimization).
    ///
    /// `global_sorted` contains ALL dataset indices sorted by each feature.
    /// Filters to only the bootstrap sample indices, then builds the tree
    /// using partitioned sorted arrays.
    pub(crate) fn fit_on_indices_presorted(
        &mut self,
        data: &Dataset,
        sample_indices: &[usize],
        global_sorted: &[Vec<usize>],
    ) -> Result<()> {
        // Filter global sorted arrays to only include bootstrap sample indices.
        let membership_len = global_sorted.first().map_or(0, Vec::len);
        let mut membership = vec![false; membership_len];
        for &i in sample_indices {
            membership[i] = true;
        }
        let sorted_by_feature = filter_sorted(global_sorted, &membership);
        self.fit_with_sorted(data, sample_indices, sorted_by_feature)
    }

    /// Internal: fit using pre-filtered, per-node sorted arrays.
    fn fit_with_sorted(
        &mut self,
        data: &Dataset,
        sample_indices: &[usize],
        sorted_by_feature: Vec<Vec<usize>>,
    ) -> Result<()> {
        let n = sample_indices.len();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        self.n_features = data.n_features();
        self.n_classes = data.n_classes();
        self.feature_importances_ = vec![0.0; self.n_features];

        // Compute per-sample weights if class_weight is non-uniform.
        let weights = match &self.class_weight {
            ClassWeight::Uniform => None,
            cw => Some(compute_sample_weights(&data.target, cw)),
        };
        self.sample_weights = weights;

        let mut feature_buf = Vec::with_capacity(self.n_features);
        let mut split_rng = crate::rng::FastRng::new(0);

        let tree = if self.sample_weights.is_some() {
            self.build_tree_weighted(
                data,
                sorted_by_feature,
                n,
                0,
                &mut feature_buf,
                &mut split_rng,
            )
        } else {
            self.build_tree(
                data,
                sorted_by_feature,
                n,
                0,
                &mut feature_buf,
                &mut split_rng,
            )
        };

        // Apply cost-complexity pruning if requested.
        let tree = if self.ccp_alpha > 0.0 {
            tree.prune_ccp(self.ccp_alpha)
        } else {
            tree
        };

        // Flatten recursive tree into contiguous array for prediction.
        let flat = FlatTree::from_tree_node(&tree, self.n_classes);
        self.flat_tree = Some(flat);

        // Normalize feature importances to sum to 1.
        let total: f64 = self.feature_importances_.iter().sum();
        if total > 0.0 {
            for imp in &mut self.feature_importances_ {
                *imp /= total;
            }
        }

        // Free training-only data.
        self.sample_weights = None;

        Ok(())
    }

    /// Predict class labels for a feature matrix.
    ///
    /// `features` is row-major: `features[sample_idx][feature_idx]`.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        crate::version::check_schema_version(self._schema_version)?;
        let ft = self.flat_tree.as_ref().ok_or(ScryLearnError::NotFitted)?;
        Ok(ft.predict(features))
    }

    /// Predict class probabilities for a feature matrix.
    pub fn predict_proba(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let ft = self.flat_tree.as_ref().ok_or(ScryLearnError::NotFitted)?;
        let n_classes = self.n_classes;
        Ok(features
            .iter()
            .map(|row| ft.predict_proba_sample(row, n_classes))
            .collect())
    }

    /// Get feature importances (sum of weighted impurity decreases).
    pub fn feature_importances(&self) -> Result<Vec<f64>> {
        if self.flat_tree.is_none() {
            return Err(ScryLearnError::NotFitted);
        }
        Ok(self.feature_importances_.clone())
    }

    /// Get the flat tree (for direct access).
    pub fn flat_tree(&self) -> Option<&FlatTree> {
        self.flat_tree.as_ref()
    }

    /// Tree depth.
    pub fn depth(&self) -> usize {
        self.flat_tree.as_ref().map_or(0, FlatTree::depth)
    }

    /// Number of leaf nodes.
    pub fn n_leaves(&self) -> usize {
        self.flat_tree.as_ref().map_or(0, FlatTree::n_leaves)
    }

    /// Number of features the model was trained on.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Number of classes.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Compute the cost-complexity pruning path for this classifier.
    ///
    /// Trains an unpruned tree, then returns `(ccp_alphas, total_impurities)`,
    /// a sequence of effective alpha values and the corresponding total tree
    /// impurity at each pruning step. Useful for selecting `ccp_alpha` via
    /// the elbow method.
    ///
    /// The classifier must be fitted before calling this method.
    pub fn cost_complexity_pruning_path(&self, data: &Dataset) -> Result<(Vec<f64>, Vec<f64>)> {
        // Build an unpruned tree to compute the path.
        let mut unpruned = self.clone();
        unpruned.ccp_alpha = 0.0;
        unpruned.fit(data)?;

        // Rebuild the recursive tree from the dataset to get the path.
        let indices: Vec<usize> = (0..data.n_samples()).collect();
        let sorted_by_feature = presort_indices(data, &indices);
        let n = indices.len();
        let mut feature_buf = Vec::with_capacity(unpruned.n_features);
        let mut split_rng = crate::rng::FastRng::new(0);

        let tree = if unpruned.sample_weights.is_some() {
            unpruned.build_tree_weighted(
                data,
                sorted_by_feature,
                n,
                0,
                &mut feature_buf,
                &mut split_rng,
            )
        } else {
            unpruned.build_tree(
                data,
                sorted_by_feature,
                n,
                0,
                &mut feature_buf,
                &mut split_rng,
            )
        };
        Ok(tree.cost_complexity_pruning_path())
    }

    // -----------------------------------------------------------------------
    // Recursive tree building (unweighted)
    // -----------------------------------------------------------------------

    /// Build tree using partitioned sorted arrays.
    ///
    /// `sorted_by_feature[feat_idx]` contains only this node's sample indices,
    /// sorted by that feature's value. No membership bitset needed.
    fn build_tree(
        &mut self,
        data: &Dataset,
        sorted_by_feature: Vec<Vec<usize>>,
        n_root_samples: usize,
        depth: usize,
        feature_buf: &mut Vec<usize>,
        split_rng: &mut crate::rng::FastRng,
    ) -> TreeNode {
        let active = &sorted_by_feature[0];
        let n_actual = active.len();

        // Collect class counts.
        let mut class_counts = vec![0usize; self.n_classes];
        for &idx in active {
            let c = data.target[idx] as usize;
            if c < self.n_classes {
                class_counts[c] += 1;
            }
        }
        let impurity = compute_impurity(&class_counts, n_actual, self.criterion);

        // Check stopping conditions.
        let max_depth_reached = self.max_depth.is_some_and(|d| depth >= d);
        let too_few_samples = n_actual < self.min_samples_split;
        let is_pure = impurity < 1e-12;

        if max_depth_reached || too_few_samples || is_pure {
            return TreeNode::Leaf {
                prediction: majority_class(&class_counts),
                n_samples: n_actual,
                class_counts,
                impurity,
            };
        }

        // Find best split.
        let best = self.find_best_split(
            data,
            &sorted_by_feature,
            &class_counts,
            n_actual,
            feature_buf,
            split_rng,
        );

        let node_prediction = majority_class(&class_counts);

        match best {
            None => TreeNode::Leaf {
                prediction: node_prediction,
                n_samples: n_actual,
                class_counts,
                impurity,
            },
            Some(split) => {
                let col = &data.features[split.feature_idx];
                let threshold = split.threshold;

                // Count left/right.
                let mut left_count = 0usize;
                let mut right_count = 0usize;
                for &idx in active {
                    if col[idx] <= threshold {
                        left_count += 1;
                    } else {
                        right_count += 1;
                    }
                }

                if left_count < self.min_samples_leaf || right_count < self.min_samples_leaf {
                    return TreeNode::Leaf {
                        prediction: node_prediction,
                        n_samples: n_actual,
                        class_counts,
                        impurity,
                    };
                }

                // Record feature importance.
                let weighted_impurity_decrease = (n_actual as f64 / n_root_samples as f64)
                    * (impurity - split.impurity_decrease);
                self.feature_importances_[split.feature_idx] += weighted_impurity_decrease.max(0.0);

                // Partition sorted arrays into left/right children.
                let (left_sorted, right_sorted) =
                    partition_sorted(sorted_by_feature, col, threshold, left_count, right_count);

                let left = self.build_tree(
                    data,
                    left_sorted,
                    n_root_samples,
                    depth + 1,
                    feature_buf,
                    split_rng,
                );
                let right = self.build_tree(
                    data,
                    right_sorted,
                    n_root_samples,
                    depth + 1,
                    feature_buf,
                    split_rng,
                );

                TreeNode::Split {
                    feature_idx: split.feature_idx,
                    threshold,
                    left: Box::new(left),
                    right: Box::new(right),
                    n_samples: n_actual,
                    impurity,
                    class_counts,
                    prediction: node_prediction,
                }
            }
        }
    }

    /// Find the best split by scanning sorted arrays — O(n) per feature.
    fn find_best_split(
        &self,
        data: &Dataset,
        sorted_by_feature: &[Vec<usize>],
        parent_counts: &[usize],
        n_parent: usize,
        feature_buf: &mut Vec<usize>,
        split_rng: &mut crate::rng::FastRng,
    ) -> Option<BestSplit> {
        let n_features = data.n_features();
        let mut best: Option<BestSplit> = None;

        fill_feature_buf(feature_buf, n_features, self.max_features, split_rng);

        for &feat_idx in feature_buf.iter() {
            let col = &data.features[feat_idx];
            let sorted = &sorted_by_feature[feat_idx];

            let mut left_counts = vec![0usize; self.n_classes];
            let mut left_n = 0;
            let mut prev_val = f64::NEG_INFINITY;

            for &idx in sorted {
                let val = col[idx];

                // Check threshold between previous and current value.
                if left_n > 0 && (val - prev_val).abs() > 1e-12 {
                    let right_n = n_parent - left_n;
                    if left_n >= self.min_samples_leaf && right_n >= self.min_samples_leaf {
                        let right_counts: Vec<usize> = parent_counts
                            .iter()
                            .zip(left_counts.iter())
                            .map(|(&p, &l)| p - l)
                            .collect();

                        let left_imp = compute_impurity(&left_counts, left_n, self.criterion);
                        let right_imp = compute_impurity(&right_counts, right_n, self.criterion);
                        let weighted_imp = (left_n as f64 * left_imp + right_n as f64 * right_imp)
                            / n_parent as f64;

                        let threshold = (prev_val + val) / 2.0;

                        let is_better = best
                            .as_ref()
                            .is_none_or(|b| weighted_imp < b.impurity_decrease);

                        if is_better {
                            best = Some(BestSplit {
                                feature_idx: feat_idx,
                                threshold,
                                impurity_decrease: weighted_imp,
                            });
                        }
                    }
                }

                // Add current sample to left side.
                let class = data.target[idx] as usize;
                if class < self.n_classes {
                    left_counts[class] += 1;
                }
                left_n += 1;
                prev_val = val;
            }
        }

        best
    }

    // -------------------------------------------------------------------
    // Weighted tree building (class_weight support)
    // -------------------------------------------------------------------

    /// Build tree using partitioned sorted arrays with per-sample weights.
    fn build_tree_weighted(
        &mut self,
        data: &Dataset,
        sorted_by_feature: Vec<Vec<usize>>,
        n_root_samples: usize,
        depth: usize,
        feature_buf: &mut Vec<usize>,
        split_rng: &mut crate::rng::FastRng,
    ) -> TreeNode {
        let weights = self.sample_weights.as_ref().expect("weights must be set");
        let active = &sorted_by_feature[0];
        let n_actual = active.len();

        // Collect weighted and unweighted class counts.
        let mut w_counts = vec![0.0_f64; self.n_classes];
        let mut w_total = 0.0_f64;
        let mut class_counts = vec![0usize; self.n_classes];

        for &idx in active {
            let c = data.target[idx] as usize;
            let w = weights[idx];
            if c < self.n_classes {
                w_counts[c] += w;
                class_counts[c] += 1;
            }
            w_total += w;
        }

        let impurity = compute_impurity_weighted(&w_counts, w_total, self.criterion);

        // Check stopping conditions.
        let max_depth_reached = self.max_depth.is_some_and(|d| depth >= d);
        let too_few_samples = n_actual < self.min_samples_split;
        let is_pure = impurity < 1e-12;

        if max_depth_reached || too_few_samples || is_pure {
            return TreeNode::Leaf {
                prediction: weighted_majority_class(&w_counts),
                n_samples: n_actual,
                class_counts,
                impurity,
            };
        }

        let best = self.find_best_split_weighted(
            data,
            &sorted_by_feature,
            &w_counts,
            w_total,
            n_actual,
            feature_buf,
            split_rng,
        );

        let node_prediction = weighted_majority_class(&w_counts);

        match best {
            None => TreeNode::Leaf {
                prediction: node_prediction,
                n_samples: n_actual,
                class_counts,
                impurity,
            },
            Some(split) => {
                let col = &data.features[split.feature_idx];
                let threshold = split.threshold;

                let mut left_count = 0usize;
                let mut right_count = 0usize;
                for &idx in active {
                    if col[idx] <= threshold {
                        left_count += 1;
                    } else {
                        right_count += 1;
                    }
                }

                if left_count < self.min_samples_leaf || right_count < self.min_samples_leaf {
                    return TreeNode::Leaf {
                        prediction: node_prediction,
                        n_samples: n_actual,
                        class_counts,
                        impurity,
                    };
                }

                // Record feature importance.
                let weighted_impurity_decrease = (n_actual as f64 / n_root_samples as f64)
                    * (impurity - split.impurity_decrease);
                self.feature_importances_[split.feature_idx] += weighted_impurity_decrease.max(0.0);

                let (left_sorted, right_sorted) =
                    partition_sorted(sorted_by_feature, col, threshold, left_count, right_count);

                let left = self.build_tree_weighted(
                    data,
                    left_sorted,
                    n_root_samples,
                    depth + 1,
                    feature_buf,
                    split_rng,
                );
                let right = self.build_tree_weighted(
                    data,
                    right_sorted,
                    n_root_samples,
                    depth + 1,
                    feature_buf,
                    split_rng,
                );

                TreeNode::Split {
                    feature_idx: split.feature_idx,
                    threshold,
                    left: Box::new(left),
                    right: Box::new(right),
                    n_samples: n_actual,
                    impurity,
                    class_counts,
                    prediction: node_prediction,
                }
            }
        }
    }

    /// Weighted variant of find_best_split — uses f64 weighted counts.
    fn find_best_split_weighted(
        &self,
        data: &Dataset,
        sorted_by_feature: &[Vec<usize>],
        parent_w_counts: &[f64],
        w_parent_total: f64,
        n_parent: usize,
        feature_buf: &mut Vec<usize>,
        split_rng: &mut crate::rng::FastRng,
    ) -> Option<BestSplit> {
        let weights = self.sample_weights.as_ref().expect("weights must be set");
        let n_features = data.n_features();
        let mut best: Option<BestSplit> = None;

        fill_feature_buf(feature_buf, n_features, self.max_features, split_rng);

        for &feat_idx in feature_buf.iter() {
            let col = &data.features[feat_idx];
            let sorted = &sorted_by_feature[feat_idx];

            let mut left_w_counts = vec![0.0_f64; self.n_classes];
            let mut left_w_total = 0.0_f64;
            let mut left_n = 0usize;
            let mut prev_val = f64::NEG_INFINITY;

            for &idx in sorted {
                let val = col[idx];
                let w = weights[idx];

                if left_n > 0 && (val - prev_val).abs() > 1e-12 {
                    let right_n = n_parent - left_n;
                    if left_n >= self.min_samples_leaf && right_n >= self.min_samples_leaf {
                        let right_w_total = w_parent_total - left_w_total;
                        let right_w_counts: Vec<f64> = parent_w_counts
                            .iter()
                            .zip(left_w_counts.iter())
                            .map(|(&p, &l)| (p - l).max(0.0))
                            .collect();

                        let left_imp =
                            compute_impurity_weighted(&left_w_counts, left_w_total, self.criterion);
                        let right_imp = compute_impurity_weighted(
                            &right_w_counts,
                            right_w_total,
                            self.criterion,
                        );
                        let weighted_imp =
                            (left_w_total * left_imp + right_w_total * right_imp) / w_parent_total;

                        let threshold = (prev_val + val) / 2.0;

                        let is_better = best
                            .as_ref()
                            .is_none_or(|b| weighted_imp < b.impurity_decrease);

                        if is_better {
                            best = Some(BestSplit {
                                feature_idx: feat_idx,
                                threshold,
                                impurity_decrease: weighted_imp,
                            });
                        }
                    }
                }

                // Add current sample to left side.
                let class = data.target[idx] as usize;
                if class < self.n_classes {
                    left_w_counts[class] += w;
                }
                left_w_total += w;
                left_n += 1;
                prev_val = val;
            }
        }

        best
    }
}

impl Default for DecisionTreeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Decision Tree Regressor
// ---------------------------------------------------------------------------

/// CART decision tree for regression.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DecisionTreeRegressor {
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: Option<usize>,
    ccp_alpha: f64,
    /// Flattened tree for cache-optimal prediction.
    pub(crate) flat_tree: Option<FlatTree>,
    n_features: usize,
    pub(crate) feature_importances_: Vec<f64>,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl DecisionTreeRegressor {
    /// Create a new regressor with default parameters.
    pub fn new() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            ccp_alpha: 0.0,
            flat_tree: None,
            n_features: 0,
            feature_importances_: Vec::new(),
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set maximum tree depth.
    pub fn max_depth(mut self, d: usize) -> Self {
        self.max_depth = Some(d);
        self
    }

    /// Set minimum samples required to split.
    pub fn min_samples_split(mut self, n: usize) -> Self {
        self.min_samples_split = n;
        self
    }

    /// Set minimum samples required in a leaf.
    pub fn min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }

    /// Set maximum features per split (for random forest).
    pub fn max_features(mut self, n: usize) -> Self {
        self.max_features = Some(n);
        self
    }

    /// Set cost-complexity pruning parameter.
    ///
    /// Subtrees with effective alpha ≤ `ccp_alpha` are pruned after
    /// tree construction. A value of 0.0 (default) disables pruning.
    /// Larger values produce smaller, more regularized trees.
    pub fn ccp_alpha(mut self, alpha: f64) -> Self {
        self.ccp_alpha = alpha;
        self
    }

    /// Train on a dataset.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        let indices: Vec<usize> = (0..data.n_samples()).collect();
        self.fit_on_indices(data, &indices)
    }

    /// Train on a dataset using a subset of sample indices.
    ///
    /// Production path for Random Forest — trains directly on indices
    /// into the original data, avoiding dataset copies.
    pub(crate) fn fit_on_indices(
        &mut self,
        data: &Dataset,
        sample_indices: &[usize],
    ) -> Result<()> {
        let n = sample_indices.len();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        self.n_features = data.n_features();
        self.feature_importances_ = vec![0.0; self.n_features];

        let sorted_by_feature = presort_indices(data, sample_indices);
        let mut feature_buf = Vec::with_capacity(self.n_features);
        let mut split_rng = crate::rng::FastRng::new(0);

        let tree = self.build_tree_reg(
            data,
            sorted_by_feature,
            n,
            0,
            &mut feature_buf,
            &mut split_rng,
        );

        // Apply cost-complexity pruning if requested.
        let tree = if self.ccp_alpha > 0.0 {
            tree.prune_ccp(self.ccp_alpha)
        } else {
            tree
        };

        // Flatten recursive tree into contiguous array for prediction.
        // Regression trees don't need class probabilities (n_classes=0).
        let flat = FlatTree::from_tree_node(&tree, 0);
        self.flat_tree = Some(flat);

        let total: f64 = self.feature_importances_.iter().sum();
        if total > 0.0 {
            for imp in &mut self.feature_importances_ {
                *imp /= total;
            }
        }
        Ok(())
    }

    /// Train using pre-sorted indices (GBT/RF optimization — sort once, reuse each round).
    ///
    /// `global_sorted` contains ALL dataset indices sorted by each feature.
    /// Filters to only the requested sample indices, then builds the tree.
    pub(crate) fn fit_on_indices_presorted(
        &mut self,
        data: &Dataset,
        sample_indices: &[usize],
        global_sorted: &[Vec<usize>],
    ) -> Result<()> {
        let n = sample_indices.len();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        self.n_features = data.n_features();
        self.feature_importances_ = vec![0.0; self.n_features];

        // Filter global sorted arrays to only include requested sample indices.
        let membership_len = global_sorted.first().map_or(0, Vec::len);
        let mut membership = vec![false; membership_len];
        for &i in sample_indices {
            membership[i] = true;
        }
        let sorted_by_feature = filter_sorted(global_sorted, &membership);
        let mut feature_buf = Vec::with_capacity(self.n_features);
        let mut split_rng = crate::rng::FastRng::new(0);

        let tree = self.build_tree_reg(
            data,
            sorted_by_feature,
            n,
            0,
            &mut feature_buf,
            &mut split_rng,
        );

        let tree = if self.ccp_alpha > 0.0 {
            tree.prune_ccp(self.ccp_alpha)
        } else {
            tree
        };

        let flat = FlatTree::from_tree_node(&tree, 0);
        self.flat_tree = Some(flat);

        let total: f64 = self.feature_importances_.iter().sum();
        if total > 0.0 {
            for imp in &mut self.feature_importances_ {
                *imp /= total;
            }
        }
        Ok(())
    }

    /// Predict values.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        crate::version::check_schema_version(self._schema_version)?;
        let ft = self.flat_tree.as_ref().ok_or(ScryLearnError::NotFitted)?;
        Ok(ft.predict(features))
    }

    /// Feature importances.
    pub fn feature_importances(&self) -> Result<Vec<f64>> {
        if self.flat_tree.is_none() {
            return Err(ScryLearnError::NotFitted);
        }
        Ok(self.feature_importances_.clone())
    }

    /// Get the flat tree.
    pub fn flat_tree(&self) -> Option<&FlatTree> {
        self.flat_tree.as_ref()
    }

    /// Number of features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Build tree using partitioned sorted arrays — no membership bitset.
    fn build_tree_reg(
        &mut self,
        data: &Dataset,
        sorted_by_feature: Vec<Vec<usize>>,
        n_root_samples: usize,
        depth: usize,
        feature_buf: &mut Vec<usize>,
        split_rng: &mut crate::rng::FastRng,
    ) -> TreeNode {
        let active = &sorted_by_feature[0];
        let n_actual = active.len();

        if n_actual == 0 {
            return TreeNode::Leaf {
                prediction: 0.0,
                n_samples: 0,
                class_counts: Vec::new(),
                impurity: 0.0,
            };
        }

        // Compute mean/MSE from active indices directly.
        let mut sum = 0.0;
        let mut sq_sum = 0.0;
        for &idx in active {
            let v = data.target[idx];
            sum += v;
            sq_sum += v * v;
        }
        let mean = sum / n_actual as f64;
        // Clamp to 0.0: the textbook formula E[X²]-E[X]² can go slightly
        // negative due to floating-point catastrophic cancellation.
        let mse = (sq_sum / n_actual as f64 - mean * mean).max(0.0);

        let max_depth_reached = self.max_depth.is_some_and(|d| depth >= d);
        let too_few = n_actual < self.min_samples_split;

        if max_depth_reached || too_few || mse < 1e-12 {
            return TreeNode::Leaf {
                prediction: mean,
                n_samples: n_actual,
                class_counts: Vec::new(),
                impurity: mse,
            };
        }

        let best = self.find_best_split_reg(
            data,
            &sorted_by_feature,
            sum,
            sq_sum,
            n_actual,
            feature_buf,
            split_rng,
        );

        match best {
            None => TreeNode::Leaf {
                prediction: mean,
                n_samples: n_actual,
                class_counts: Vec::new(),
                impurity: mse,
            },
            Some(split) => {
                let col = &data.features[split.feature_idx];
                let threshold = split.threshold;

                let mut left_count = 0usize;
                let mut right_count = 0usize;
                for &idx in active {
                    if col[idx] <= threshold {
                        left_count += 1;
                    } else {
                        right_count += 1;
                    }
                }

                if left_count < self.min_samples_leaf || right_count < self.min_samples_leaf {
                    return TreeNode::Leaf {
                        prediction: mean,
                        n_samples: n_actual,
                        class_counts: Vec::new(),
                        impurity: mse,
                    };
                }

                let decrease =
                    (n_actual as f64 / n_root_samples as f64) * (mse - split.impurity_decrease);
                self.feature_importances_[split.feature_idx] += decrease.max(0.0);

                let (left_sorted, right_sorted) =
                    partition_sorted(sorted_by_feature, col, threshold, left_count, right_count);

                let left = self.build_tree_reg(
                    data,
                    left_sorted,
                    n_root_samples,
                    depth + 1,
                    feature_buf,
                    split_rng,
                );
                let right = self.build_tree_reg(
                    data,
                    right_sorted,
                    n_root_samples,
                    depth + 1,
                    feature_buf,
                    split_rng,
                );

                TreeNode::Split {
                    feature_idx: split.feature_idx,
                    threshold,
                    left: Box::new(left),
                    right: Box::new(right),
                    n_samples: n_actual,
                    impurity: mse,
                    class_counts: Vec::new(),
                    prediction: mean,
                }
            }
        }
    }

    /// Find best regression split using incremental variance — O(n) per feature.
    fn find_best_split_reg(
        &self,
        data: &Dataset,
        sorted_by_feature: &[Vec<usize>],
        total_sum: f64,
        total_sq: f64,
        n_parent: usize,
        feature_buf: &mut Vec<usize>,
        split_rng: &mut crate::rng::FastRng,
    ) -> Option<BestSplit> {
        let n_features = data.n_features();
        let mut best: Option<BestSplit> = None;

        fill_feature_buf(feature_buf, n_features, self.max_features, split_rng);

        for &feat_idx in feature_buf.iter() {
            let col = &data.features[feat_idx];
            let sorted = &sorted_by_feature[feat_idx];

            let mut left_sum = 0.0;
            let mut left_sq_sum = 0.0;
            let mut left_n = 0usize;
            let mut prev_val = f64::NEG_INFINITY;

            for &idx in sorted {
                let feat_val = col[idx];

                // Check threshold between previous and current.
                if left_n > 0 && (feat_val - prev_val).abs() > 1e-12 {
                    let right_n = n_parent - left_n;
                    if left_n >= self.min_samples_leaf && right_n >= self.min_samples_leaf {
                        let left_mse = (left_sq_sum / left_n as f64
                            - (left_sum / left_n as f64).powi(2))
                        .max(0.0);
                        let right_sum = total_sum - left_sum;
                        let right_sq = total_sq - left_sq_sum;
                        let right_mse = (right_sq / right_n as f64
                            - (right_sum / right_n as f64).powi(2))
                        .max(0.0);

                        let weighted = (left_n as f64 * left_mse + right_n as f64 * right_mse)
                            / n_parent as f64;

                        let threshold = (prev_val + feat_val) / 2.0;

                        let is_better =
                            best.as_ref().is_none_or(|b| weighted < b.impurity_decrease);
                        if is_better {
                            best = Some(BestSplit {
                                feature_idx: feat_idx,
                                threshold,
                                impurity_decrease: weighted,
                            });
                        }
                    }
                }

                let target_val = data.target[idx];
                left_sum += target_val;
                left_sq_sum += target_val * target_val;
                left_n += 1;
                prev_val = feat_val;
            }
        }
        best
    }
}

impl Default for DecisionTreeRegressor {
    fn default() -> Self {
        Self::new()
    }
}
