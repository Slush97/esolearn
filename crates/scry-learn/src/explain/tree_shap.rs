// SPDX-License-Identifier: MIT OR Apache-2.0
//! TreeSHAP: polynomial-time exact Shapley values for tree ensembles.
//!
//! Implements the path-dependent TreeSHAP algorithm from
//! Lundberg & Lee (2018) "Consistent Individualized Feature Attribution
//! for Tree Ensembles".

use crate::tree::FlatTree;

/// A single entry in the SHAP path tracking structure.
#[derive(Clone, Copy, Debug)]
struct PathEntry {
    /// Index of the feature at this path position, or `usize::MAX` for "no feature".
    feature_index: usize,
    /// Fraction of zero-valued paths through this node.
    zero_fraction: f64,
    /// Fraction of one-valued paths through this node.
    one_fraction: f64,
    /// Path weight.
    pweight: f64,
}

/// Compute SHAP values for a single sample on a single `FlatTree`.
///
/// # Arguments
///
/// * `tree` - A flattened decision tree.
/// * `sample` - Feature values for the sample to explain (length = n_features).
/// * `node_counts` - Number of training samples reaching each DFS node,
///   same length as `tree.nodes`. Available as `tree.node_counts`.
///
/// # Returns
///
/// A `Vec<f64>` of SHAP values, one per feature. The sum of SHAP values
/// plus the expected value (root prediction weighted by coverage) equals
/// the model prediction for this sample.
pub fn tree_shap(tree: &FlatTree, sample: &[f64], node_counts: &[usize]) -> Vec<f64> {
    let n_features = sample.len();
    let mut phi = vec![0.0; n_features];

    if tree.nodes.is_empty() {
        return phi;
    }

    // Initial path: single entry at [0] with pweight=1 (base).
    let mut path: Vec<PathEntry> = vec![PathEntry {
        feature_index: usize::MAX,
        zero_fraction: 1.0,
        one_fraction: 1.0,
        pweight: 1.0,
    }];

    recurse(
        tree,
        sample,
        node_counts,
        0,
        &mut path,
        0,      // unique_depth
        1.0,    // pz
        1.0,    // po
        -1_i64, // no incoming feature
        &mut phi,
    );

    phi
}

/// Compute SHAP values for a single sample across an ensemble of trees.
///
/// # Arguments
///
/// * `trees` - Slice of `(tree, node_counts)` pairs.
/// * `sample` - Feature values for the sample.
/// * `n_features` - Number of features.
///
/// # Returns
///
/// A `Vec<f64>` of averaged SHAP values across all trees.
pub fn ensemble_tree_shap(
    trees: &[(&FlatTree, &[usize])],
    sample: &[f64],
    n_features: usize,
) -> Vec<f64> {
    if trees.is_empty() {
        return vec![0.0; n_features];
    }

    let mut phi = vec![0.0; n_features];

    for &(tree, counts) in trees {
        let tree_phi = tree_shap(tree, sample, counts);
        for (i, &v) in tree_phi.iter().enumerate() {
            if i < n_features {
                phi[i] += v;
            }
        }
    }

    let n = trees.len() as f64;
    for v in &mut phi {
        *v /= n;
    }

    phi
}

/// Recursive path-dependent TreeSHAP traversal (Algorithm 2 from Lundberg 2018).
///
/// `incoming_feature` is -1 for the initial call, otherwise the feature index
/// from the parent split that led to this node.
#[allow(clippy::too_many_arguments)]
fn recurse(
    tree: &FlatTree,
    sample: &[f64],
    node_counts: &[usize],
    node_idx: usize,
    path: &mut Vec<PathEntry>,
    unique_depth: usize,
    pz: f64,
    po: f64,
    incoming_feature: i64,
    phi: &mut [f64],
) {
    // Extend the path if we have an incoming feature.
    if incoming_feature >= 0 {
        extend_path(path, unique_depth, pz, po, incoming_feature as usize);
    }

    let node = &tree.nodes[node_idx];

    if node.right == u32::MAX {
        // Leaf node: accumulate SHAP contributions.
        let leaf_idx = node.feature_idx as usize;
        let leaf_val = tree.predictions[leaf_idx];

        for i in 1..=unique_depth {
            let w = unwound_path_sum(path, unique_depth, i);
            let entry = &path[i];
            if entry.feature_index < phi.len() {
                let contrib = w * (entry.one_fraction - entry.zero_fraction) * leaf_val;
                phi[entry.feature_index] += contrib;
            }
        }
    } else {
        // Internal node: split and recurse.
        let split_feature = node.feature_idx as usize;
        let threshold = node.threshold;
        let left_idx = node_idx + 1;
        let right_idx = node.right as usize;

        let parent_count = node_counts[node_idx] as f64;
        let left_count = if left_idx < node_counts.len() {
            node_counts[left_idx] as f64
        } else {
            0.0
        };
        let right_count = if right_idx < node_counts.len() {
            node_counts[right_idx] as f64
        } else {
            0.0
        };

        // Determine which child the sample goes to ("hot" path).
        let goes_left = split_feature < sample.len() && sample[split_feature] <= threshold;

        let (hot_idx, cold_idx, hot_count, cold_count) = if goes_left {
            (left_idx, right_idx, left_count, right_count)
        } else {
            (right_idx, left_idx, right_count, left_count)
        };

        let hot_zero_fraction = if parent_count > 0.0 {
            hot_count / parent_count
        } else {
            0.5
        };
        let cold_zero_fraction = if parent_count > 0.0 {
            cold_count / parent_count
        } else {
            0.5
        };

        // Check if this feature already appears in the path.
        let mut incoming_zero = 1.0;
        let mut incoming_one = 1.0;
        let mut found_idx = None;

        for i in 1..=unique_depth {
            if path[i].feature_index == split_feature {
                incoming_zero = path[i].zero_fraction;
                incoming_one = path[i].one_fraction;
                found_idx = Some(i);
                break;
            }
        }

        let mut next_depth = unique_depth;

        if let Some(fi) = found_idx {
            unwind_path(path, next_depth, fi);
            next_depth -= 1;
        }

        // Save path state for restoration after recursion.
        let saved_path = path.clone();

        // Recurse into hot path (one_fraction = incoming_one for hot).
        recurse(
            tree,
            sample,
            node_counts,
            hot_idx,
            path,
            next_depth + 1,
            incoming_zero * hot_zero_fraction,
            incoming_one,
            split_feature as i64,
            phi,
        );

        // Restore path.
        *path = saved_path.clone();

        // Recurse into cold path (one_fraction = 0 for cold).
        recurse(
            tree,
            sample,
            node_counts,
            cold_idx,
            path,
            next_depth + 1,
            incoming_zero * cold_zero_fraction,
            0.0,
            split_feature as i64,
            phi,
        );

        // Restore path.
        *path = saved_path;
    }
}

/// Extend the path with a new entry (Lundberg Algorithm 3).
fn extend_path(
    path: &mut Vec<PathEntry>,
    unique_depth: usize,
    zero_fraction: f64,
    one_fraction: f64,
    feature: usize,
) {
    // Ensure path is large enough.
    while path.len() <= unique_depth {
        path.push(PathEntry {
            feature_index: usize::MAX,
            zero_fraction: 0.0,
            one_fraction: 0.0,
            pweight: 0.0,
        });
    }

    path[unique_depth] = PathEntry {
        feature_index: feature,
        zero_fraction,
        one_fraction,
        pweight: if unique_depth == 0 { 1.0 } else { 0.0 },
    };

    // Update path weights using the binomial recurrence.
    if unique_depth > 0 {
        let d = unique_depth; // the new depth index
        for i in (0..d).rev() {
            let old_pw = path[i].pweight;
            path[i + 1].pweight += one_fraction * old_pw * ((i + 1) as f64) / ((d + 1) as f64);
            path[i].pweight = zero_fraction * old_pw * ((d - i) as f64) / ((d + 1) as f64);
        }
    }
}

/// Remove a feature from the path and adjust weights (Lundberg Algorithm 4).
fn unwind_path(path: &mut [PathEntry], unique_depth: usize, path_index: usize) {
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;

    let n = unique_depth;
    let mut next_one = path[n].pweight;

    for i in (0..n).rev() {
        if one_fraction.abs() > 1e-30 {
            let tmp = next_one * ((n + 1) as f64) / (((i + 1) as f64) * one_fraction);
            next_one = path[i].pweight - tmp * zero_fraction * ((n - i) as f64) / ((n + 1) as f64);
            path[i].pweight = tmp;
        } else {
            path[i].pweight =
                path[i].pweight * ((n + 1) as f64) / (zero_fraction * ((n - i) as f64));
        }
    }

    // Shift entries to remove the unwound position.
    for i in path_index..n {
        path[i] = path[i + 1];
    }
}

/// Compute the unwound path weight sum for a feature (Lundberg Algorithm 5).
fn unwound_path_sum(path: &[PathEntry], unique_depth: usize, path_index: usize) -> f64 {
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;

    let n = unique_depth;
    let mut next_one = path[n].pweight;
    let mut total = 0.0;

    if one_fraction.abs() < 1e-30 && zero_fraction.abs() < 1e-30 {
        return 0.0;
    }

    for i in (0..n).rev() {
        if one_fraction.abs() > 1e-30 {
            let tmp = next_one * ((n + 1) as f64) / (((i + 1) as f64) * one_fraction);
            total += tmp;
            next_one = path[i].pweight - tmp * zero_fraction * ((n - i) as f64) / ((n + 1) as f64);
        } else {
            total += (path[i].pweight / zero_fraction) / (((n - i) as f64) / ((n + 1) as f64));
        }
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{FlatTree, TreeNode};

    /// Build a simple tree: if x[0] <= 0.5 then 1.0 else 3.0.
    fn simple_tree() -> (FlatTree, Vec<usize>) {
        let root = TreeNode::Split {
            feature_idx: 0,
            threshold: 0.5,
            left: Box::new(TreeNode::Leaf {
                prediction: 1.0,
                n_samples: 50,
                class_counts: vec![],
                impurity: 0.0,
            }),
            right: Box::new(TreeNode::Leaf {
                prediction: 3.0,
                n_samples: 50,
                class_counts: vec![],
                impurity: 0.0,
            }),
            n_samples: 100,
            impurity: 0.5,
            class_counts: vec![],
            prediction: 2.0,
        };
        let flat = FlatTree::from_tree_node(&root, 0);
        let counts = flat.node_counts.clone();
        (flat, counts)
    }

    #[test]
    fn test_tree_shap_single_split() {
        let (tree, counts) = simple_tree();

        // Sample goes left (x[0] = 0.2 <= 0.5), prediction = 1.0.
        let sample = vec![0.2];
        let phi = tree_shap(&tree, &sample, &counts);

        assert_eq!(phi.len(), 1);
        // Expected = (50*1 + 50*3)/100 = 2.0, prediction = 1.0.
        // So phi[0] should be exactly -1.0.
        assert!(
            (phi[0] - (-1.0)).abs() < 1e-10,
            "SHAP for left sample should be -1.0, got {}",
            phi[0]
        );
    }

    #[test]
    fn test_tree_shap_right_path() {
        let (tree, counts) = simple_tree();

        // Sample goes right (x[0] = 0.8 > 0.5), prediction = 3.0.
        let sample = vec![0.8];
        let phi = tree_shap(&tree, &sample, &counts);

        assert_eq!(phi.len(), 1);
        // Expected = 2.0, prediction = 3.0, so phi[0] ~ 1.0.
        assert!(
            (phi[0] - 1.0).abs() < 1e-10,
            "SHAP for right sample should be 1.0, got {}",
            phi[0]
        );
    }

    #[test]
    fn test_tree_shap_empty_tree() {
        let tree = FlatTree {
            nodes: vec![],
            predictions: vec![],
            leaf_probas: vec![],
            n_classes_stored: 0,
            node_counts: vec![],
        };
        let phi = tree_shap(&tree, &[1.0, 2.0], &[]);
        assert_eq!(phi, vec![0.0, 0.0]);
    }

    #[test]
    fn test_ensemble_tree_shap() {
        let (tree1, counts1) = simple_tree();
        let (tree2, counts2) = simple_tree();

        let trees: Vec<(&FlatTree, &[usize])> = vec![(&tree1, &counts1), (&tree2, &counts2)];
        let sample = vec![0.2];
        let phi = ensemble_tree_shap(&trees, &sample, 1);

        assert_eq!(phi.len(), 1);
        // Should be the average of two identical trees.
        let single_phi = tree_shap(&tree1, &sample, &counts1);
        assert!(
            (phi[0] - single_phi[0]).abs() < 1e-10,
            "Ensemble of identical trees should match single: {} vs {}",
            phi[0],
            single_phi[0]
        );
    }

    #[test]
    fn test_tree_shap_additivity() {
        // SHAP additivity: sum of phi values = prediction - E[f(x)]
        let (tree, counts) = simple_tree();

        // E[f(x)] = (50*1 + 50*3) / 100 = 2.0
        let expected = 2.0;

        // Left sample
        let sample_left = vec![0.2];
        let phi_left = tree_shap(&tree, &sample_left, &counts);
        let pred_left = tree.predict_sample(&sample_left);
        let phi_sum_left: f64 = phi_left.iter().sum();
        assert!(
            (phi_sum_left - (pred_left - expected)).abs() < 1e-10,
            "SHAP additivity: sum={}, pred-E[f]={}",
            phi_sum_left,
            pred_left - expected,
        );

        // Right sample
        let sample_right = vec![0.8];
        let phi_right = tree_shap(&tree, &sample_right, &counts);
        let pred_right = tree.predict_sample(&sample_right);
        let phi_sum_right: f64 = phi_right.iter().sum();
        assert!(
            (phi_sum_right - (pred_right - expected)).abs() < 1e-10,
            "SHAP additivity: sum={}, pred-E[f]={}",
            phi_sum_right,
            pred_right - expected,
        );
    }
}
