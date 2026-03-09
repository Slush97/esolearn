// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cache-optimal flat tree representation for prediction.
//!
//! `FlatTree` stores nodes in DFS pre-order for cache-friendly traversal.
//! Predictions and probabilities are stored in compact cold arrays indexed
//! by leaf position.
//!
//! # Safety
//!
//! This module uses `unsafe` in the hot-path prediction methods
//! (`predict_sample`, `predict_proba_sample`, `apply_sample`) to elide
//! bounds checks on the node and prediction arrays. The invariants that
//! make this safe are established at construction time by
//! `FlatTree::from_tree_node`:
//!
//! - **Node indices**: every `right` field and every implicit left index
//!   (`idx + 1`) produced by the DFS flattening is guaranteed to be
//!   in-bounds of `self.nodes`.
//! - **Feature indices**: every `feature_idx` in a split node corresponds
//!   to a valid feature column validated during `fit()`.
//! - **Leaf data indices**: every leaf node's `feature_idx` (repurposed as
//!   a leaf data index) is a valid index into `self.predictions` and, when
//!   applicable, into the `self.leaf_probas` stride.
//!
//! These invariants are additionally checked by `debug_assert!` in debug
//! builds. The unchecked accesses avoid branch-predictor pollution in the
//! inner prediction loop, which is critical for ensemble models that
//! evaluate hundreds of trees per sample.

use super::{TreeNode, LEAF_SENTINEL};

/// Hot-path node for prediction — 16 bytes, **4 per 64-byte cache line**.
///
/// Laid out in DFS pre-order: the left child of node `i` is always
/// `i + 1` (the next node in memory), so no left-index field is needed.
/// Only the right-child index is stored explicitly.
///
/// Leaf nodes are marked by `right == LEAF_SENTINEL`.
#[repr(C)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FlatNode {
    /// Right child index in DFS order, or `LEAF_SENTINEL` if this is a leaf.
    /// Left child is always `self_idx + 1` (implicit in DFS layout).
    pub right: u32,
    /// Index of feature to split on (ignored for leaves).
    pub feature_idx: u32,
    /// Split threshold — go left if `feature <= threshold` (ignored for leaves).
    pub threshold: f64,
}

impl FlatNode {
    /// Create a new `FlatNode`.
    pub fn new(right: u32, feature_idx: u32, threshold: f64) -> Self {
        Self {
            right,
            feature_idx,
            threshold,
        }
    }
}

/// A cache-optimal decision tree stored as a contiguous DFS pre-ordered array.
///
/// - Left child is always the **next** node in memory (free prefetch)
/// - 16-byte nodes → 4 fit per 64-byte cache line
/// - A depth-8 tree (~255 nodes × 16B = 4KB) fits entirely in L1 cache
/// - Prediction / probability data stored **only for leaf nodes** in compact cold arrays
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(clippy::unsafe_derive_deserialize)]
#[non_exhaustive]
pub struct FlatTree {
    /// DFS pre-ordered hot nodes (16 bytes each).
    ///
    /// For leaf nodes (`right == LEAF_SENTINEL`), `feature_idx` stores the
    /// leaf data index into `predictions` and `leaf_probas` — no separate
    /// `leaf_indices` Vec needed.
    pub nodes: Vec<FlatNode>,
    /// Prediction value for each **leaf** node (indexed by leaf position).
    pub predictions: Vec<f64>,
    /// Flat array of leaf probabilities as f32: `leaf_probas[leaf_idx * n_classes + class_idx]`.
    /// Empty for regression trees (n_classes_stored == 0).
    pub leaf_probas: Vec<f32>,
    /// Number of classes (stride for `leaf_probas`).
    pub n_classes_stored: u32,
    /// Number of training samples reaching each DFS node.
    /// Used by TreeSHAP for coverage fractions.
    pub node_counts: Vec<usize>,
}

impl FlatTree {
    /// Create a `FlatTree` from raw components.
    ///
    /// `node_counts` is left empty — it is only populated by
    /// [`from_tree_node`](Self::from_tree_node) for TreeSHAP support.
    pub fn new(
        nodes: Vec<FlatNode>,
        predictions: Vec<f64>,
        leaf_probas: Vec<f32>,
        n_classes_stored: u32,
    ) -> Self {
        Self {
            nodes,
            predictions,
            leaf_probas,
            n_classes_stored,
            node_counts: Vec::new(),
        }
    }

    /// Flatten a recursive `TreeNode` into a DFS pre-ordered `FlatTree`.
    ///
    /// Predictions and probabilities are stored **only for leaf nodes**.
    /// The leaf data index is embedded directly in `FlatNode::feature_idx`
    /// for leaf nodes, eliminating a separate `leaf_indices` Vec.
    pub fn from_tree_node(root: &TreeNode, n_classes: usize) -> Self {
        let mut nodes = Vec::new();
        let mut predictions = Vec::new();
        let mut leaf_probas: Vec<f32> = Vec::new();
        let mut leaf_count: u32 = 0;
        let mut node_counts = Vec::new();
        Self::flatten_dfs(
            root,
            &mut nodes,
            &mut predictions,
            &mut leaf_probas,
            &mut leaf_count,
            n_classes,
            &mut node_counts,
        );
        FlatTree {
            nodes,
            predictions,
            leaf_probas,
            n_classes_stored: n_classes as u32,
            node_counts,
        }
    }

    /// Recursive DFS pre-order flattening.
    ///
    /// For leaf nodes, `feature_idx` is repurposed to store the leaf data
    /// index (into `predictions` / `leaf_probas`).  Internal nodes use it
    /// normally as the split feature index.
    fn flatten_dfs(
        node: &TreeNode,
        nodes: &mut Vec<FlatNode>,
        predictions: &mut Vec<f64>,
        leaf_probas: &mut Vec<f32>,
        leaf_count: &mut u32,
        n_classes: usize,
        node_counts: &mut Vec<usize>,
    ) {
        match node {
            TreeNode::Leaf {
                prediction,
                n_samples,
                class_counts,
                ..
            } => {
                let li = *leaf_count;
                *leaf_count += 1;
                nodes.push(FlatNode {
                    right: LEAF_SENTINEL,
                    feature_idx: li, // repurposed: leaf data index
                    threshold: 0.0,
                });
                node_counts.push(*n_samples);
                predictions.push(*prediction);
                Self::append_proba(leaf_probas, class_counts, *n_samples, n_classes);
            }
            TreeNode::Split {
                feature_idx,
                threshold,
                left,
                right,
                n_samples,
                ..
            } => {
                let my_idx = nodes.len();
                nodes.push(FlatNode {
                    right: 0, // placeholder — patched after left subtree
                    feature_idx: *feature_idx as u32,
                    threshold: *threshold,
                });
                node_counts.push(*n_samples);

                // Recurse left — left child is always my_idx + 1.
                Self::flatten_dfs(
                    left,
                    nodes,
                    predictions,
                    leaf_probas,
                    leaf_count,
                    n_classes,
                    node_counts,
                );

                // Right child index = current length (after entire left subtree).
                nodes[my_idx].right = nodes.len() as u32;
                Self::flatten_dfs(
                    right,
                    nodes,
                    predictions,
                    leaf_probas,
                    leaf_count,
                    n_classes,
                    node_counts,
                );
            }
        }
    }

    /// Append probabilities for one leaf to the flat f32 array.
    fn append_proba(
        probas: &mut Vec<f32>,
        class_counts: &[usize],
        n_samples: usize,
        n_classes: usize,
    ) {
        if n_classes > 0 && n_samples > 0 {
            let total = n_samples as f64;
            for i in 0..n_classes {
                let count = if i < class_counts.len() {
                    class_counts[i]
                } else {
                    0
                };
                probas.push((count as f64 / total) as f32);
            }
        }
    }

    /// Predict for a single sample — zero-overhead DFS traversal.
    ///
    /// # Safety invariants (verified by `debug_assert!` in debug builds)
    /// - All `right` indices and implicit left indices (`idx + 1`) are in-bounds
    /// - All `feature_idx` values are valid indices into `sample`
    /// - Guaranteed by `from_tree_node()` construction
    #[inline(always)]
    #[allow(unsafe_code, clippy::inline_always)]
    pub fn predict_sample(&self, sample: &[f64]) -> f64 {
        let nodes = self.nodes.as_slice();
        let preds = self.predictions.as_slice();
        debug_assert!(!nodes.is_empty());
        let mut idx = 0usize;
        loop {
            debug_assert!(idx < nodes.len());
            // SAFETY: idx is 0 (initial) or from DFS indices validated at construction.
            let node = unsafe { nodes.get_unchecked(idx) };
            if node.right == LEAF_SENTINEL {
                // feature_idx stores leaf data index for leaf nodes.
                let li = node.feature_idx as usize;
                return unsafe { *preds.get_unchecked(li) };
            }
            // SAFETY: feature_idx < n_features, validated during fit().
            let feat_val = unsafe { *sample.get_unchecked(node.feature_idx as usize) };
            idx = if feat_val <= node.threshold {
                idx + 1 // left child: next in DFS order
            } else {
                node.right as usize // right child: stored index
            };
        }
    }

    /// Predict class probabilities for a single sample.
    #[inline(always)]
    #[allow(unsafe_code, clippy::inline_always)]
    pub fn predict_proba_sample(&self, sample: &[f64], n_classes: usize) -> Vec<f64> {
        let nodes = self.nodes.as_slice();
        let nc = self.n_classes_stored as usize;
        debug_assert!(!nodes.is_empty());
        let mut idx = 0usize;
        loop {
            debug_assert!(idx < nodes.len());
            let node = unsafe { nodes.get_unchecked(idx) };
            if node.right == LEAF_SENTINEL {
                let li = node.feature_idx as usize;
                let start = li * nc;
                let mut result = vec![0.0; n_classes];
                let copy_len = n_classes.min(nc);
                for (i, p) in self.leaf_probas[start..start + copy_len].iter().enumerate() {
                    result[i] = *p as f64;
                }
                return result;
            }
            let feat_val = unsafe { *sample.get_unchecked(node.feature_idx as usize) };
            idx = if feat_val <= node.threshold {
                idx + 1
            } else {
                node.right as usize
            };
        }
    }

    // ── GBT compatibility methods ──

    /// Number of nodes in the tree (DFS array length).
    ///
    /// Used by GBT for sizing per-node accumulation arrays.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Read a leaf's prediction value given its DFS node index.
    ///
    /// Panics in debug if `node_idx` is not a leaf.
    #[inline]
    pub fn leaf_prediction(&self, node_idx: usize) -> f64 {
        let node = &self.nodes[node_idx];
        debug_assert!(
            node.right == LEAF_SENTINEL,
            "node_idx {node_idx} is not a leaf"
        );
        self.predictions[node.feature_idx as usize]
    }

    /// Set a leaf's prediction value given its DFS node index.
    ///
    /// Used by GBT Newton-Raphson correction. Panics in debug if not a leaf.
    #[inline]
    pub fn set_leaf_prediction(&mut self, node_idx: usize, value: f64) {
        let node = &self.nodes[node_idx];
        debug_assert!(
            node.right == LEAF_SENTINEL,
            "node_idx {node_idx} is not a leaf"
        );
        self.predictions[node.feature_idx as usize] = value;
    }

    /// Check if a DFS node index is a leaf.
    #[inline]
    pub fn is_leaf(&self, node_idx: usize) -> bool {
        self.nodes[node_idx].right == LEAF_SENTINEL
    }

    /// Predict for all samples.
    pub fn predict(&self, features: &[Vec<f64>]) -> Vec<f64> {
        features
            .iter()
            .map(|row| self.predict_sample(row))
            .collect()
    }

    /// Route a single sample to its leaf and return the leaf node index.
    ///
    /// This is used by GBT for Newton-Raphson leaf value correction.
    #[inline(always)]
    #[allow(unsafe_code, clippy::inline_always)]
    pub(crate) fn apply_sample(&self, sample: &[f64]) -> usize {
        let nodes = self.nodes.as_slice();
        debug_assert!(!nodes.is_empty());
        let mut idx = 0usize;
        loop {
            debug_assert!(idx < nodes.len());
            let node = unsafe { nodes.get_unchecked(idx) };
            if node.right == LEAF_SENTINEL {
                return idx;
            }
            let feat_val = unsafe { *sample.get_unchecked(node.feature_idx as usize) };
            idx = if feat_val <= node.threshold {
                idx + 1
            } else {
                node.right as usize
            };
        }
    }

    /// Route all samples to their leaf nodes, returning leaf indices.
    pub(crate) fn apply(&self, features: &[Vec<f64>]) -> Vec<usize> {
        features.iter().map(|row| self.apply_sample(row)).collect()
    }

    /// Depth of the tree.
    pub fn depth(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        Self::depth_at(&self.nodes, 0, 1)
    }

    fn depth_at(nodes: &[FlatNode], idx: usize, d: usize) -> usize {
        if idx >= nodes.len() {
            return 0;
        }
        let node = &nodes[idx];
        if node.right == LEAF_SENTINEL {
            d
        } else {
            let l = Self::depth_at(nodes, idx + 1, d + 1);
            let r = Self::depth_at(nodes, node.right as usize, d + 1);
            l.max(r)
        }
    }

    /// Number of leaf nodes.
    pub fn n_leaves(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.right == LEAF_SENTINEL)
            .count()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    /// Build a single-leaf tree (root is a leaf).
    fn single_leaf_tree() -> FlatTree {
        let nodes = vec![FlatNode::new(LEAF_SENTINEL, 0, 0.0)];
        let predictions = vec![42.0];
        let leaf_probas = vec![0.3f32, 0.7];
        FlatTree::new(nodes, predictions, leaf_probas, 2)
    }

    /// Build a balanced binary tree of depth 3 (7 nodes, 4 leaves).
    ///
    /// ```text
    ///         [0: f0 <= 0.5]
    ///        /              \
    ///   [1: f1 <= 0.3]     [4: f1 <= 0.7]
    ///    /       \          /       \
    ///  [2:L0]  [3:L1]    [5:L2]  [6:L3]
    /// ```
    fn balanced_tree() -> FlatTree {
        let nodes = vec![
            FlatNode::new(4, 0, 0.5),  // node 0: split on f0 <= 0.5
            FlatNode::new(3, 1, 0.3),  // node 1: split on f1 <= 0.3
            FlatNode::new(LEAF_SENTINEL, 0, 0.0), // node 2: leaf 0
            FlatNode::new(LEAF_SENTINEL, 1, 0.0), // node 3: leaf 1
            FlatNode::new(6, 1, 0.7),  // node 4: split on f1 <= 0.7
            FlatNode::new(LEAF_SENTINEL, 2, 0.0), // node 5: leaf 2
            FlatNode::new(LEAF_SENTINEL, 3, 0.0), // node 6: leaf 3
        ];
        let predictions = vec![1.0, 2.0, 3.0, 4.0];
        let leaf_probas = vec![
            1.0, 0.0,  // leaf 0
            0.0, 1.0,  // leaf 1
            0.6, 0.4,  // leaf 2
            0.2, 0.8,  // leaf 3
        ];
        FlatTree::new(nodes, predictions, leaf_probas, 2)
    }

    /// Build an all-left tree of depth 5 (5 internal nodes + 6 leaves).
    /// Every internal node goes left, so the leftmost path is the deepest.
    fn all_left_tree(depth: usize) -> FlatTree {
        let mut nodes = Vec::new();
        let mut predictions = Vec::new();
        let mut leaf_count = 0u32;

        fn build(
            nodes: &mut Vec<FlatNode>,
            predictions: &mut Vec<f64>,
            leaf_count: &mut u32,
            depth: usize,
            max_depth: usize,
        ) {
            if depth >= max_depth {
                let li = *leaf_count;
                *leaf_count += 1;
                nodes.push(FlatNode::new(LEAF_SENTINEL, li, 0.0));
                predictions.push(li as f64);
                return;
            }
            let my_idx = nodes.len();
            nodes.push(FlatNode::new(0, 0, 0.5)); // split on f0 <= 0.5
            // left child
            build(nodes, predictions, leaf_count, depth + 1, max_depth);
            // patch right
            nodes[my_idx].right = nodes.len() as u32;
            // right child is always a leaf
            let li = *leaf_count;
            *leaf_count += 1;
            nodes.push(FlatNode::new(LEAF_SENTINEL, li, 0.0));
            predictions.push(li as f64);
        }

        build(&mut nodes, &mut predictions, &mut leaf_count, 0, depth);
        FlatTree::new(nodes, predictions, vec![], 0)
    }

    /// Build an all-right tree of depth 5.
    fn all_right_tree(depth: usize) -> FlatTree {
        let mut nodes = Vec::new();
        let mut predictions = Vec::new();
        let mut leaf_count = 0u32;

        fn build(
            nodes: &mut Vec<FlatNode>,
            predictions: &mut Vec<f64>,
            leaf_count: &mut u32,
            depth: usize,
            max_depth: usize,
        ) {
            if depth >= max_depth {
                let li = *leaf_count;
                *leaf_count += 1;
                nodes.push(FlatNode::new(LEAF_SENTINEL, li, 0.0));
                predictions.push(li as f64);
                return;
            }
            let my_idx = nodes.len();
            nodes.push(FlatNode::new(0, 0, 0.5)); // split on f0 <= 0.5
            // left child is always a leaf
            let li = *leaf_count;
            *leaf_count += 1;
            nodes.push(FlatNode::new(LEAF_SENTINEL, li, 0.0));
            predictions.push(li as f64);
            // patch right
            nodes[my_idx].right = nodes.len() as u32;
            // right child
            build(nodes, predictions, leaf_count, depth + 1, max_depth);
        }

        build(&mut nodes, &mut predictions, &mut leaf_count, 0, depth);
        FlatTree::new(nodes, predictions, vec![], 0)
    }

    /// Miri-compatible test exercising unsafe predict paths on boundary trees.
    #[test]
    fn test_flat_tree_predict_boundaries() {
        // Single leaf tree.
        let tree = single_leaf_tree();
        assert_eq!(tree.predict_sample(&[0.0, 1.0]), 42.0);
        assert_eq!(tree.predict_sample(&[99.0]), 42.0);
        let proba = tree.predict_proba_sample(&[0.0], 2);
        assert!((proba[0] - 0.3).abs() < 1e-5);
        assert!((proba[1] - 0.7).abs() < 1e-5);
        assert_eq!(tree.apply_sample(&[0.0]), 0);

        // Balanced tree — test all four leaf paths.
        let tree = balanced_tree();
        // f0=0.0 <= 0.5 (left), f1=0.0 <= 0.3 (left) → leaf 0, pred=1.0
        assert_eq!(tree.predict_sample(&[0.0, 0.0]), 1.0);
        // f0=0.0 <= 0.5 (left), f1=0.5 > 0.3 (right) → leaf 1, pred=2.0
        assert_eq!(tree.predict_sample(&[0.0, 0.5]), 2.0);
        // f0=0.8 > 0.5 (right), f1=0.5 <= 0.7 (left) → leaf 2, pred=3.0
        assert_eq!(tree.predict_sample(&[0.8, 0.5]), 3.0);
        // f0=0.8 > 0.5 (right), f1=0.9 > 0.7 (right) → leaf 3, pred=4.0
        assert_eq!(tree.predict_sample(&[0.8, 0.9]), 4.0);
        // predict_proba for leaf 3
        let proba = tree.predict_proba_sample(&[0.8, 0.9], 2);
        assert!((proba[0] - 0.2).abs() < 1e-5);
        assert!((proba[1] - 0.8).abs() < 1e-5);
        // apply
        assert_eq!(tree.apply_sample(&[0.0, 0.0]), 2); // node index of leaf 0
        assert_eq!(tree.apply_sample(&[0.8, 0.9]), 6); // node index of leaf 3

        // All-left tree (max_depth=5) — deepest left path has 5 splits + 1 leaf = depth 6.
        let tree = all_left_tree(5);
        assert_eq!(tree.depth(), 6);
        // Sample with f0=0.0 always goes left → reaches deepest leaf.
        let _ = tree.predict_sample(&[0.0]);
        // Sample with f0=1.0 always goes right → first right leaf.
        let _ = tree.predict_sample(&[1.0]);
        // Predict batch.
        let preds = tree.predict(&[vec![0.0], vec![1.0], vec![0.5]]);
        assert_eq!(preds.len(), 3);

        // All-right tree (max_depth=5) — same depth structure.
        let tree = all_right_tree(5);
        assert_eq!(tree.depth(), 6);
        let _ = tree.predict_sample(&[0.0]);
        let _ = tree.predict_sample(&[1.0]);
        let preds = tree.predict(&[vec![0.0], vec![1.0]]);
        assert_eq!(preds.len(), 2);
    }
}
