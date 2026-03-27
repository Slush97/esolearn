// SPDX-License-Identifier: MIT OR Apache-2.0
//! Histogram-based Gradient Boosted Trees — O(n) split finding.
//!
//! This module implements the core innovation behind XGBoost, LightGBM, and
//! CatBoost: instead of sorting features at each split, the data is pre-binned
//! into 256 `u8` bins and gradients are accumulated into fixed-size histograms.
//! Split finding becomes O(256) per feature per leaf, regardless of dataset size.
//!
//! ## Key Optimizations
//!
//! - **Histogram subtraction trick**: parent − left = right (halves histogram
//!   construction cost).
//! - **Leaf-wise (best-first) growth**: grows the leaf with the highest gain,
//!   matching LightGBM's strategy for deeper, more accurate trees.
//! - **SIMD-friendly layout**: histograms are contiguous `[HistBin; 256]` arrays,
//!   enabling auto-vectorization.
//! - **Rayon parallelism** for histogram construction across features.
//!
//! # Example
//! ```
//! use scry_learn::dataset::Dataset;
//! use scry_learn::tree::HistGradientBoostingRegressor;
//!
//! let features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
//! let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
//! let data = Dataset::new(features, target, vec!["x".into()], "y");
//!
//! let mut model = HistGradientBoostingRegressor::new()
//!     .n_estimators(100)
//!     .learning_rate(0.1)
//!     .max_leaf_nodes(31);
//! model.fit(&data).unwrap();
//!
//! let preds = model.predict(&[vec![3.0]]).unwrap();
//! assert!((preds[0] - 6.0).abs() < 1.0);
//! ```

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::tree::binning::FeatureBinner;

use rayon::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Histogram data structures
// ═══════════════════════════════════════════════════════════════════════════

/// Number of histogram bins.
const NUM_BINS: usize = 256;

/// A single histogram bin accumulating gradient statistics.
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct HistBin {
    grad_sum: f64,
    hess_sum: f64,
    count: u32,
}

/// Histogram for one feature: 256 bins of gradient/hessian sums.
///
/// Contiguous layout for SIMD-friendly access during split search.
type FeatureHistogram = [HistBin; NUM_BINS];

/// Build histograms for all features from the binned data.
///
/// When the `gpu` feature is enabled and a GPU backend is available, this
/// delegates to [`ComputeBackend::build_histograms`] for acceleration.
/// Otherwise it falls back to the Rayon-parallel CPU path.
fn build_histograms(
    binned: &[Vec<u8>], // [feature][sample]
    gradients: &[f64],
    hessians: &[f64],
    sample_indices: &[usize],
    n_features: usize,
) -> Vec<FeatureHistogram> {
    // Try GPU-accelerated path when the feature is enabled.
    #[cfg(feature = "scry-gpu")]
    {
        if let Ok(gpu) = crate::accel::ScryGpuBackend::new() {
            use crate::accel::ComputeBackend;
            let accel_hists = gpu.build_histograms(
                binned,
                gradients,
                hessians,
                sample_indices,
                n_features,
                NUM_BINS,
            );
            return accel_hists
                .into_iter()
                .map(|feat_bins| {
                    let mut hist: FeatureHistogram = [HistBin::default(); NUM_BINS];
                    for (b, &(g, h, c)) in feat_bins.iter().enumerate() {
                        if b < NUM_BINS {
                            hist[b].grad_sum = g;
                            hist[b].hess_sum = h;
                            hist[b].count = c as u32;
                        }
                    }
                    hist
                })
                .collect();
        }
    }

    // CPU fallback: parallel across features via Rayon.
    (0..n_features)
        .into_par_iter()
        .map(|f| {
            let col = &binned[f];
            let mut hist: FeatureHistogram = [HistBin::default(); NUM_BINS];
            for &idx in sample_indices {
                let bin = col[idx] as usize;
                hist[bin].grad_sum += gradients[idx];
                hist[bin].hess_sum += hessians[idx];
                hist[bin].count += 1;
            }
            hist
        })
        .collect()
}

/// Histogram subtraction: parent − left = right.
fn subtract_histograms(
    parent: &[FeatureHistogram],
    left: &[FeatureHistogram],
) -> Vec<FeatureHistogram> {
    parent
        .iter()
        .zip(left.iter())
        .map(|(p, l)| {
            let mut right = [HistBin::default(); NUM_BINS];
            for b in 0..NUM_BINS {
                right[b].grad_sum = p[b].grad_sum - l[b].grad_sum;
                right[b].hess_sum = p[b].hess_sum - l[b].hess_sum;
                right[b].count = p[b].count.saturating_sub(l[b].count);
            }
            right
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal tree representation
// ═══════════════════════════════════════════════════════════════════════════

/// A node in the histogram-based tree.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum HistNode {
    /// Leaf node with a prediction value.
    Leaf { value: f64 },
    /// Internal split node.
    Split {
        feature: usize,
        bin_threshold: u8,
        left: usize, // index into HistTree::nodes
        right: usize,
        gain: f64,
    },
}

/// Public view of a HistNode for ONNX export, with bin thresholds converted
/// to raw feature value thresholds.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum HistNodeView {
    /// Leaf node.
    Leaf {
        /// Prediction value.
        value: f64,
    },
    /// Internal split node.
    Split {
        /// Feature index.
        feature: usize,
        /// Raw feature threshold (≤ goes left).
        threshold: f64,
        /// Left child index.
        left: usize,
        /// Right child index.
        right: usize,
    },
}

/// A tree built from histogram-based splits.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct HistTree {
    nodes: Vec<HistNode>,
}

impl HistTree {
    /// Predict for a single sample.
    fn predict_one(&self, sample_binned: &[u8]) -> f64 {
        let mut node_idx = 0;
        loop {
            match &self.nodes[node_idx] {
                HistNode::Leaf { value } => return *value,
                HistNode::Split {
                    feature,
                    bin_threshold,
                    left,
                    right,
                    ..
                } => {
                    if sample_binned[*feature] <= *bin_threshold {
                        node_idx = *left;
                    } else {
                        node_idx = *right;
                    }
                }
            }
        }
    }

    /// Predict for a single raw (unbinned) sample using bin edges.
    fn predict_one_raw(&self, sample: &[f64], binner: &FeatureBinner) -> f64 {
        let mut node_idx = 0;
        loop {
            match &self.nodes[node_idx] {
                HistNode::Leaf { value } => return *value,
                HistNode::Split {
                    feature,
                    bin_threshold,
                    left,
                    right,
                    ..
                } => {
                    let val = sample[*feature];
                    let bin = if val.is_nan() {
                        0u8
                    } else {
                        let edges = &binner.bin_edges()[*feature];
                        let pos = match edges.binary_search_by(|edge| {
                            edge.partial_cmp(&val).unwrap_or(std::cmp::Ordering::Equal)
                        }) {
                            Ok(p) => p + 1,
                            Err(p) => p,
                        };
                        (pos + 1).min(255) as u8
                    };
                    if bin <= *bin_threshold {
                        node_idx = *left;
                    } else {
                        node_idx = *right;
                    }
                }
            }
        }
    }

    /// Collect feature importance (total gain) from this tree.
    fn feature_importances(&self, n_features: usize) -> Vec<f64> {
        let mut imp = vec![0.0; n_features];
        for node in &self.nodes {
            if let HistNode::Split { feature, gain, .. } = node {
                if *feature < n_features {
                    imp[*feature] += gain;
                }
            }
        }
        imp
    }

    /// Convert to public HistNodeView representation, translating bin
    /// thresholds to raw feature value thresholds using the binner.
    fn to_node_views(&self, binner: &FeatureBinner) -> Vec<HistNodeView> {
        let edges = binner.bin_edges();
        self.nodes
            .iter()
            .map(|node| match node {
                HistNode::Leaf { value } => HistNodeView::Leaf { value: *value },
                HistNode::Split {
                    feature,
                    bin_threshold,
                    left,
                    right,
                    ..
                } => {
                    // Convert bin threshold to raw value threshold.
                    // bin k corresponds to values in [edges[k-2], edges[k-1]).
                    // bin <= bin_threshold means val < edges[bin_threshold - 1].
                    // For ONNX BRANCH_LEQ (val <= T), use edges[bin_threshold - 1].
                    let threshold = if *bin_threshold == 0 || *feature >= edges.len() {
                        f64::NEG_INFINITY
                    } else {
                        let feat_edges = &edges[*feature];
                        let idx = (*bin_threshold as usize).saturating_sub(1);
                        if idx < feat_edges.len() {
                            feat_edges[idx]
                        } else if !feat_edges.is_empty() {
                            feat_edges[feat_edges.len() - 1]
                        } else {
                            0.0
                        }
                    };
                    HistNodeView::Split {
                        feature: *feature,
                        threshold,
                        left: *left,
                        right: *right,
                    }
                }
            })
            .collect()
    }
}

/// Candidate leaf for best-first (leaf-wise) growth.
struct LeafCandidate {
    /// Index into the tree's nodes Vec (this is a Leaf node).
    node_idx: usize,
    /// Sample indices falling into this leaf.
    sample_indices: Vec<usize>,
    /// Pre-computed histograms for this leaf.
    histograms: Vec<FeatureHistogram>,
    /// Total gradient sum in this leaf.
    grad_sum: f64,
    /// Total hessian sum in this leaf.
    hess_sum: f64,
    /// Depth of this leaf.
    depth: usize,
}

/// Result of scanning one leaf for the best split.
struct SplitResult {
    feature: usize,
    bin_threshold: u8,
    gain: f64,
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
    left_value: f64,
    right_value: f64,
    left_grad_sum: f64,
    left_hess_sum: f64,
    right_grad_sum: f64,
    right_hess_sum: f64,
}

/// L2-regularized leaf value: −G / (H + λ).
///
/// Guards against near-zero denominator which would produce extreme
/// leaf values that destabilize the boosting ensemble.
#[inline]
fn leaf_value(grad_sum: f64, hess_sum: f64, l2_reg: f64) -> f64 {
    let denom = hess_sum + l2_reg;
    if denom.abs() < 1e-10 {
        0.0
    } else {
        -grad_sum / denom
    }
}

/// Split gain: G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ).
#[inline]
fn split_gain(
    grad_left: f64,
    hess_left: f64,
    grad_right: f64,
    hess_right: f64,
    l2_reg: f64,
) -> f64 {
    let left_term = grad_left * grad_left / (hess_left + l2_reg);
    let right_term = grad_right * grad_right / (hess_right + l2_reg);
    let parent_grad = grad_left + grad_right;
    let parent_hess = hess_left + hess_right;
    let parent_term = parent_grad * parent_grad / (parent_hess + l2_reg);
    0.5 * (left_term + right_term - parent_term)
}

/// Find the best split across all features for a given leaf.
#[allow(clippy::too_many_arguments)]
fn find_best_split(
    histograms: &[FeatureHistogram],
    binned: &[Vec<u8>],
    sample_indices: &[usize],
    grad_sum: f64,
    hess_sum: f64,
    min_samples_leaf: usize,
    l2_reg: f64,
    n_features: usize,
) -> Option<SplitResult> {
    let mut best_gain = 0.0; // only accept positive gains
    let mut best_feature = 0;
    let mut best_threshold: u8 = 0;
    let mut best_left_grad = 0.0;
    let mut best_left_hess = 0.0;

    for (f, hist) in histograms.iter().enumerate().take(n_features) {
        let mut running_grad = 0.0;
        let mut running_hess = 0.0;
        let mut running_count: u32 = 0;
        let total_count = sample_indices.len() as u32;

        // Scan bins left-to-right (including bin 0 for NaN).
        for bin in 0..255u8 {
            let b = bin as usize;
            running_grad += hist[b].grad_sum;
            running_hess += hist[b].hess_sum;
            running_count += hist[b].count;

            let right_count = total_count.saturating_sub(running_count);
            if (running_count as usize) < min_samples_leaf
                || (right_count as usize) < min_samples_leaf
            {
                continue;
            }

            let right_grad = grad_sum - running_grad;
            let right_hess = hess_sum - running_hess;

            let gain = split_gain(running_grad, running_hess, right_grad, right_hess, l2_reg);

            if gain > best_gain {
                best_gain = gain;
                best_feature = f;
                best_threshold = bin;
                best_left_grad = running_grad;
                best_left_hess = running_hess;
            }
        }
    }

    if best_gain <= 0.0 {
        return None;
    }

    // Split sample indices.
    let col = &binned[best_feature];
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();
    for &idx in sample_indices {
        if col[idx] <= best_threshold {
            left_indices.push(idx);
        } else {
            right_indices.push(idx);
        }
    }

    let best_right_grad = grad_sum - best_left_grad;
    let best_right_hess = hess_sum - best_left_hess;

    // Build left histogram to enable histogram subtraction trick.
    // (The caller will rebuild with the actual gradients.)

    Some(SplitResult {
        feature: best_feature,
        bin_threshold: best_threshold,
        gain: best_gain,
        left_indices,
        right_indices,
        left_value: leaf_value(best_left_grad, best_left_hess, l2_reg),
        right_value: leaf_value(best_right_grad, best_right_hess, l2_reg),
        left_grad_sum: best_left_grad,
        left_hess_sum: best_left_hess,
        right_grad_sum: best_right_grad,
        right_hess_sum: best_right_hess,
    })
}

/// Build a single tree using leaf-wise (best-first) growth.
#[allow(clippy::too_many_arguments)]
fn build_tree_leaf_wise(
    binned: &[Vec<u8>],
    gradients: &[f64],
    hessians: &[f64],
    sample_indices: &[usize],
    max_leaf_nodes: usize,
    min_samples_leaf: usize,
    max_depth: usize,
    l2_reg: f64,
    n_features: usize,
) -> HistTree {
    let mut nodes: Vec<HistNode> = Vec::new();

    // Compute initial sums.
    let total_grad: f64 = sample_indices.iter().map(|&i| gradients[i]).sum();
    let total_hess: f64 = sample_indices.iter().map(|&i| hessians[i]).sum();

    let root_value = leaf_value(total_grad, total_hess, l2_reg);
    nodes.push(HistNode::Leaf { value: root_value });

    // Build root histograms.
    let root_histograms = build_histograms(binned, gradients, hessians, sample_indices, n_features);

    // Priority queue of splittable leaves (sorted by best gain).
    let mut candidates: Vec<LeafCandidate> = Vec::new();
    candidates.push(LeafCandidate {
        node_idx: 0,
        sample_indices: sample_indices.to_vec(),
        histograms: root_histograms,
        grad_sum: total_grad,
        hess_sum: total_hess,
        depth: 0,
    });

    let mut n_leaves = 1usize;

    while n_leaves < max_leaf_nodes && !candidates.is_empty() {
        // Find the candidate with the best split gain.
        let mut best_cand_idx = 0;
        let mut best_gain = f64::NEG_INFINITY;

        for (c_idx, cand) in candidates.iter().enumerate() {
            if cand.depth >= max_depth {
                continue;
            }
            if cand.sample_indices.len() < 2 * min_samples_leaf {
                continue;
            }
            // Find best split for this candidate.
            if let Some(split) = find_best_split(
                &cand.histograms,
                binned,
                &cand.sample_indices,
                cand.grad_sum,
                cand.hess_sum,
                min_samples_leaf,
                l2_reg,
                n_features,
            ) {
                if split.gain > best_gain {
                    best_gain = split.gain;
                    best_cand_idx = c_idx;
                }
            }
        }

        if best_gain <= 0.0 {
            break;
        }

        let cand = candidates.remove(best_cand_idx);

        // Re-compute best split for the chosen candidate (we need the full result).
        let split = find_best_split(
            &cand.histograms,
            binned,
            &cand.sample_indices,
            cand.grad_sum,
            cand.hess_sum,
            min_samples_leaf,
            l2_reg,
            n_features,
        );

        let Some(split) = split else {
            continue;
        };

        // Create child leaf nodes.
        let left_idx = nodes.len();
        nodes.push(HistNode::Leaf {
            value: split.left_value,
        });
        let right_idx = nodes.len();
        nodes.push(HistNode::Leaf {
            value: split.right_value,
        });

        // Convert parent leaf → split node.
        nodes[cand.node_idx] = HistNode::Split {
            feature: split.feature,
            bin_threshold: split.bin_threshold,
            left: left_idx,
            right: right_idx,
            gain: split.gain,
        };

        n_leaves += 1; // one leaf became two (net +1)

        // Build histograms for children using subtraction trick:
        // smaller child gets full histogram build, larger gets subtraction.
        let (small_indices, _large_indices, small_is_left) =
            if split.left_indices.len() <= split.right_indices.len() {
                (&split.left_indices, &split.right_indices, true)
            } else {
                (&split.right_indices, &split.left_indices, false)
            };

        let small_histograms =
            build_histograms(binned, gradients, hessians, small_indices, n_features);
        let large_histograms = subtract_histograms(&cand.histograms, &small_histograms);

        let (left_hist, right_hist) = if small_is_left {
            (small_histograms, large_histograms)
        } else {
            (large_histograms, small_histograms)
        };

        let new_depth = cand.depth + 1;

        // Add children as new candidates.
        if split.left_indices.len() >= 2 * min_samples_leaf && new_depth < max_depth {
            candidates.push(LeafCandidate {
                node_idx: left_idx,
                sample_indices: split.left_indices,
                histograms: left_hist,
                grad_sum: split.left_grad_sum,
                hess_sum: split.left_hess_sum,
                depth: new_depth,
            });
        }

        if split.right_indices.len() >= 2 * min_samples_leaf && new_depth < max_depth {
            candidates.push(LeafCandidate {
                node_idx: right_idx,
                sample_indices: split.right_indices,
                histograms: right_hist,
                grad_sum: split.right_grad_sum,
                hess_sum: split.right_hess_sum,
                depth: new_depth,
            });
        }
    }

    HistTree { nodes }
}

// ═══════════════════════════════════════════════════════════════════════════
// Histogram Gradient Boosting Regressor
// ═══════════════════════════════════════════════════════════════════════════

/// Histogram-based Gradient Boosting for regression.
///
/// Uses pre-binned features and O(256) histogram scans for split finding,
/// delivering 5-10× speedup over standard GBT on large datasets. This is
/// the same algorithmic approach as LightGBM/XGBoost/CatBoost, implemented
/// in pure Rust with no external BLAS dependency.
///
/// # Example
/// ```
/// use scry_learn::dataset::Dataset;
/// use scry_learn::tree::HistGradientBoostingRegressor;
///
/// let features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
/// let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
/// let data = Dataset::new(features, target, vec!["x".into()], "y");
///
/// let mut model = HistGradientBoostingRegressor::new()
///     .n_estimators(100)
///     .learning_rate(0.1)
///     .max_leaf_nodes(31);
/// model.fit(&data).unwrap();
///
/// let preds = model.predict(&[vec![3.0]]).unwrap();
/// assert!((preds[0] - 6.0).abs() < 1.0);
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct HistGradientBoostingRegressor {
    n_estimators: usize,
    learning_rate: f64,
    max_leaf_nodes: usize,
    min_samples_leaf: usize,
    max_depth: usize,
    max_bins: usize,
    l2_regularization: f64,
    seed: u64,
    // Fitted state
    trees: Vec<HistTree>,
    binner: FeatureBinner,
    init_prediction: f64,
    n_features: usize,
    fitted: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl HistGradientBoostingRegressor {
    /// Create a new regressor with default parameters.
    ///
    /// # Example
    /// ```
    /// use scry_learn::tree::HistGradientBoostingRegressor;
    ///
    /// let model = HistGradientBoostingRegressor::new()
    ///     .n_estimators(200)
    ///     .learning_rate(0.05);
    /// ```
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_leaf_nodes: 31,
            min_samples_leaf: 20,
            max_depth: 8,
            max_bins: NUM_BINS,
            l2_regularization: 0.0,
            seed: 42,
            trees: Vec::new(),
            binner: FeatureBinner::new(),
            init_prediction: 0.0,
            n_features: 0,
            fitted: false,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set number of boosting rounds (default: 100).
    pub fn n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Set learning rate / shrinkage (default: 0.1).
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set maximum number of leaf nodes per tree (default: 31).
    ///
    /// This controls tree complexity. LightGBM default is 31.
    pub fn max_leaf_nodes(mut self, n: usize) -> Self {
        self.max_leaf_nodes = n;
        self
    }

    /// Set minimum samples required in a leaf (default: 20).
    pub fn min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }

    /// Set maximum tree depth (default: 8). Acts as a secondary depth limit.
    pub fn max_depth(mut self, d: usize) -> Self {
        self.max_depth = d;
        self
    }

    /// Set maximum number of bins (2..=256, default: 256).
    pub fn max_bins(mut self, bins: usize) -> Self {
        self.max_bins = bins.clamp(2, NUM_BINS);
        self
    }

    /// Set L2 regularization (default: 0.0).
    pub fn l2_regularization(mut self, l2: f64) -> Self {
        self.l2_regularization = l2;
        self
    }

    /// Set random seed (default: 42).
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Train the histogram-based gradient boosting regressor.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_no_inf()?;
        let n = data.n_samples();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err(ScryLearnError::InvalidParameter(
                "learning_rate must be in (0, 1]".into(),
            ));
        }

        self.n_features = data.n_features();

        // Bin features.
        self.binner = FeatureBinner::new().max_bins(self.max_bins);
        let binned = self.binner.fit_transform(data)?;

        // Initial prediction: mean of targets.
        let mean: f64 = data.target.iter().sum::<f64>() / n as f64;
        self.init_prediction = mean;

        let mut predictions = vec![mean; n];
        let all_indices: Vec<usize> = (0..n).collect();

        self.trees = Vec::with_capacity(self.n_estimators);

        // Adjust min_samples_leaf for small datasets.
        let effective_min_leaf = self.min_samples_leaf.min(n / 4).max(1);

        for _ in 0..self.n_estimators {
            // Compute gradients (negative residuals) and hessians.
            let gradients: Vec<f64> = (0..n).map(|i| -(data.target[i] - predictions[i])).collect();
            let hessians = vec![1.0; n]; // squared error: hessian = 1

            let tree = build_tree_leaf_wise(
                &binned,
                &gradients,
                &hessians,
                &all_indices,
                self.max_leaf_nodes,
                effective_min_leaf,
                self.max_depth,
                self.l2_regularization,
                self.n_features,
            );

            // Update predictions.
            for &i in &all_indices {
                let sample: Vec<u8> = binned.iter().map(|col| col[i]).collect();
                predictions[i] += self.learning_rate * tree.predict_one(&sample);
            }

            self.trees.push(tree);
        }

        self.fitted = true;
        Ok(())
    }

    /// Predict values for new samples.
    ///
    /// `features` is row-major: `features[sample_idx][feature_idx]`.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        crate::version::check_schema_version(self._schema_version)?;
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let n = features.len();
        let mut preds = vec![self.init_prediction; n];

        for tree in &self.trees {
            for (i, sample) in features.iter().enumerate() {
                preds[i] += self.learning_rate * tree.predict_one_raw(sample, &self.binner);
            }
        }

        Ok(preds)
    }

    /// Feature importances (total gain, normalized).
    pub fn feature_importances(&self) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let m = self.n_features;
        let mut imp = vec![0.0; m];
        for tree in &self.trees {
            let ti = tree.feature_importances(m);
            for (i, &v) in ti.iter().enumerate() {
                imp[i] += v;
            }
        }
        let total: f64 = imp.iter().sum();
        if total > 0.0 {
            for v in &mut imp {
                *v /= total;
            }
        }
        Ok(imp)
    }

    /// Number of trees in the ensemble.
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Number of features the model was trained on.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Learning rate value.
    pub fn learning_rate_val(&self) -> f64 {
        self.learning_rate
    }

    /// Initial (base) prediction value.
    pub fn init_prediction_val(&self) -> f64 {
        self.init_prediction
    }

    /// Convert internal HistTree nodes to public HistNodeView arrays for ONNX export.
    /// Bin thresholds are converted to raw feature thresholds using the binner.
    pub fn tree_node_views(&self) -> Vec<Vec<HistNodeView>> {
        self.trees
            .iter()
            .map(|tree| tree.to_node_views(&self.binner))
            .collect()
    }
}

impl Default for HistGradientBoostingRegressor {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Histogram Gradient Boosting Classifier
// ═══════════════════════════════════════════════════════════════════════════

/// Histogram-based Gradient Boosting for classification (binary + multiclass).
///
/// Uses the same O(256) histogram approach as the regressor, with log-loss
/// for binary classification and softmax for multiclass. Leaf-wise tree growth
/// with Newton-Raphson leaf correction.
///
/// # Example
/// ```
/// use scry_learn::dataset::Dataset;
/// use scry_learn::tree::HistGradientBoostingClassifier;
///
/// let features = vec![
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
///     vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
/// ];
/// let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
/// let data = Dataset::new(features, target, vec!["x1".into(), "x2".into()], "class");
///
/// let mut model = HistGradientBoostingClassifier::new()
///     .n_estimators(50)
///     .learning_rate(0.1)
///     .max_leaf_nodes(31);
/// model.fit(&data).unwrap();
///
/// let preds = model.predict(&[vec![1.5, 0.15], vec![5.5, 0.55]]).unwrap();
/// assert_eq!(preds[0], 0.0);
/// assert_eq!(preds[1], 1.0);
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct HistGradientBoostingClassifier {
    n_estimators: usize,
    learning_rate: f64,
    max_leaf_nodes: usize,
    min_samples_leaf: usize,
    max_depth: usize,
    max_bins: usize,
    l2_regularization: f64,
    seed: u64,
    // Fitted state — trees[class_idx][estimator_idx]
    trees: Vec<Vec<HistTree>>,
    binner: FeatureBinner,
    init_predictions: Vec<f64>,
    n_classes: usize,
    n_features: usize,
    fitted: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl HistGradientBoostingClassifier {
    /// Create a new classifier with default parameters.
    ///
    /// # Example
    /// ```
    /// use scry_learn::tree::HistGradientBoostingClassifier;
    ///
    /// let model = HistGradientBoostingClassifier::new()
    ///     .n_estimators(200)
    ///     .learning_rate(0.05);
    /// ```
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_leaf_nodes: 31,
            min_samples_leaf: 20,
            max_depth: 8,
            max_bins: NUM_BINS,
            l2_regularization: 0.0,
            seed: 42,
            trees: Vec::new(),
            binner: FeatureBinner::new(),
            init_predictions: Vec::new(),
            n_classes: 0,
            n_features: 0,
            fitted: false,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set number of boosting rounds (default: 100).
    pub fn n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Set learning rate / shrinkage (default: 0.1).
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set maximum leaf nodes per tree (default: 31).
    pub fn max_leaf_nodes(mut self, n: usize) -> Self {
        self.max_leaf_nodes = n;
        self
    }

    /// Set minimum samples per leaf (default: 20).
    pub fn min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }

    /// Set maximum tree depth (default: 8).
    pub fn max_depth(mut self, d: usize) -> Self {
        self.max_depth = d;
        self
    }

    /// Set maximum bins (2..=256, default: 256).
    pub fn max_bins(mut self, bins: usize) -> Self {
        self.max_bins = bins.clamp(2, NUM_BINS);
        self
    }

    /// Set L2 regularization (default: 0.0).
    pub fn l2_regularization(mut self, l2: f64) -> Self {
        self.l2_regularization = l2;
        self
    }

    /// Set random seed (default: 42).
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Train the histogram-based gradient boosting classifier.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_no_inf()?;
        let n = data.n_samples();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err(ScryLearnError::InvalidParameter(
                "learning_rate must be in (0, 1]".into(),
            ));
        }

        self.n_features = data.n_features();
        self.n_classes = data.n_classes();
        let k = self.n_classes;

        if k < 2 {
            return Err(ScryLearnError::InvalidParameter(
                "need at least 2 classes for classification".into(),
            ));
        }

        // Bin features.
        self.binner = FeatureBinner::new().max_bins(self.max_bins);
        let binned = self.binner.fit_transform(data)?;

        let all_indices: Vec<usize> = (0..n).collect();

        // Adjust min_samples_leaf for small datasets.
        let effective_min_leaf = self.min_samples_leaf.min(n / 4).max(1);

        if k == 2 {
            self.fit_binary(data, n, &binned, &all_indices, effective_min_leaf)
        } else {
            self.fit_multiclass(data, n, k, &binned, &all_indices, effective_min_leaf)
        }
    }

    /// Binary classification via log-loss.
    #[allow(clippy::unnecessary_wraps)]
    fn fit_binary(
        &mut self,
        data: &Dataset,
        n: usize,
        binned: &[Vec<u8>],
        all_indices: &[usize],
        min_leaf: usize,
    ) -> Result<()> {
        // Initial prediction: log-odds of positive class.
        let pos_count = data.target.iter().filter(|&&y| y > 0.5).count();
        let p = (pos_count as f64 / n as f64).clamp(1e-7, 1.0 - 1e-7);
        let f0 = (p / (1.0 - p)).ln();
        self.init_predictions = vec![f0];

        let mut f_vals = vec![f0; n];
        let mut trees_seq = Vec::with_capacity(self.n_estimators);

        for _ in 0..self.n_estimators {
            // Compute gradients and hessians for log-loss.
            let probs: Vec<f64> = f_vals.iter().map(|&f| sigmoid(f)).collect();
            let gradients: Vec<f64> = (0..n).map(|i| probs[i] - data.target[i]).collect();
            let hessians: Vec<f64> = probs.iter().map(|&p| (p * (1.0 - p)).max(1e-10)).collect();

            let tree = build_tree_leaf_wise(
                binned,
                &gradients,
                &hessians,
                all_indices,
                self.max_leaf_nodes,
                min_leaf,
                self.max_depth,
                self.l2_regularization,
                self.n_features,
            );

            // Update predictions.
            for &i in all_indices {
                let sample: Vec<u8> = binned.iter().map(|col| col[i]).collect();
                f_vals[i] += self.learning_rate * tree.predict_one(&sample);
            }

            trees_seq.push(tree);
        }

        self.trees = vec![trees_seq];
        self.fitted = true;
        Ok(())
    }

    /// Multiclass via softmax (K tree sequences).
    #[allow(clippy::unnecessary_wraps)]
    fn fit_multiclass(
        &mut self,
        data: &Dataset,
        n: usize,
        k: usize,
        binned: &[Vec<u8>],
        all_indices: &[usize],
        min_leaf: usize,
    ) -> Result<()> {
        // One-hot targets.
        let y_onehot: Vec<Vec<f64>> = (0..k)
            .map(|cls| {
                data.target
                    .iter()
                    .map(|&y| if (y as usize) == cls { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();

        // Initial predictions: log of class priors.
        let class_counts: Vec<usize> = (0..k)
            .map(|cls| data.target.iter().filter(|&&y| (y as usize) == cls).count())
            .collect();
        let init_preds: Vec<f64> = class_counts
            .iter()
            .map(|&c| (c as f64 / n as f64).clamp(1e-7, 1.0 - 1e-7).ln())
            .collect();
        self.init_predictions.clone_from(&init_preds);

        // f_vals[class][sample]
        let mut f_vals: Vec<Vec<f64>> = (0..k).map(|c| vec![init_preds[c]; n]).collect();
        let mut trees_all: Vec<Vec<HistTree>> = (0..k)
            .map(|_| Vec::with_capacity(self.n_estimators))
            .collect();

        for _ in 0..self.n_estimators {
            // Softmax probabilities.
            let probs = softmax_matrix(&f_vals, n, k);

            for cls in 0..k {
                // Gradients: p_k - y_k; Hessians: p_k * (1 - p_k).
                let gradients: Vec<f64> =
                    (0..n).map(|i| probs[cls][i] - y_onehot[cls][i]).collect();
                let hessians: Vec<f64> = (0..n)
                    .map(|i| (probs[cls][i] * (1.0 - probs[cls][i])).max(1e-10))
                    .collect();

                let tree = build_tree_leaf_wise(
                    binned,
                    &gradients,
                    &hessians,
                    all_indices,
                    self.max_leaf_nodes,
                    min_leaf,
                    self.max_depth,
                    self.l2_regularization,
                    self.n_features,
                );

                for &i in all_indices {
                    let sample: Vec<u8> = binned.iter().map(|col| col[i]).collect();
                    f_vals[cls][i] += self.learning_rate * tree.predict_one(&sample);
                }

                trees_all[cls].push(tree);
            }
        }

        self.trees = trees_all;
        self.fitted = true;
        Ok(())
    }

    /// Predict class labels for new samples.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        crate::version::check_schema_version(self._schema_version)?;
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let proba = self.predict_proba(features)?;
        Ok(proba
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0.0, |(idx, _)| idx as f64)
            })
            .collect())
    }

    /// Predict class probabilities for new samples.
    pub fn predict_proba(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let n = features.len();
        let k = self.n_classes;

        if k == 2 {
            // Binary: single tree sequence, use sigmoid.
            let mut f_vals = vec![self.init_predictions[0]; n];
            for tree in &self.trees[0] {
                for (i, sample) in features.iter().enumerate() {
                    f_vals[i] += self.learning_rate * tree.predict_one_raw(sample, &self.binner);
                }
            }
            Ok(f_vals
                .iter()
                .map(|&f| {
                    let p = sigmoid(f);
                    vec![1.0 - p, p]
                })
                .collect())
        } else {
            // Multiclass: K tree sequences, use softmax.
            let mut f_vals: Vec<Vec<f64>> =
                (0..k).map(|c| vec![self.init_predictions[c]; n]).collect();

            for (cls_vals, cls_trees) in f_vals.iter_mut().zip(self.trees.iter()).take(k) {
                for tree in cls_trees {
                    for (i, sample) in features.iter().enumerate() {
                        cls_vals[i] +=
                            self.learning_rate * tree.predict_one_raw(sample, &self.binner);
                    }
                }
            }

            let probs = softmax_matrix(&f_vals, n, k);
            // Transpose from [class][sample] to [sample][class].
            Ok((0..n)
                .map(|i| (0..k).map(|c| probs[c][i]).collect())
                .collect())
        }
    }

    /// Feature importances (total gain, normalized).
    pub fn feature_importances(&self) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let m = self.n_features;
        let mut imp = vec![0.0; m];
        for tree_seq in &self.trees {
            for tree in tree_seq {
                let ti = tree.feature_importances(m);
                for (i, &v) in ti.iter().enumerate() {
                    imp[i] += v;
                }
            }
        }
        let total: f64 = imp.iter().sum();
        if total > 0.0 {
            for v in &mut imp {
                *v /= total;
            }
        }
        Ok(imp)
    }

    /// Number of trees in the ensemble.
    pub fn n_trees(&self) -> usize {
        self.trees.iter().map(Vec::len).sum()
    }

    /// Number of classes.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Number of features the model was trained on.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Learning rate value.
    pub fn learning_rate_val(&self) -> f64 {
        self.learning_rate
    }

    /// Initial predictions per class.
    pub fn init_predictions_val(&self) -> &[f64] {
        &self.init_predictions
    }

    /// Convert internal HistTree nodes to public HistNodeView arrays for ONNX export.
    /// Returns `class_tree_views[class_idx][tree_idx]` = Vec of HistNodeView.
    pub fn class_tree_node_views(&self) -> Vec<Vec<Vec<HistNodeView>>> {
        self.trees
            .iter()
            .map(|class_trees| {
                class_trees
                    .iter()
                    .map(|tree| tree.to_node_views(&self.binner))
                    .collect()
            })
            .collect()
    }
}

impl Default for HistGradientBoostingClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Utility functions
// ═══════════════════════════════════════════════════════════════════════════

/// Sigmoid function.
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Softmax over class×sample matrix. Input/output: `[class][sample]`.
fn softmax_matrix(f_vals: &[Vec<f64>], n: usize, k: usize) -> Vec<Vec<f64>> {
    let mut result: Vec<Vec<f64>> = vec![vec![0.0; n]; k];

    for i in 0..n {
        let max_f = (0..k)
            .map(|c| f_vals[c][i])
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = (0..k).map(|c| (f_vals[c][i] - max_f).exp()).sum();
        for c in 0..k {
            result[c][i] = (f_vals[c][i] - max_f).exp() / exp_sum;
        }
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{accuracy, r2_score};

    fn simple_regression_data() -> Dataset {
        // y = 2x + 1
        let x: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|&v| 2.0 * v + 1.0).collect();
        Dataset::new(vec![x], y, vec!["x".into()], "y")
    }

    fn simple_classification_data() -> Dataset {
        let n = 200;
        let mut f1 = Vec::with_capacity(n);
        let mut f2 = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        let mut rng = crate::rng::FastRng::new(42);

        for _ in 0..n / 2 {
            f1.push(rng.f64() * 2.0);
            f2.push(rng.f64() * 2.0);
            target.push(0.0);
        }
        for _ in 0..n / 2 {
            f1.push(5.0 + rng.f64() * 2.0);
            f2.push(5.0 + rng.f64() * 2.0);
            target.push(1.0);
        }

        Dataset::new(
            vec![f1, f2],
            target,
            vec!["f1".into(), "f2".into()],
            "class",
        )
    }

    #[test]
    fn test_hist_gbr_fit_predict() {
        let data = simple_regression_data();
        let mut model = HistGradientBoostingRegressor::new()
            .n_estimators(50)
            .learning_rate(0.1)
            .max_leaf_nodes(15)
            .min_samples_leaf(5);
        model.fit(&data).unwrap();

        let test_x = vec![vec![3.0], vec![5.0], vec![7.0]];
        let preds = model.predict(&test_x).unwrap();
        assert_eq!(preds.len(), 3);

        // Should approximate y = 2x + 1.
        assert!((preds[0] - 7.0).abs() < 1.5, "got {}", preds[0]);
        assert!((preds[1] - 11.0).abs() < 1.5, "got {}", preds[1]);
    }

    #[test]
    fn test_hist_gbr_r2() {
        let data = simple_regression_data();
        let mut model = HistGradientBoostingRegressor::new()
            .n_estimators(100)
            .learning_rate(0.1)
            .max_leaf_nodes(31)
            .min_samples_leaf(3);
        model.fit(&data).unwrap();

        let features = data.feature_matrix();
        let preds = model.predict(&features).unwrap();
        let r2 = r2_score(&data.target, &preds);
        assert!(r2 > 0.95, "R² should be > 0.95, got {r2:.4}");
    }

    #[test]
    fn test_hist_gbc_binary() {
        let data = simple_classification_data();
        let mut model = HistGradientBoostingClassifier::new()
            .n_estimators(50)
            .learning_rate(0.1)
            .max_leaf_nodes(15)
            .min_samples_leaf(5);
        model.fit(&data).unwrap();

        let features = data.feature_matrix();
        let preds = model.predict(&features).unwrap();
        let acc = accuracy(&data.target, &preds);
        assert!(
            acc > 0.90,
            "accuracy should be > 90%, got {:.1}%",
            acc * 100.0
        );
    }

    #[test]
    fn test_hist_gbc_multiclass() {
        let n_per_class = 50;
        let mut rng = crate::rng::FastRng::new(42);
        let mut f1 = Vec::new();
        let mut f2 = Vec::new();
        let mut target = Vec::new();

        for cls in 0..3 {
            let offset = cls as f64 * 5.0;
            for _ in 0..n_per_class {
                f1.push(offset + rng.f64() * 2.0);
                f2.push(offset + rng.f64() * 2.0);
                target.push(cls as f64);
            }
        }

        let data = Dataset::new(
            vec![f1, f2],
            target,
            vec!["f1".into(), "f2".into()],
            "class",
        );

        let mut model = HistGradientBoostingClassifier::new()
            .n_estimators(50)
            .learning_rate(0.1)
            .max_leaf_nodes(15)
            .min_samples_leaf(3);
        model.fit(&data).unwrap();

        let features = data.feature_matrix();
        let preds = model.predict(&features).unwrap();
        let acc = accuracy(&data.target, &preds);
        assert!(
            acc > 0.90,
            "multiclass accuracy > 90%, got {:.1}%",
            acc * 100.0
        );
    }

    #[test]
    fn test_hist_gbc_predict_proba() {
        let data = simple_classification_data();
        let mut model = HistGradientBoostingClassifier::new()
            .n_estimators(30)
            .learning_rate(0.1)
            .min_samples_leaf(5);
        model.fit(&data).unwrap();

        let features = data.feature_matrix();
        let proba = model.predict_proba(&features).unwrap();
        for row in &proba {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "probabilities should sum to 1.0");
            for &p in row {
                assert!((0.0..=1.0).contains(&p), "probability out of range: {p}");
            }
        }
    }

    #[test]
    fn test_hist_gbr_not_fitted() {
        let model = HistGradientBoostingRegressor::new();
        let result = model.predict(&[vec![1.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_hist_gbc_not_fitted() {
        let model = HistGradientBoostingClassifier::new();
        let result = model.predict(&[vec![1.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_hist_gbr_feature_importances() {
        let data = simple_regression_data();
        let mut model = HistGradientBoostingRegressor::new()
            .n_estimators(50)
            .min_samples_leaf(3);
        model.fit(&data).unwrap();

        let imp = model.feature_importances().unwrap();
        assert_eq!(imp.len(), 1);
        let sum: f64 = imp.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6 || sum == 0.0);
    }

    #[test]
    fn test_hist_gbr_with_nan() {
        let x: Vec<f64> = (0..100)
            .map(|i| {
                if i % 10 == 0 {
                    f64::NAN
                } else {
                    i as f64 * 0.1
                }
            })
            .collect();
        let y: Vec<f64> = (0..100).map(|i| i as f64 * 0.2 + 1.0).collect();
        let data = Dataset::new(vec![x], y, vec!["x".into()], "y");

        let mut model = HistGradientBoostingRegressor::new()
            .n_estimators(50)
            .min_samples_leaf(3);
        model.fit(&data).unwrap();

        // Predict with NaN — should not panic.
        let preds = model.predict(&[vec![f64::NAN], vec![5.0]]).unwrap();
        assert_eq!(preds.len(), 2);
        assert!(
            !preds[0].is_nan(),
            "NaN input should produce a finite prediction"
        );
        assert!(!preds[1].is_nan());
    }
}
