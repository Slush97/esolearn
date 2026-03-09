// SPDX-License-Identifier: MIT OR Apache-2.0
//! Isolation Forest for anomaly/outlier detection.
//!
//! Detects anomalies by building an ensemble of random isolation trees.
//! Anomalies are isolated in fewer splits, yielding shorter average path
//! lengths and higher anomaly scores.
//!
//! # Examples
//!
//! ```ignore
//! use scry_learn::anomaly::IsolationForest;
//!
//! let mut ifo = IsolationForest::new()
//!     .n_estimators(100)
//!     .contamination(0.1);
//!
//! let data = vec![
//!     vec![1.0, 2.0],
//!     vec![1.1, 2.1],
//!     vec![100.0, 200.0], // outlier
//! ];
//! ifo.fit(&data).unwrap();
//!
//! let scores = ifo.predict(&data);
//! let labels = ifo.predict_labels(&data);
//! assert_eq!(labels[2], -1); // outlier
//! ```

use crate::error::{Result, ScryLearnError};

// ---------------------------------------------------------------------------
// Isolation Tree internals
// ---------------------------------------------------------------------------

/// A single node in an isolation tree.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum ITreeNode {
    /// Internal split node.
    Split {
        /// Feature index used for the split.
        feature: usize,
        /// Split threshold.
        threshold: f64,
        /// Left child (values < threshold).
        left: Box<ITreeNode>,
        /// Right child (values >= threshold).
        right: Box<ITreeNode>,
    },
    /// Leaf node reached when max depth hit or only one sample remains.
    Leaf {
        /// Number of samples that reached this leaf during training.
        size: usize,
    },
}

/// A single isolation tree.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct IsolationTree {
    root: ITreeNode,
}

impl IsolationTree {
    /// Build an isolation tree on a subsample.
    fn build(data: &[Vec<f64>], max_depth: usize, rng: &mut crate::rng::FastRng) -> Self {
        let root = Self::build_node(data, 0, max_depth, rng);
        Self { root }
    }

    fn build_node(
        data: &[Vec<f64>],
        depth: usize,
        max_depth: usize,
        rng: &mut crate::rng::FastRng,
    ) -> ITreeNode {
        let n = data.len();
        if n <= 1 || depth >= max_depth {
            return ITreeNode::Leaf { size: n };
        }

        let n_features = data[0].len();
        if n_features == 0 {
            return ITreeNode::Leaf { size: n };
        }

        // Pick a random feature.
        let feature = rng.usize(0..n_features);

        // Find min/max for this feature.
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        for sample in data {
            let v = sample[feature];
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }

        // If all values are the same, can't split further.
        if (max_val - min_val).abs() < f64::EPSILON {
            return ITreeNode::Leaf { size: n };
        }

        // Random split value between min and max.
        let threshold = min_val + rng.f64() * (max_val - min_val);

        // Partition data.
        let mut left_data = Vec::new();
        let mut right_data = Vec::new();
        for sample in data {
            if sample[feature] < threshold {
                left_data.push(sample.clone());
            } else {
                right_data.push(sample.clone());
            }
        }

        // If one side is empty, treat as leaf (shouldn't happen with proper threshold).
        if left_data.is_empty() || right_data.is_empty() {
            return ITreeNode::Leaf { size: n };
        }

        let left = Self::build_node(&left_data, depth + 1, max_depth, rng);
        let right = Self::build_node(&right_data, depth + 1, max_depth, rng);

        ITreeNode::Split {
            feature,
            threshold,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Compute the path length for a single sample.
    fn path_length(&self, sample: &[f64]) -> f64 {
        Self::path_length_node(&self.root, sample, 0)
    }

    fn path_length_node(node: &ITreeNode, sample: &[f64], depth: usize) -> f64 {
        match node {
            ITreeNode::Leaf { size } => depth as f64 + average_path_length(*size),
            ITreeNode::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if sample[*feature] < *threshold {
                    Self::path_length_node(left, sample, depth + 1)
                } else {
                    Self::path_length_node(right, sample, depth + 1)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Average path length of unsuccessful search in BST
// ---------------------------------------------------------------------------

/// Average path length of an unsuccessful search in a Binary Search Tree
/// with `n` elements: `c(n) = 2 * (H(n-1)) - 2*(n-1)/n`
/// where `H(k) = ln(k) + 0.5772156649` (Euler-Mascheroni constant).
fn average_path_length(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let n_f = n as f64;
    2.0 * (((n_f - 1.0).ln()) + 0.577_215_664_9) - 2.0 * (n_f - 1.0) / n_f
}

// ---------------------------------------------------------------------------
// IsolationForest
// ---------------------------------------------------------------------------

/// Isolation Forest for anomaly detection.
///
/// Builds an ensemble of random isolation trees. Points that are isolated
/// in fewer splits (shorter path lengths) receive higher anomaly scores.
///
/// # Algorithm
///
/// 1. **Fit**: Build `n_estimators` isolation trees, each trained on a random
///    subsample of `max_samples` points.
/// 2. **Score**: For each point, compute the average path length across all
///    trees. Normalize to `[0, 1]` using `score = 2^(-E[h(x)] / c(max_samples))`.
/// 3. **Label**: Scores above the threshold (determined by `contamination`)
///    are labelled as anomalies (`-1`), others as normal (`1`).
///
/// # Examples
///
/// ```ignore
/// use scry_learn::anomaly::IsolationForest;
///
/// let mut ifo = IsolationForest::new()
///     .n_estimators(100)
///     .contamination(0.05);
/// ifo.fit(&data).unwrap();
/// let labels = ifo.predict_labels(&data);
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct IsolationForest {
    /// Number of isolation trees to build.
    n_estimators: usize,
    /// Number of samples to draw for each tree.
    max_samples: usize,
    /// Expected proportion of outliers in the data (used for threshold).
    contamination: f64,
    /// Random seed.
    random_state: Option<u64>,
    /// Trained trees (populated after fit).
    trees: Vec<IsolationTree>,
    /// The anomaly score threshold (set after fit).
    threshold: f64,
    /// The subsample size used during training (for normalization constant).
    training_sub_size: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl IsolationForest {
    /// Create a new `IsolationForest` with default parameters.
    ///
    /// Defaults: 100 estimators, 256 max samples, 0.1 contamination.
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            max_samples: 256,
            contamination: 0.1,
            random_state: None,
            trees: Vec::new(),
            threshold: 0.5,
            training_sub_size: 0,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set the number of isolation trees (default: 100).
    pub fn n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Set the subsample size per tree (default: 256).
    pub fn max_samples(mut self, n: usize) -> Self {
        self.max_samples = n;
        self
    }

    /// Set the expected contamination ratio (default: 0.1).
    ///
    /// Must be in `(0, 0.5]`. Used to determine the anomaly score threshold.
    pub fn contamination(mut self, c: f64) -> Self {
        self.contamination = c;
        self
    }

    /// Set the random seed for reproducibility.
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Alias for [`random_state`](Self::random_state) (consistent with other models).
    pub fn seed(self, s: u64) -> Self {
        self.random_state(s)
    }

    /// Build isolation trees on the provided feature matrix.
    ///
    /// `features` is row-major: `features[sample_idx][feature_idx]`.
    ///
    /// # Errors
    ///
    /// Returns [`ScryLearnError::EmptyDataset`] if `features` is empty.
    /// Returns [`ScryLearnError::InvalidParameter`] if `contamination` is out of range.
    pub fn fit(&mut self, features: &[Vec<f64>]) -> Result<()> {
        for (i, row) in features.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if !v.is_finite() {
                    return Err(ScryLearnError::InvalidData(format!(
                        "non-finite value ({v}) in feature[{j}] at sample {i}"
                    )));
                }
            }
        }
        if features.is_empty() {
            return Err(ScryLearnError::EmptyDataset);
        }
        if self.contamination <= 0.0 || self.contamination > 0.5 {
            return Err(ScryLearnError::InvalidParameter(format!(
                "contamination must be in (0, 0.5], got {}",
                self.contamination
            )));
        }

        let n = features.len();
        let sub_size = self.max_samples.min(n);
        let max_depth = (sub_size as f64).log2().ceil() as usize;
        let seed = self.random_state.unwrap_or(42);

        let mut trees = Vec::with_capacity(self.n_estimators);

        for i in 0..self.n_estimators {
            let mut rng = crate::rng::FastRng::new(seed.wrapping_add(i as u64));

            // Sample `sub_size` points randomly (with replacement).
            let subsample: Vec<Vec<f64>> = (0..sub_size)
                .map(|_| features[rng.usize(0..n)].clone())
                .collect();

            let tree = IsolationTree::build(&subsample, max_depth, &mut rng);
            trees.push(tree);
        }

        self.trees = trees;
        self.training_sub_size = sub_size;

        // Compute scores on training data to determine the threshold.
        let mut scores = self.predict(features);
        // Sort descending (highest score = most anomalous).
        scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Threshold: the score at the `contamination` quantile.
        let cutoff_idx = ((self.contamination * n as f64).ceil() as usize)
            .min(n)
            .max(1);
        self.threshold = scores[cutoff_idx - 1];

        Ok(())
    }

    /// Compute anomaly scores for each sample.
    ///
    /// Returns a `Vec<f64>` where values closer to 1.0 indicate anomalies
    /// and values closer to 0.0 indicate normal points.
    ///
    /// Score formula: `score = 2^(-E[h(x)] / c(ψ))`
    /// where `E[h(x)]` is the average path length and `c(ψ)` is the
    /// average BST path length for the training subsample size `ψ`.
    pub fn predict(&self, features: &[Vec<f64>]) -> Vec<f64> {
        let n = features.len();
        // Use the training subsample size for normalization (Liu et al.).
        // During fit, training_sub_size is set; before fit, fall back to max_samples.
        let sub_size = if self.training_sub_size > 0 {
            self.training_sub_size
        } else {
            self.max_samples.min(n.max(1))
        };
        let c = average_path_length(sub_size);

        if c.abs() < f64::EPSILON || self.trees.is_empty() {
            return vec![0.5; n];
        }

        features
            .iter()
            .map(|sample| {
                let avg_path: f64 = self
                    .trees
                    .iter()
                    .map(|t| t.path_length(sample))
                    .sum::<f64>()
                    / self.trees.len() as f64;
                2.0_f64.powf(-avg_path / c)
            })
            .collect()
    }

    /// Predict anomaly labels: `1` for normal, `-1` for anomaly.
    ///
    /// Uses the contamination-based threshold computed during `fit`.
    /// Returns all `1` (normal) if the model has not been fitted.
    pub fn predict_labels(&self, features: &[Vec<f64>]) -> Vec<i8> {
        if self.trees.is_empty() {
            return vec![1; features.len()];
        }
        let scores = self.predict(features);
        scores
            .into_iter()
            .map(|s| if s >= self.threshold { -1 } else { 1 })
            .collect()
    }

    /// Returns the anomaly score threshold determined during fit.
    pub fn score_threshold(&self) -> f64 {
        self.threshold
    }
}

impl Default for IsolationForest {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate `n_normal` normal points centered around origin and
    /// `n_outliers` outliers far from origin.
    fn make_test_data(n_normal: usize, n_outliers: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = crate::rng::FastRng::new(seed);
        let mut data = Vec::with_capacity(n_normal + n_outliers);

        // Normal cluster around (0, 0).
        for _ in 0..n_normal {
            data.push(vec![rng.f64() * 2.0 - 1.0, rng.f64() * 2.0 - 1.0]);
        }

        // Outliers far away.
        for _ in 0..n_outliers {
            data.push(vec![10.0 + rng.f64() * 5.0, 10.0 + rng.f64() * 5.0]);
        }

        data
    }

    #[test]
    fn test_iforest_detects_outliers() {
        let data = make_test_data(90, 10, 42);
        let mut ifo = IsolationForest::new()
            .n_estimators(100)
            .max_samples(64)
            .contamination(0.1)
            .random_state(42);

        ifo.fit(&data).unwrap();
        let scores = ifo.predict(&data);

        // Outliers (last 10 points) should have higher scores than normal points.
        let normal_mean: f64 = scores[..90].iter().sum::<f64>() / 90.0;
        let outlier_mean: f64 = scores[90..].iter().sum::<f64>() / 10.0;

        assert!(
            outlier_mean > normal_mean,
            "outlier mean score ({:.3}) should be higher than normal mean ({:.3})",
            outlier_mean,
            normal_mean,
        );
    }

    #[test]
    fn test_iforest_labels_recall() {
        let data = make_test_data(90, 10, 123);
        let mut ifo = IsolationForest::new()
            .n_estimators(100)
            .max_samples(64)
            .contamination(0.1)
            .random_state(123);

        ifo.fit(&data).unwrap();
        let labels = ifo.predict_labels(&data);

        // Count how many of the last 10 (outliers) were labeled -1.
        let outlier_detected = labels[90..].iter().filter(|&&l| l == -1).count();
        let recall = outlier_detected as f64 / 10.0;

        assert!(
            recall >= 0.7,
            "expected outlier recall ≥ 0.70, got {:.2}",
            recall,
        );
    }

    #[test]
    fn test_iforest_single_feature() {
        let mut data: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64 * 0.1]).collect();
        // Add an outlier.
        data.push(vec![1000.0]);

        let mut ifo = IsolationForest::new()
            .n_estimators(50)
            .max_samples(64)
            .contamination(0.05)
            .random_state(7);

        ifo.fit(&data).unwrap();
        let scores = ifo.predict(&data);

        // The outlier (last) should have the highest score.
        let max_score_idx = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        assert_eq!(
            max_score_idx,
            data.len() - 1,
            "outlier should have highest anomaly score"
        );
    }

    #[test]
    fn test_iforest_multi_feature() {
        let mut rng = crate::rng::FastRng::new(99);
        let mut data: Vec<Vec<f64>> = (0..100)
            .map(|_| {
                vec![
                    rng.f64() * 2.0,
                    rng.f64() * 2.0,
                    rng.f64() * 2.0,
                    rng.f64() * 2.0,
                ]
            })
            .collect();
        // Add outliers in 4D.
        for _ in 0..5 {
            data.push(vec![50.0, 50.0, 50.0, 50.0]);
        }

        let mut ifo = IsolationForest::new()
            .n_estimators(80)
            .max_samples(64)
            .contamination(0.05)
            .random_state(99);

        ifo.fit(&data).unwrap();
        let labels = ifo.predict_labels(&data);

        let outlier_detected = labels[100..].iter().filter(|&&l| l == -1).count();
        assert!(
            outlier_detected >= 3,
            "expected ≥ 3 of 5 outliers detected, got {}",
            outlier_detected,
        );
    }

    #[test]
    fn test_iforest_empty_input() {
        let mut ifo = IsolationForest::new();
        let result = ifo.fit(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_iforest_invalid_contamination() {
        let data = make_test_data(10, 0, 1);
        let mut ifo = IsolationForest::new().contamination(0.0);
        assert!(ifo.fit(&data).is_err());

        let mut ifo2 = IsolationForest::new().contamination(0.6);
        assert!(ifo2.fit(&data).is_err());
    }

    #[test]
    fn test_iforest_default() {
        let ifo = IsolationForest::default();
        assert_eq!(ifo.n_estimators, 100);
        assert_eq!(ifo.max_samples, 256);
        assert!((ifo.contamination - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_average_path_length() {
        assert!((average_path_length(0) - 0.0).abs() < f64::EPSILON);
        assert!((average_path_length(1) - 0.0).abs() < f64::EPSILON);
        // c(2) = 2*(ln(1) + 0.5772) - 2*1/2 = 1.1544 - 1 ≈ 0.1544
        let c2 = average_path_length(2);
        assert!((c2 - 0.1544).abs() < 0.01, "c(2) = {c2}");
        // c(256) ≈ 9.21
        let c256 = average_path_length(256);
        assert!(c256 > 8.0 && c256 < 12.0, "c(256) = {c256}");
    }
}
