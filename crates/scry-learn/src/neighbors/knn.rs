// SPDX-License-Identifier: MIT OR Apache-2.0
//! K-Nearest Neighbors classifier and regressor.
//!
//! Supports two search algorithms:
//! - **Brute force**: O(n) per query. Always correct, any metric.
//! - **KD-tree**: O(log n) average per query. Euclidean only, best for < 20 features.
//!
//! Optimizations:
//! - Uses squared Euclidean distance (avoids sqrt — monotonic, same ordering).
//! - Uses `select_nth_unstable` for partial sort (O(n) vs O(n·log n)).
//! - Fixed-size vote array avoids HashMap overhead.

use rayon::prelude::*;

use crate::accel;
use crate::constants::KNN_PAR_THRESHOLD;
use crate::dataset::Dataset;
use crate::distance::{
    cosine_distance, euclidean_sq, manhattan, sparse_cosine, sparse_euclidean_sq, sparse_manhattan,
};
use crate::error::{Result, ScryLearnError};
use crate::neighbors::kdtree::KdTree;
use crate::sparse::{CsrMatrix, SparseRow};
use crate::weights::{compute_sample_weights, ClassWeight};

/// Distance metric for KNN.
///
/// # Example
///
/// ```
/// use scry_learn::neighbors::DistanceMetric;
///
/// let metric = DistanceMetric::Cosine;
/// ```
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum DistanceMetric {
    /// Euclidean distance (L2).
    Euclidean,
    /// Manhattan distance (L1).
    Manhattan,
    /// Cosine distance: `1 − cos(θ)`, range `[0, 2]`.
    Cosine,
}

/// Weighting function for neighbor votes.
///
/// Controls how the k-nearest neighbors contribute to predictions.
///
/// # Example
///
/// ```
/// use scry_learn::neighbors::{KnnClassifier, WeightFunction};
///
/// let knn = KnnClassifier::new()
///     .k(5)
///     .weights(WeightFunction::Distance);
/// ```
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum WeightFunction {
    /// All neighbors have equal vote weight.
    #[default]
    Uniform,
    /// Closer neighbors contribute more: weight = `1 / distance`.
    ///
    /// When distance is zero (exact match), that neighbor gets all the weight.
    Distance,
}

/// Algorithm used for nearest-neighbor search.
///
/// # Example
///
/// ```
/// use scry_learn::neighbors::{KnnClassifier, Algorithm};
///
/// let knn = KnnClassifier::new()
///     .k(5)
///     .algorithm(Algorithm::KDTree);
/// ```
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Algorithm {
    /// Automatically choose the best algorithm based on data and metric.
    ///
    /// Uses KD-tree for Euclidean distance with < 20 features,
    /// brute-force otherwise.
    #[default]
    Auto,
    /// Brute-force O(n) search. Works with all distance metrics.
    BruteForce,
    /// KD-tree O(log n) average-case search. Euclidean distance only.
    ///
    /// Falls back to brute-force if a non-Euclidean metric is selected.
    KDTree,
}

// ─────────────────────────────────────────────────────────────────
// KNN Classifier
// ─────────────────────────────────────────────────────────────────

/// K-Nearest Neighbors classifier.
///
/// Uses brute-force distance computation — fast enough for datasets up to ~100k samples.
///
/// # Example
///
/// ```
/// use scry_learn::dataset::Dataset;
/// use scry_learn::neighbors::{KnnClassifier, WeightFunction};
///
/// let features = vec![
///     vec![0.0, 0.0, 10.0, 10.0],
///     vec![0.0, 0.0, 10.0, 10.0],
/// ];
/// let target = vec![0.0, 0.0, 1.0, 1.0];
/// let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");
///
/// let mut knn = KnnClassifier::new()
///     .k(3)
///     .weights(WeightFunction::Distance);
/// knn.fit(&data).unwrap();
///
/// let preds = knn.predict(&[vec![1.0, 1.0]]).unwrap();
/// assert_eq!(preds[0] as usize, 0);
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct KnnClassifier {
    k: usize,
    metric: DistanceMetric,
    weight_fn: WeightFunction,
    class_weight: ClassWeight,
    algorithm: Algorithm,
    train_features: Vec<Vec<f64>>, // [n_samples][n_features]
    train_target: Vec<f64>,
    train_weights: Vec<f64>,
    n_classes: usize,
    kdtree: Option<KdTree>,
    /// Sparse training data (CSR) for sparse-native distance computation.
    train_sparse: Option<CsrMatrix>,
    fitted: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl KnnClassifier {
    /// Create a new KNN classifier with k=5.
    pub fn new() -> Self {
        Self {
            k: 5,
            metric: DistanceMetric::Euclidean,
            weight_fn: WeightFunction::Uniform,
            class_weight: ClassWeight::Uniform,
            algorithm: Algorithm::Auto,
            train_features: Vec::new(),
            train_target: Vec::new(),
            train_weights: Vec::new(),
            n_classes: 0,
            kdtree: None,
            train_sparse: None,
            fitted: false,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set the number of neighbors.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the distance metric.
    pub fn metric(mut self, m: DistanceMetric) -> Self {
        self.metric = m;
        self
    }

    /// Set the neighbor weighting function.
    ///
    /// - [`WeightFunction::Uniform`]: every neighbor's vote counts equally.
    /// - [`WeightFunction::Distance`]: closer neighbors contribute more (weight = `1/d`).
    pub fn weights(mut self, w: WeightFunction) -> Self {
        self.weight_fn = w;
        self
    }

    /// Set class weighting strategy for imbalanced datasets.
    pub fn class_weight(mut self, cw: ClassWeight) -> Self {
        self.class_weight = cw;
        self
    }

    /// Set the nearest-neighbor search algorithm.
    ///
    /// - [`Algorithm::Auto`] (default): uses KD-tree for Euclidean distance
    ///   with fewer than 20 features, brute-force otherwise.
    /// - [`Algorithm::BruteForce`]: always O(n) brute-force scan.
    /// - [`Algorithm::KDTree`]: builds a KD-tree for O(log n) queries;
    ///   falls back to brute-force if a non-Euclidean metric is set.
    pub fn algorithm(mut self, algo: Algorithm) -> Self {
        self.algorithm = algo;
        self
    }

    /// Store training data. Builds a KD-tree if the selected algorithm
    /// (or `Auto` heuristic) calls for it.
    ///
    /// If the dataset uses sparse storage, the CSR representation is stored
    /// for efficient sparse distance computation in [`KnnClassifier::predict_sparse`].
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        if data.n_samples() == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        // Store sparse training data if available.
        if let Some(csr) = data.sparse_csr() {
            self.train_sparse = Some(csr);
            self.train_features = Vec::new(); // no dense copy needed
        } else {
            self.train_sparse = None;
            self.train_features = data.feature_matrix();
        }

        self.train_target.clone_from(&data.target);
        self.train_weights = compute_sample_weights(&data.target, &self.class_weight);
        self.n_classes = data.n_classes();

        // Build KD-tree if appropriate (only for dense data).
        self.kdtree = if self.train_sparse.is_none()
            && should_use_kdtree(self.algorithm, self.metric, data.n_features())
        {
            Some(KdTree::build(&self.train_features))
        } else {
            None
        };

        self.fitted = true;
        Ok(())
    }

    /// Predict class labels.
    ///
    /// Uses partial sort (`select_nth_unstable`) to find k nearest neighbors
    /// in O(n) instead of full O(n·log n) sort. Euclidean distances skip sqrt
    /// since we only need relative ordering (unless distance weighting is on).
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        crate::version::check_schema_version(self._schema_version)?;
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        if self.train_features.is_empty() && self.train_sparse.is_some() {
            return Err(ScryLearnError::InvalidParameter(
                "model was trained on sparse data; use predict_sparse() instead".into(),
            ));
        }
        let probas = self.compute_votes(features);
        Ok(probas
            .into_iter()
            .map(|votes| {
                // Fold to keep the *first* class with max votes on ties
                // (sklearn picks lowest class index).
                votes
                    .iter()
                    .enumerate()
                    .fold((0usize, f64::NEG_INFINITY), |(best_i, best_v), (i, &v)| {
                        if v > best_v {
                            (i, v)
                        } else {
                            (best_i, best_v)
                        }
                    })
                    .0 as f64
            })
            .collect())
    }

    /// Predict class probability distribution for each sample.
    ///
    /// Returns a `Vec<Vec<f64>>` where `result[i][c]` is the estimated
    /// probability that sample `i` belongs to class `c`. Probabilities
    /// sum to 1.0 for each sample.
    ///
    /// # Example
    ///
    /// ```
    /// use scry_learn::dataset::Dataset;
    /// use scry_learn::neighbors::KnnClassifier;
    ///
    /// let features = vec![
    ///     vec![0.0, 0.0, 10.0, 10.0],
    ///     vec![0.0, 0.0, 10.0, 10.0],
    /// ];
    /// let target = vec![0.0, 0.0, 1.0, 1.0];
    /// let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");
    ///
    /// let mut knn = KnnClassifier::new().k(3);
    /// knn.fit(&data).unwrap();
    ///
    /// let probas = knn.predict_proba(&[vec![1.0, 1.0]]).unwrap();
    /// let sum: f64 = probas[0].iter().sum();
    /// assert!((sum - 1.0).abs() < 1e-9);
    /// ```
    pub fn predict_proba(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        if self.train_features.is_empty() && self.train_sparse.is_some() {
            return Err(ScryLearnError::InvalidParameter(
                "model was trained on sparse data; use predict_sparse() instead".into(),
            ));
        }
        let votes = self.compute_votes(features);
        Ok(votes
            .into_iter()
            .map(|v| {
                let total: f64 = v.iter().sum();
                if total > 0.0 {
                    v.iter().map(|&x| x / total).collect()
                } else {
                    // Fallback: uniform distribution.
                    let n = v.len() as f64;
                    vec![1.0 / n; v.len()]
                }
            })
            .collect())
    }

    /// Core voting logic shared by `predict` and `predict_proba`.
    ///
    /// Returns raw weighted vote counts per class for each query sample.
    ///
    /// When the metric is Euclidean and no KD-tree is in use, distances
    /// are computed in a single batch via [`ComputeBackend`], which
    /// uses GPU compute shaders when the `gpu` feature is enabled and
    /// the dataset is large enough.
    #[allow(clippy::option_if_let_else)]
    fn compute_votes(&self, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let k = self.k.min(self.train_features.len());
        let use_actual_dist = matches!(self.weight_fn, WeightFunction::Distance);
        let metric = self.metric;

        // Try batched backend path for Euclidean brute-force.
        let batched = if self.kdtree.is_none() && matches!(metric, DistanceMetric::Euclidean) {
            batched_brute_force_neighbors(features, &self.train_features, k, use_actual_dist)
        } else {
            None
        };

        if let Some(all_neighbors) = batched {
            // Batched path — distances already computed.
            all_neighbors
                .into_iter()
                .map(|neighbors| {
                    aggregate_votes(
                        &neighbors,
                        &self.train_target,
                        &self.train_weights,
                        self.n_classes,
                        use_actual_dist,
                    )
                })
                .collect()
        } else {
            // Per-sample path (KD-tree or non-Euclidean metric).
            let n_train = self.train_features.len();
            let n_features = if n_train > 0 { self.train_features[0].len() } else { 0 };
            let use_par = self.kdtree.is_none()
                && features.len() * n_train * n_features >= KNN_PAR_THRESHOLD;

            let vote_fn = |query: &Vec<f64>| {
                let neighbors: Vec<(f64, usize)> = if let Some(ref tree) = self.kdtree {
                    let raw = tree.query_k_nearest(query, k, &self.train_features);
                    if use_actual_dist {
                        raw.into_iter().map(|(d2, i)| (d2.sqrt(), i)).collect()
                    } else {
                        raw
                    }
                } else {
                    scalar_brute_force(query, &self.train_features, k, metric, use_actual_dist)
                };

                aggregate_votes(
                    &neighbors,
                    &self.train_target,
                    &self.train_weights,
                    self.n_classes,
                    use_actual_dist,
                )
            };

            if use_par {
                features.par_iter().map(vote_fn).collect()
            } else {
                features.iter().map(vote_fn).collect()
            }
        }
    }
}

impl KnnClassifier {
    /// Predict class labels from sparse features (CSR format).
    ///
    /// Uses true sparse distance computation via merge-join on sorted indices,
    /// avoiding densification. Supports Euclidean, Manhattan, and Cosine metrics.
    pub fn predict_sparse(&self, features: &CsrMatrix) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let n_train = self.train_target.len();
        let k = self.k.min(n_train);
        let use_actual_dist = matches!(self.weight_fn, WeightFunction::Distance);

        Ok((0..features.n_rows())
            .map(|i| {
                let query = features.row(i);
                let neighbors = if let Some(ref train_csr) = self.train_sparse {
                    sparse_brute_force(&query, train_csr, k, self.metric, use_actual_dist)
                } else {
                    let dense = sparse_row_to_dense(&query, features.n_cols());
                    scalar_brute_force(
                        &dense,
                        &self.train_features,
                        k,
                        self.metric,
                        use_actual_dist,
                    )
                };
                let votes = aggregate_votes(
                    &neighbors,
                    &self.train_target,
                    &self.train_weights,
                    self.n_classes,
                    use_actual_dist,
                );
                votes
                    .iter()
                    .enumerate()
                    .fold((0usize, f64::NEG_INFINITY), |(best_i, best_v), (i, &v)| {
                        if v > best_v {
                            (i, v)
                        } else {
                            (best_i, best_v)
                        }
                    })
                    .0 as f64
            })
            .collect())
    }
}

impl Default for KnnClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────
// KNN Regressor
// ─────────────────────────────────────────────────────────────────

/// K-Nearest Neighbors regressor.
///
/// Predicts the (optionally distance-weighted) mean of the k-nearest
/// training targets for each query point.
///
/// # Example
///
/// ```
/// use scry_learn::dataset::Dataset;
/// use scry_learn::neighbors::{KnnRegressor, WeightFunction};
///
/// let features = vec![vec![1.0, 2.0, 3.0]];
/// let target = vec![10.0, 20.0, 30.0];
/// let data = Dataset::new(features, target, vec!["x".into()], "y");
///
/// let mut knn = KnnRegressor::new()
///     .k(2)
///     .weights(WeightFunction::Uniform);
/// knn.fit(&data).unwrap();
///
/// let preds = knn.predict(&[vec![2.5]]).unwrap();
/// // Nearest neighbors are x=2 (y=20) and x=3 (y=30) → mean = 25
/// assert!((preds[0] - 25.0).abs() < 1e-9);
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct KnnRegressor {
    k: usize,
    metric: DistanceMetric,
    weight_fn: WeightFunction,
    algorithm: Algorithm,
    train_features: Vec<Vec<f64>>, // [n_samples][n_features]
    train_target: Vec<f64>,
    kdtree: Option<KdTree>,
    /// Sparse training data (CSR) for sparse-native distance computation.
    train_sparse: Option<CsrMatrix>,
    fitted: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl KnnRegressor {
    /// Create a new KNN regressor with k=5.
    pub fn new() -> Self {
        Self {
            k: 5,
            metric: DistanceMetric::Euclidean,
            weight_fn: WeightFunction::Uniform,
            algorithm: Algorithm::Auto,
            train_features: Vec::new(),
            train_target: Vec::new(),
            kdtree: None,
            train_sparse: None,
            fitted: false,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set the number of neighbors.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the distance metric.
    pub fn metric(mut self, m: DistanceMetric) -> Self {
        self.metric = m;
        self
    }

    /// Set the neighbor weighting function.
    ///
    /// - [`WeightFunction::Uniform`]: all k neighbors contribute equally to the mean.
    /// - [`WeightFunction::Distance`]: closer neighbors are weighted by `1/distance`.
    pub fn weights(mut self, w: WeightFunction) -> Self {
        self.weight_fn = w;
        self
    }

    /// Set the nearest-neighbor search algorithm.
    ///
    /// See [`Algorithm`] for details.
    pub fn algorithm(mut self, algo: Algorithm) -> Self {
        self.algorithm = algo;
        self
    }

    /// Store training data. Builds KD-tree if appropriate.
    ///
    /// If the dataset uses sparse storage, the CSR representation is stored
    /// for efficient sparse distance computation in [`KnnRegressor::predict_sparse`].
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        if data.n_samples() == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        if let Some(csr) = data.sparse_csr() {
            self.train_sparse = Some(csr);
            self.train_features = Vec::new();
        } else {
            self.train_sparse = None;
            self.train_features = data.feature_matrix();
        }

        self.train_target.clone_from(&data.target);

        self.kdtree = if self.train_sparse.is_none()
            && should_use_kdtree(self.algorithm, self.metric, data.n_features())
        {
            Some(KdTree::build(&self.train_features))
        } else {
            None
        };

        self.fitted = true;
        Ok(())
    }

    /// Predict continuous target values.
    ///
    /// For each query point, finds the k nearest training samples and returns
    /// their mean (or distance-weighted mean) target value.
    ///
    /// When the metric is Euclidean and no KD-tree is in use, distances
    /// are computed in a single batch via `ComputeBackend`.
    #[allow(clippy::option_if_let_else)]
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        crate::version::check_schema_version(self._schema_version)?;
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        if self.train_features.is_empty() && self.train_sparse.is_some() {
            return Err(ScryLearnError::InvalidParameter(
                "model was trained on sparse data; use predict_sparse() instead".into(),
            ));
        }

        let k = self.k.min(self.train_features.len());
        let use_actual_dist = matches!(self.weight_fn, WeightFunction::Distance);
        let metric = self.metric;

        // Try batched backend path for Euclidean brute-force.
        let batched = if self.kdtree.is_none() && matches!(metric, DistanceMetric::Euclidean) {
            batched_brute_force_neighbors(features, &self.train_features, k, use_actual_dist)
        } else {
            None
        };

        let get_neighbors = |query: &Vec<f64>| -> Vec<(f64, usize)> {
            if let Some(ref tree) = self.kdtree {
                let raw = tree.query_k_nearest(query, k, &self.train_features);
                if use_actual_dist {
                    raw.into_iter().map(|(d2, i)| (d2.sqrt(), i)).collect()
                } else {
                    raw
                }
            } else {
                scalar_brute_force(query, &self.train_features, k, metric, use_actual_dist)
            }
        };

        if let Some(ref all) = batched {
            // Batched path — already computed.
            Ok(features
                .iter()
                .enumerate()
                .map(|(qi, _query)| {
                    aggregate_regression(&all[qi], &self.train_target, use_actual_dist, k)
                })
                .collect())
        } else {
            let n_train = self.train_features.len();
            let n_features = if n_train > 0 { self.train_features[0].len() } else { 0 };
            let use_par = self.kdtree.is_none()
                && features.len() * n_train * n_features >= KNN_PAR_THRESHOLD;

            let predict_fn = |query: &Vec<f64>| {
                let neighbors = get_neighbors(query);
                aggregate_regression(&neighbors, &self.train_target, use_actual_dist, k)
            };

            if use_par {
                Ok(features.par_iter().map(predict_fn).collect())
            } else {
                Ok(features.iter().map(predict_fn).collect())
            }
        }
    }
}

impl KnnRegressor {
    /// Predict from sparse features (CSR format).
    ///
    /// Uses true sparse distance computation via merge-join on sorted indices.
    pub fn predict_sparse(&self, features: &CsrMatrix) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let n_train = self.train_target.len();
        let k = self.k.min(n_train);
        let use_actual_dist = matches!(self.weight_fn, WeightFunction::Distance);

        Ok((0..features.n_rows())
            .map(|i| {
                let query = features.row(i);
                let neighbors = if let Some(ref train_csr) = self.train_sparse {
                    sparse_brute_force(&query, train_csr, k, self.metric, use_actual_dist)
                } else {
                    let dense = sparse_row_to_dense(&query, features.n_cols());
                    scalar_brute_force(
                        &dense,
                        &self.train_features,
                        k,
                        self.metric,
                        use_actual_dist,
                    )
                };
                aggregate_regression(&neighbors, &self.train_target, use_actual_dist, k)
            })
            .collect())
    }
}

impl Default for KnnRegressor {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────

/// Per-sample brute-force distance computation.
///
/// Used when the batched backend path is not applicable (non-Euclidean metric).
fn scalar_brute_force(
    query: &[f64],
    train: &[Vec<f64>],
    k: usize,
    metric: DistanceMetric,
    use_actual_dist: bool,
) -> Vec<(f64, usize)> {
    let mut dists: Vec<(f64, usize)> = train
        .iter()
        .enumerate()
        .map(|(i, train_row)| {
            let d = if use_actual_dist {
                actual_distance(query, train_row, metric)
            } else {
                distance_for_compare(query, train_row, metric)
            };
            (d, i)
        })
        .collect();

    if k < dists.len() {
        dists.select_nth_unstable_by(k - 1, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    dists.truncate(k);
    // Stable sort by (distance, index) to deterministically prefer
    // lower-index training samples on ties (matches sklearn behavior).
    dists.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    dists
}

/// Batched brute-force using `ComputeBackend::pairwise_distances_squared()`.
///
/// Returns `Some(neighbors)` where `neighbors[i]` is a `Vec<(dist, idx)>` of
/// k-nearest for query `i`, or `None` if batch threshold isn't met.
///
/// Only valid for Euclidean distance (squared distances preserve ordering).
fn batched_brute_force_neighbors(
    queries: &[Vec<f64>],
    train: &[Vec<f64>],
    k: usize,
    use_actual_dist: bool,
) -> Option<Vec<Vec<(f64, usize)>>> {
    let n_q = queries.len();
    let n_t = train.len();
    if n_q == 0 || n_t == 0 {
        return None;
    }
    let dim = queries[0].len();

    // Only worth batching for reasonably sized problems.
    // The backend has its own internal thresholds too.
    if n_q * n_t < 256 {
        return None;
    }

    // Flatten row-major: queries[n_q][dim] → flat[n_q * dim]
    let q_flat: Vec<f64> = queries.iter().flat_map(|r| r.iter().copied()).collect();
    let t_flat: Vec<f64> = train.iter().flat_map(|r| r.iter().copied()).collect();

    let backend = accel::auto();
    let dist_matrix = backend.pairwise_distances_squared(&q_flat, &t_flat, n_q, n_t, dim);

    let result: Vec<Vec<(f64, usize)>> = (0..n_q)
        .map(|qi| {
            let row = &dist_matrix[qi * n_t..(qi + 1) * n_t];
            let mut indexed: Vec<(f64, usize)> = row
                .iter()
                .enumerate()
                .map(|(j, &d2)| {
                    let d = if use_actual_dist { d2.sqrt() } else { d2 };
                    (d, j)
                })
                .collect();

            let k_eff = k.min(indexed.len());
            if k_eff < indexed.len() {
                indexed.select_nth_unstable_by(k_eff - 1, |a, b| {
                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            indexed.truncate(k_eff);
            // Stable sort by (distance, index) — matches sklearn tie-breaking.
            indexed.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.cmp(&b.1))
            });
            indexed
        })
        .collect();

    Some(result)
}

/// Aggregate weighted votes for classification.
fn aggregate_votes(
    neighbors: &[(f64, usize)],
    target: &[f64],
    weights: &[f64],
    n_classes: usize,
    use_actual_dist: bool,
) -> Vec<f64> {
    let mut votes = vec![0.0_f64; n_classes.max(1)];

    if use_actual_dist {
        let has_exact = neighbors.iter().any(|&(d, _)| d < f64::EPSILON);
        if has_exact {
            for &(d, idx) in neighbors {
                if d < f64::EPSILON {
                    let class = target[idx] as usize;
                    let w = weights[idx];
                    if class < votes.len() {
                        votes[class] += w;
                    }
                }
            }
        } else {
            for &(d, idx) in neighbors {
                let class = target[idx] as usize;
                let w = weights[idx];
                if class < votes.len() {
                    votes[class] += w / d;
                }
            }
        }
    } else {
        for &(_, idx) in neighbors {
            let class = target[idx] as usize;
            let w = weights[idx];
            if class < votes.len() {
                votes[class] += w;
            }
        }
    }

    votes
}

/// Aggregate predictions for regression.
fn aggregate_regression(
    neighbors: &[(f64, usize)],
    target: &[f64],
    use_actual_dist: bool,
    k: usize,
) -> f64 {
    if use_actual_dist {
        let has_exact = neighbors.iter().any(|&(d, _)| d < f64::EPSILON);
        if has_exact {
            let (sum, count) = neighbors.iter().fold((0.0, 0usize), |(s, c), &(d, idx)| {
                if d < f64::EPSILON {
                    (s + target[idx], c + 1)
                } else {
                    (s, c)
                }
            });
            sum / count as f64
        } else {
            let (weighted_sum, total_w) =
                neighbors.iter().fold((0.0, 0.0), |(ws, tw), &(d, idx)| {
                    let w = 1.0 / d;
                    (ws + w * target[idx], tw + w)
                });
            weighted_sum / total_w
        }
    } else {
        let sum: f64 = neighbors.iter().map(|&(_, idx)| target[idx]).sum();
        sum / k as f64
    }
}

// ─────────────────────────────────────────────────────────────────
// Distance functions
// ─────────────────────────────────────────────────────────────────

/// Compute distance for comparison purposes (skips sqrt for Euclidean).
///
/// For Euclidean, returns squared distance (monotonic — preserves ordering).
/// For Manhattan and Cosine, returns the actual distance.
#[inline]
fn distance_for_compare(a: &[f64], b: &[f64], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => euclidean_sq(a, b),
        DistanceMetric::Manhattan => manhattan(a, b),
        DistanceMetric::Cosine => cosine_distance(a, b),
    }
}

/// Compute the actual distance (with sqrt for Euclidean).
///
/// Used when `WeightFunction::Distance` is active, since we need true
/// distances for the `1/d` weighting.
#[inline]
fn actual_distance(a: &[f64], b: &[f64], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => euclidean_sq(a, b).sqrt(),
        DistanceMetric::Manhattan => manhattan(a, b),
        DistanceMetric::Cosine => cosine_distance(a, b),
    }
}

/// Convert a sparse row view to a dense vector.
fn sparse_row_to_dense(row: &SparseRow<'_>, n_cols: usize) -> Vec<f64> {
    let mut dense = vec![0.0; n_cols];
    for (col, val) in row.iter() {
        dense[col] = val;
    }
    dense
}

/// Compute sparse distance for comparison (skips sqrt for Euclidean).
#[inline]
fn sparse_distance_for_compare(
    a: &SparseRow<'_>,
    b: &SparseRow<'_>,
    metric: DistanceMetric,
) -> f64 {
    match metric {
        DistanceMetric::Euclidean => sparse_euclidean_sq(a, b),
        DistanceMetric::Manhattan => sparse_manhattan(a, b),
        DistanceMetric::Cosine => sparse_cosine(a, b),
    }
}

/// Compute actual sparse distance (with sqrt for Euclidean).
#[inline]
fn sparse_actual_distance(a: &SparseRow<'_>, b: &SparseRow<'_>, metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => sparse_euclidean_sq(a, b).sqrt(),
        DistanceMetric::Manhattan => sparse_manhattan(a, b),
        DistanceMetric::Cosine => sparse_cosine(a, b),
    }
}

/// Brute-force k-nearest on sparse training data.
fn sparse_brute_force(
    query: &SparseRow<'_>,
    train: &CsrMatrix,
    k: usize,
    metric: DistanceMetric,
    use_actual_dist: bool,
) -> Vec<(f64, usize)> {
    let n = train.n_rows();
    let mut dists: Vec<(f64, usize)> = (0..n)
        .map(|i| {
            let train_row = train.row(i);
            let d = if use_actual_dist {
                sparse_actual_distance(query, &train_row, metric)
            } else {
                sparse_distance_for_compare(query, &train_row, metric)
            };
            (d, i)
        })
        .collect();

    if k < dists.len() {
        dists.select_nth_unstable_by(k - 1, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    dists.truncate(k);
    dists.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    dists
}

/// Decide whether to use the KD-tree based on algorithm selection, metric, and dimensionality.
fn should_use_kdtree(algo: Algorithm, metric: DistanceMetric, n_features: usize) -> bool {
    match algo {
        Algorithm::BruteForce => false,
        Algorithm::KDTree => matches!(metric, DistanceMetric::Euclidean),
        Algorithm::Auto => matches!(metric, DistanceMetric::Euclidean) && n_features < 20,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_simple() {
        // Two clusters: class 0 near origin, class 1 near (10, 10).
        let features = vec![
            vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut knn = KnnClassifier::new().k(3);
        knn.fit(&data).unwrap();

        let preds = knn.predict(&[vec![1.0, 1.0], vec![9.0, 9.0]]).unwrap();
        assert!((preds[0] - 0.0).abs() < 1e-6);
        assert!((preds[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_knn_distance_weights() {
        // 3 class-0 samples far away, 2 class-1 samples very close to query.
        // With distance weights, class 1 should win (closer neighbors dominate).
        // Query at x=0.15: class-1 at 0.1 (d=0.05), 0.2 (d=0.05); class-0 at 5, 10, 10 (far).
        let features = vec![vec![5.0, 10.0, 10.0, 0.1, 0.2]];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into()], "class");

        let mut knn_dist = KnnClassifier::new().k(5).weights(WeightFunction::Distance);
        knn_dist.fit(&data).unwrap();
        let preds_d = knn_dist.predict(&[vec![0.15]]).unwrap();
        assert_eq!(
            preds_d[0] as usize, 1,
            "Distance-weighted should pick closer class 1"
        );
    }

    #[test]
    fn test_knn_predict_proba() {
        let features = vec![vec![0.0, 0.0, 10.0, 10.0], vec![0.0, 0.0, 10.0, 10.0]];
        let target = vec![0.0, 0.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut knn = KnnClassifier::new().k(4);
        knn.fit(&data).unwrap();

        let probas = knn
            .predict_proba(&[vec![1.0, 1.0], vec![5.0, 5.0]])
            .unwrap();
        for p in &probas {
            let sum: f64 = p.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "Probabilities must sum to 1.0, got {sum}"
            );
        }

        // Point near class 0 should have higher probability for class 0.
        assert!(
            probas[0][0] > 0.4,
            "Expected high prob for class 0 at (1,1)"
        );
    }

    #[test]
    fn test_knn_cosine() {
        // Cosine distance ignores magnitude — direction matters.
        // [1, 0] and [100, 0] have same direction → distance ≈ 0.
        // [1, 0] and [0, 1] are orthogonal → distance ≈ 1.
        let d_same = cosine_distance(&[1.0, 0.0], &[100.0, 0.0]);
        let d_orth = cosine_distance(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(
            d_same < 1e-9,
            "Same direction should have ~0 distance, got {d_same}"
        );
        assert!(
            (d_orth - 1.0).abs() < 1e-9,
            "Orthogonal should have distance ~1, got {d_orth}"
        );

        // Use cosine metric in classifier.
        let features = vec![vec![1.0, 100.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 100.0]];
        let target = vec![0.0, 0.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut knn = KnnClassifier::new().k(2).metric(DistanceMetric::Cosine);
        knn.fit(&data).unwrap();

        // Query [50, 0] has same direction as class 0.
        let preds = knn.predict(&[vec![50.0, 0.0]]).unwrap();
        assert_eq!(
            preds[0] as usize, 0,
            "Cosine metric should match class 0 by direction"
        );
    }

    #[test]
    fn test_knn_regressor_simple() {
        // 3 points: x=1→y=10, x=5→y=50, x=9→y=90
        let features = vec![vec![1.0, 5.0, 9.0]];
        let target = vec![10.0, 50.0, 90.0];
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut knn = KnnRegressor::new().k(2);
        knn.fit(&data).unwrap();

        // Query x=3: nearest are x=1(y=10) and x=5(y=50) → mean=30
        let preds = knn.predict(&[vec![3.0]]).unwrap();
        assert!(
            (preds[0] - 30.0).abs() < 1e-9,
            "Expected 30.0, got {}",
            preds[0]
        );

        // Query x=7: nearest are x=5(y=50) and x=9(y=90) → mean=70
        let preds2 = knn.predict(&[vec![7.0]]).unwrap();
        assert!(
            (preds2[0] - 70.0).abs() < 1e-9,
            "Expected 70.0, got {}",
            preds2[0]
        );
    }

    #[test]
    fn test_knn_regressor_distance_weights() {
        // x=0→y=0, x=10→y=100. Query at x=1 (much closer to x=0).
        // Uniform: mean(0, 100) = 50.
        // Distance: weighted toward x=0 → should be << 50.
        let features = vec![vec![0.0, 10.0]];
        let target = vec![0.0, 100.0];
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut knn_u = KnnRegressor::new().k(2);
        knn_u.fit(&data).unwrap();
        let pred_u = knn_u.predict(&[vec![1.0]]).unwrap()[0];
        assert!((pred_u - 50.0).abs() < 1e-9, "Uniform should give 50.0");

        let mut knn_d = KnnRegressor::new().k(2).weights(WeightFunction::Distance);
        knn_d.fit(&data).unwrap();
        let pred_d = knn_d.predict(&[vec![1.0]]).unwrap()[0];
        // 1/1 * 0 + 1/9 * 100 = 11.11... / (1 + 0.111...) = ~10
        assert!(
            pred_d < 20.0,
            "Distance-weighted should favor x=0, got {pred_d}"
        );
    }

    #[test]
    fn test_knn_not_fitted() {
        let knn = KnnClassifier::new();
        assert!(knn.predict(&[vec![1.0]]).is_err());
        assert!(knn.predict_proba(&[vec![1.0]]).is_err());

        let knn_r = KnnRegressor::new();
        assert!(knn_r.predict(&[vec![1.0]]).is_err());
    }

    #[test]
    fn test_knn_predict_sparse_matches_dense() {
        let features = vec![
            vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut knn = KnnClassifier::new().k(3);
        knn.fit(&data).unwrap();

        let test = vec![vec![1.0, 1.0], vec![9.0, 9.0]];
        let preds_dense = knn.predict(&test).unwrap();
        let csr = CsrMatrix::from_dense(&test);
        let preds_sparse = knn.predict_sparse(&csr).unwrap();

        for (d, s) in preds_dense.iter().zip(preds_sparse.iter()) {
            assert!((d - s).abs() < 1e-6, "Dense={d} vs Sparse={s}");
        }
    }

    #[test]
    fn test_knn_regressor_predict_sparse() {
        let features = vec![vec![1.0, 5.0, 9.0]];
        let target = vec![10.0, 50.0, 90.0];
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut knn = KnnRegressor::new().k(2);
        knn.fit(&data).unwrap();

        let test = vec![vec![3.0], vec![7.0]];
        let preds_dense = knn.predict(&test).unwrap();
        let csr = CsrMatrix::from_dense(&test);
        let preds_sparse = knn.predict_sparse(&csr).unwrap();

        for (d, s) in preds_dense.iter().zip(preds_sparse.iter()) {
            assert!((d - s).abs() < 1e-6, "Dense={d} vs Sparse={s}");
        }
    }

    #[test]
    fn test_sparse_euclidean_matches_dense() {
        // Dense: d²([1,0,3], [0,2,3]) = 1 + 4 + 0 = 5
        let a = CsrMatrix::from_dense(&[vec![1.0, 0.0, 3.0]]);
        let b = CsrMatrix::from_dense(&[vec![0.0, 2.0, 3.0]]);
        let d2 = sparse_euclidean_sq(&a.row(0), &b.row(0));
        assert!((d2 - 5.0).abs() < 1e-10, "Expected 5.0, got {d2}");
    }

    #[test]
    fn test_sparse_manhattan_matches_dense() {
        // Dense: d([1,0,3], [0,2,3]) = 1 + 2 + 0 = 3
        let a = CsrMatrix::from_dense(&[vec![1.0, 0.0, 3.0]]);
        let b = CsrMatrix::from_dense(&[vec![0.0, 2.0, 3.0]]);
        let d = sparse_manhattan(&a.row(0), &b.row(0));
        assert!((d - 3.0).abs() < 1e-10, "Expected 3.0, got {d}");
    }

    #[test]
    fn test_sparse_cosine_matches_dense() {
        // Same direction → distance ≈ 0
        let a = CsrMatrix::from_dense(&[vec![1.0, 0.0]]);
        let b = CsrMatrix::from_dense(&[vec![100.0, 0.0]]);
        let d = sparse_cosine(&a.row(0), &b.row(0));
        assert!(d < 1e-9, "Same direction should be ~0, got {d}");

        // Orthogonal → distance ≈ 1
        let c = CsrMatrix::from_dense(&[vec![0.0, 1.0]]);
        let d_orth = sparse_cosine(&a.row(0), &c.row(0));
        assert!(
            (d_orth - 1.0).abs() < 1e-9,
            "Orthogonal should be ~1, got {d_orth}"
        );
    }

    #[test]
    fn test_sparse_knn_end_to_end() {
        // Train on dense, predict_sparse with CSR — results should match.
        use crate::sparse::CscMatrix;
        let features = vec![
            vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let data = Dataset::new(
            features.clone(),
            target.clone(),
            vec!["x".into(), "y".into()],
            "class",
        );

        // Fit on dense.
        let mut knn_dense = KnnClassifier::new().k(3);
        knn_dense.fit(&data).unwrap();

        // Fit on sparse.
        let csc = CscMatrix::from_dense(&features);
        let data_sparse = Dataset::from_sparse(csc, target, vec!["x".into(), "y".into()], "class");
        let mut knn_sparse = KnnClassifier::new().k(3);
        knn_sparse.fit(&data_sparse).unwrap();
        assert!(knn_sparse.train_sparse.is_some());

        // Query with sparse input.
        let test = vec![vec![1.0, 1.0], vec![9.0, 9.0]];
        let preds_dense = knn_dense.predict(&test).unwrap();
        let csr = CsrMatrix::from_dense(&test);
        let preds_sparse = knn_sparse.predict_sparse(&csr).unwrap();

        for (d, s) in preds_dense.iter().zip(preds_sparse.iter()) {
            assert!((d - s).abs() < 1e-6, "Dense={d} vs Sparse={s}");
        }
    }

    #[test]
    fn test_high_dimensional_sparse_knn() {
        // 100×5000 matrix with ~2% density — should complete without OOM.
        // (Would require 100 × 400KB per query if densifying.)
        use crate::sparse::CscMatrix;
        let n_train = 100;
        let n_feat = 5000;
        let mut rng = crate::rng::FastRng::new(42);

        // Build sparse training data as column-major.
        let mut cols: Vec<Vec<f64>> = vec![vec![0.0; n_train]; n_feat];
        for col in &mut cols {
            for x in col.iter_mut() {
                if rng.f64() < 0.02 {
                    *x = rng.f64() * 10.0;
                }
            }
        }
        let target: Vec<f64> = (0..n_train).map(|i| (i % 3) as f64).collect();
        let csc = CscMatrix::from_dense(&cols);
        let names: Vec<String> = (0..n_feat).map(|j| format!("f{j}")).collect();
        let data = Dataset::from_sparse(csc, target, names, "class");

        let mut knn = KnnClassifier::new().k(5);
        knn.fit(&data).unwrap();
        assert!(knn.train_sparse.is_some());

        // Build sparse query.
        let mut query_row = vec![0.0; n_feat];
        for x in &mut query_row {
            if rng.f64() < 0.02 {
                *x = rng.f64() * 10.0;
            }
        }
        let query_csr = CsrMatrix::from_dense(&[query_row]);
        let preds = knn.predict_sparse(&query_csr).unwrap();
        assert_eq!(preds.len(), 1);
        assert!(preds[0] >= 0.0 && preds[0] < 3.0);
    }
}

#[cfg(all(test, feature = "scry-gpu"))]
mod gpu_tests {
    use super::*;

    #[test]
    fn gpu_knn_classifier_batched_matches_scalar() {
        // 100 training samples × 5 features, 10 queries → 1000 pairs (above 256 threshold)
        let n_train = 100;
        let n_feat = 5;
        let mut features_col: Vec<Vec<f64>> = Vec::with_capacity(n_feat);
        for j in 0..n_feat {
            let col: Vec<f64> = (0..n_train)
                .map(|i| ((i * (j + 3)) % 37) as f64 * 0.5)
                .collect();
            features_col.push(col);
        }
        let target: Vec<f64> = (0..n_train).map(|i| (i % 3) as f64).collect();
        let names: Vec<String> = (0..n_feat).map(|j| format!("f{j}")).collect();
        let data = Dataset::new(features_col, target, names, "class");

        let mut knn = KnnClassifier::new().k(5).algorithm(Algorithm::BruteForce);
        knn.fit(&data).unwrap();

        // 10 queries — enough to trigger batched path
        let queries: Vec<Vec<f64>> = (0..10)
            .map(|i| (0..n_feat).map(|j| ((i + j) % 17) as f64 * 0.3).collect())
            .collect();

        let preds = knn.predict(&queries).unwrap();
        assert_eq!(preds.len(), 10);
        for p in &preds {
            assert!(
                *p >= 0.0 && *p < 3.0,
                "prediction must be a valid class: {p}"
            );
        }
    }

    #[test]
    fn gpu_knn_regressor_batched_matches_scalar() {
        let n_train = 100;
        let n_feat = 5;
        let mut features_col: Vec<Vec<f64>> = Vec::with_capacity(n_feat);
        for j in 0..n_feat {
            let col: Vec<f64> = (0..n_train)
                .map(|i| ((i * (j + 2)) % 41) as f64 * 0.2)
                .collect();
            features_col.push(col);
        }
        let target: Vec<f64> = (0..n_train).map(|i| (i % 50) as f64).collect();
        let names: Vec<String> = (0..n_feat).map(|j| format!("f{j}")).collect();
        let data = Dataset::new(features_col, target, names, "y");

        let mut knn = KnnRegressor::new().k(5).algorithm(Algorithm::BruteForce);
        knn.fit(&data).unwrap();

        let queries: Vec<Vec<f64>> = (0..10)
            .map(|i| (0..n_feat).map(|j| ((i + j) % 19) as f64 * 0.4).collect())
            .collect();

        let preds = knn.predict(&queries).unwrap();
        assert_eq!(preds.len(), 10);
        for p in &preds {
            assert!(p.is_finite(), "prediction must be finite: {p}");
        }
    }
}
