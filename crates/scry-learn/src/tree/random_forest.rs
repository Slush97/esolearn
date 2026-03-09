// SPDX-License-Identifier: MIT OR Apache-2.0
//! Random Forest — parallel ensemble of CART decision trees.
//!
//! Uses bootstrap sampling and random feature subsets for each tree,
//! trained in parallel via rayon.

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::tree::cart::{DecisionTreeClassifier, DecisionTreeRegressor};
use crate::weights::ClassWeight;
use rayon::prelude::*;

/// Strategy for selecting the number of features per split.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum MaxFeatures {
    /// `√n_features` (default for classification).
    Sqrt,
    /// `log₂(n_features)`.
    Log2,
    /// Use all features (no bagging).
    All,
    /// A fixed count.
    Fixed(usize),
}

impl MaxFeatures {
    fn resolve(self, n_features: usize) -> usize {
        match self {
            Self::Sqrt => (n_features as f64).sqrt().ceil() as usize,
            Self::Log2 => (n_features as f64).log2().ceil() as usize,
            Self::All => n_features,
            Self::Fixed(n) => n.min(n_features),
        }
        .max(1)
    }
}

// ---------------------------------------------------------------------------
// Random Forest Classifier
// ---------------------------------------------------------------------------

/// Random Forest for classification.
///
/// Trains an ensemble of decision trees in parallel, each on a bootstrap
/// sample with a random subset of features. Predictions are by majority vote.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct RandomForestClassifier {
    n_estimators: usize,
    max_depth: Option<usize>,
    max_features: MaxFeatures,
    min_samples_split: usize,
    min_samples_leaf: usize,
    bootstrap: bool,
    seed: u64,
    class_weight: ClassWeight,
    trees: Vec<DecisionTreeClassifier>,
    n_classes: usize,
    n_features: usize,
    feature_importances_: Vec<f64>,
    oob_score_: Option<f64>,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl RandomForestClassifier {
    /// Create a new random forest with default parameters.
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            max_depth: None,
            max_features: MaxFeatures::Sqrt,
            min_samples_split: 2,
            min_samples_leaf: 1,
            bootstrap: true,
            seed: 42,
            class_weight: ClassWeight::Uniform,
            trees: Vec::new(),
            n_classes: 0,
            n_features: 0,
            feature_importances_: Vec::new(),
            oob_score_: None,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set number of trees.
    pub fn n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Set maximum depth per tree.
    pub fn max_depth(mut self, d: usize) -> Self {
        self.max_depth = Some(d);
        self
    }

    /// Set feature selection strategy.
    pub fn max_features(mut self, mf: MaxFeatures) -> Self {
        self.max_features = mf;
        self
    }

    /// Set minimum samples to split.
    pub fn min_samples_split(mut self, n: usize) -> Self {
        self.min_samples_split = n;
        self
    }

    /// Set minimum samples per leaf.
    pub fn min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }

    /// Enable/disable bootstrap sampling.
    pub fn bootstrap(mut self, b: bool) -> Self {
        self.bootstrap = b;
        self
    }

    /// Set the random seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Set class weighting strategy for imbalanced datasets.
    pub fn class_weight(mut self, cw: ClassWeight) -> Self {
        self.class_weight = cw;
        self
    }

    /// Train the random forest.
    ///
    /// OOB votes are accumulated into a shared atomic array during parallel build,
    /// avoiding retention of per-tree vote arrays or bootstrap indices.
    /// Dataset indices are pre-sorted once and shared across all trees to avoid
    /// per-tree `sorted_by_feature` allocation (~6 MB savings with 16 threads).
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        use std::sync::atomic::{AtomicU32, Ordering};

        if data.n_samples() == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        self.n_features = data.n_features();
        self.n_classes = data.n_classes();
        let max_feats = self.max_features.resolve(self.n_features);
        let do_bootstrap = self.bootstrap;
        let n_samples = data.n_samples();
        let n_classes = self.n_classes;
        let feature_matrix = data.feature_matrix();
        let n_features = data.n_features();

        // Pre-sort ALL dataset indices by each feature once (shared read-only).
        // Each tree filters via membership bitset for its bootstrap sample.
        let global_sorted: Vec<Vec<usize>> = (0..n_features)
            .map(|feat_idx| {
                let col = &data.features[feat_idx];
                let mut sorted: Vec<usize> = (0..n_samples).collect();
                sorted.sort_unstable_by(|&a, &b| {
                    col[a]
                        .partial_cmp(&col[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                sorted
            })
            .collect();
        let global_sorted_ref = &global_sorted;

        // Shared OOB accumulator: oob_votes[sample * n_classes + class].
        // Atomic u32 so multiple threads can update without locking.
        let oob_votes: Vec<AtomicU32> = (0..n_samples * n_classes)
            .map(|_| AtomicU32::new(0))
            .collect();
        let oob_votes_ref = &oob_votes;

        // Train trees in parallel. OOB votes are merged directly into the
        // shared accumulator — no per-tree vote arrays are ever stored.
        let mut trees: Vec<DecisionTreeClassifier> = (0..self.n_estimators)
            .into_par_iter()
            .map(|tree_idx| {
                let mut rng = crate::rng::FastRng::new(self.seed.wrapping_add(tree_idx as u64));
                let n = n_samples;

                // Bootstrap sample.
                let indices: Vec<usize> = if do_bootstrap {
                    (0..n).map(|_| rng.usize(0..n)).collect()
                } else {
                    (0..n).collect()
                };

                let mut tree = DecisionTreeClassifier::new()
                    .max_features(max_feats)
                    .min_samples_split(self.min_samples_split)
                    .min_samples_leaf(self.min_samples_leaf)
                    .class_weight(self.class_weight.clone());

                if let Some(d) = self.max_depth {
                    tree = tree.max_depth(d);
                }

                // Train using shared pre-sorted indices — no per-tree sort allocation.
                tree.fit_on_indices_presorted(data, &indices, global_sorted_ref)
                    .ok();

                // Compute OOB votes inline and merge into shared accumulator.
                // Bootstrap indices and bitset are dropped at end of closure.
                if do_bootstrap {
                    if let Some(ref ft) = tree.flat_tree {
                        // Build compact bitset of in-bag samples.
                        let n_words = n.div_ceil(64);
                        let mut in_bag = vec![0u64; n_words];
                        for &idx in &indices {
                            in_bag[idx / 64] |= 1u64 << (idx % 64);
                        }

                        // Vote for OOB samples, merging directly into shared accumulator.
                        for sample_idx in 0..n {
                            if in_bag[sample_idx / 64] & (1u64 << (sample_idx % 64)) != 0 {
                                continue;
                            }
                            let pred = ft.predict_sample(&feature_matrix[sample_idx]) as usize;
                            if pred < n_classes {
                                oob_votes_ref[sample_idx * n_classes + pred]
                                    .fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }

                tree
            })
            .collect();

        // Aggregate feature importances.
        self.feature_importances_ = vec![0.0; self.n_features];
        for tree in &trees {
            if let Ok(imp) = tree.feature_importances() {
                for (i, &v) in imp.iter().enumerate() {
                    self.feature_importances_[i] += v;
                }
            }
        }
        let n_trees = trees.len() as f64;
        for imp in &mut self.feature_importances_ {
            *imp /= n_trees;
        }

        // Compute OOB accuracy from accumulated atomic votes.
        self.oob_score_ = if do_bootstrap {
            // Convert atomics to plain u32 for scoring.
            let totals: Vec<u32> = oob_votes
                .iter()
                .map(|a| a.load(Ordering::Relaxed))
                .collect();
            Self::oob_accuracy_from_votes(&totals, n_samples, n_classes, &data.target)
        } else {
            None
        };

        // Clear per-tree training-only data to save memory.
        for tree in &mut trees {
            tree.sample_weights = None;
            tree.feature_importances_ = Vec::new();
        }

        self.trees = trees;
        Ok(())
    }

    /// Compute OOB accuracy from flat vote accumulation array.
    fn oob_accuracy_from_votes(
        oob_total: &[u32],
        n_samples: usize,
        n_classes: usize,
        target: &[f64],
    ) -> Option<f64> {
        let mut correct = 0usize;
        let mut total = 0usize;
        for sample_idx in 0..n_samples {
            let row = &oob_total[sample_idx * n_classes..(sample_idx + 1) * n_classes];
            let vote_count: u32 = row.iter().sum();
            if vote_count == 0 {
                continue;
            }
            let predicted_class = row
                .iter()
                .enumerate()
                .max_by_key(|&(_, &v)| v)
                .map_or(0, |(idx, _)| idx);
            let true_class = target[sample_idx] as usize;
            if predicted_class == true_class {
                correct += 1;
            }
            total += 1;
        }

        if total > 0 {
            Some(correct as f64 / total as f64)
        } else {
            None
        }
    }

    /// Predict class labels by majority vote.
    ///
    /// Uses `FlatTree::predict_sample` for cache-optimal traversal.
    /// Parallelized across samples via rayon.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        crate::version::check_schema_version(self._schema_version)?;
        if self.trees.is_empty() {
            return Err(ScryLearnError::NotFitted);
        }

        let n_classes = self.n_classes;
        let predictions: Vec<f64> = features
            .par_iter()
            .map(|sample| {
                let mut votes = vec![0usize; n_classes];
                for tree in &self.trees {
                    if let Some(ref ft) = tree.flat_tree {
                        let class = ft.predict_sample(sample) as usize;
                        if class < n_classes {
                            votes[class] += 1;
                        }
                    }
                }
                votes
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, &v)| v)
                    .map_or(0.0, |(idx, _)| idx as f64)
            })
            .collect();

        Ok(predictions)
    }

    /// Predict class probabilities (average across trees).
    ///
    /// Parallelized across samples via rayon.
    pub fn predict_proba(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if self.trees.is_empty() {
            return Err(ScryLearnError::NotFitted);
        }

        let n_classes = self.n_classes;
        let n_trees = self.trees.len() as f64;

        let probas: Vec<Vec<f64>> = features
            .par_iter()
            .map(|sample| {
                let mut proba = vec![0.0; n_classes];
                for tree in &self.trees {
                    if let Some(ref ft) = tree.flat_tree {
                        let tree_proba = ft.predict_proba_sample(sample, n_classes);
                        for (j, p) in tree_proba.into_iter().enumerate() {
                            if j < n_classes {
                                proba[j] += p;
                            }
                        }
                    }
                }
                for p in &mut proba {
                    *p /= n_trees;
                }
                proba
            })
            .collect();

        Ok(probas)
    }

    /// Feature importances averaged across all trees.
    pub fn feature_importances(&self) -> Result<Vec<f64>> {
        if self.trees.is_empty() {
            return Err(ScryLearnError::NotFitted);
        }
        Ok(self.feature_importances_.clone())
    }

    /// Out-of-bag accuracy score (available after fit with bootstrap=true).
    pub fn oob_score(&self) -> Option<f64> {
        self.oob_score_
    }

    /// Number of trained trees.
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Get individual trees (for visualization or inspection).
    pub fn trees(&self) -> &[DecisionTreeClassifier] {
        &self.trees
    }

    /// Number of classes the model was trained on.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Number of features the model was trained on.
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

impl Default for RandomForestClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Random Forest Regressor
// ---------------------------------------------------------------------------

/// Random Forest for regression (mean of tree predictions).
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct RandomForestRegressor {
    n_estimators: usize,
    max_depth: Option<usize>,
    max_features: MaxFeatures,
    min_samples_split: usize,
    min_samples_leaf: usize,
    bootstrap: bool,
    seed: u64,
    trees: Vec<DecisionTreeRegressor>,
    n_features: usize,
    feature_importances_: Vec<f64>,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl RandomForestRegressor {
    /// Create a new regressor forest.
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            max_depth: None,
            max_features: MaxFeatures::All,
            min_samples_split: 2,
            min_samples_leaf: 1,
            bootstrap: true,
            seed: 42,
            trees: Vec::new(),
            n_features: 0,
            feature_importances_: Vec::new(),
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set number of trees.
    pub fn n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Set maximum depth.
    pub fn max_depth(mut self, d: usize) -> Self {
        self.max_depth = Some(d);
        self
    }

    /// Set feature selection strategy.
    pub fn max_features(mut self, mf: MaxFeatures) -> Self {
        self.max_features = mf;
        self
    }

    /// Set random seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Train the forest.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        if data.n_samples() == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        self.n_features = data.n_features();
        let max_feats = self.max_features.resolve(self.n_features);

        let mut trees: Vec<DecisionTreeRegressor> = (0..self.n_estimators)
            .into_par_iter()
            .map(|tree_idx| {
                let mut rng = crate::rng::FastRng::new(self.seed.wrapping_add(tree_idx as u64));
                let n = data.n_samples();

                let indices: Vec<usize> = if self.bootstrap {
                    (0..n).map(|_| rng.usize(0..n)).collect()
                } else {
                    (0..n).collect()
                };

                let mut tree = DecisionTreeRegressor::new()
                    .max_features(max_feats)
                    .min_samples_split(self.min_samples_split)
                    .min_samples_leaf(self.min_samples_leaf);

                if let Some(d) = self.max_depth {
                    tree = tree.max_depth(d);
                }

                // Train directly on indices — no data copy.
                tree.fit_on_indices(data, &indices).ok();
                tree
            })
            .collect();

        self.feature_importances_ = vec![0.0; self.n_features];
        for tree in &trees {
            if let Ok(imp) = tree.feature_importances() {
                for (i, &v) in imp.iter().enumerate() {
                    self.feature_importances_[i] += v;
                }
            }
        }
        let n_trees = trees.len() as f64;
        for imp in &mut self.feature_importances_ {
            *imp /= n_trees;
        }

        // Clear per-tree training-only data to save memory.
        for tree in &mut trees {
            tree.feature_importances_ = Vec::new();
        }

        self.trees = trees;
        Ok(())
    }

    /// Predict values (mean across trees).
    ///
    /// Parallelized across samples via rayon.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        crate::version::check_schema_version(self._schema_version)?;
        if self.trees.is_empty() {
            return Err(ScryLearnError::NotFitted);
        }

        let n_trees = self.trees.len() as f64;

        let predictions: Vec<f64> = features
            .par_iter()
            .map(|sample| {
                let mut sum = 0.0;
                for tree in &self.trees {
                    if let Some(ref ft) = tree.flat_tree {
                        sum += ft.predict_sample(sample);
                    }
                }
                sum / n_trees
            })
            .collect();

        Ok(predictions)
    }

    /// Feature importances.
    pub fn feature_importances(&self) -> Result<Vec<f64>> {
        if self.trees.is_empty() {
            return Err(ScryLearnError::NotFitted);
        }
        Ok(self.feature_importances_.clone())
    }

    /// Get individual trees (for inspection or ONNX export).
    pub fn trees(&self) -> &[DecisionTreeRegressor] {
        &self.trees
    }

    /// Number of features the model was trained on.
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

impl Default for RandomForestRegressor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_classification_data() -> Dataset {
        // Two features, clear separation.
        let n = 100;
        let mut f1 = Vec::with_capacity(n);
        let mut f2 = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        let mut rng = crate::rng::FastRng::new(42);

        for _ in 0..n / 2 {
            f1.push(rng.f64() * 3.0);
            f2.push(rng.f64() * 3.0);
            target.push(0.0);
        }
        for _ in 0..n / 2 {
            f1.push(rng.f64() * 3.0 + 5.0);
            f2.push(rng.f64() * 3.0 + 5.0);
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
    fn test_random_forest_classification() {
        let data = make_classification_data();
        let mut rf = RandomForestClassifier::new()
            .n_estimators(20)
            .max_depth(5)
            .seed(42);
        rf.fit(&data).unwrap();

        let matrix = data.feature_matrix();
        let preds = rf.predict(&matrix).unwrap();
        let acc = preds
            .iter()
            .zip(data.target.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-6)
            .count() as f64
            / data.n_samples() as f64;

        assert!(
            acc >= 0.90,
            "expected ≥90% accuracy, got {:.1}%",
            acc * 100.0
        );
    }

    #[test]
    fn test_feature_importances_valid() {
        let data = make_classification_data();
        let mut rf = RandomForestClassifier::new().n_estimators(10).seed(42);
        rf.fit(&data).unwrap();

        let imp = rf.feature_importances().unwrap();
        assert_eq!(imp.len(), 2);
        assert!(imp.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_predict_proba() {
        let data = make_classification_data();
        let mut rf = RandomForestClassifier::new().n_estimators(10).seed(42);
        rf.fit(&data).unwrap();

        let sample = vec![1.0, 1.0]; // should be class 0
        let proba = rf.predict_proba(&[sample]).unwrap();
        assert!(proba[0][0] > 0.5, "should predict class 0 with >50%");
    }

    #[test]
    fn test_oob_score_with_bootstrap() {
        let data = make_classification_data();
        let mut rf = RandomForestClassifier::new()
            .n_estimators(50)
            .max_depth(5)
            .bootstrap(true)
            .seed(42);
        rf.fit(&data).unwrap();

        let oob = rf.oob_score();
        assert!(
            oob.is_some(),
            "OOB score should be available with bootstrap=true"
        );
        let score = oob.unwrap();
        assert!(score >= 0.80, "expected OOB score ≥ 0.80, got {:.3}", score);
        assert!(score <= 1.0, "OOB score should be ≤ 1.0, got {:.3}", score);
    }

    #[test]
    fn test_oob_score_without_bootstrap() {
        let data = make_classification_data();
        let mut rf = RandomForestClassifier::new()
            .n_estimators(10)
            .bootstrap(false)
            .seed(42);
        rf.fit(&data).unwrap();

        assert!(
            rf.oob_score().is_none(),
            "OOB score should be None when bootstrap=false"
        );
    }
}
