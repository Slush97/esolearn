// SPDX-License-Identifier: MIT OR Apache-2.0
//! Randomized search with cross-validation.

use std::collections::HashMap;

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::metrics::accuracy;
use crate::split::{k_fold, stratified_k_fold, ScoringFn};

use super::{cartesian_product, evaluate_combo, CvResult, ParamGrid, ParamValue, Tunable};

/// Randomized search over a hyperparameter grid with cross-validation.
///
/// Samples `n_iter` random combinations from the grid instead of trying
/// every one — much faster for large grids.
///
/// # Examples
///
/// ```ignore
/// use scry_learn::prelude::*;
/// use scry_learn::search::*;
///
/// let mut grid = ParamGrid::new();
/// grid.insert("max_depth".into(), vec![
///     ParamValue::Int(2), ParamValue::Int(4),
///     ParamValue::Int(6), ParamValue::Int(8),
/// ]);
///
/// let result = RandomizedSearchCV::new(DecisionTreeClassifier::new(), grid)
///     .n_iter(5)
///     .cv(3)
///     .fit(&data)
///     .unwrap();
/// ```
#[non_exhaustive]
pub struct RandomizedSearchCV {
    base_model: Box<dyn Tunable>,
    param_grid: ParamGrid,
    n_iter: usize,
    cv: usize,
    scorer: ScoringFn,
    seed: u64,
    stratified: bool,
    best_params_: Option<HashMap<String, ParamValue>>,
    best_score_: f64,
    cv_results_: Vec<CvResult>,
}

impl RandomizedSearchCV {
    /// Create a randomized search with `n_iter` random samples.
    ///
    /// Defaults: 10 iterations, 5-fold CV, accuracy scorer, seed 42, non-stratified.
    pub fn new(model: impl Tunable + 'static, grid: ParamGrid) -> Self {
        Self {
            base_model: Box::new(model),
            param_grid: grid,
            n_iter: 10,
            cv: 5,
            scorer: accuracy,
            seed: 42,
            stratified: false,
            best_params_: None,
            best_score_: f64::NEG_INFINITY,
            cv_results_: Vec::new(),
        }
    }

    /// Set the number of random combinations to try (default: 10).
    pub fn n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }

    /// Set the number of cross-validation folds (default: 5).
    pub fn cv(mut self, k: usize) -> Self {
        self.cv = k;
        self
    }

    /// Set the scoring function (default: `accuracy`).
    pub fn scoring(mut self, scorer: ScoringFn) -> Self {
        self.scorer = scorer;
        self
    }

    /// Set the random seed (default: 42).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Enable stratified k-fold CV (default: `false`).
    ///
    /// When `true`, uses [`stratified_k_fold`](crate::split::stratified_k_fold)
    /// to preserve class proportions in each fold.
    pub fn stratified(mut self, stratified: bool) -> Self {
        self.stratified = stratified;
        self
    }

    /// Run the randomized search.
    ///
    /// Samples up to `n_iter` random parameter combinations from the grid.
    pub fn fit(mut self, data: &Dataset) -> Result<Self> {
        if self.cv < 2 {
            return Err(ScryLearnError::InvalidParameter(format!(
                "cv must be >= 2, got {}",
                self.cv
            )));
        }
        let all_combos = cartesian_product(&self.param_grid);
        if all_combos.is_empty() {
            return Err(ScryLearnError::InvalidParameter(
                "parameter grid is empty".into(),
            ));
        }

        let folds = if self.stratified {
            stratified_k_fold(data, self.cv, self.seed)
        } else {
            k_fold(data, self.cv, self.seed)
        };
        let mut rng = crate::rng::FastRng::new(self.seed);

        // Sample n_iter unique combos (or all if grid is smaller).
        let n = self.n_iter.min(all_combos.len());
        let mut indices: Vec<usize> = (0..all_combos.len()).collect();
        // Fisher-Yates shuffle and take first n.
        for i in (1..indices.len()).rev() {
            let j = rng.usize(0..=i);
            indices.swap(i, j);
        }

        for &idx in &indices[..n] {
            let combo = &all_combos[idx];
            let result = evaluate_combo(&*self.base_model, combo, &folds, self.scorer)?;

            if result.mean_score.is_finite()
                && (self.best_params_.is_none() || result.mean_score > self.best_score_)
            {
                self.best_score_ = result.mean_score;
                self.best_params_ = Some(result.params.clone());
            }
            self.cv_results_.push(result);
        }

        if self.best_params_.is_none() {
            return Err(ScryLearnError::InvalidParameter(
                "all parameter combinations produced NaN scores".into(),
            ));
        }

        Ok(self)
    }

    /// The best parameter combination found.
    ///
    /// # Panics
    ///
    /// Panics if called before [`fit`](Self::fit).
    pub fn best_params(&self) -> &HashMap<String, ParamValue> {
        self.best_params_.as_ref().expect("call fit() first")
    }

    /// The best mean CV score achieved.
    pub fn best_score(&self) -> f64 {
        self.best_score_
    }

    /// All evaluated combinations with their scores.
    pub fn cv_results(&self) -> &[CvResult] {
        &self.cv_results_
    }
}
