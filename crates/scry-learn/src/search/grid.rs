// SPDX-License-Identifier: MIT OR Apache-2.0
//! Exhaustive grid search with cross-validation.

use std::collections::HashMap;

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::metrics::accuracy;
use crate::split::{k_fold, stratified_k_fold, ScoringFn};

use super::{cartesian_product, evaluate_combo, CvResult, ParamGrid, ParamValue, Tunable};

/// Exhaustive search over a hyperparameter grid with cross-validation.
///
/// Tries every combination in the grid, evaluates each with k-fold CV,
/// and reports the best-performing parameter set.
///
/// # Examples
///
/// ```ignore
/// use scry_learn::prelude::*;
/// use scry_learn::search::*;
///
/// let mut grid = ParamGrid::new();
/// grid.insert("max_depth".into(), vec![
///     ParamValue::Int(2), ParamValue::Int(4), ParamValue::Int(8),
/// ]);
///
/// let result = GridSearchCV::new(DecisionTreeClassifier::new(), grid)
///     .cv(5)
///     .scoring(accuracy)
///     .fit(&data)
///     .unwrap();
///
/// println!("Best: {:?} → {:.3}", result.best_params(), result.best_score());
/// ```
#[non_exhaustive]
pub struct GridSearchCV {
    base_model: Box<dyn Tunable>,
    param_grid: ParamGrid,
    cv: usize,
    scorer: ScoringFn,
    seed: u64,
    stratified: bool,
    // Results (populated after fit)
    best_params_: Option<HashMap<String, ParamValue>>,
    best_score_: f64,
    cv_results_: Vec<CvResult>,
}

impl GridSearchCV {
    /// Create a grid search over the given model and parameter grid.
    ///
    /// Defaults: 5-fold CV, accuracy scorer, seed 42, non-stratified.
    pub fn new(model: impl Tunable + 'static, grid: ParamGrid) -> Self {
        Self {
            base_model: Box::new(model),
            param_grid: grid,
            cv: 5,
            scorer: accuracy,
            seed: 42,
            stratified: false,
            best_params_: None,
            best_score_: f64::NEG_INFINITY,
            cv_results_: Vec::new(),
        }
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

    /// Set the random seed for fold generation (default: 42).
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

    /// Run the exhaustive grid search.
    ///
    /// Returns `self` for chained accessor calls.
    pub fn fit(mut self, data: &Dataset) -> Result<Self> {
        if self.cv < 2 {
            return Err(ScryLearnError::InvalidParameter(format!(
                "cv must be >= 2, got {}",
                self.cv
            )));
        }
        let combos = cartesian_product(&self.param_grid);
        if combos.is_empty() {
            return Err(ScryLearnError::InvalidParameter(
                "parameter grid is empty".into(),
            ));
        }

        let folds = if self.stratified {
            stratified_k_fold(data, self.cv, self.seed)
        } else {
            k_fold(data, self.cv, self.seed)
        };

        for combo in &combos {
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
