// SPDX-License-Identifier: MIT OR Apache-2.0
//! Hyperparameter search via cross-validation.
//!
//! [`GridSearchCV`] performs exhaustive search over a parameter grid,
//! while [`RandomizedSearchCV`] samples random combinations for faster
//! exploration of large search spaces.
//!
//! # Examples
//!
//! ```ignore
//! use scry_learn::prelude::*;
//! use scry_learn::search::*;
//!
//! let mut grid = ParamGrid::new();
//! grid.insert("max_depth".into(), vec![ParamValue::Int(2), ParamValue::Int(6)]);
//!
//! let result = GridSearchCV::new(DecisionTreeClassifier::new(), grid)
//!     .cv(5)
//!     .scoring(accuracy)
//!     .fit(&data)
//!     .unwrap();
//!
//! println!("Best score: {}", result.best_score());
//! ```

mod bayes;
mod grid;
mod random;
mod tunable;

pub use bayes::{BayesSearchCV, ParamDistribution, ParamSpace};
pub use grid::GridSearchCV;
pub use random::RandomizedSearchCV;
pub use tunable::Tunable;

use std::collections::HashMap;

use crate::dataset::Dataset;
use crate::error::Result;
use crate::split::ScoringFn;

// ---------------------------------------------------------------------------
// ParamValue + ParamGrid
// ---------------------------------------------------------------------------

/// A single hyperparameter value.
///
/// # Examples
///
/// ```
/// use scry_learn::search::ParamValue;
///
/// let depth = ParamValue::Int(5);
/// let lr = ParamValue::Float(0.01);
/// let flag = ParamValue::Bool(true);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ParamValue {
    /// Integer parameter (e.g. `max_depth`, `n_estimators`).
    Int(usize),
    /// Floating-point parameter (e.g. `learning_rate`).
    Float(f64),
    /// Boolean parameter (e.g. `bootstrap`).
    Bool(bool),
    /// Categorical / string parameter (e.g. `criterion = "gini"`).
    Categorical(String),
}

impl std::fmt::Display for ParamValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamValue::Int(v) => write!(f, "{v}"),
            ParamValue::Float(v) => write!(f, "{v}"),
            ParamValue::Bool(v) => write!(f, "{v}"),
            ParamValue::Categorical(v) => write!(f, "{v}"),
        }
    }
}

/// A grid of hyperparameter values to search over.
///
/// Keys are parameter names (e.g. `"max_depth"`), values are lists of
/// candidate values to try.
///
/// # Examples
///
/// ```
/// use scry_learn::search::{ParamGrid, ParamValue};
///
/// let mut grid = ParamGrid::new();
/// grid.insert("max_depth".into(), vec![
///     ParamValue::Int(2),
///     ParamValue::Int(4),
///     ParamValue::Int(8),
/// ]);
/// ```
pub type ParamGrid = HashMap<String, Vec<ParamValue>>;

// ---------------------------------------------------------------------------
// CvResult
// ---------------------------------------------------------------------------

/// Result of a single parameter combination evaluated via cross-validation.
///
/// # Examples
///
/// ```ignore
/// for r in search_result.cv_results() {
///     println!("params={:?}  mean_score={:.3}", r.params, r.mean_score);
/// }
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CvResult {
    /// The parameter combination that was evaluated.
    pub params: HashMap<String, ParamValue>,
    /// Mean score across all CV folds.
    pub mean_score: f64,
    /// Individual fold scores.
    pub fold_scores: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate the cartesian product of all parameter lists.
pub(super) fn cartesian_product(grid: &ParamGrid) -> Vec<HashMap<String, ParamValue>> {
    let keys: Vec<&String> = grid.keys().collect();
    if keys.is_empty() {
        return Vec::new();
    }

    let mut combos: Vec<HashMap<String, ParamValue>> = vec![HashMap::new()];

    for key in &keys {
        let values = &grid[*key];
        let mut new_combos = Vec::with_capacity(combos.len() * values.len());
        for combo in &combos {
            for val in values {
                let mut c = combo.clone();
                c.insert((*key).clone(), val.clone());
                new_combos.push(c);
            }
        }
        combos = new_combos;
    }

    combos
}

/// Evaluate a single parameter combination via k-fold CV.
pub(super) fn evaluate_combo(
    base: &dyn Tunable,
    params: &HashMap<String, ParamValue>,
    folds: &[(Dataset, Dataset)],
    scorer: ScoringFn,
) -> Result<CvResult> {
    let mut scores = Vec::with_capacity(folds.len());

    for (train, test) in folds {
        let mut model = base.clone_box();
        for (name, value) in params {
            model.set_param(name, value.clone())?;
        }
        model.fit(train)?;
        let features = test.feature_matrix();
        let preds = model.predict(&features)?;
        scores.push(scorer(&test.target, &preds));
    }

    let mean = scores.iter().sum::<f64>() / scores.len() as f64;

    Ok(CvResult {
        params: params.clone(),
        mean_score: mean,
        fold_scores: scores,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{DecisionTreeClassifier, RandomForestClassifier};

    /// Build an Iris-like dataset with 3 well-separated classes.
    fn iris_like() -> Dataset {
        let n_per_class = 30;
        let n = n_per_class * 3;
        let mut f0 = Vec::with_capacity(n);
        let mut f1 = Vec::with_capacity(n);
        let mut f2 = Vec::with_capacity(n);
        let mut f3 = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);

        let mut rng = crate::rng::FastRng::new(123);

        for _ in 0..n_per_class {
            // Class 0: small values
            f0.push(1.0 + rng.f64() * 0.5);
            f1.push(1.0 + rng.f64() * 0.5);
            f2.push(0.5 + rng.f64() * 0.3);
            f3.push(0.1 + rng.f64() * 0.2);
            target.push(0.0);
        }
        for _ in 0..n_per_class {
            // Class 1: medium values
            f0.push(5.0 + rng.f64() * 0.5);
            f1.push(3.0 + rng.f64() * 0.5);
            f2.push(3.5 + rng.f64() * 0.5);
            f3.push(1.0 + rng.f64() * 0.3);
            target.push(1.0);
        }
        for _ in 0..n_per_class {
            // Class 2: large values
            f0.push(6.5 + rng.f64() * 0.5);
            f1.push(3.0 + rng.f64() * 0.5);
            f2.push(5.5 + rng.f64() * 0.5);
            f3.push(2.0 + rng.f64() * 0.3);
            target.push(2.0);
        }

        Dataset::new(
            vec![f0, f1, f2, f3],
            target,
            vec![
                "sepal_len".into(),
                "sepal_wid".into(),
                "petal_len".into(),
                "petal_wid".into(),
            ],
            "species",
        )
    }

    #[test]
    fn test_grid_search_dt() {
        let data = iris_like();
        let mut grid = ParamGrid::new();
        grid.insert(
            "max_depth".into(),
            vec![
                ParamValue::Int(2),
                ParamValue::Int(4),
                ParamValue::Int(6),
                ParamValue::Int(8),
            ],
        );

        let result = GridSearchCV::new(DecisionTreeClassifier::new(), grid)
            .cv(3)
            .scoring(crate::metrics::accuracy)
            .seed(42)
            .fit(&data)
            .unwrap();

        // Should find a reasonable best score on well-separated data.
        assert!(
            result.best_score() > 0.7,
            "best score {:.3} too low",
            result.best_score()
        );
        // Should have evaluated all 4 combos.
        assert_eq!(result.cv_results().len(), 4);
        // Best params must include max_depth.
        assert!(result.best_params().contains_key("max_depth"));
    }

    #[test]
    fn test_randomized_search_rf() {
        let data = iris_like();
        let mut grid = ParamGrid::new();
        grid.insert(
            "n_estimators".into(),
            vec![ParamValue::Int(3), ParamValue::Int(5), ParamValue::Int(10)],
        );
        grid.insert(
            "max_depth".into(),
            vec![ParamValue::Int(2), ParamValue::Int(4), ParamValue::Int(6)],
        );

        let result = RandomizedSearchCV::new(RandomForestClassifier::new(), grid)
            .n_iter(5)
            .cv(3)
            .seed(99)
            .fit(&data)
            .unwrap();

        // Should have evaluated exactly 5 combos (out of 9 total).
        assert_eq!(result.cv_results().len(), 5);
        assert!(
            result.best_score() > 0.5,
            "randomized best score too low: {:.3}",
            result.best_score()
        );
        assert!(result.best_params().contains_key("n_estimators"));
        assert!(result.best_params().contains_key("max_depth"));
    }

    #[test]
    fn test_cartesian_product() {
        let mut grid = ParamGrid::new();
        grid.insert("a".into(), vec![ParamValue::Int(1), ParamValue::Int(2)]);
        grid.insert(
            "b".into(),
            vec![ParamValue::Float(0.1), ParamValue::Float(0.2)],
        );
        let combos = cartesian_product(&grid);
        assert_eq!(combos.len(), 4);
    }

    #[test]
    fn test_invalid_param() {
        let mut dt = DecisionTreeClassifier::new();
        let err = dt.set_param("max_depth", ParamValue::Float(3.5));
        assert!(err.is_err());
        let err = dt.set_param("nonexistent", ParamValue::Int(3));
        assert!(err.is_err());
    }

    #[test]
    fn test_empty_grid() {
        let data = iris_like();
        let grid = ParamGrid::new();
        let result = GridSearchCV::new(DecisionTreeClassifier::new(), grid).fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_grid_search_logistic() {
        let data = iris_like();
        let mut grid = ParamGrid::new();
        grid.insert(
            "max_iter".into(),
            vec![ParamValue::Int(50), ParamValue::Int(200)],
        );
        let result = GridSearchCV::new(crate::linear::LogisticRegression::new(), grid)
            .cv(3)
            .scoring(crate::metrics::accuracy)
            .fit(&data)
            .unwrap();

        assert_eq!(result.cv_results().len(), 2);
        assert!(
            result.best_score() > 0.5,
            "logistic best score too low: {:.3}",
            result.best_score()
        );
        assert!(result.best_params().contains_key("max_iter"));
    }

    #[test]
    fn test_grid_search_knn() {
        let data = iris_like();
        let mut grid = ParamGrid::new();
        grid.insert(
            "k".into(),
            vec![ParamValue::Int(1), ParamValue::Int(3), ParamValue::Int(5)],
        );
        let result = GridSearchCV::new(crate::neighbors::KnnClassifier::new(), grid)
            .cv(3)
            .scoring(crate::metrics::accuracy)
            .fit(&data)
            .unwrap();

        assert_eq!(result.cv_results().len(), 3);
        assert!(
            result.best_score() > 0.7,
            "knn best score too low: {:.3}",
            result.best_score()
        );
        assert!(result.best_params().contains_key("k"));
    }

    #[test]
    fn test_grid_search_gbc() {
        let data = iris_like();
        let mut grid = ParamGrid::new();
        grid.insert(
            "n_estimators".into(),
            vec![ParamValue::Int(10), ParamValue::Int(20)],
        );
        grid.insert(
            "max_depth".into(),
            vec![ParamValue::Int(2), ParamValue::Int(3)],
        );
        let result = GridSearchCV::new(crate::tree::GradientBoostingClassifier::new(), grid)
            .cv(3)
            .scoring(crate::metrics::accuracy)
            .fit(&data)
            .unwrap();

        assert_eq!(result.cv_results().len(), 4);
        assert!(
            result.best_score() > 0.6,
            "gbc best score too low: {:.3}",
            result.best_score()
        );
        assert!(result.best_params().contains_key("n_estimators"));
        assert!(result.best_params().contains_key("max_depth"));
    }

    #[test]
    fn test_grid_search_lasso() {
        // Regression dataset: y = 2*x + noise.
        let n = 60;
        let mut rng = crate::rng::FastRng::new(42);
        let x: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();
        let target: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + rng.f64() * 0.5).collect();
        let data = crate::dataset::Dataset::new(vec![x], target, vec!["x".into()], "y");
        let mut grid = ParamGrid::new();
        grid.insert(
            "alpha".into(),
            vec![
                ParamValue::Float(0.01),
                ParamValue::Float(0.1),
                ParamValue::Float(1.0),
            ],
        );
        let result = GridSearchCV::new(crate::linear::LassoRegression::new(), grid)
            .cv(3)
            .scoring(crate::metrics::r2_score)
            .fit(&data)
            .unwrap();

        assert_eq!(result.cv_results().len(), 3);
        assert!(
            result.best_score() > 0.5,
            "lasso r2 too low: {:.3}",
            result.best_score()
        );
        assert!(result.best_params().contains_key("alpha"));
    }

    #[test]
    fn test_categorical_display() {
        let v = ParamValue::Categorical("gini".into());
        assert_eq!(format!("{v}"), "gini");
    }

    #[test]
    fn test_grid_search_stratified() {
        let data = iris_like();
        let mut grid = ParamGrid::new();
        grid.insert(
            "max_depth".into(),
            vec![ParamValue::Int(2), ParamValue::Int(4)],
        );

        let result = GridSearchCV::new(DecisionTreeClassifier::new(), grid)
            .cv(3)
            .stratified(true)
            .scoring(crate::metrics::accuracy)
            .seed(42)
            .fit(&data)
            .unwrap();

        assert_eq!(result.cv_results().len(), 2);
        assert!(
            result.best_score() > 0.7,
            "stratified best score {:.3} too low",
            result.best_score()
        );
    }

    #[test]
    fn test_randomized_search_stratified() {
        let data = iris_like();
        let mut grid = ParamGrid::new();
        grid.insert(
            "max_depth".into(),
            vec![ParamValue::Int(2), ParamValue::Int(4), ParamValue::Int(6)],
        );

        let result = RandomizedSearchCV::new(DecisionTreeClassifier::new(), grid)
            .n_iter(2)
            .cv(3)
            .stratified(true)
            .seed(99)
            .fit(&data)
            .unwrap();

        assert_eq!(result.cv_results().len(), 2);
        assert!(
            result.best_score() > 0.5,
            "stratified randomized best score {:.3} too low",
            result.best_score()
        );
    }
}
