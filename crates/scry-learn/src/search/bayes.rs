// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bayesian hyperparameter optimization via Tree-structured Parzen Estimator (TPE).
//!
//! [`BayesSearchCV`] uses a TPE surrogate model to guide the search towards
//! promising regions of the hyperparameter space, typically finding good
//! configurations in fewer evaluations than grid or random search.
//!
//! # Examples
//!
//! ```ignore
//! use scry_learn::prelude::*;
//! use scry_learn::search::*;
//!
//! let mut space = ParamSpace::new();
//! space.insert("max_depth".into(), ParamDistribution::IntUniform { low: 2, high: 10 });
//! space.insert("learning_rate".into(), ParamDistribution::LogUniform { low: 0.001, high: 1.0 });
//!
//! let result = BayesSearchCV::new(GradientBoostingClassifier::new(), space)
//!     .n_iter(30)
//!     .cv(5)
//!     .scoring(accuracy)
//!     .fit(&data)
//!     .unwrap();
//!
//! println!("Best: {:?} → {:.3}", result.best_params(), result.best_score());
//! ```

use std::collections::HashMap;

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::metrics::accuracy;
use crate::rng::FastRng;
use crate::split::{k_fold, stratified_k_fold, ScoringFn};

use super::{evaluate_combo, CvResult, ParamValue, Tunable};

// ---------------------------------------------------------------------------
// ParamDistribution + ParamSpace
// ---------------------------------------------------------------------------

/// A distribution from which hyperparameter values can be sampled.
///
/// Used with [`BayesSearchCV`] to define a continuous or discrete search space
/// for each hyperparameter.
///
/// # Examples
///
/// ```
/// use scry_learn::search::ParamDistribution;
///
/// let lr = ParamDistribution::LogUniform { low: 0.001, high: 1.0 };
/// let depth = ParamDistribution::IntUniform { low: 2, high: 10 };
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ParamDistribution {
    /// A set of discrete candidate values (any [`ParamValue`] variant).
    Categorical(Vec<ParamValue>),
    /// Continuous uniform distribution over `[low, high]`.
    Uniform {
        /// Lower bound (inclusive).
        low: f64,
        /// Upper bound (inclusive).
        high: f64,
    },
    /// Log-uniform distribution over `[low, high]` (sampled in log space).
    /// Both `low` and `high` must be positive.
    LogUniform {
        /// Lower bound (inclusive, positive).
        low: f64,
        /// Upper bound (inclusive, positive).
        high: f64,
    },
    /// Discrete uniform distribution over integers `[low, high]`.
    IntUniform {
        /// Lower bound (inclusive).
        low: usize,
        /// Upper bound (inclusive).
        high: usize,
    },
}

/// A mapping from parameter names to their search distributions.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use scry_learn::search::{ParamDistribution, ParamSpace};
///
/// let mut space = ParamSpace::new();
/// space.insert("max_depth".into(), ParamDistribution::IntUniform { low: 2, high: 10 });
/// ```
pub type ParamSpace = HashMap<String, ParamDistribution>;

// ---------------------------------------------------------------------------
// BayesSearchCV
// ---------------------------------------------------------------------------

/// Bayesian hyperparameter optimization with cross-validation.
///
/// Uses a Tree-structured Parzen Estimator (TPE) to model the objective
/// function and focus evaluations on the most promising hyperparameter
/// combinations.
///
/// # Algorithm
///
/// 1. Evaluate `n_initial` random samples to bootstrap the surrogate model.
/// 2. For each remaining iteration, split observed results at the `gamma`
///    quantile into "good" and "bad" groups.
/// 3. Build factored 1D kernel density estimates for each group.
/// 4. Draw 100 random candidates and pick the one maximizing `l(x) / g(x)`.
/// 5. Evaluate the chosen candidate and add it to the history.
///
/// # Examples
///
/// ```ignore
/// use scry_learn::prelude::*;
/// use scry_learn::search::*;
///
/// let mut space = ParamSpace::new();
/// space.insert("max_depth".into(), ParamDistribution::IntUniform { low: 2, high: 10 });
///
/// let result = BayesSearchCV::new(DecisionTreeClassifier::new(), space)
///     .n_iter(20)
///     .cv(3)
///     .fit(&data)
///     .unwrap();
///
/// println!("Best score: {:.3}", result.best_score());
/// ```
#[non_exhaustive]
pub struct BayesSearchCV {
    base_model: Box<dyn Tunable>,
    param_space: ParamSpace,
    n_iter: usize,
    n_initial: usize,
    gamma: f64,
    cv: usize,
    scorer: ScoringFn,
    seed: u64,
    stratified: bool,
    // Results (populated after fit)
    best_params_: Option<HashMap<String, ParamValue>>,
    best_score_: f64,
    cv_results_: Vec<CvResult>,
}

impl std::fmt::Debug for BayesSearchCV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BayesSearchCV")
            .field("n_iter", &self.n_iter)
            .field("n_initial", &self.n_initial)
            .field("gamma", &self.gamma)
            .field("cv", &self.cv)
            .field("seed", &self.seed)
            .field("stratified", &self.stratified)
            .field("best_score_", &self.best_score_)
            .field("cv_results_len", &self.cv_results_.len())
            .finish()
    }
}

impl BayesSearchCV {
    /// Create a Bayesian search over the given model and parameter space.
    ///
    /// Defaults: 30 iterations, 10 initial random samples, gamma 0.25,
    /// 5-fold CV, accuracy scorer, seed 42, non-stratified.
    pub fn new(model: impl Tunable + 'static, param_space: ParamSpace) -> Self {
        Self {
            base_model: Box::new(model),
            param_space,
            n_iter: 30,
            n_initial: 10,
            gamma: 0.25,
            cv: 5,
            scorer: accuracy,
            seed: 42,
            stratified: false,
            best_params_: None,
            best_score_: f64::NEG_INFINITY,
            cv_results_: Vec::new(),
        }
    }

    /// Set the total number of iterations (default: 30).
    pub fn n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }

    /// Set the number of initial random exploration samples (default: 10).
    pub fn n_initial(mut self, n: usize) -> Self {
        self.n_initial = n;
        self
    }

    /// Set the quantile threshold for splitting good/bad observations (default: 0.25).
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
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

    /// Run the Bayesian optimization search.
    ///
    /// Returns `self` for chained accessor calls.
    pub fn fit(mut self, data: &Dataset) -> Result<Self> {
        if self.cv < 2 {
            return Err(ScryLearnError::InvalidParameter(format!(
                "cv must be >= 2, got {}",
                self.cv
            )));
        }
        if self.param_space.is_empty() {
            return Err(ScryLearnError::InvalidParameter(
                "parameter space is empty".into(),
            ));
        }
        if self.n_iter == 0 {
            return Err(ScryLearnError::InvalidParameter(
                "n_iter must be >= 1".into(),
            ));
        }

        let folds = if self.stratified {
            stratified_k_fold(data, self.cv, self.seed)
        } else {
            k_fold(data, self.cv, self.seed)
        };

        let mut rng = FastRng::new(self.seed);

        // Sorted parameter names for deterministic ordering.
        let param_names: Vec<String> = {
            let mut names: Vec<String> = self.param_space.keys().cloned().collect();
            names.sort();
            names
        };

        // Phase 1: random exploration.
        let n_initial = self.n_initial.min(self.n_iter);
        for _ in 0..n_initial {
            let combo = sample_random(&self.param_space, &param_names, &mut rng);
            let result = evaluate_combo(&*self.base_model, &combo, &folds, self.scorer)?;
            self.update_best(&result);
            self.cv_results_.push(result);
        }

        // Phase 2: TPE-guided search.
        let n_tpe = self.n_iter - n_initial;
        for _ in 0..n_tpe {
            // Split history into good/bad at gamma quantile.
            let mut scores: Vec<f64> = self
                .cv_results_
                .iter()
                .map(|r| r.mean_score)
                .filter(|s| s.is_finite())
                .collect();
            scores.sort_by(|a, b| a.total_cmp(b));

            let n_good = ((scores.len() as f64 * self.gamma).ceil() as usize).max(1);
            let threshold = scores[scores.len().saturating_sub(n_good)];

            let (good, bad): (Vec<&CvResult>, Vec<&CvResult>) = self
                .cv_results_
                .iter()
                .filter(|r| r.mean_score.is_finite())
                .partition(|r| r.mean_score >= threshold);

            // If all observations are "good" (e.g. equal scores), fall back to random.
            let combo = if bad.is_empty() {
                sample_random(&self.param_space, &param_names, &mut rng)
            } else {
                // Build KDEs for good and bad, sample candidates, pick best EI.
                let good_kde = build_factored_kde(&good, &param_names, &self.param_space);
                let bad_kde = build_factored_kde(&bad, &param_names, &self.param_space);

                let n_candidates = 100;
                let mut best_candidate = sample_random(&self.param_space, &param_names, &mut rng);
                let mut best_ei = f64::NEG_INFINITY;

                for _ in 0..n_candidates {
                    let candidate = sample_random(&self.param_space, &param_names, &mut rng);
                    let l = evaluate_kde(&good_kde, &candidate, &param_names, &self.param_space);
                    let g = evaluate_kde(&bad_kde, &candidate, &param_names, &self.param_space);
                    let ei = if g > 1e-300 { l / g } else { l * 1e300 };
                    if ei > best_ei {
                        best_ei = ei;
                        best_candidate = candidate;
                    }
                }
                best_candidate
            };

            let result = evaluate_combo(&*self.base_model, &combo, &folds, self.scorer)?;
            self.update_best(&result);
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

    fn update_best(&mut self, result: &CvResult) {
        if result.mean_score.is_finite()
            && (self.best_params_.is_none() || result.mean_score > self.best_score_)
        {
            self.best_score_ = result.mean_score;
            self.best_params_ = Some(result.params.clone());
        }
    }
}

// ---------------------------------------------------------------------------
// Sampling helpers
// ---------------------------------------------------------------------------

/// Sample a random parameter combination from the search space.
fn sample_random(
    space: &ParamSpace,
    param_names: &[String],
    rng: &mut FastRng,
) -> HashMap<String, ParamValue> {
    let mut combo = HashMap::new();
    for name in param_names {
        let dist = &space[name];
        let value = match dist {
            ParamDistribution::Categorical(values) => {
                let idx = rng.usize(0..values.len());
                values[idx].clone()
            }
            ParamDistribution::Uniform { low, high } => {
                ParamValue::Float(low + rng.f64() * (high - low))
            }
            ParamDistribution::LogUniform { low, high } => {
                let log_low = low.ln();
                let log_high = high.ln();
                ParamValue::Float((log_low + rng.f64() * (log_high - log_low)).exp())
            }
            ParamDistribution::IntUniform { low, high } => {
                if high > low {
                    ParamValue::Int(low + rng.usize(0..=(high - low)))
                } else {
                    ParamValue::Int(*low)
                }
            }
        };
        combo.insert(name.clone(), value);
    }
    combo
}

// ---------------------------------------------------------------------------
// Factored KDE (1D Gaussian kernels per dimension)
// ---------------------------------------------------------------------------

/// A factored KDE: for each parameter we store either continuous observations
/// (normalized to [0,1]) or categorical frequency counts.
enum ParamKde {
    /// Normalized observations in [0,1] plus the Scott's-rule bandwidth.
    Continuous {
        observations: Vec<f64>,
        bandwidth: f64,
    },
    /// Frequency of each categorical index, with Laplace smoothing applied.
    Categorical {
        /// Probability for each index.
        probs: Vec<f64>,
    },
}

/// One KDE per parameter dimension (factored assumption).
struct FactoredKde {
    kdes: Vec<(String, ParamKde)>,
}

/// Build a factored KDE from a set of CvResult observations.
fn build_factored_kde(
    observations: &[&CvResult],
    param_names: &[String],
    space: &ParamSpace,
) -> FactoredKde {
    let mut kdes = Vec::with_capacity(param_names.len());

    for name in param_names {
        let dist = &space[name];
        if let ParamDistribution::Categorical(values) = dist {
            let n_categories = values.len();
            // Count frequencies with Laplace smoothing.
            let mut counts = vec![1.0_f64; n_categories]; // Laplace prior
            for obs in observations {
                if let Some(val) = obs.params.get(name) {
                    if let Some(idx) = values.iter().position(|v| v == val) {
                        counts[idx] += 1.0;
                    }
                }
            }
            let total: f64 = counts.iter().sum();
            let probs: Vec<f64> = counts.iter().map(|c| c / total).collect();
            kdes.push((name.clone(), ParamKde::Categorical { probs }));
        } else {
            // Normalize observations to [0,1].
            let obs_normalized: Vec<f64> = observations
                .iter()
                .filter_map(|r| r.params.get(name))
                .map(|v| normalize_param(v, dist))
                .collect();

            // Scott's rule bandwidth: n^(-1/(d+4)) where d=1 for 1D.
            let bw = if obs_normalized.is_empty() {
                1.0
            } else {
                (obs_normalized.len() as f64).powf(-1.0 / 5.0)
            };

            kdes.push((
                name.clone(),
                ParamKde::Continuous {
                    observations: obs_normalized,
                    bandwidth: bw,
                },
            ));
        }
    }

    FactoredKde { kdes }
}

/// Evaluate the factored KDE density at a candidate point.
fn evaluate_kde(
    kde: &FactoredKde,
    candidate: &HashMap<String, ParamValue>,
    _param_names: &[String],
    space: &ParamSpace,
) -> f64 {
    let mut log_density = 0.0_f64;

    for (name, param_kde) in &kde.kdes {
        let Some(val) = candidate.get(name) else {
            continue;
        };
        let dist = &space[name];

        match param_kde {
            ParamKde::Continuous {
                observations,
                bandwidth,
            } => {
                let x = normalize_param(val, dist);
                let n = observations.len() as f64;
                if n < 1.0 {
                    continue;
                }
                // Mean of Gaussian kernel values.
                let mut density_sum = 0.0_f64;
                for &obs in observations {
                    let z = (x - obs) / bandwidth;
                    density_sum += (-0.5 * z * z).exp();
                }
                let density = density_sum / (n * bandwidth * (std::f64::consts::TAU).sqrt());
                // Clamp to avoid log(0).
                log_density += density.max(1e-300).ln();
            }
            ParamKde::Categorical { probs } => {
                if let ParamDistribution::Categorical(values) = dist {
                    if let Some(idx) = values.iter().position(|v| v == val) {
                        log_density += probs[idx].max(1e-300).ln();
                    } else {
                        // Unknown category — use uniform fallback.
                        log_density += (1.0 / probs.len() as f64).ln();
                    }
                }
            }
        }
    }

    log_density.exp()
}

/// Normalize a parameter value to [0, 1] given its distribution.
fn normalize_param(value: &ParamValue, dist: &ParamDistribution) -> f64 {
    match (value, dist) {
        (ParamValue::Float(v), ParamDistribution::Uniform { low, high }) => {
            if (high - low).abs() < 1e-300 {
                0.5
            } else {
                (v - low) / (high - low)
            }
        }
        (ParamValue::Float(v), ParamDistribution::LogUniform { low, high }) => {
            let log_low = low.ln();
            let log_high = high.ln();
            if (log_high - log_low).abs() < 1e-300 {
                0.5
            } else {
                (v.ln() - log_low) / (log_high - log_low)
            }
        }
        (ParamValue::Int(v), ParamDistribution::IntUniform { low, high }) => {
            if high == low {
                0.5
            } else {
                (*v as f64 - *low as f64) / (*high as f64 - *low as f64)
            }
        }
        // Fallback: treat the raw float-like value as already in some space.
        (ParamValue::Float(v), _) => v.clamp(0.0, 1.0),
        (ParamValue::Int(v), _) => (*v as f64).clamp(0.0, 1.0),
        _ => 0.5,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::DecisionTreeClassifier;

    /// Build an Iris-like dataset with 3 well-separated classes.
    fn iris_like() -> Dataset {
        let n_per_class = 30;
        let n = n_per_class * 3;
        let mut f0 = Vec::with_capacity(n);
        let mut f1 = Vec::with_capacity(n);
        let mut f2 = Vec::with_capacity(n);
        let mut f3 = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);

        let mut rng = FastRng::new(123);

        for _ in 0..n_per_class {
            f0.push(1.0 + rng.f64() * 0.5);
            f1.push(1.0 + rng.f64() * 0.5);
            f2.push(0.5 + rng.f64() * 0.3);
            f3.push(0.1 + rng.f64() * 0.2);
            target.push(0.0);
        }
        for _ in 0..n_per_class {
            f0.push(5.0 + rng.f64() * 0.5);
            f1.push(3.0 + rng.f64() * 0.5);
            f2.push(3.5 + rng.f64() * 0.5);
            f3.push(1.0 + rng.f64() * 0.3);
            target.push(1.0);
        }
        for _ in 0..n_per_class {
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
    fn test_bayes_search_int_uniform() {
        let data = iris_like();
        let mut space = ParamSpace::new();
        space.insert(
            "max_depth".into(),
            ParamDistribution::IntUniform { low: 2, high: 10 },
        );

        let result = BayesSearchCV::new(DecisionTreeClassifier::new(), space)
            .n_iter(15)
            .n_initial(5)
            .cv(3)
            .seed(42)
            .fit(&data)
            .unwrap();

        assert!(
            result.best_score() > 0.7,
            "bayes best score {:.3} too low",
            result.best_score()
        );
        assert_eq!(result.cv_results().len(), 15);
        assert!(result.best_params().contains_key("max_depth"));
    }

    #[test]
    fn test_bayes_search_categorical() {
        let data = iris_like();
        let mut space = ParamSpace::new();
        space.insert(
            "max_depth".into(),
            ParamDistribution::Categorical(vec![
                ParamValue::Int(2),
                ParamValue::Int(4),
                ParamValue::Int(6),
                ParamValue::Int(8),
            ]),
        );

        let result = BayesSearchCV::new(DecisionTreeClassifier::new(), space)
            .n_iter(10)
            .n_initial(4)
            .cv(3)
            .seed(99)
            .fit(&data)
            .unwrap();

        assert!(
            result.best_score() > 0.5,
            "bayes categorical best score {:.3} too low",
            result.best_score()
        );
        assert!(result.best_params().contains_key("max_depth"));
    }

    #[test]
    fn test_bayes_search_mixed_space() {
        let data = iris_like();
        let mut space = ParamSpace::new();
        space.insert(
            "max_depth".into(),
            ParamDistribution::IntUniform { low: 2, high: 8 },
        );
        space.insert(
            "min_samples_split".into(),
            ParamDistribution::IntUniform { low: 2, high: 10 },
        );

        let result = BayesSearchCV::new(DecisionTreeClassifier::new(), space)
            .n_iter(12)
            .n_initial(5)
            .cv(3)
            .seed(42)
            .fit(&data)
            .unwrap();

        assert_eq!(result.cv_results().len(), 12);
        assert!(result.best_params().contains_key("max_depth"));
        assert!(result.best_params().contains_key("min_samples_split"));
    }

    #[test]
    fn test_bayes_search_stratified() {
        let data = iris_like();
        let mut space = ParamSpace::new();
        space.insert(
            "max_depth".into(),
            ParamDistribution::IntUniform { low: 2, high: 8 },
        );

        let result = BayesSearchCV::new(DecisionTreeClassifier::new(), space)
            .n_iter(10)
            .n_initial(5)
            .cv(3)
            .stratified(true)
            .seed(42)
            .fit(&data)
            .unwrap();

        assert!(
            result.best_score() > 0.7,
            "stratified bayes best score {:.3} too low",
            result.best_score()
        );
    }

    #[test]
    fn test_bayes_search_empty_space() {
        let data = iris_like();
        let space = ParamSpace::new();
        let result = BayesSearchCV::new(DecisionTreeClassifier::new(), space).fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_bayes_search_n_iter_zero() {
        let data = iris_like();
        let mut space = ParamSpace::new();
        space.insert(
            "max_depth".into(),
            ParamDistribution::IntUniform { low: 2, high: 8 },
        );
        let result = BayesSearchCV::new(DecisionTreeClassifier::new(), space)
            .n_iter(0)
            .fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_bayes_search_all_initial() {
        // When n_initial >= n_iter, all samples are random (no TPE phase).
        let data = iris_like();
        let mut space = ParamSpace::new();
        space.insert(
            "max_depth".into(),
            ParamDistribution::IntUniform { low: 2, high: 6 },
        );

        let result = BayesSearchCV::new(DecisionTreeClassifier::new(), space)
            .n_iter(5)
            .n_initial(10)
            .cv(3)
            .seed(42)
            .fit(&data)
            .unwrap();

        assert_eq!(result.cv_results().len(), 5);
    }

    #[test]
    fn test_bayes_search_gbc_log_uniform() {
        let data = iris_like();
        let mut space = ParamSpace::new();
        space.insert(
            "n_estimators".into(),
            ParamDistribution::Categorical(vec![
                ParamValue::Int(5),
                ParamValue::Int(10),
                ParamValue::Int(20),
            ]),
        );
        space.insert(
            "max_depth".into(),
            ParamDistribution::IntUniform { low: 2, high: 4 },
        );

        let result = BayesSearchCV::new(crate::tree::GradientBoostingClassifier::new(), space)
            .n_iter(10)
            .n_initial(5)
            .cv(3)
            .scoring(crate::metrics::accuracy)
            .seed(42)
            .fit(&data)
            .unwrap();

        assert!(
            result.best_score() > 0.5,
            "gbc bayes best score {:.3} too low",
            result.best_score()
        );
    }

    #[test]
    fn test_normalize_param() {
        let dist = ParamDistribution::Uniform {
            low: 0.0,
            high: 10.0,
        };
        let val = ParamValue::Float(5.0);
        let norm = normalize_param(&val, &dist);
        assert!((norm - 0.5).abs() < 1e-10);

        let dist_int = ParamDistribution::IntUniform { low: 0, high: 10 };
        let val_int = ParamValue::Int(5);
        let norm_int = normalize_param(&val_int, &dist_int);
        assert!((norm_int - 0.5).abs() < 1e-10);
    }
}
