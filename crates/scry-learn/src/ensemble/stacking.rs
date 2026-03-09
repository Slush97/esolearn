// SPDX-License-Identifier: MIT OR Apache-2.0
//! Voting and Stacking ensemble classifiers.
//!
//! [`VotingClassifier`] combines multiple classifiers via hard (majority
//! vote) or soft (probability averaging) voting.
//!
//! [`StackingClassifier`] trains a meta-learner on out-of-fold predictions
//! from a set of base estimators.

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};

// ---------------------------------------------------------------------------
// Voting strategy
// ---------------------------------------------------------------------------

/// Voting strategy for [`VotingClassifier`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Voting {
    /// Majority vote on predicted class labels.
    Hard,
    /// Average predicted probabilities, then take argmax.
    ///
    /// Requires all estimators to support `predict_proba`.
    Soft,
}

// ---------------------------------------------------------------------------
// Classifier wrapper — trait object for ensemble base learners
// ---------------------------------------------------------------------------

/// Trait object for classifiers that can be used in ensembles.
///
/// Covers the common interface: fit on a [`Dataset`], predict on features,
/// and optionally predict class probabilities.
pub trait EnsembleClassifier: Send + Sync {
    /// Train on a dataset.
    fn fit(&mut self, data: &Dataset) -> Result<()>;

    /// Predict class labels for the given feature matrix.
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>>;

    /// Predict class probabilities (required for soft voting / stacking).
    ///
    /// Default implementation returns an error indicating the model does not
    /// support probability predictions.
    fn predict_proba(&self, _features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        Err(ScryLearnError::InvalidParameter(
            "this estimator does not support predict_proba".into(),
        ))
    }

    /// Clone into a boxed trait object.
    fn clone_box(&self) -> Box<dyn EnsembleClassifier>;
}

impl Clone for Box<dyn EnsembleClassifier> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// ---------------------------------------------------------------------------
// EnsembleClassifier implementations for existing models
// ---------------------------------------------------------------------------

macro_rules! impl_ensemble_no_proba {
    ($ty:path) => {
        impl EnsembleClassifier for $ty {
            fn fit(&mut self, data: &Dataset) -> Result<()> {
                self.fit(data)
            }
            fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
                self.predict(features)
            }
            fn clone_box(&self) -> Box<dyn EnsembleClassifier> {
                Box::new(self.clone())
            }
        }
    };
}

macro_rules! impl_ensemble_with_proba {
    ($ty:path) => {
        impl EnsembleClassifier for $ty {
            fn fit(&mut self, data: &Dataset) -> Result<()> {
                self.fit(data)
            }
            fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
                self.predict(features)
            }
            fn predict_proba(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
                self.predict_proba(features)
            }
            fn clone_box(&self) -> Box<dyn EnsembleClassifier> {
                Box::new(self.clone())
            }
        }
    };
}

// Models that support predict_proba:
impl_ensemble_with_proba!(crate::tree::DecisionTreeClassifier);
impl_ensemble_with_proba!(crate::tree::RandomForestClassifier);
impl_ensemble_with_proba!(crate::naive_bayes::GaussianNb);
impl_ensemble_with_proba!(crate::naive_bayes::BernoulliNB);
impl_ensemble_with_proba!(crate::naive_bayes::MultinomialNB);

// Models without predict_proba:
impl_ensemble_no_proba!(crate::tree::DecisionTreeRegressor);
impl_ensemble_no_proba!(crate::linear::LogisticRegression);
impl_ensemble_no_proba!(crate::linear::LinearRegression);
impl_ensemble_no_proba!(crate::linear::LassoRegression);
impl_ensemble_no_proba!(crate::linear::ElasticNet);
impl_ensemble_no_proba!(crate::neighbors::KnnClassifier);
impl_ensemble_no_proba!(crate::neighbors::KnnRegressor);
impl_ensemble_no_proba!(crate::svm::LinearSVC);
impl_ensemble_no_proba!(crate::svm::LinearSVR);
#[cfg(feature = "experimental")]
impl_ensemble_no_proba!(crate::svm::KernelSVC);
#[cfg(feature = "experimental")]
impl_ensemble_no_proba!(crate::svm::KernelSVR);

// ---------------------------------------------------------------------------
// VotingClassifier
// ---------------------------------------------------------------------------

/// Combines multiple classifiers via voting.
///
/// In [`Voting::Hard`] mode, each estimator votes for a class and the majority
/// wins. In [`Voting::Soft`] mode, predicted probabilities are averaged and
/// the class with the highest average probability is selected.
///
/// # Examples
///
/// ```ignore
/// use scry_learn::ensemble::{VotingClassifier, Voting};
/// use scry_learn::tree::DecisionTreeClassifier;
///
/// let vc = VotingClassifier::new(vec![
///     Box::new(DecisionTreeClassifier::new().max_depth(3)),
///     Box::new(DecisionTreeClassifier::new().max_depth(5)),
///     Box::new(DecisionTreeClassifier::new().max_depth(7)),
/// ]).voting(Voting::Hard);
/// ```
#[derive(Clone)]
#[non_exhaustive]
pub struct VotingClassifier {
    /// Base estimators.
    estimators: Vec<Box<dyn EnsembleClassifier>>,
    /// Voting strategy.
    voting_strategy: Voting,
    /// Optional weights for each estimator.
    weights: Option<Vec<f64>>,
    /// Whether the model has been fitted.
    fitted: bool,
    /// Number of unique classes seen during fit.
    n_classes: usize,
}

impl std::fmt::Debug for VotingClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VotingClassifier")
            .field("n_estimators", &self.estimators.len())
            .field("voting", &self.voting_strategy)
            .field("weights", &self.weights)
            .field("fitted", &self.fitted)
            .finish()
    }
}

impl VotingClassifier {
    /// Create a new voting classifier with the given base estimators.
    pub fn new(estimators: Vec<Box<dyn EnsembleClassifier>>) -> Self {
        Self {
            estimators,
            voting_strategy: Voting::Hard,
            weights: None,
            fitted: false,
            n_classes: 0,
        }
    }

    /// Set the voting strategy (default: [`Voting::Hard`]).
    pub fn voting(mut self, v: Voting) -> Self {
        self.voting_strategy = v;
        self
    }

    /// Set weights for each estimator (default: equal weights).
    pub fn weights(mut self, w: Vec<f64>) -> Self {
        self.weights = Some(w);
        self
    }

    /// Fit all base estimators on the given dataset.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        if data.n_samples() == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        if self.estimators.is_empty() {
            return Err(ScryLearnError::InvalidParameter(
                "VotingClassifier requires at least one estimator".into(),
            ));
        }
        if let Some(ref w) = self.weights {
            if w.len() != self.estimators.len() {
                return Err(ScryLearnError::InvalidParameter(format!(
                    "weights length ({}) must match estimators length ({})",
                    w.len(),
                    self.estimators.len(),
                )));
            }
        }

        self.n_classes = data.n_classes();

        for est in &mut self.estimators {
            est.fit(data)?;
        }
        self.fitted = true;
        Ok(())
    }

    /// Predict class labels via voting.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }

        match self.voting_strategy {
            Voting::Hard => self.predict_hard(features),
            Voting::Soft => self.predict_soft(features),
        }
    }

    /// Hard voting: majority vote across estimator predictions.
    fn predict_hard(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        let n = features.len();
        let n_classes = self.n_classes;

        // Collect predictions from all estimators.
        let all_preds: Vec<Vec<f64>> = self
            .estimators
            .iter()
            .map(|est| est.predict(features))
            .collect::<Result<_>>()?;

        let weights = self.uniform_weights();

        let mut result = Vec::with_capacity(n);
        for sample_idx in 0..n {
            let mut votes = vec![0.0_f64; n_classes.max(1)];
            for (est_idx, preds) in all_preds.iter().enumerate() {
                let class = preds[sample_idx] as usize;
                if class < votes.len() {
                    votes[class] += weights[est_idx];
                }
            }
            let best_class = votes
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx);
            result.push(best_class as f64);
        }

        Ok(result)
    }

    /// Soft voting: average predict_proba across estimators, take argmax.
    fn predict_soft(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        let n = features.len();
        let n_classes = self.n_classes;
        let weights = self.uniform_weights();

        let mut avg_proba = vec![vec![0.0; n_classes]; n];

        for (est_idx, est) in self.estimators.iter().enumerate() {
            let probas = est.predict_proba(features)?;
            for (sample_idx, proba) in probas.iter().enumerate() {
                for (class_idx, &p) in proba.iter().enumerate() {
                    if class_idx < n_classes {
                        avg_proba[sample_idx][class_idx] += p * weights[est_idx];
                    }
                }
            }
        }

        let result: Vec<f64> = avg_proba
            .iter()
            .map(|proba| {
                proba
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0.0, |(idx, _)| idx as f64)
            })
            .collect();

        Ok(result)
    }

    /// Return weights (user-provided or uniform).
    fn uniform_weights(&self) -> Vec<f64> {
        self.weights
            .clone()
            .unwrap_or_else(|| vec![1.0; self.estimators.len()])
    }
}

// ---------------------------------------------------------------------------
// StackingClassifier
// ---------------------------------------------------------------------------

/// Stacking (stacked generalization) classifier.
///
/// Trains base estimators via k-fold cross-validation, collects out-of-fold
/// predictions as meta-features, then trains a final meta-learner on those
/// meta-features. At predict time, base estimator predictions are fed to
/// the meta-learner.
///
/// # Examples
///
/// ```ignore
/// use scry_learn::ensemble::StackingClassifier;
/// use scry_learn::tree::DecisionTreeClassifier;
/// use scry_learn::linear::LogisticRegression;
///
/// let sc = StackingClassifier::new(
///     vec![
///         Box::new(DecisionTreeClassifier::new().max_depth(3)),
///         Box::new(DecisionTreeClassifier::new().max_depth(7)),
///     ],
///     Box::new(LogisticRegression::new()),
/// ).cv(5);
/// ```
#[derive(Clone)]
#[non_exhaustive]
pub struct StackingClassifier {
    /// Base learners.
    estimators: Vec<Box<dyn EnsembleClassifier>>,
    /// Meta-learner trained on out-of-fold predictions.
    final_estimator: Box<dyn EnsembleClassifier>,
    /// Number of cross-validation folds.
    cv: usize,
    /// Random seed for fold generation.
    seed: u64,
    /// Whether the model has been fitted.
    fitted: bool,
}

impl std::fmt::Debug for StackingClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StackingClassifier")
            .field("n_estimators", &self.estimators.len())
            .field("cv", &self.cv)
            .field("fitted", &self.fitted)
            .finish()
    }
}

impl StackingClassifier {
    /// Create a new stacking classifier.
    ///
    /// `estimators` are the base learners; `final_estimator` is the meta-learner.
    pub fn new(
        estimators: Vec<Box<dyn EnsembleClassifier>>,
        final_estimator: Box<dyn EnsembleClassifier>,
    ) -> Self {
        Self {
            estimators,
            final_estimator,
            cv: 5,
            seed: 42,
            fitted: false,
        }
    }

    /// Set the number of CV folds (default: 5).
    pub fn cv(mut self, k: usize) -> Self {
        self.cv = k;
        self
    }

    /// Set the random seed for fold generation (default: 42).
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Fit the stacking classifier.
    ///
    /// 1. Split data into `cv` folds.
    /// 2. For each fold, train base learners on training folds and collect
    ///    out-of-fold predictions.
    /// 3. Assemble meta-features from out-of-fold predictions.
    /// 4. Train the final estimator on meta-features.
    /// 5. Re-train all base learners on the full dataset for prediction.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        if data.n_samples() == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        if self.estimators.is_empty() {
            return Err(ScryLearnError::InvalidParameter(
                "StackingClassifier requires at least one base estimator".into(),
            ));
        }
        if self.cv < 2 {
            return Err(ScryLearnError::InvalidParameter(
                "cv must be at least 2".into(),
            ));
        }

        let n_samples = data.n_samples();
        let n_estimators = self.estimators.len();

        // Generate fold indices.
        let folds = generate_fold_indices(n_samples, self.cv, self.seed);

        // Meta-feature matrix: n_samples rows × n_estimators columns.
        let mut meta_features = vec![vec![0.0; n_estimators]; n_samples];

        for (fold_idx, test_indices) in folds.iter().enumerate() {
            let train_indices: Vec<usize> = (0..n_samples)
                .filter(|i| !test_indices.contains(i))
                .collect();

            let train_data = data.subset(&train_indices);
            let test_features = Self::extract_features(data, test_indices);

            for (est_idx, est_template) in self.estimators.iter().enumerate() {
                let mut est = est_template.clone_box();
                est.fit(&train_data)?;
                let preds = est.predict(&test_features)?;

                for (local_idx, &global_idx) in test_indices.iter().enumerate() {
                    meta_features[global_idx][est_idx] = preds[local_idx];
                }

                // Drop to free memory.
                let _ = fold_idx;
            }
        }

        // Build meta-dataset: features = meta_features, target = original target.
        let meta_columns: Vec<Vec<f64>> = (0..n_estimators)
            .map(|est_idx| meta_features.iter().map(|row| row[est_idx]).collect())
            .collect();
        let feature_names: Vec<String> = (0..n_estimators).map(|i| format!("est_{i}")).collect();

        let meta_dataset = Dataset::new(meta_columns, data.target.clone(), feature_names, "target");

        // Train the final estimator on meta-features.
        self.final_estimator.fit(&meta_dataset)?;

        // Re-train base learners on the full dataset for prediction time.
        for est in &mut self.estimators {
            est.fit(data)?;
        }

        self.fitted = true;
        Ok(())
    }

    /// Predict class labels using the stacking ensemble.
    ///
    /// Gets predictions from all base learners, then feeds them to the
    /// meta-learner.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }

        let n = features.len();
        let n_estimators = self.estimators.len();

        // Get base predictions.
        let base_preds: Vec<Vec<f64>> = self
            .estimators
            .iter()
            .map(|est| est.predict(features))
            .collect::<Result<_>>()?;

        // Assemble meta-features.
        let meta_features: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n_estimators).map(|j| base_preds[j][i]).collect())
            .collect();

        self.final_estimator.predict(&meta_features)
    }

    /// Extract row-major features for specific sample indices.
    fn extract_features(data: &Dataset, indices: &[usize]) -> Vec<Vec<f64>> {
        indices.iter().map(|&i| data.sample(i)).collect()
    }
}

/// Generate fold indices for k-fold cross-validation.
fn generate_fold_indices(n: usize, k: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = crate::rng::FastRng::new(seed);

    // Fisher-Yates shuffle.
    for i in (1..indices.len()).rev() {
        let j = rng.usize(0..=i);
        indices.swap(i, j);
    }

    let fold_size = n / k;
    let remainder = n % k;
    let mut folds = Vec::with_capacity(k);
    let mut start = 0;
    for fold in 0..k {
        let extra = usize::from(fold < remainder);
        let end = start + fold_size + extra;
        folds.push(indices[start..end].to_vec());
        start = end;
    }

    folds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::DecisionTreeClassifier;

    fn make_iris_like_data() -> Dataset {
        // 3-class classification with clear separation.
        let mut f1 = Vec::new();
        let mut f2 = Vec::new();
        let mut target = Vec::new();
        let mut rng = crate::rng::FastRng::new(42);

        // Class 0: cluster around (1, 1)
        for _ in 0..40 {
            f1.push(1.0 + rng.f64() * 0.5);
            f2.push(1.0 + rng.f64() * 0.5);
            target.push(0.0);
        }
        // Class 1: cluster around (5, 5)
        for _ in 0..40 {
            f1.push(5.0 + rng.f64() * 0.5);
            f2.push(5.0 + rng.f64() * 0.5);
            target.push(1.0);
        }
        // Class 2: cluster around (1, 5)
        for _ in 0..40 {
            f1.push(1.0 + rng.f64() * 0.5);
            f2.push(5.0 + rng.f64() * 0.5);
            target.push(2.0);
        }

        Dataset::new(
            vec![f1, f2],
            target,
            vec!["f1".into(), "f2".into()],
            "class",
        )
    }

    #[test]
    fn test_voting_hard_basic() {
        let data = make_iris_like_data();

        let mut vc = VotingClassifier::new(vec![
            Box::new(DecisionTreeClassifier::new().max_depth(3)),
            Box::new(DecisionTreeClassifier::new().max_depth(5)),
            Box::new(DecisionTreeClassifier::new().max_depth(7)),
        ])
        .voting(Voting::Hard);

        vc.fit(&data).unwrap();
        let features = data.feature_matrix();
        let preds = vc.predict(&features).unwrap();

        let acc = preds
            .iter()
            .zip(data.target.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-6)
            .count() as f64
            / data.n_samples() as f64;

        assert!(
            acc >= 0.85,
            "VotingClassifier hard vote accuracy should be ≥ 85%, got {:.1}%",
            acc * 100.0,
        );
    }

    #[test]
    fn test_voting_soft_basic() {
        let data = make_iris_like_data();

        // Use DecisionTreeClassifier which supports predict_proba.
        let mut vc = VotingClassifier::new(vec![
            Box::new(DecisionTreeClassifier::new().max_depth(3)),
            Box::new(DecisionTreeClassifier::new().max_depth(5)),
            Box::new(DecisionTreeClassifier::new().max_depth(7)),
        ])
        .voting(Voting::Soft);

        vc.fit(&data).unwrap();
        let features = data.feature_matrix();
        let preds = vc.predict(&features).unwrap();

        let acc = preds
            .iter()
            .zip(data.target.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-6)
            .count() as f64
            / data.n_samples() as f64;

        assert!(
            acc >= 0.85,
            "VotingClassifier soft vote accuracy should be ≥ 85%, got {:.1}%",
            acc * 100.0,
        );
    }

    #[test]
    fn test_voting_weighted() {
        let data = make_iris_like_data();

        let mut vc = VotingClassifier::new(vec![
            Box::new(DecisionTreeClassifier::new().max_depth(3)),
            Box::new(DecisionTreeClassifier::new().max_depth(5)),
        ])
        .voting(Voting::Hard)
        .weights(vec![1.0, 2.0]);

        vc.fit(&data).unwrap();
        let features = data.feature_matrix();
        let preds = vc.predict(&features).unwrap();
        assert_eq!(preds.len(), data.n_samples());
    }

    #[test]
    fn test_voting_not_fitted() {
        let vc = VotingClassifier::new(vec![Box::new(DecisionTreeClassifier::new())]);
        let result = vc.predict(&[vec![1.0, 2.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_voting_empty_estimators() {
        let data = make_iris_like_data();
        let mut vc = VotingClassifier::new(vec![]);
        assert!(vc.fit(&data).is_err());
    }

    #[test]
    fn test_voting_weights_mismatch() {
        let data = make_iris_like_data();
        let mut vc = VotingClassifier::new(vec![Box::new(DecisionTreeClassifier::new())])
            .weights(vec![1.0, 2.0]); // mismatch: 2 weights for 1 estimator
        assert!(vc.fit(&data).is_err());
    }

    #[test]
    fn test_stacking_basic() {
        let data = make_iris_like_data();

        let mut sc = StackingClassifier::new(
            vec![
                Box::new(DecisionTreeClassifier::new().max_depth(3)),
                Box::new(DecisionTreeClassifier::new().max_depth(7)),
            ],
            Box::new(DecisionTreeClassifier::new().max_depth(5)),
        )
        .cv(3)
        .seed(42);

        sc.fit(&data).unwrap();
        let features = data.feature_matrix();
        let preds = sc.predict(&features).unwrap();

        assert_eq!(preds.len(), data.n_samples());

        let acc = preds
            .iter()
            .zip(data.target.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-6)
            .count() as f64
            / data.n_samples() as f64;

        assert!(
            acc >= 0.70,
            "StackingClassifier accuracy should be ≥ 70%, got {:.1}%",
            acc * 100.0,
        );
    }

    #[test]
    fn test_stacking_not_fitted() {
        let sc = StackingClassifier::new(
            vec![Box::new(DecisionTreeClassifier::new())],
            Box::new(DecisionTreeClassifier::new()),
        );
        let result = sc.predict(&[vec![1.0, 2.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_stacking_empty_estimators() {
        let data = make_iris_like_data();
        let mut sc = StackingClassifier::new(vec![], Box::new(DecisionTreeClassifier::new()));
        assert!(sc.fit(&data).is_err());
    }

    #[test]
    fn test_stacking_cv_too_small() {
        let data = make_iris_like_data();
        let mut sc = StackingClassifier::new(
            vec![Box::new(DecisionTreeClassifier::new())],
            Box::new(DecisionTreeClassifier::new()),
        )
        .cv(1);
        assert!(sc.fit(&data).is_err());
    }

    #[test]
    fn test_generate_fold_indices() {
        let folds = generate_fold_indices(10, 3, 42);
        assert_eq!(folds.len(), 3);
        let total: usize = folds.iter().map(std::vec::Vec::len).sum();
        assert_eq!(total, 10);
        // All indices present.
        let mut all: Vec<usize> = folds.into_iter().flatten().collect();
        all.sort_unstable();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_voting_accuracy_ge_individual() {
        let data = make_iris_like_data();
        let features = data.feature_matrix();

        // Train individual trees and get their accuracies.
        let mut dt1 = DecisionTreeClassifier::new().max_depth(2);
        dt1.fit(&data).unwrap();
        let preds1 = dt1.predict(&features).unwrap();
        let acc1 = preds1
            .iter()
            .zip(data.target.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-6)
            .count() as f64
            / data.n_samples() as f64;

        // Voting with 3 trees — accuracy should generally be >= worst individual.
        let mut vc = VotingClassifier::new(vec![
            Box::new(DecisionTreeClassifier::new().max_depth(2)),
            Box::new(DecisionTreeClassifier::new().max_depth(4)),
            Box::new(DecisionTreeClassifier::new().max_depth(6)),
        ])
        .voting(Voting::Hard);

        vc.fit(&data).unwrap();
        let preds_vc = vc.predict(&features).unwrap();
        let acc_vc = preds_vc
            .iter()
            .zip(data.target.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-6)
            .count() as f64
            / data.n_samples() as f64;

        // Ensemble should be at least as good as shallow tree.
        assert!(
            acc_vc >= acc1 - 0.05,
            "VotingClassifier ({:.1}%) should be ≥ individual DT ({:.1}%) - 5%",
            acc_vc * 100.0,
            acc1 * 100.0,
        );
    }
}
