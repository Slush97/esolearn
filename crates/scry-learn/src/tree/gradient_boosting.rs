// SPDX-License-Identifier: MIT OR Apache-2.0
//! Gradient Boosted Trees — sequential ensemble for classification and regression.
//!
//! Each boosting round fits a shallow regression tree to the negative gradient
//! (pseudo-residuals) of the loss function. Prediction is the sum of all trees
//! scaled by `learning_rate` plus the initial prediction.
//!
//! Uses **Newton-Raphson leaf correction** for classification (second-order
//! gradient step), matching sklearn's `GradientBoostingClassifier` behavior.
//!
//! Internally reuses [`DecisionTreeRegressor`] with [`FlatTree`] for
//! cache-optimal prediction of each weak learner.

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::neural::callback::{CallbackAction, EpochMetrics, TrainingCallback, TrainingHistory};
use crate::tree::cart::{presort_indices, DecisionTreeRegressor};
use crate::weights::{compute_sample_weights, ClassWeight};

// ═══════════════════════════════════════════════════════════════════════════
// Regression Loss Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Loss function for gradient boosting regression.
///
/// Controls how pseudo-residuals are computed and how leaf values are
/// determined. Different losses provide different robustness properties.
///
/// - `SquaredError` (default): standard MSE loss, optimal for Gaussian noise.
/// - `AbsoluteError`: L1 loss (MAE), more robust to outliers.
/// - `Huber { alpha }`: hybrid of squared and absolute error; `alpha` is
///   the quantile at which the transition occurs (default 0.9).
/// - `Quantile { alpha }`: predicts the `alpha`-quantile of the conditional
///   distribution (default 0.5 = median).
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum RegressionLoss {
    /// Least-squares loss (MSE). Default.
    #[default]
    SquaredError,
    /// Least-absolute-deviation loss (MAE).
    AbsoluteError,
    /// Huber loss — squared error for small residuals, absolute for large.
    /// `alpha` is the quantile threshold (typically 0.9).
    Huber {
        /// Quantile threshold for switching between squared and absolute.
        alpha: f64,
    },
    /// Quantile loss — predicts the `alpha`-quantile.
    /// `alpha` = 0.5 gives the median.
    Quantile {
        /// Target quantile in (0, 1).
        alpha: f64,
    },
}

impl RegressionLoss {
    /// Compute the initial (constant) prediction F₀.
    fn initial_prediction(&self, y: &[f64]) -> f64 {
        match self {
            Self::SquaredError => {
                let sum: f64 = y.iter().sum();
                sum / y.len() as f64
            }
            Self::AbsoluteError | Self::Huber { .. } => median(y),
            Self::Quantile { alpha } => quantile(y, *alpha),
        }
    }

    /// Compute negative gradient (pseudo-residuals) for sample `i`.
    ///
    /// `y` is the true target, `f` is the current prediction.
    fn negative_gradient(&self, y: f64, f: f64, delta: f64) -> f64 {
        match self {
            Self::SquaredError => y - f,
            Self::AbsoluteError => {
                if y > f {
                    1.0
                } else if y < f {
                    -1.0
                } else {
                    0.0
                }
            }
            Self::Huber { .. } => {
                let r = y - f;
                if r.abs() <= delta {
                    r
                } else {
                    delta * r.signum()
                }
            }
            Self::Quantile { alpha } => {
                if y > f {
                    *alpha
                } else if y < f {
                    -(1.0 - alpha)
                } else {
                    0.0
                }
            }
        }
    }

    /// Compute optimal leaf value for terminal regions.
    ///
    /// For SquaredError, mean of residuals is already correct (tree default).
    /// For other losses, we override leaf predictions.
    fn update_terminal_value(
        &self,
        residuals: &[f64],
        y_in_leaf: &[f64],
        f_in_leaf: &[f64],
        delta: f64,
    ) -> f64 {
        match self {
            Self::SquaredError => {
                // Tree already computes mean — no override needed.
                if residuals.is_empty() {
                    0.0
                } else {
                    residuals.iter().sum::<f64>() / residuals.len() as f64
                }
            }
            Self::AbsoluteError => median(residuals),
            Self::Huber { .. } => {
                // Median of residuals + mean of clipped tails.
                let med = median(residuals);
                let correction: f64 = residuals
                    .iter()
                    .map(|&r| {
                        let diff = r - med;
                        diff.clamp(-delta, delta)
                    })
                    .sum::<f64>()
                    / residuals.len().max(1) as f64;
                med + correction
            }
            Self::Quantile { alpha } => {
                // Compute residuals from current predictions.
                let diffs: Vec<f64> = y_in_leaf
                    .iter()
                    .zip(f_in_leaf.iter())
                    .map(|(&y, &f)| y - f)
                    .collect();
                quantile(&diffs, *alpha)
            }
        }
    }

    /// Whether this loss needs terminal region updates (overriding tree leaf values).
    fn needs_terminal_update(&self) -> bool {
        !matches!(self, Self::SquaredError)
    }
}

/// Compute the median of a slice.
fn median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Compute the `alpha`-quantile of a slice (linear interpolation).
fn quantile(data: &[f64], alpha: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = alpha * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Gradient Boosting Regressor
// ═══════════════════════════════════════════════════════════════════════════

/// Gradient Boosting for regression.
///
/// Builds an additive ensemble of shallow decision trees, each fitting the
/// negative gradient (pseudo-residuals) of the loss function. Supports
/// stochastic subsampling and multiple loss functions.
///
/// # Example
/// ```
/// use scry_learn::dataset::Dataset;
/// use scry_learn::tree::GradientBoostingRegressor;
///
/// let features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
/// let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
/// let data = Dataset::new(features, target, vec!["x".into()], "y");
///
/// let mut gbr = GradientBoostingRegressor::new()
///     .n_estimators(50)
///     .learning_rate(0.1)
///     .max_depth(3);
/// gbr.fit(&data).unwrap();
///
/// let preds = gbr.predict(&[vec![3.0]]).unwrap();
/// assert!((preds[0] - 6.0).abs() < 1.0);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct GradientBoostingRegressor {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    subsample: f64,
    seed: u64,
    loss: RegressionLoss,
    validation_fraction: f64,
    n_iter_no_change: Option<usize>,
    tol: f64,
    // Fitted state
    trees: Vec<DecisionTreeRegressor>,
    init_prediction: f64,
    n_features: usize,
    fitted: bool,
    n_estimators_used: usize,
    history: Option<TrainingHistory>,
    /// User-supplied training callbacks (not cloned or serialized).
    #[cfg_attr(feature = "serde", serde(skip))]
    callbacks: Vec<Box<dyn TrainingCallback>>,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl Clone for GradientBoostingRegressor {
    fn clone(&self) -> Self {
        Self {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            subsample: self.subsample,
            seed: self.seed,
            loss: self.loss.clone(),
            validation_fraction: self.validation_fraction,
            n_iter_no_change: self.n_iter_no_change,
            tol: self.tol,
            trees: self.trees.clone(),
            init_prediction: self.init_prediction,
            n_features: self.n_features,
            fitted: self.fitted,
            n_estimators_used: self.n_estimators_used,
            history: self.history.clone(),
            callbacks: Vec::new(),
            _schema_version: self._schema_version,
        }
    }
}

impl GradientBoostingRegressor {
    /// Create a new regressor with default parameters.
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: 3,
            min_samples_split: 2,
            min_samples_leaf: 1,
            subsample: 1.0,
            seed: 42,
            loss: RegressionLoss::SquaredError,
            validation_fraction: 0.1,
            n_iter_no_change: None,
            tol: crate::constants::DEFAULT_TOL,
            trees: Vec::new(),
            init_prediction: 0.0,
            n_features: 0,
            fitted: false,
            n_estimators_used: 0,
            history: None,
            callbacks: Vec::new(),
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set number of boosting rounds.
    pub fn n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Set learning rate (shrinkage). Lower values need more estimators.
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set maximum depth per tree (default: 3, shallow stumps).
    pub fn max_depth(mut self, d: usize) -> Self {
        self.max_depth = d;
        self
    }

    /// Set minimum samples required to split an internal node.
    pub fn min_samples_split(mut self, n: usize) -> Self {
        self.min_samples_split = n;
        self
    }

    /// Set minimum samples required in a leaf node.
    pub fn min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }

    /// Set subsample fraction (0.0, 1.0] for stochastic GBT.
    pub fn subsample(mut self, s: f64) -> Self {
        self.subsample = s;
        self
    }

    /// Set random seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Enable early stopping. Training stops when validation loss does not
    /// improve for `n` consecutive rounds.
    pub fn n_iter_no_change(mut self, n: usize) -> Self {
        self.n_iter_no_change = Some(n);
        self
    }

    /// Set fraction of training data to use as validation for early stopping
    /// (default: 0.1).
    pub fn validation_fraction(mut self, frac: f64) -> Self {
        self.validation_fraction = frac;
        self
    }

    /// Set tolerance for early stopping (default: 1e-4).
    pub fn tol(mut self, t: f64) -> Self {
        self.tol = t;
        self
    }

    /// Add a training callback (invoked after each boosting round).
    pub fn callback(mut self, cb: Box<dyn TrainingCallback>) -> Self {
        self.callbacks.push(cb);
        self
    }

    /// Number of estimators actually used (may be less than `n_estimators`
    /// if early stopping triggered).
    pub fn n_estimators_used(&self) -> usize {
        self.n_estimators_used
    }

    /// Set the regression loss function.
    ///
    /// # Example
    /// ```
    /// use scry_learn::tree::{GradientBoostingRegressor, RegressionLoss};
    ///
    /// let gbr = GradientBoostingRegressor::new()
    ///     .loss(RegressionLoss::Huber { alpha: 0.9 });
    /// ```
    pub fn loss(mut self, l: RegressionLoss) -> Self {
        self.loss = l;
        self
    }

    /// Train the gradient boosting ensemble.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        let n = data.n_samples();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err(ScryLearnError::InvalidParameter(
                "learning_rate must be in (0, 1]".into(),
            ));
        }
        if self.subsample <= 0.0 || self.subsample > 1.0 {
            return Err(ScryLearnError::InvalidParameter(
                "subsample must be in (0, 1]".into(),
            ));
        }

        self.n_features = data.n_features();

        // ── Early stopping: split into train / validation ──
        let (train_data, val_data) = if self.n_iter_no_change.is_some() {
            let (t, v) = crate::split::train_test_split(data, self.validation_fraction, self.seed);
            (t, Some(v))
        } else {
            (data.clone(), None)
        };
        let n_train = train_data.n_samples();

        // F₀ = loss-specific initial prediction
        let init = self.loss.initial_prediction(&train_data.target);
        self.init_prediction = init;

        // Current predictions for each training sample.
        let mut f_vals = vec![init; n_train];

        // Compute Huber delta (quantile of |y - F₀|) — used throughout training.
        let delta = match &self.loss {
            RegressionLoss::Huber { alpha } => {
                let abs_resid: Vec<f64> = train_data
                    .target
                    .iter()
                    .zip(f_vals.iter())
                    .map(|(&y, &f)| (y - f).abs())
                    .collect();
                quantile(&abs_resid, *alpha)
            }
            _ => 0.0, // unused for other losses
        };

        let mut rng = crate::rng::FastRng::new(self.seed);
        let all_indices: Vec<usize> = (0..n_train).collect();
        self.trees = Vec::with_capacity(self.n_estimators);

        // Reusable dataset — share features, replace target each round.
        let mut temp_data = Dataset::new(
            train_data.features.clone(),
            vec![0.0; n_train],
            train_data.feature_names.clone(),
            "residual",
        );
        let row_major = train_data.feature_matrix();

        // Pre-sort indices once — reused across all boosting rounds.
        // Feature values don't change between rounds (only targets/residuals do),
        // so the sorted order is valid throughout training.
        let global_sorted = presort_indices(&temp_data, &all_indices);

        // Early stopping state.
        let mut best_val_loss = f64::INFINITY;
        let mut no_improve_count = 0usize;
        let patience = self.n_iter_no_change.unwrap_or(usize::MAX);

        let mut history = TrainingHistory::new();
        let mut callbacks = std::mem::take(&mut self.callbacks);

        for round in 0..self.n_estimators {
            let round_start = std::time::Instant::now();

            // Compute negative gradient (pseudo-residuals).
            for (i, fv) in f_vals.iter().enumerate().take(n_train) {
                temp_data.target[i] = self
                    .loss
                    .negative_gradient(train_data.target[i], *fv, delta);
            }

            // Subsample indices.
            let indices = subsample_indices(n_train, self.subsample, &mut rng, &all_indices);

            // Fit a shallow regression tree to the pseudo-residuals.
            let mut tree = DecisionTreeRegressor::new()
                .max_depth(self.max_depth)
                .min_samples_split(self.min_samples_split)
                .min_samples_leaf(self.min_samples_leaf);
            tree.fit_on_indices_presorted(&temp_data, &indices, &global_sorted)?;

            // For non-squared-error losses, override leaf values with
            // the loss-specific optimal terminal region update.
            if self.loss.needs_terminal_update() {
                if let Some(ref mut flat) = tree.flat_tree {
                    // Compute leaf assignments for training samples.
                    let leaf_ids = flat.apply(&row_major);
                    let n_nodes = flat.n_nodes();
                    let mut leaf_residuals: Vec<Vec<f64>> = vec![Vec::new(); n_nodes];
                    let mut leaf_y: Vec<Vec<f64>> = vec![Vec::new(); n_nodes];
                    let mut leaf_f: Vec<Vec<f64>> = vec![Vec::new(); n_nodes];
                    for (i, &lid) in leaf_ids.iter().enumerate() {
                        leaf_residuals[lid].push(temp_data.target[i]);
                        leaf_y[lid].push(train_data.target[i]);
                        leaf_f[lid].push(f_vals[i]);
                    }
                    for node_id in 0..n_nodes {
                        if !leaf_residuals[node_id].is_empty() {
                            let new_val = self.loss.update_terminal_value(
                                &leaf_residuals[node_id],
                                &leaf_y[node_id],
                                &leaf_f[node_id],
                                delta,
                            );
                            flat.set_leaf_prediction(node_id, new_val);
                        }
                    }
                }
            }

            // Update predictions: F(x_i) += η × tree.predict(x_i)
            let tree_preds = tree.predict(&row_major)?;
            for (f_val, &tp) in f_vals.iter_mut().zip(tree_preds.iter()) {
                *f_val += self.learning_rate * tp;
            }

            self.trees.push(tree);

            // Compute training loss (MSE on training set).
            let train_mse: f64 = train_data
                .target
                .iter()
                .zip(f_vals.iter())
                .map(|(&y, &f)| (y - f).powi(2))
                .sum::<f64>()
                / n_train as f64;

            // Gradient norm: L2 norm of pseudo-residuals (approximation for trees).
            let grad_norm: f64 = temp_data
                .target
                .iter()
                .take(n_train)
                .map(|&r| r * r)
                .sum::<f64>()
                .sqrt();

            let elapsed = round_start.elapsed().as_millis() as u64;

            let metrics = EpochMetrics {
                epoch: round,
                train_loss: train_mse,
                val_loss: None, // updated below if early stopping
                train_metric: None,
                val_metric: None,
                learning_rate: self.learning_rate,
                grad_norm,
                elapsed_ms: elapsed,
            };

            let mut cb_stop = false;
            for cb in &mut callbacks {
                if cb.on_epoch_end(&metrics) == CallbackAction::Stop {
                    cb_stop = true;
                }
            }

            history.push(metrics);

            if cb_stop {
                self.n_estimators_used = round + 1;
                self.fitted = true;
                for cb in &mut callbacks {
                    cb.on_training_end();
                }
                self.callbacks = callbacks;
                self.history = Some(history);
                return Ok(());
            }

            // ── Check early stopping ──
            if let Some(ref val) = val_data {
                let val_features = val.feature_matrix();
                let mut val_preds = vec![self.init_prediction; val_features.len()];
                for t in &self.trees {
                    if let Ok(tp) = t.predict(&val_features) {
                        for (p, &v) in val_preds.iter_mut().zip(tp.iter()) {
                            *p += self.learning_rate * v;
                        }
                    }
                }
                let val_mse: f64 = val
                    .target
                    .iter()
                    .zip(val_preds.iter())
                    .map(|(&y, &p)| (y - p).powi(2))
                    .sum::<f64>()
                    / val.target.len() as f64;

                // Record val_loss in history.
                if let Some(last) = history.epochs.last_mut() {
                    last.val_loss = Some(val_mse);
                }

                if val_mse + self.tol < best_val_loss {
                    best_val_loss = val_mse;
                    no_improve_count = 0;
                } else {
                    no_improve_count += 1;
                    if no_improve_count >= patience {
                        self.n_estimators_used = round + 1;
                        self.fitted = true;
                        for cb in &mut callbacks {
                            cb.on_training_end();
                        }
                        self.callbacks = callbacks;
                        self.history = Some(history);
                        return Ok(());
                    }
                }
            }
        }

        self.n_estimators_used = self.trees.len();
        self.fitted = true;
        for cb in &mut callbacks {
            cb.on_training_end();
        }
        self.callbacks = callbacks;
        self.history = Some(history);
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
            let tp = tree.predict(features)?;
            for (p, &t) in preds.iter_mut().zip(tp.iter()) {
                *p += self.learning_rate * t;
            }
        }
        Ok(preds)
    }

    /// Feature importances averaged across all trees.
    pub fn feature_importances(&self) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let m = self.n_features;
        let mut importances = vec![0.0; m];
        let n_trees = self.trees.len() as f64;
        for tree in &self.trees {
            if let Ok(imp) = tree.feature_importances() {
                for (i, &v) in imp.iter().enumerate() {
                    if i < m {
                        importances[i] += v / n_trees;
                    }
                }
            }
        }
        // Normalize.
        let total: f64 = importances.iter().sum();
        if total > 0.0 {
            for v in &mut importances {
                *v /= total;
            }
        }
        Ok(importances)
    }

    /// Number of estimators (trees) in the ensemble.
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Whether early stopping was triggered.
    pub fn early_stopped(&self) -> bool {
        self.n_iter_no_change.is_some() && self.n_estimators_used < self.n_estimators
    }

    /// Return training history (populated after `fit()`).
    pub fn history(&self) -> Option<&TrainingHistory> {
        self.history.as_ref()
    }

    /// Get individual trees (for inspection or ONNX export).
    pub fn trees(&self) -> &[DecisionTreeRegressor] {
        &self.trees
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
}

impl Default for GradientBoostingRegressor {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Gradient Boosting Classifier
// ═══════════════════════════════════════════════════════════════════════════

/// Gradient Boosting for classification (binary + multiclass).
///
/// - Binary: fits a single sequence of trees to log-loss pseudo-residuals.
/// - Multiclass (K > 2): fits K sequences of trees (one-vs-all softmax).
///
/// Uses **Newton-Raphson leaf correction** (second-order gradient step) for
/// optimal leaf values, matching sklearn's `GradientBoostingClassifier`.
///
/// # Example
/// ```
/// use scry_learn::dataset::Dataset;
/// use scry_learn::tree::GradientBoostingClassifier;
///
/// // Simple linearly separable data.
/// let features = vec![
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
///     vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
/// ];
/// let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
/// let data = Dataset::new(features, target, vec!["x1".into(), "x2".into()], "class");
///
/// let mut gbc = GradientBoostingClassifier::new()
///     .n_estimators(50)
///     .learning_rate(0.1)
///     .max_depth(2);
/// gbc.fit(&data).unwrap();
///
/// let preds = gbc.predict(&[vec![1.5, 0.15], vec![5.5, 0.55]]).unwrap();
/// assert_eq!(preds[0], 0.0);
/// assert_eq!(preds[1], 1.0);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct GradientBoostingClassifier {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    subsample: f64,
    seed: u64,
    class_weight: ClassWeight,
    // Fitted state — trees[class_idx][estimator_idx]
    trees: Vec<Vec<DecisionTreeRegressor>>,
    init_predictions: Vec<f64>,
    n_classes: usize,
    n_features: usize,
    fitted: bool,
    history: Option<TrainingHistory>,
    /// User-supplied training callbacks (not cloned or serialized).
    #[cfg_attr(feature = "serde", serde(skip))]
    callbacks: Vec<Box<dyn TrainingCallback>>,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl Clone for GradientBoostingClassifier {
    fn clone(&self) -> Self {
        Self {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            subsample: self.subsample,
            seed: self.seed,
            class_weight: self.class_weight.clone(),
            trees: self.trees.clone(),
            init_predictions: self.init_predictions.clone(),
            n_classes: self.n_classes,
            n_features: self.n_features,
            fitted: self.fitted,
            history: self.history.clone(),
            callbacks: Vec::new(),
            _schema_version: self._schema_version,
        }
    }
}

impl GradientBoostingClassifier {
    /// Create a new classifier with default parameters.
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: 3,
            min_samples_split: 2,
            min_samples_leaf: 1,
            subsample: 1.0,
            seed: 42,
            class_weight: ClassWeight::Uniform,
            trees: Vec::new(),
            init_predictions: Vec::new(),
            n_classes: 0,
            n_features: 0,
            fitted: false,
            history: None,
            callbacks: Vec::new(),
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set number of boosting rounds.
    pub fn n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Set learning rate (shrinkage).
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set maximum depth per tree.
    pub fn max_depth(mut self, d: usize) -> Self {
        self.max_depth = d;
        self
    }

    /// Set minimum samples required to split.
    pub fn min_samples_split(mut self, n: usize) -> Self {
        self.min_samples_split = n;
        self
    }

    /// Set minimum samples required in a leaf.
    pub fn min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }

    /// Set subsample fraction for stochastic GBT.
    pub fn subsample(mut self, s: f64) -> Self {
        self.subsample = s;
        self
    }

    /// Set random seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Set class weighting strategy for imbalanced datasets.
    pub fn class_weight(mut self, cw: ClassWeight) -> Self {
        self.class_weight = cw;
        self
    }

    /// Add a training callback (invoked after each boosting round).
    pub fn callback(mut self, cb: Box<dyn TrainingCallback>) -> Self {
        self.callbacks.push(cb);
        self
    }

    /// Train the gradient boosting classifier.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        let n = data.n_samples();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err(ScryLearnError::InvalidParameter(
                "learning_rate must be in (0, 1]".into(),
            ));
        }
        if self.subsample <= 0.0 || self.subsample > 1.0 {
            return Err(ScryLearnError::InvalidParameter(
                "subsample must be in (0, 1]".into(),
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

        let mut rng = crate::rng::FastRng::new(self.seed);
        let all_indices: Vec<usize> = (0..n).collect();
        let row_major = data.feature_matrix();
        let sample_weights = compute_sample_weights(&data.target, &self.class_weight);

        if k == 2 {
            // ── Binary classification via log-loss ──
            self.fit_binary(data, n, &mut rng, &all_indices, &row_major, &sample_weights)?;
        } else {
            // ── Multiclass via softmax (K one-vs-all) ──
            self.fit_multiclass(
                data,
                n,
                k,
                &mut rng,
                &all_indices,
                &row_major,
                &sample_weights,
            )?;
        }
        Ok(())
    }

    /// Binary classification: single sequence of trees with Newton leaf correction.
    fn fit_binary(
        &mut self,
        data: &Dataset,
        n: usize,
        rng: &mut crate::rng::FastRng,
        all_indices: &[usize],
        row_major: &[Vec<f64>],
        sample_weights: &[f64],
    ) -> Result<()> {
        // Class prior: p = count(y=1) / n
        let pos_count = data.target.iter().filter(|&&y| y > 0.5).count();
        let p = (pos_count as f64) / (n as f64);
        let p_clamped = p.clamp(crate::constants::GBT_PROB_CLAMP, 1.0 - crate::constants::GBT_PROB_CLAMP);
        let f0 = (p_clamped / (1.0 - p_clamped)).ln(); // log-odds
        self.init_predictions = vec![f0];

        let mut f_vals = vec![f0; n];
        let mut trees_seq = Vec::with_capacity(self.n_estimators);
        let mut history = TrainingHistory::new();
        let mut callbacks = std::mem::take(&mut self.callbacks);

        // Reusable dataset — share features, replace target each round.
        let mut temp_data = Dataset::new(
            data.features.clone(),
            vec![0.0; n],
            data.feature_names.clone(),
            "residual",
        );

        // Pre-sort indices once — reused across all boosting rounds.
        let global_sorted = presort_indices(&temp_data, all_indices);

        for round in 0..self.n_estimators {
            let round_start = std::time::Instant::now();

            // Compute probabilities and pseudo-residuals.
            let probs: Vec<f64> = f_vals.iter().map(|&f| sigmoid(f)).collect();

            // pseudo-residuals: r_i = weight_i * (y_i - sigmoid(F(x_i)))
            for i in 0..n {
                temp_data.target[i] = sample_weights[i] * (data.target[i] - probs[i]);
            }

            let indices = subsample_indices(n, self.subsample, rng, all_indices);

            let mut tree = DecisionTreeRegressor::new()
                .max_depth(self.max_depth)
                .min_samples_split(self.min_samples_split)
                .min_samples_leaf(self.min_samples_leaf);
            tree.fit_on_indices_presorted(&temp_data, &indices, &global_sorted)?;

            // ── Newton-Raphson leaf correction ──
            // For each leaf, replace the mean residual with:
            //   leaf_value = Σ(residual_i) / Σ(p_i × (1 - p_i))
            // This is the optimal Newton step for log-loss.
            if let Some(ref mut flat) = tree.flat_tree {
                let leaf_indices = flat.apply(row_major);
                newton_correct_binary_leaves(
                    flat,
                    &leaf_indices,
                    &temp_data.target, // residuals
                    &probs,
                );
            }

            let tp = tree.predict(row_major)?;
            for (f_val, &t) in f_vals.iter_mut().zip(tp.iter()) {
                *f_val += self.learning_rate * t;
            }

            trees_seq.push(tree);

            // Binary cross-entropy loss: -mean(y*log(p) + (1-y)*log(1-p))
            let probs_after: Vec<f64> = f_vals.iter().map(|&f| sigmoid(f)).collect();
            let train_loss: f64 = data
                .target
                .iter()
                .zip(probs_after.iter())
                .map(|(&y, &p)| {
                    let p_c = p.clamp(crate::constants::NEAR_ZERO, 1.0 - crate::constants::NEAR_ZERO);
                    -(y * p_c.ln() + (1.0 - y) * (1.0 - p_c).ln())
                })
                .sum::<f64>()
                / n as f64;

            // Gradient norm: L2 norm of residuals.
            let grad_norm: f64 = temp_data
                .target
                .iter()
                .take(n)
                .map(|&r| r * r)
                .sum::<f64>()
                .sqrt();

            let elapsed = round_start.elapsed().as_millis() as u64;

            let metrics = EpochMetrics {
                epoch: round,
                train_loss,
                val_loss: None,
                train_metric: None,
                val_metric: None,
                learning_rate: self.learning_rate,
                grad_norm,
                elapsed_ms: elapsed,
            };

            let mut cb_stop = false;
            for cb in &mut callbacks {
                if cb.on_epoch_end(&metrics) == CallbackAction::Stop {
                    cb_stop = true;
                }
            }

            history.push(metrics);

            if cb_stop {
                break;
            }
        }

        self.trees = vec![trees_seq];
        self.fitted = true;
        for cb in &mut callbacks {
            cb.on_training_end();
        }
        self.callbacks = callbacks;
        self.history = Some(history);
        Ok(())
    }

    /// Multiclass classification: K parallel tree sequences (softmax) with Newton correction.
    #[allow(clippy::too_many_arguments)]
    fn fit_multiclass(
        &mut self,
        data: &Dataset,
        n: usize,
        k: usize,
        rng: &mut crate::rng::FastRng,
        all_indices: &[usize],
        row_major: &[Vec<f64>],
        sample_weights: &[f64],
    ) -> Result<()> {
        // Build one-hot targets: y_k[i] = 1 if target[i] == k, else 0.
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
            .map(|&c| {
                let p = (c as f64 / n as f64).clamp(crate::constants::GBT_PROB_CLAMP, 1.0 - crate::constants::GBT_PROB_CLAMP);
                p.ln()
            })
            .collect();
        self.init_predictions.clone_from(&init_preds);

        // f_vals[class][sample]
        let mut f_vals: Vec<Vec<f64>> = (0..k).map(|c| vec![init_preds[c]; n]).collect();

        let mut trees_all: Vec<Vec<DecisionTreeRegressor>> = (0..k)
            .map(|_| Vec::with_capacity(self.n_estimators))
            .collect();
        let mut history = TrainingHistory::new();
        let mut callbacks = std::mem::take(&mut self.callbacks);

        // Reusable dataset — share features, replace target each round.
        let mut temp_data = Dataset::new(
            data.features.clone(),
            vec![0.0; n],
            data.feature_names.clone(),
            "residual",
        );

        // Pre-sort indices once — reused across all boosting rounds.
        let global_sorted = presort_indices(&temp_data, all_indices);

        for round in 0..self.n_estimators {
            let round_start = std::time::Instant::now();
            // Compute softmax probabilities.
            let probs = softmax_matrix(&f_vals, n, k);

            let indices = subsample_indices(n, self.subsample, rng, all_indices);

            // Fit one tree per class.
            for cls in 0..k {
                // pseudo-residuals: r_i = weight_i * (y_k[i] - p_k[i])
                for i in 0..n {
                    temp_data.target[i] = sample_weights[i] * (y_onehot[cls][i] - probs[cls][i]);
                }

                let mut tree = DecisionTreeRegressor::new()
                    .max_depth(self.max_depth)
                    .min_samples_split(self.min_samples_split)
                    .min_samples_leaf(self.min_samples_leaf);
                tree.fit_on_indices_presorted(&temp_data, &indices, &global_sorted)?;

                // ── Newton-Raphson leaf correction for multiclass ──
                // For each leaf:
                //   leaf_value = (K-1)/K × Σ(residual_i) / Σ(p_i × (1 - p_i))
                if let Some(ref mut flat) = tree.flat_tree {
                    let leaf_indices = flat.apply(row_major);
                    newton_correct_multiclass_leaves(
                        flat,
                        &leaf_indices,
                        &temp_data.target, // residuals
                        &probs[cls],       // softmax probabilities for this class
                        k,
                    );
                }

                let tp = tree.predict(row_major)?;
                for (f_val, &t) in f_vals[cls].iter_mut().zip(tp.iter()) {
                    *f_val += self.learning_rate * t;
                }

                trees_all[cls].push(tree);
            }

            // Cross-entropy loss: -mean(sum_k y_k * log(p_k))
            let probs_after = softmax_matrix(&f_vals, n, k);
            let train_loss: f64 = (0..n)
                .map(|i| {
                    let cls_i = data.target[i] as usize;
                    let p = probs_after[cls_i][i].clamp(crate::constants::NEAR_ZERO, 1.0 - crate::constants::NEAR_ZERO);
                    -p.ln()
                })
                .sum::<f64>()
                / n as f64;

            // Gradient norm: L2 norm of last class's residuals (representative).
            let grad_norm: f64 = temp_data
                .target
                .iter()
                .take(n)
                .map(|&r| r * r)
                .sum::<f64>()
                .sqrt();

            let elapsed = round_start.elapsed().as_millis() as u64;

            let metrics = EpochMetrics {
                epoch: round,
                train_loss,
                val_loss: None,
                train_metric: None,
                val_metric: None,
                learning_rate: self.learning_rate,
                grad_norm,
                elapsed_ms: elapsed,
            };

            let mut cb_stop = false;
            for cb in &mut callbacks {
                if cb.on_epoch_end(&metrics) == CallbackAction::Stop {
                    cb_stop = true;
                }
            }

            history.push(metrics);

            if cb_stop {
                break;
            }
        }

        self.trees = trees_all;
        self.fitted = true;
        for cb in &mut callbacks {
            cb.on_training_end();
        }
        self.callbacks = callbacks;
        self.history = Some(history);
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
            // Binary: single tree sequence.
            let mut f_vals = vec![self.init_predictions[0]; n];
            for tree in &self.trees[0] {
                let tp = tree.predict(features)?;
                for (f, &t) in f_vals.iter_mut().zip(tp.iter()) {
                    *f += self.learning_rate * t;
                }
            }
            Ok(f_vals
                .iter()
                .map(|&f| {
                    let p1 = sigmoid(f);
                    vec![1.0 - p1, p1]
                })
                .collect())
        } else {
            // Multiclass: K tree sequences → softmax.
            let mut f_vals: Vec<Vec<f64>> =
                (0..k).map(|c| vec![self.init_predictions[c]; n]).collect();
            for (cls_fvals, cls_trees) in f_vals.iter_mut().zip(self.trees.iter()).take(k) {
                for tree in cls_trees {
                    let tp = tree.predict(features)?;
                    for (f, &t) in cls_fvals.iter_mut().zip(tp.iter()) {
                        *f += self.learning_rate * t;
                    }
                }
            }
            // Softmax across classes for each sample.
            let mut result = Vec::with_capacity(n);
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let logits: Vec<f64> = (0..k).map(|c| f_vals[c][i]).collect();
                let max_l = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = logits.iter().map(|&l| (l - max_l).exp()).collect();
                let sum: f64 = exps.iter().sum();
                result.push(exps.iter().map(|&e| e / sum).collect());
            }
            Ok(result)
        }
    }

    /// Feature importances averaged across all trees.
    pub fn feature_importances(&self) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let m = self.n_features;
        let mut importances = vec![0.0; m];
        let mut total_trees = 0.0;
        for class_trees in &self.trees {
            for tree in class_trees {
                if let Ok(imp) = tree.feature_importances() {
                    for (i, &v) in imp.iter().enumerate() {
                        if i < m {
                            importances[i] += v;
                        }
                    }
                }
                total_trees += 1.0;
            }
        }
        if total_trees > 0.0 {
            for v in &mut importances {
                *v /= total_trees;
            }
        }
        let total: f64 = importances.iter().sum();
        if total > 0.0 {
            for v in &mut importances {
                *v /= total;
            }
        }
        Ok(importances)
    }

    /// Number of classes.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Total number of trees across all class sequences.
    pub fn n_trees(&self) -> usize {
        self.trees.iter().map(Vec::len).sum()
    }

    /// Return training history (populated after `fit()`).
    pub fn history(&self) -> Option<&TrainingHistory> {
        self.history.as_ref()
    }

    /// Get tree sequences per class (for inspection or ONNX export).
    /// `class_trees()[class_idx][estimator_idx]` is the tree for that class/round.
    pub fn class_trees(&self) -> &[Vec<DecisionTreeRegressor>] {
        &self.trees
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
}

impl Default for GradientBoostingClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Newton-Raphson leaf correction
// ═══════════════════════════════════════════════════════════════════════════

/// Newton-Raphson correction for binary log-loss.
///
/// For each leaf, replace the mean residual with:
///   leaf_value = Σ(residual_i) / Σ(p_i × (1 - p_i))
///
/// where p_i = sigmoid(F(x_i)) and residual_i = y_i - p_i.
///
/// This is the optimal second-order correction step from Friedman (2001).
fn newton_correct_binary_leaves(
    flat: &mut crate::tree::cart::FlatTree,
    leaf_indices: &[usize],
    residuals: &[f64],
    probs: &[f64],
) {
    use std::collections::HashMap;

    // Accumulate numerator (Σresid) and denominator (Σp*(1-p)) per leaf.
    let mut leaf_num: HashMap<usize, f64> = HashMap::new();
    let mut leaf_den: HashMap<usize, f64> = HashMap::new();

    for (i, &leaf_idx) in leaf_indices.iter().enumerate() {
        let r = residuals[i];
        let p = probs[i];
        let hessian = p * (1.0 - p);
        *leaf_num.entry(leaf_idx).or_insert(0.0) += r;
        *leaf_den.entry(leaf_idx).or_insert(0.0) += hessian;
    }

    // Overwrite leaf predictions with Newton-corrected values.
    for (&leaf_idx, &num) in &leaf_num {
        let den = leaf_den[&leaf_idx];
        // Avoid division by zero; fall back to gradient mean.
        if den.abs() > crate::constants::SINGULAR_THRESHOLD {
            flat.set_leaf_prediction(leaf_idx, num / den);
        }
    }
}

/// Newton-Raphson correction for multiclass softmax.
///
/// For each leaf, replace the mean residual with:
///   leaf_value = (K-1)/K × Σ(residual_i) / Σ(p_i × (1 - p_i))
///
/// where p_i is the softmax probability for the current class.
/// Uses the exact diagonal Hessian p(1-p), matching sklearn, XGBoost,
/// and LightGBM (not the Friedman 2001 |r|(1-|r|) approximation).
fn newton_correct_multiclass_leaves(
    flat: &mut crate::tree::cart::FlatTree,
    leaf_indices: &[usize],
    residuals: &[f64],
    probs: &[f64],
    k: usize,
) {
    use std::collections::HashMap;

    let factor = (k - 1) as f64 / k as f64;

    let mut leaf_num: HashMap<usize, f64> = HashMap::new();
    let mut leaf_den: HashMap<usize, f64> = HashMap::new();

    for (i, &leaf_idx) in leaf_indices.iter().enumerate() {
        let r = residuals[i];
        let p = probs[i];
        let hessian = (p * (1.0 - p)).max(crate::constants::SINGULAR_THRESHOLD);
        *leaf_num.entry(leaf_idx).or_insert(0.0) += r;
        *leaf_den.entry(leaf_idx).or_insert(0.0) += hessian;
    }

    for (&leaf_idx, &num) in &leaf_num {
        let den = leaf_den[&leaf_idx];
        if den.abs() > crate::constants::SINGULAR_THRESHOLD {
            flat.set_leaf_prediction(leaf_idx, factor * num / den);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper functions
// ═══════════════════════════════════════════════════════════════════════════

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute softmax probabilities: probs[class][sample].
fn softmax_matrix(f_vals: &[Vec<f64>], n: usize, k: usize) -> Vec<Vec<f64>> {
    let mut probs = vec![vec![0.0; n]; k];
    for i in 0..n {
        let logits: Vec<f64> = (0..k).map(|c| f_vals[c][i]).collect();
        let max_l = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&l| (l - max_l).exp()).collect();
        let sum: f64 = exps.iter().sum();
        for c in 0..k {
            probs[c][i] = exps[c] / sum;
        }
    }
    probs
}

/// Subsample indices using Fisher-Yates partial shuffle.
fn subsample_indices(
    n: usize,
    subsample: f64,
    rng: &mut crate::rng::FastRng,
    all_indices: &[usize],
) -> Vec<usize> {
    if subsample >= 1.0 {
        return all_indices.to_vec();
    }
    let k = ((n as f64) * subsample).ceil() as usize;
    let mut idx = all_indices.to_vec();
    for i in 0..k.min(n) {
        let j = rng.usize(i..n);
        idx.swap(i, j);
    }
    idx.truncate(k);
    idx
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn make_linear_data(n: usize) -> Dataset {
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| 2.0 * v + 1.0).collect(); // y = 2x + 1
        Dataset::new(vec![x], y, vec!["x".into()], "y")
    }

    fn make_binary_data() -> Dataset {
        // Two linearly separable clusters.
        let mut f1 = Vec::new();
        let mut f2 = Vec::new();
        let mut target = Vec::new();
        for i in 0..50 {
            let v = i as f64 / 50.0;
            f1.push(v);
            f2.push(v * 0.5);
            target.push(0.0);
        }
        for i in 0..50 {
            let v = 1.0 + i as f64 / 50.0;
            f1.push(v);
            f2.push(v * 0.5);
            target.push(1.0);
        }
        Dataset::new(vec![f1, f2], target, vec!["f1".into(), "f2".into()], "cls")
    }

    fn make_multiclass_data() -> Dataset {
        let mut f1 = Vec::new();
        let mut f2 = Vec::new();
        let mut target = Vec::new();
        for i in 0..30 {
            f1.push(i as f64 / 30.0);
            f2.push(0.0);
            target.push(0.0);
        }
        for i in 0..30 {
            f1.push(2.0 + i as f64 / 30.0);
            f2.push(0.0);
            target.push(1.0);
        }
        for i in 0..30 {
            f1.push(4.0 + i as f64 / 30.0);
            f2.push(0.0);
            target.push(2.0);
        }
        Dataset::new(vec![f1, f2], target, vec!["f1".into(), "f2".into()], "cls")
    }

    // ─── Regressor tests ───

    #[test]
    fn regressor_learns_linear() {
        let data = make_linear_data(100);
        let mut gbr = GradientBoostingRegressor::new()
            .n_estimators(100)
            .learning_rate(0.1)
            .max_depth(3);
        gbr.fit(&data).unwrap();

        let preds = gbr.predict(&[vec![50.0], vec![75.0]]).unwrap();
        // y = 2x + 1 → 101, 151
        assert!((preds[0] - 101.0).abs() < 10.0, "pred={}", preds[0]);
        assert!((preds[1] - 151.0).abs() < 15.0, "pred={}", preds[1]);
    }

    #[test]
    fn regressor_not_fitted_error() {
        let gbr = GradientBoostingRegressor::new();
        assert!(gbr.predict(&[vec![1.0]]).is_err());
        assert!(gbr.feature_importances().is_err());
    }

    #[test]
    fn regressor_subsample() {
        let data = make_linear_data(100);
        let mut gbr = GradientBoostingRegressor::new()
            .n_estimators(50)
            .subsample(0.7)
            .learning_rate(0.1)
            .max_depth(3);
        gbr.fit(&data).unwrap();
        let preds = gbr.predict(&[vec![25.0]]).unwrap();
        // Should still learn something reasonable.
        assert!((preds[0] - 51.0).abs() < 15.0, "pred={}", preds[0]);
    }

    #[test]
    fn regressor_feature_importances() {
        let data = make_linear_data(100);
        let mut gbr = GradientBoostingRegressor::new()
            .n_estimators(20)
            .max_depth(2);
        gbr.fit(&data).unwrap();
        let imp = gbr.feature_importances().unwrap();
        assert_eq!(imp.len(), 1);
        assert!(
            (imp[0] - 1.0).abs() < 1e-6,
            "single feature should have importance 1.0"
        );
    }

    #[test]
    fn regressor_invalid_params() {
        let data = make_linear_data(10);
        let mut gbr = GradientBoostingRegressor::new().learning_rate(0.0);
        assert!(gbr.fit(&data).is_err());

        let mut gbr = GradientBoostingRegressor::new().subsample(1.5);
        assert!(gbr.fit(&data).is_err());
    }

    #[test]
    fn regressor_early_stopping() {
        // Use noisy data where overfitting will occur with aggressive settings.
        let mut rng = crate::rng::FastRng::new(42);
        let n = 50;
        let x: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect();
        // y = sin(x) + heavy noise — tree will overfit noise.
        let y: Vec<f64> = x.iter().map(|&v| v.sin() + rng.f64() * 5.0).collect();
        let data = Dataset::new(vec![x], y, vec!["x".into()], "y");

        let mut gbr = GradientBoostingRegressor::new()
            .n_estimators(1000)
            .learning_rate(0.5)
            .max_depth(5)
            .n_iter_no_change(5)
            .validation_fraction(0.3)
            .tol(0.0);
        gbr.fit(&data).unwrap();

        // With 1000 max estimators, heavy noise, and patience of 5,
        // early stopping should kick in well before 1000.
        assert!(
            gbr.n_trees() < 1000,
            "Expected early stopping, but used all {} estimators",
            gbr.n_trees()
        );
        assert!(gbr.early_stopped(), "early_stopped() should be true");
        assert!(gbr.n_estimators_used() < 1000);
    }

    // ─── Classifier tests ───

    #[test]
    fn classifier_binary() {
        let data = make_binary_data();
        let mut gbc = GradientBoostingClassifier::new()
            .n_estimators(50)
            .learning_rate(0.1)
            .max_depth(2);
        gbc.fit(&data).unwrap();

        let test = vec![vec![0.2, 0.1], vec![1.5, 0.75]];
        let preds = gbc.predict(&test).unwrap();
        assert_eq!(preds[0], 0.0, "low values -> class 0");
        assert_eq!(preds[1], 1.0, "high values -> class 1");
    }

    #[test]
    fn classifier_binary_proba() {
        let data = make_binary_data();
        let mut gbc = GradientBoostingClassifier::new()
            .n_estimators(50)
            .learning_rate(0.1)
            .max_depth(2);
        gbc.fit(&data).unwrap();

        let probas = gbc.predict_proba(&[vec![0.2, 0.1]]).unwrap();
        assert_eq!(probas[0].len(), 2);
        let sum: f64 = probas[0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "probabilities should sum to 1");
        assert!(probas[0][0] > probas[0][1], "class 0 should be more likely");
    }

    #[test]
    fn classifier_multiclass() {
        let data = make_multiclass_data();
        let mut gbc = GradientBoostingClassifier::new()
            .n_estimators(100)
            .learning_rate(0.1)
            .max_depth(3);
        gbc.fit(&data).unwrap();

        let test = vec![vec![0.5, 0.0], vec![2.5, 0.0], vec![4.5, 0.0]];
        let preds = gbc.predict(&test).unwrap();
        assert_eq!(preds[0], 0.0, "should be class 0");
        assert_eq!(preds[1], 1.0, "should be class 1");
        assert_eq!(preds[2], 2.0, "should be class 2");
    }

    #[test]
    fn classifier_multiclass_proba() {
        let data = make_multiclass_data();
        let mut gbc = GradientBoostingClassifier::new()
            .n_estimators(50)
            .learning_rate(0.1)
            .max_depth(2);
        gbc.fit(&data).unwrap();

        let probas = gbc.predict_proba(&[vec![0.5, 0.0]]).unwrap();
        assert_eq!(probas[0].len(), 3);
        let sum: f64 = probas[0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "probabilities should sum to 1");
    }

    #[test]
    fn classifier_subsample() {
        let data = make_binary_data();
        let mut gbc = GradientBoostingClassifier::new()
            .n_estimators(50)
            .subsample(0.8)
            .learning_rate(0.1)
            .max_depth(2);
        gbc.fit(&data).unwrap();

        let test = vec![vec![0.2, 0.1], vec![1.5, 0.75]];
        let preds = gbc.predict(&test).unwrap();
        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 1.0);
    }

    #[test]
    fn classifier_feature_importances() {
        let data = make_binary_data();
        let mut gbc = GradientBoostingClassifier::new()
            .n_estimators(20)
            .max_depth(2);
        gbc.fit(&data).unwrap();
        let imp = gbc.feature_importances().unwrap();
        assert_eq!(imp.len(), 2);
        let sum: f64 = imp.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "importances should sum to 1");
    }

    #[test]
    fn classifier_not_fitted_error() {
        let gbc = GradientBoostingClassifier::new();
        assert!(gbc.predict(&[vec![1.0, 2.0]]).is_err());
        assert!(gbc.predict_proba(&[vec![1.0, 2.0]]).is_err());
        assert!(gbc.feature_importances().is_err());
    }

    #[test]
    fn classifier_n_trees_binary() {
        let data = make_binary_data();
        let mut gbc = GradientBoostingClassifier::new()
            .n_estimators(25)
            .max_depth(2);
        gbc.fit(&data).unwrap();
        assert_eq!(gbc.n_trees(), 25, "binary: 1 class × 25 rounds");
    }

    #[test]
    fn classifier_n_trees_multiclass() {
        let data = make_multiclass_data();
        let mut gbc = GradientBoostingClassifier::new()
            .n_estimators(10)
            .max_depth(2);
        gbc.fit(&data).unwrap();
        assert_eq!(gbc.n_trees(), 30, "multiclass: 3 classes × 10 rounds");
    }

    // ─── Loss function tests ───

    #[test]
    fn regressor_loss_squared_error_default() {
        // Verify default behaviour is unchanged: SquaredError.
        let data = make_linear_data(100);
        let mut gbr = GradientBoostingRegressor::new()
            .n_estimators(100)
            .loss(RegressionLoss::SquaredError)
            .learning_rate(0.1)
            .max_depth(3);
        gbr.fit(&data).unwrap();
        let preds = gbr.predict(&[vec![50.0]]).unwrap();
        assert!(
            (preds[0] - 101.0).abs() < 10.0,
            "SquaredError pred={}",
            preds[0]
        );
    }

    #[test]
    fn regressor_loss_absolute_error() {
        let data = make_linear_data(100);
        let mut gbr = GradientBoostingRegressor::new()
            .n_estimators(200)
            .loss(RegressionLoss::AbsoluteError)
            .learning_rate(0.1)
            .max_depth(3);
        gbr.fit(&data).unwrap();
        let preds = gbr.predict(&[vec![50.0]]).unwrap();
        // y = 2x + 1 → 101
        assert!(
            (preds[0] - 101.0).abs() < 20.0,
            "AbsoluteError pred={}",
            preds[0]
        );
    }

    #[test]
    fn regressor_loss_huber() {
        let data = make_linear_data(100);
        let mut gbr = GradientBoostingRegressor::new()
            .n_estimators(200)
            .loss(RegressionLoss::Huber { alpha: 0.9 })
            .learning_rate(0.1)
            .max_depth(3);
        gbr.fit(&data).unwrap();
        let preds = gbr.predict(&[vec![50.0]]).unwrap();
        assert!((preds[0] - 101.0).abs() < 20.0, "Huber pred={}", preds[0]);
    }

    #[test]
    fn regressor_loss_quantile_median() {
        let data = make_linear_data(100);
        let mut gbr = GradientBoostingRegressor::new()
            .n_estimators(200)
            .loss(RegressionLoss::Quantile { alpha: 0.5 })
            .learning_rate(0.1)
            .max_depth(3);
        gbr.fit(&data).unwrap();
        let preds = gbr.predict(&[vec![50.0]]).unwrap();
        assert!(
            (preds[0] - 101.0).abs() < 25.0,
            "Quantile(0.5) pred={}",
            preds[0]
        );
    }

    #[test]
    fn test_median_helper() {
        assert!((median(&[1.0, 3.0, 5.0]) - 3.0).abs() < 1e-12);
        assert!((median(&[1.0, 3.0, 5.0, 7.0]) - 4.0).abs() < 1e-12);
        assert!((median(&[42.0]) - 42.0).abs() < 1e-12);
        assert!((median(&[]) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_helper() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((quantile(&data, 0.5) - 3.0).abs() < 1e-12);
        assert!((quantile(&data, 0.0) - 1.0).abs() < 1e-12);
        assert!((quantile(&data, 1.0) - 5.0).abs() < 1e-12);
        assert!((quantile(&data, 0.25) - 2.0).abs() < 1e-12);
    }

    // ─── Training history tests ───

    #[test]
    fn regressor_history_populated() {
        let data = make_linear_data(50);
        let mut gbr = GradientBoostingRegressor::new()
            .n_estimators(10)
            .learning_rate(0.1)
            .max_depth(3);
        gbr.fit(&data).unwrap();

        let history = gbr.history().expect("history should be populated");
        assert_eq!(history.len(), 10);
        // Loss should decrease over rounds.
        assert!(history.epochs[0].train_loss > history.epochs[9].train_loss);
        // Grad norms should be positive.
        assert!(history.epochs[0].grad_norm > 0.0);
    }

    #[test]
    fn classifier_binary_history_populated() {
        let data = make_binary_data();
        let mut gbc = GradientBoostingClassifier::new()
            .n_estimators(10)
            .learning_rate(0.1)
            .max_depth(2);
        gbc.fit(&data).unwrap();

        let history = gbc.history().expect("history should be populated");
        assert_eq!(history.len(), 10);
        assert!(history.epochs[0].train_loss > 0.0);
    }

    #[test]
    fn classifier_multiclass_history_populated() {
        let data = make_multiclass_data();
        let mut gbc = GradientBoostingClassifier::new()
            .n_estimators(10)
            .learning_rate(0.1)
            .max_depth(2);
        gbc.fit(&data).unwrap();

        let history = gbc.history().expect("history should be populated");
        assert_eq!(history.len(), 10);
        assert!(history.epochs[0].train_loss > 0.0);
    }

    #[test]
    fn regressor_early_stopping_history() {
        let mut rng = crate::rng::FastRng::new(42);
        let n = 50;
        let x: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect();
        let y: Vec<f64> = x.iter().map(|&v| v.sin() + rng.f64() * 5.0).collect();
        let data = Dataset::new(vec![x], y, vec!["x".into()], "y");

        let mut gbr = GradientBoostingRegressor::new()
            .n_estimators(1000)
            .learning_rate(0.5)
            .max_depth(5)
            .n_iter_no_change(5)
            .validation_fraction(0.3)
            .tol(0.0);
        gbr.fit(&data).unwrap();

        let history = gbr.history().expect("history should be populated");
        // History length should match n_estimators_used.
        assert_eq!(history.len(), gbr.n_estimators_used());
        // Some epochs should have val_loss.
        assert!(history.epochs.last().unwrap().val_loss.is_some());
    }
}
