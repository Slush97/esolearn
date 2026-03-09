// SPDX-License-Identifier: MIT OR Apache-2.0
//! Multi-layer perceptron regressor.
//!
//! Sklearn-compatible API with builder pattern.
//!
//! ```ignore
//! let mut reg = MLPRegressor::new()
//!     .hidden_layers(&[100, 50])
//!     .max_iter(200)
//!     .seed(42);
//! reg.fit(&data)?;
//! let preds = reg.predict(&test_features)?;
//! ```

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::neural::activation::Activation;
use crate::neural::callback::{
    self, CallbackAction, EpochMetrics, TrainingCallback, TrainingHistory,
};
use crate::neural::layer::FastRng;
use crate::neural::network::{self, Network};
use crate::neural::optimizer::{LearningRateSchedule, OptimizerKind, OptimizerState};
use crate::partial_fit::PartialFit;

/// Multi-layer perceptron regressor.
///
/// Trains a feedforward neural network for regression using
/// backpropagation with MSE loss.
///
/// Defaults match sklearn: `hidden_layers=[100]`, Adam, lr=0.001,
/// `max_iter=200`, `batch_size=200`, `alpha=0.0001`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct MLPRegressor {
    hidden_layers: Vec<usize>,
    activation: Activation,
    optimizer_kind: OptimizerKind,
    learning_rate: f64,
    max_iter: usize,
    batch_size: usize,
    alpha: f64,
    tolerance: f64,
    early_stopping: bool,
    validation_fraction: f64,
    n_iter_no_change: usize,
    seed: u64,
    /// Dropout probability applied between hidden layers (0.0 = no dropout).
    dropout_rate: f64,
    /// Learning rate schedule.
    lr_schedule: LearningRateSchedule,
    // ── Fitted state ──
    fitted: bool,
    n_features: usize,
    network_weights: Vec<(Vec<f64>, Vec<f64>)>,
    network_dims: Vec<(usize, usize)>,
    /// Training loss curve (one entry per epoch).
    pub loss_curve: Vec<f64>,
    /// Structured training history with per-epoch metrics.
    training_history: TrainingHistory,
    /// User-supplied training callbacks (not cloned — session-specific).
    #[cfg_attr(feature = "serde", serde(skip))]
    callbacks: Vec<Box<dyn TrainingCallback>>,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl Clone for MLPRegressor {
    fn clone(&self) -> Self {
        Self {
            hidden_layers: self.hidden_layers.clone(),
            activation: self.activation,
            optimizer_kind: self.optimizer_kind,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            batch_size: self.batch_size,
            alpha: self.alpha,
            tolerance: self.tolerance,
            early_stopping: self.early_stopping,
            validation_fraction: self.validation_fraction,
            n_iter_no_change: self.n_iter_no_change,
            seed: self.seed,
            dropout_rate: self.dropout_rate,
            lr_schedule: self.lr_schedule,
            fitted: self.fitted,
            n_features: self.n_features,
            network_weights: self.network_weights.clone(),
            network_dims: self.network_dims.clone(),
            loss_curve: self.loss_curve.clone(),
            training_history: self.training_history.clone(),
            callbacks: Vec::new(),
            _schema_version: 0,
        }
    }
}

impl MLPRegressor {
    /// Create a new MLP regressor with sklearn defaults.
    pub fn new() -> Self {
        Self {
            hidden_layers: vec![100],
            activation: Activation::Relu,
            optimizer_kind: OptimizerKind::default(),
            learning_rate: 0.001,
            max_iter: 200,
            batch_size: 200,
            alpha: 0.0001,
            tolerance: 1e-4,
            early_stopping: false,
            validation_fraction: 0.1,
            n_iter_no_change: 10,
            seed: 42,
            dropout_rate: 0.0,
            lr_schedule: LearningRateSchedule::Constant,
            fitted: false,
            n_features: 0,
            network_weights: Vec::new(),
            network_dims: Vec::new(),
            loss_curve: Vec::new(),
            training_history: TrainingHistory::new(),
            callbacks: Vec::new(),
            _schema_version: 0,
        }
    }

    /// Set hidden layer sizes. Default: `&[100]`.
    pub fn hidden_layers(mut self, sizes: &[usize]) -> Self {
        self.hidden_layers = sizes.to_vec();
        self
    }

    /// Set activation function for hidden layers. Default: ReLU.
    pub fn activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Set optimizer algorithm. Default: Adam.
    pub fn optimizer(mut self, kind: OptimizerKind) -> Self {
        self.optimizer_kind = kind;
        self
    }

    /// Set learning rate. Default: 0.001.
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set maximum training iterations (epochs). Default: 200.
    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Set mini-batch size. Default: 200.
    pub fn batch_size(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }

    /// Set L2 regularization strength. Default: 0.0001.
    pub fn alpha(mut self, a: f64) -> Self {
        self.alpha = a;
        self
    }

    /// Set convergence tolerance. Default: 1e-4.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Alias for [`tolerance`](Self::tolerance) (sklearn convention).
    pub fn tol(self, t: f64) -> Self {
        self.tolerance(t)
    }

    /// Enable early stopping. Default: false.
    pub fn early_stopping(mut self, enable: bool) -> Self {
        self.early_stopping = enable;
        self
    }

    /// Set validation fraction for early stopping. Default: 0.1.
    pub fn validation_fraction(mut self, frac: f64) -> Self {
        self.validation_fraction = frac;
        self
    }

    /// Set patience for early stopping. Default: 10.
    pub fn n_iter_no_change(mut self, n: usize) -> Self {
        self.n_iter_no_change = n;
        self
    }

    /// Set random seed. Default: 42.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Set learning rate schedule. Default: [`LearningRateSchedule::Constant`].
    ///
    /// Use [`LearningRateSchedule::adaptive()`] for reduce-on-plateau behavior.
    pub fn learning_rate_schedule(mut self, schedule: LearningRateSchedule) -> Self {
        self.lr_schedule = schedule;
        self
    }

    /// Set dropout probability applied between hidden layers.
    ///
    /// `p` is the fraction of activations to zero out (e.g. 0.5 for 50%).
    /// Applied only during training; inference is unaffected.
    /// Default: 0.0 (no dropout).
    pub fn dropout(mut self, p: f64) -> Self {
        self.dropout_rate = p;
        self
    }

    /// Add a training callback (invoked after each epoch).
    pub fn callback(mut self, cb: Box<dyn TrainingCallback>) -> Self {
        self.callbacks.push(cb);
        self
    }

    /// Train the regressor on a dataset.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        let n_samples = data.n_samples();
        let n_features = data.n_features();

        if n_samples == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        // Build row-major feature matrix
        let x = build_row_major(&data.features, n_samples, n_features);
        let y = data.target.clone();

        // Split train/val if early stopping
        let (train_x, train_y, val_x, val_y) = if self.early_stopping {
            let mut rng = FastRng::new(self.seed);
            let val_size = (n_samples as f64 * self.validation_fraction).max(1.0) as usize;
            let train_size = n_samples - val_size;
            let mut indices: Vec<usize> = (0..n_samples).collect();
            rng.shuffle(&mut indices);

            let mut tx = Vec::with_capacity(train_size * n_features);
            let mut ty = Vec::with_capacity(train_size);
            let mut vx = Vec::with_capacity(val_size * n_features);
            let mut vy = Vec::with_capacity(val_size);

            for &i in &indices[..train_size] {
                tx.extend_from_slice(&x[i * n_features..(i + 1) * n_features]);
                ty.push(y[i]);
            }
            for &i in &indices[train_size..] {
                vx.extend_from_slice(&x[i * n_features..(i + 1) * n_features]);
                vy.push(y[i]);
            }
            (tx, ty, Some(vx), Some(vy))
        } else {
            (x, y, None, None)
        };

        let train_n = train_y.len();

        // Build network: output is 1 neuron for regression
        let mut sizes = Vec::with_capacity(self.hidden_layers.len() + 2);
        sizes.push(n_features);
        sizes.extend_from_slice(&self.hidden_layers);
        sizes.push(1);

        let mut net = Network::new_with_dropout(&sizes, self.activation, self.seed, self.dropout_rate);
        let param_sizes = net.param_group_sizes();
        let mut optimizer =
            OptimizerState::new_with_schedule(self.optimizer_kind, self.learning_rate, &param_sizes, self.lr_schedule);

        let batch_size = self.batch_size.min(train_n);
        let mut rng = FastRng::new(self.seed.wrapping_add(1));
        let mut indices: Vec<usize> = (0..train_n).collect();

        self.loss_curve.clear();
        self.training_history = TrainingHistory::new();
        let mut best_val_loss = f64::INFINITY;
        let mut best_weights: Option<Vec<(Vec<f64>, Vec<f64>)>> = None;
        let mut no_improve = 0;

        let mut callbacks = std::mem::take(&mut self.callbacks);

        for epoch_idx in 0..self.max_iter {
            let epoch_start = std::time::Instant::now();
            rng.shuffle(&mut indices);

            let mut epoch_loss = 0.0;
            let mut n_batches = 0;
            let mut last_grad_norm = 0.0;
            let mut epoch_ss_res = 0.0;
            let mut epoch_ss_tot = 0.0;
            let mut epoch_sum_y = 0.0;
            let mut epoch_n = 0usize;

            // First pass: compute mean of targets for R²
            for &i in &indices {
                epoch_sum_y += train_y[i];
                epoch_n += 1;
            }
            let mean_y = epoch_sum_y / epoch_n as f64;

            for chunk in indices.chunks(batch_size) {
                let b = chunk.len();
                let mut batch_x = Vec::with_capacity(b * n_features);
                let mut batch_y = Vec::with_capacity(b);
                for &i in chunk {
                    batch_x.extend_from_slice(&train_x[i * n_features..(i + 1) * n_features]);
                    batch_y.push(train_y[i]);
                }

                let output = net.forward(&batch_x, b, true);
                let (loss, grad) = network::mse_loss(&output, &batch_y, b);
                epoch_loss += loss;
                n_batches += 1;

                // Accumulate R² components
                for (pred, &actual) in output.iter().zip(batch_y.iter()) {
                    epoch_ss_res += (pred - actual).powi(2);
                    epoch_ss_tot += (actual - mean_y).powi(2);
                }

                // Reshape gradient to [batch, 1] for backward pass
                let grad_2d: Vec<f64> = grad;
                let layer_grads = net.backward(&grad_2d, self.alpha);
                last_grad_norm = callback::compute_grad_norm(&layer_grads);
                optimizer.tick();
                net.apply_gradients(&layer_grads, &mut optimizer);
            }

            let avg_loss = epoch_loss / n_batches as f64;
            self.loss_curve.push(avg_loss);

            // Adjust learning rate based on schedule.
            optimizer.adjust_lr(avg_loss);

            let train_r2 = if epoch_ss_tot > 0.0 {
                Some(1.0 - epoch_ss_res / epoch_ss_tot)
            } else {
                None
            };

            // Early stopping check + validation metrics
            let mut val_loss_epoch = None;
            let mut val_metric_epoch = None;

            if self.early_stopping {
                if let (Some(ref vx), Some(ref vy)) = (&val_x, &val_y) {
                    let val_n = vy.len();
                    let val_output = net.forward(vx, val_n, false);
                    let (val_loss, _) = network::mse_loss(&val_output, vy, val_n);
                    val_loss_epoch = Some(val_loss);

                    // Validation R²
                    let val_mean: f64 = vy.iter().sum::<f64>() / val_n as f64;
                    let val_ss_res: f64 = val_output
                        .iter()
                        .zip(vy.iter())
                        .map(|(p, a)| (p - a).powi(2))
                        .sum();
                    let val_ss_tot: f64 = vy.iter().map(|a| (a - val_mean).powi(2)).sum();
                    if val_ss_tot > 0.0 {
                        val_metric_epoch = Some(1.0 - val_ss_res / val_ss_tot);
                    }

                    if val_loss < best_val_loss - self.tolerance {
                        best_val_loss = val_loss;
                        best_weights = Some(net.save_weights());
                        no_improve = 0;
                    } else {
                        no_improve += 1;
                    }
                }
            } else {
                let n = self.loss_curve.len();
                if n >= 2 {
                    let improvement = self.loss_curve[n - 2] - self.loss_curve[n - 1];
                    if improvement.abs() < self.tolerance {
                        no_improve += 1;
                    } else {
                        no_improve = 0;
                    }
                }
            }

            let elapsed = epoch_start.elapsed();
            let metrics = EpochMetrics {
                epoch: epoch_idx,
                train_loss: avg_loss,
                val_loss: val_loss_epoch,
                train_metric: train_r2,
                val_metric: val_metric_epoch,
                learning_rate: optimizer.current_lr(),
                grad_norm: last_grad_norm,
                elapsed_ms: elapsed.as_millis() as u64,
            };

            let mut cb_stop = false;
            for cb in &mut callbacks {
                if cb.on_epoch_end(&metrics) == CallbackAction::Stop {
                    cb_stop = true;
                }
            }

            self.training_history.push(metrics);

            if cb_stop {
                break;
            }

            if no_improve >= self.n_iter_no_change
                && (self.early_stopping || self.loss_curve.len() >= 2)
            {
                break;
            }
        }

        for cb in &mut callbacks {
            cb.on_training_end();
        }
        self.callbacks = callbacks;

        if let Some(ref best) = best_weights {
            net.restore_weights(best);
        }

        self.network_weights = net.save_weights();
        self.network_dims = net.layer_dims();
        self.n_features = n_features;
        self.fitted = true;

        Ok(())
    }

    /// Predict target values for input samples.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }

        let batch = features.len();
        if batch == 0 {
            return Ok(Vec::new());
        }

        let n_feat = features[0].len();
        if n_feat != self.n_features {
            return Err(ScryLearnError::ShapeMismatch {
                expected: self.n_features,
                got: n_feat,
            });
        }

        let mut net = self.rebuild_network();
        let x: Vec<f64> = features
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let output = net.forward(&x, batch, false);
        Ok(output)
    }

    /// Number of features the model was trained on.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Training loss per epoch.
    pub fn loss_curve(&self) -> &[f64] {
        &self.loss_curve
    }

    /// Structured training history with per-epoch metrics.
    ///
    /// Returns `None` if the model has not been fitted yet.
    pub fn history(&self) -> Option<&TrainingHistory> {
        if self.training_history.is_empty() {
            None
        } else {
            Some(&self.training_history)
        }
    }

    /// Saved network weights (for visualization).
    pub fn weights(&self) -> &[(Vec<f64>, Vec<f64>)] {
        &self.network_weights
    }

    /// Layer dimensions (for visualization).
    pub fn layer_dims(&self) -> &[(usize, usize)] {
        &self.network_dims
    }

    /// Hidden-layer activation function.
    pub fn activation_fn(&self) -> Activation {
        self.activation
    }

    fn rebuild_network(&self) -> Network {
        let mut sizes = Vec::with_capacity(self.network_dims.len() + 1);
        sizes.push(self.network_dims[0].0);
        for &(_, out) in &self.network_dims {
            sizes.push(out);
        }
        let mut net = Network::new_with_dropout(&sizes, self.activation, 0, self.dropout_rate);
        net.restore_weights(&self.network_weights);
        net
    }
}

impl PartialFit for MLPRegressor {
    /// Run one epoch of mini-batch SGD (MSE loss) on the given data.
    ///
    /// On the first call, initializes the network architecture. Subsequent
    /// calls preserve network weights and continue training.
    fn partial_fit(&mut self, data: &Dataset) -> Result<()> {
        let n_samples = data.n_samples();
        let n_features = data.n_features();
        if n_samples == 0 {
            if self.is_initialized() {
                return Ok(());
            }
            return Err(ScryLearnError::EmptyDataset);
        }

        if !self.is_initialized() {
            let mut sizes = Vec::with_capacity(self.hidden_layers.len() + 2);
            sizes.push(n_features);
            sizes.extend_from_slice(&self.hidden_layers);
            sizes.push(1);

            let net = Network::new_with_dropout(&sizes, self.activation, self.seed, self.dropout_rate);
            self.network_weights = net.save_weights();
            self.network_dims = net.layer_dims();
            self.n_features = n_features;
            self.loss_curve.clear();
        } else if n_features != self.n_features {
            return Err(ScryLearnError::ShapeMismatch {
                expected: self.n_features,
                got: n_features,
            });
        }

        let x = build_row_major(&data.features, n_samples, n_features);
        let y = data.target.clone();

        let mut net = self.rebuild_network();
        let param_sizes = net.param_group_sizes();
        let mut optimizer =
            OptimizerState::new(self.optimizer_kind, self.learning_rate, &param_sizes);

        let batch_size = self.batch_size.min(n_samples);
        let mut rng = FastRng::new(self.seed.wrapping_add(self.loss_curve.len() as u64));
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // One epoch.
        rng.shuffle(&mut indices);
        let mut epoch_loss = 0.0;
        let mut n_batches = 0;

        for chunk in indices.chunks(batch_size) {
            let b = chunk.len();
            let mut batch_x = Vec::with_capacity(b * n_features);
            let mut batch_y = Vec::with_capacity(b);
            for &i in chunk {
                batch_x.extend_from_slice(&x[i * n_features..(i + 1) * n_features]);
                batch_y.push(y[i]);
            }

            let output = net.forward(&batch_x, b, true);
            let (loss, grad) = network::mse_loss(&output, &batch_y, b);
            epoch_loss += loss;
            n_batches += 1;

            let layer_grads = net.backward(&grad, self.alpha);
            optimizer.tick();
            net.apply_gradients(&layer_grads, &mut optimizer);
        }

        self.loss_curve.push(epoch_loss / n_batches as f64);
        self.network_weights = net.save_weights();
        self.fitted = true;
        Ok(())
    }

    fn is_initialized(&self) -> bool {
        !self.network_weights.is_empty()
    }
}

impl Default for MLPRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MLPRegressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MLPRegressor")
            .field("hidden_layers", &self.hidden_layers)
            .field("activation", &self.activation)
            .field("fitted", &self.fitted)
            .finish()
    }
}

/// Build row-major feature matrix from column-major Dataset.
fn build_row_major(features: &[Vec<f64>], n_samples: usize, n_features: usize) -> Vec<f64> {
    let mut x = vec![0.0; n_samples * n_features];
    for j in 0..n_features {
        for i in 0..n_samples {
            x[i * n_features + j] = features[j][i];
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_dataset() -> Dataset {
        // y = 2x + 1
        let n = 100;
        let x_vals: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y_vals: Vec<f64> = x_vals.iter().map(|&x| 2.0 * x + 1.0).collect();
        Dataset::new(vec![x_vals], y_vals, vec!["x".into()], "y")
    }

    #[test]
    fn not_fitted_error() {
        let reg = MLPRegressor::new();
        let result = reg.predict(&[vec![1.0]]);
        assert!(matches!(result, Err(ScryLearnError::NotFitted)));
    }

    #[test]
    fn regression_y_equals_2x_plus_1() {
        let data = linear_dataset();
        let mut reg = MLPRegressor::new()
            .hidden_layers(&[20, 10])
            .learning_rate(0.01)
            .max_iter(500)
            .batch_size(32)
            .seed(42);
        reg.fit(&data).unwrap();

        let test_x = vec![vec![0.0], vec![0.5], vec![1.0]];
        let preds = reg.predict(&test_x).unwrap();

        // Check R²
        let actual = [1.0, 2.0, 3.0];
        let mean_y: f64 = actual.iter().sum::<f64>() / actual.len() as f64;
        let ss_res: f64 = preds
            .iter()
            .zip(actual.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum();
        let ss_tot: f64 = actual.iter().map(|a| (a - mean_y).powi(2)).sum();
        let r2 = 1.0 - ss_res / ss_tot;

        assert!(r2 > 0.9, "R² should be > 0.9, got {r2:.4}, preds={preds:?}");
    }

    #[test]
    fn early_stopping_regression() {
        let data = linear_dataset();
        let mut reg = MLPRegressor::new()
            .hidden_layers(&[20])
            .max_iter(1000)
            .early_stopping(true)
            .n_iter_no_change(5)
            .seed(42);
        reg.fit(&data).unwrap();

        assert!(
            reg.loss_curve.len() < 1000,
            "expected early stop, got {} epochs",
            reg.loss_curve.len()
        );
    }

    #[test]
    fn loss_decreases() {
        let data = linear_dataset();
        let mut reg = MLPRegressor::new()
            .hidden_layers(&[20])
            .max_iter(50)
            .seed(42);
        reg.fit(&data).unwrap();

        let curve = reg.loss_curve();
        assert!(curve.len() >= 2);
        assert!(curve.first().unwrap() > curve.last().unwrap());
    }

    #[test]
    fn partial_fit_is_initialized() {
        let mut reg = MLPRegressor::new();
        assert!(!reg.is_initialized());

        let data = linear_dataset();
        reg.partial_fit(&data).unwrap();
        assert!(reg.is_initialized());
    }

    #[test]
    fn partial_fit_loss_decreases() {
        let data = linear_dataset();
        let mut reg = MLPRegressor::new()
            .hidden_layers(&[20])
            .learning_rate(0.01)
            .batch_size(32)
            .seed(42);

        for _ in 0..10 {
            reg.partial_fit(&data).unwrap();
        }

        let curve = reg.loss_curve();
        assert_eq!(curve.len(), 10);
        assert!(
            curve.first().unwrap() > curve.last().unwrap(),
            "loss should decrease: first={} last={}",
            curve.first().unwrap(),
            curve.last().unwrap()
        );
    }
}
