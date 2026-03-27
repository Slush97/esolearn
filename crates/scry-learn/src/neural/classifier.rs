// SPDX-License-Identifier: MIT OR Apache-2.0
//! Multi-layer perceptron classifier.
//!
//! Sklearn-compatible API with builder pattern.
//!
//! ```ignore
//! let mut clf = MLPClassifier::new()
//!     .hidden_layers(&[100, 50])
//!     .activation(Activation::Relu)
//!     .optimizer(OptimizerKind::Adam)
//!     .learning_rate(0.001)
//!     .max_iter(200)
//!     .batch_size(32)
//!     .early_stopping(true)
//!     .seed(42);
//! clf.fit(&train_data)?;
//! let preds = clf.predict(&test_features)?;
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

/// Multi-layer perceptron classifier.
///
/// Trains a feedforward neural network for classification using
/// backpropagation with configurable optimizers and activations.
///
/// Defaults match sklearn: `hidden_layers=[100]`, Adam, lr=0.001,
/// `max_iter=200`, `batch_size=200`, `alpha=0.0001`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct MLPClassifier {
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
    n_classes: usize,
    class_labels: Vec<f64>,
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

impl Clone for MLPClassifier {
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
            n_classes: self.n_classes,
            class_labels: self.class_labels.clone(),
            network_weights: self.network_weights.clone(),
            network_dims: self.network_dims.clone(),
            loss_curve: self.loss_curve.clone(),
            training_history: self.training_history.clone(),
            // Callbacks are session-specific and not cloned.
            callbacks: Vec::new(),
            _schema_version: 0,
        }
    }
}

impl MLPClassifier {
    /// Create a new MLP classifier with sklearn defaults.
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
            n_classes: 0,
            class_labels: Vec::new(),
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

    /// Enable early stopping with validation split. Default: false.
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

    /// Train the classifier on a dataset.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        let n_samples = data.n_samples();
        let n_features = data.n_features();

        if n_samples == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        // Discover classes
        let mut class_labels: Vec<f64> = data.target.clone();
        class_labels.sort_by(|a, b| a.total_cmp(b));
        class_labels.dedup();
        let n_classes = class_labels.len();

        if n_classes < 2 {
            return Err(ScryLearnError::InvalidParameter(
                "need at least 2 classes".into(),
            ));
        }

        // Build row-major feature matrix
        let x = build_row_major(&data.features, n_samples, n_features);

        // Map targets to class indices
        let y: Vec<f64> = data
            .target
            .iter()
            .map(|&t| {
                class_labels
                    .iter()
                    .position(|&c| (c - t).abs() < f64::EPSILON)
                    .expect("target value must appear in class_labels") as f64
            })
            .collect();

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

        // Build network
        let mut sizes = Vec::with_capacity(self.hidden_layers.len() + 2);
        sizes.push(n_features);
        sizes.extend_from_slice(&self.hidden_layers);
        sizes.push(n_classes);

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

        // Take callbacks out so we can mutably borrow them during training.
        let mut callbacks = std::mem::take(&mut self.callbacks);

        for epoch_idx in 0..self.max_iter {
            let epoch_start = std::time::Instant::now();
            rng.shuffle(&mut indices);

            let mut epoch_loss = 0.0;
            let mut n_batches = 0;
            let mut last_grad_norm = 0.0;
            let mut epoch_correct = 0usize;
            let mut epoch_total = 0usize;

            for chunk in indices.chunks(batch_size) {
                let b = chunk.len();
                let mut batch_x = Vec::with_capacity(b * n_features);
                let mut batch_y = Vec::with_capacity(b);
                for &i in chunk {
                    batch_x.extend_from_slice(&train_x[i * n_features..(i + 1) * n_features]);
                    batch_y.push(train_y[i]);
                }

                let logits = net.forward(&batch_x, b, true);
                let (loss, grad) = network::cross_entropy_loss(&logits, &batch_y, b, n_classes);
                epoch_loss += loss;
                n_batches += 1;

                // Compute training accuracy for this mini-batch
                let preds = network::argmax_predictions(&logits, b, n_classes);
                for (p, t) in preds.iter().zip(batch_y.iter()) {
                    if (*p - *t).abs() < f64::EPSILON {
                        epoch_correct += 1;
                    }
                    epoch_total += 1;
                }

                let layer_grads = net.backward(&grad, self.alpha);
                last_grad_norm = callback::compute_grad_norm(&layer_grads);
                optimizer.tick();
                net.apply_gradients(&layer_grads, &mut optimizer);
            }

            let avg_loss = epoch_loss / n_batches as f64;
            self.loss_curve.push(avg_loss);

            // Adjust learning rate based on schedule.
            optimizer.adjust_lr(avg_loss);

            let train_accuracy = if epoch_total > 0 {
                Some(epoch_correct as f64 / epoch_total as f64)
            } else {
                None
            };

            // Early stopping check + validation metrics
            let mut val_loss_epoch = None;
            let mut val_metric_epoch = None;

            if self.early_stopping {
                if let (Some(ref vx), Some(ref vy)) = (&val_x, &val_y) {
                    let val_n = vy.len();
                    let val_logits = net.forward(vx, val_n, false);
                    let (val_loss, _) =
                        network::cross_entropy_loss(&val_logits, vy, val_n, n_classes);
                    val_loss_epoch = Some(val_loss);

                    // Validation accuracy
                    let val_preds = network::argmax_predictions(&val_logits, val_n, n_classes);
                    let val_correct = val_preds
                        .iter()
                        .zip(vy.iter())
                        .filter(|(p, t)| (**p - **t).abs() < f64::EPSILON)
                        .count();
                    val_metric_epoch = Some(val_correct as f64 / val_n as f64);

                    if val_loss < best_val_loss - self.tolerance {
                        best_val_loss = val_loss;
                        best_weights = Some(net.save_weights());
                        no_improve = 0;
                    } else {
                        no_improve += 1;
                    }
                }
            } else {
                // Check training loss convergence
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
                train_metric: train_accuracy,
                val_metric: val_metric_epoch,
                learning_rate: optimizer.current_lr(),
                grad_norm: last_grad_norm,
                elapsed_ms: elapsed.as_millis() as u64,
            };

            // Invoke user callbacks.
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

        // Notify callbacks that training is done, then put them back.
        for cb in &mut callbacks {
            cb.on_training_end();
        }
        self.callbacks = callbacks;

        // Restore best weights if early stopping found an improvement
        if let Some(ref best) = best_weights {
            net.restore_weights(best);
        }

        // Save fitted state
        self.network_weights = net.save_weights();
        self.network_dims = net.layer_dims();
        self.n_features = n_features;
        self.n_classes = n_classes;
        self.class_labels = class_labels;
        self.fitted = true;

        Ok(())
    }

    /// Predict class labels for input samples.
    ///
    /// `features` is `&[Vec<f64>]` where each inner vec is one sample (row-major).
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        let proba = self.predict_proba(features)?;
        let batch = features.len();
        let preds = network::argmax_predictions(&proba, batch, self.n_classes);
        // Map indices back to original class labels
        Ok(preds
            .iter()
            .map(|&i| self.class_labels[i as usize])
            .collect())
    }

    /// Predict class probabilities (softmax output).
    ///
    /// Returns a flat `[batch * n_classes]` row-major probability matrix.
    pub fn predict_proba(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
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
        let logits = net.forward(&x, batch, false);
        Ok(network::softmax(&logits, batch, self.n_classes))
    }

    /// Number of classes discovered during fit.
    pub fn n_classes(&self) -> usize {
        self.n_classes
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

    /// Rebuild a Network from saved weights.
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

impl PartialFit for MLPClassifier {
    /// Run one epoch of mini-batch SGD on the given data.
    ///
    /// On the first call, initializes the network architecture from the data
    /// dimensions. Subsequent calls preserve network weights and continue
    /// training.
    fn partial_fit(&mut self, data: &Dataset) -> Result<()> {
        let n_samples = data.n_samples();
        let n_features = data.n_features();
        if n_samples == 0 {
            if self.is_initialized() {
                return Ok(());
            }
            return Err(ScryLearnError::EmptyDataset);
        }

        // Discover classes from this batch.
        let mut batch_labels: Vec<f64> = data.target.clone();
        batch_labels.sort_by(|a, b| a.total_cmp(b));
        batch_labels.dedup();

        if self.is_initialized() {
            if n_features != self.n_features {
                return Err(ScryLearnError::ShapeMismatch {
                    expected: self.n_features,
                    got: n_features,
                });
            }
            // Check for new classes not seen during initialization.
            for &label in &batch_labels {
                if !self
                    .class_labels
                    .iter()
                    .any(|&c| (c - label).abs() < f64::EPSILON)
                {
                    return Err(ScryLearnError::InvalidParameter(format!(
                        "partial_fit encountered new class {label} not seen during \
                         initialization (known classes: {:?}). MLPClassifier cannot add \
                         classes after network initialization — pass all possible classes \
                         in the first batch.",
                        self.class_labels
                    )));
                }
            }
        } else {
            let n_classes = batch_labels.len();
            if n_classes < 2 {
                return Err(ScryLearnError::InvalidParameter(
                    "need at least 2 classes".into(),
                ));
            }

            // Build and initialize network.
            let mut sizes = Vec::with_capacity(self.hidden_layers.len() + 2);
            sizes.push(n_features);
            sizes.extend_from_slice(&self.hidden_layers);
            sizes.push(n_classes);

            let net = Network::new(&sizes, self.activation, self.seed);
            self.network_weights = net.save_weights();
            self.network_dims = net.layer_dims();
            self.n_features = n_features;
            self.n_classes = n_classes;
            self.class_labels = batch_labels;
            self.loss_curve.clear();
        }

        // Build row-major data.
        let x = build_row_major(&data.features, n_samples, n_features);
        let y: Vec<f64> = data
            .target
            .iter()
            .map(|&t| {
                self.class_labels
                    .iter()
                    .position(|&c| (c - t).abs() < f64::EPSILON)
                    .unwrap_or(0) as f64
            })
            .collect();

        // Rebuild network from saved weights.
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

            let logits = net.forward(&batch_x, b, true);
            let (loss, grad) = network::cross_entropy_loss(&logits, &batch_y, b, self.n_classes);
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

impl Default for MLPClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MLPClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MLPClassifier")
            .field("hidden_layers", &self.hidden_layers)
            .field("activation", &self.activation)
            .field("fitted", &self.fitted)
            .field("n_classes", &self.n_classes)
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

    fn xor_dataset() -> Dataset {
        Dataset::new(
            vec![vec![0.0, 0.0, 1.0, 1.0], vec![0.0, 1.0, 0.0, 1.0]],
            vec![0.0, 1.0, 1.0, 0.0],
            vec!["x1".into(), "x2".into()],
            "xor",
        )
    }

    fn linearly_separable() -> Dataset {
        let mut f1 = Vec::new();
        let mut f2 = Vec::new();
        let mut target = Vec::new();
        for i in 0..50 {
            let v = i as f64 * 0.1;
            f1.push(v);
            f2.push(v + 0.5);
            target.push(0.0);
            f1.push(v + 5.0);
            f2.push(v + 5.5);
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
    fn not_fitted_error() {
        let clf = MLPClassifier::new();
        let result = clf.predict(&[vec![1.0, 2.0]]);
        assert!(matches!(result, Err(ScryLearnError::NotFitted)));
    }

    #[test]
    fn xor_problem() {
        // XOR requires non-linear separation — proves the network works
        let data = xor_dataset();
        let mut clf = MLPClassifier::new()
            .hidden_layers(&[10, 10])
            .learning_rate(0.01)
            .max_iter(1000)
            .batch_size(4)
            .seed(42);
        clf.fit(&data).unwrap();

        let preds = clf
            .predict(&[
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ])
            .unwrap();

        let correct = preds
            .iter()
            .zip([0.0, 1.0, 1.0, 0.0].iter())
            .filter(|(p, t)| (**p - **t).abs() < f64::EPSILON)
            .count();

        assert!(
            correct >= 3,
            "XOR: got {correct}/4 correct, preds={preds:?}"
        );
    }

    #[test]
    fn linearly_separable_data() {
        let data = linearly_separable();
        let mut clf = MLPClassifier::new()
            .hidden_layers(&[20])
            .max_iter(200)
            .seed(42);
        clf.fit(&data).unwrap();

        let test_x = vec![vec![0.5, 1.0], vec![5.5, 6.0]];
        let preds = clf.predict(&test_x).unwrap();
        assert!((preds[0] - 0.0).abs() < f64::EPSILON);
        assert!((preds[1] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn early_stopping_halts() {
        let data = linearly_separable();
        let mut clf = MLPClassifier::new()
            .hidden_layers(&[20])
            .max_iter(500)
            .early_stopping(true)
            .n_iter_no_change(5)
            .seed(42);
        clf.fit(&data).unwrap();

        // Should have stopped well before 500 epochs
        assert!(
            clf.loss_curve.len() < 500,
            "expected early stop, got {} epochs",
            clf.loss_curve.len()
        );
    }

    #[test]
    fn predict_proba_sums_to_one() {
        let data = linearly_separable();
        let mut clf = MLPClassifier::new()
            .hidden_layers(&[10])
            .max_iter(50)
            .seed(42);
        clf.fit(&data).unwrap();

        let proba = clf.predict_proba(&[vec![1.0, 1.5]]).unwrap();
        let sum: f64 = proba.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn shape_mismatch_error() {
        let data = linearly_separable();
        let mut clf = MLPClassifier::new()
            .hidden_layers(&[10])
            .max_iter(10)
            .seed(42);
        clf.fit(&data).unwrap();

        let result = clf.predict(&[vec![1.0, 2.0, 3.0]]); // 3 features, expected 2
        assert!(matches!(result, Err(ScryLearnError::ShapeMismatch { .. })));
    }

    #[test]
    fn loss_decreases() {
        let data = linearly_separable();
        let mut clf = MLPClassifier::new()
            .hidden_layers(&[20])
            .max_iter(50)
            .seed(42);
        clf.fit(&data).unwrap();

        let curve = clf.loss_curve();
        assert!(curve.len() >= 2);
        // First loss should be higher than last
        assert!(curve.first().unwrap() > curve.last().unwrap());
    }

    #[test]
    fn partial_fit_is_initialized() {
        let mut clf = MLPClassifier::new();
        assert!(!clf.is_initialized());

        let data = linearly_separable();
        clf.partial_fit(&data).unwrap();
        assert!(clf.is_initialized());
    }

    #[test]
    fn partial_fit_loss_decreases() {
        let data = linearly_separable();
        let mut clf = MLPClassifier::new()
            .hidden_layers(&[20])
            .learning_rate(0.01)
            .batch_size(32)
            .seed(42);

        // Run 10 partial_fit calls on the same data.
        for _ in 0..10 {
            clf.partial_fit(&data).unwrap();
        }

        let curve = clf.loss_curve();
        assert!(curve.len() == 10);
        // Overall trend: first loss > last loss
        assert!(
            curve.first().unwrap() > curve.last().unwrap(),
            "loss should decrease: first={} last={}",
            curve.first().unwrap(),
            curve.last().unwrap()
        );
    }

    #[test]
    fn partial_fit_classifies_after_batches() {
        let mut clf = MLPClassifier::new()
            .hidden_layers(&[20])
            .learning_rate(0.01)
            .batch_size(32)
            .seed(42);

        let data = linearly_separable();
        for _ in 0..50 {
            clf.partial_fit(&data).unwrap();
        }

        let preds = clf.predict(&[vec![0.5, 1.0], vec![5.5, 6.0]]).unwrap();
        assert!(
            (preds[0] - 0.0).abs() < f64::EPSILON,
            "x=0.5 should be class 0"
        );
        assert!(
            (preds[1] - 1.0).abs() < f64::EPSILON,
            "x=5.5 should be class 1"
        );
    }
}
