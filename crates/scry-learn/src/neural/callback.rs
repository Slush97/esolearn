// SPDX-License-Identifier: MIT OR Apache-2.0
//! Training callback system — structured per-epoch metrics and user hooks.
//!
//! Every iterative model (MLP, gradient boosting) populates a
//! [`TrainingHistory`] during `fit()`. Users can also inject custom
//! [`TrainingCallback`] implementations to log, visualize, or early-stop
//! based on arbitrary criteria.
//!
//! # Example
//!
//! ```ignore
//! let mut clf = MLPClassifier::new().max_iter(100);
//! clf.fit(&train)?;
//!
//! let history = clf.history().unwrap();
//! println!("Final loss: {:.4}", history.epochs.last().unwrap().train_loss);
//! ```

/// Snapshot of metrics at the end of one training epoch.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EpochMetrics {
    /// Zero-indexed epoch number.
    pub epoch: usize,
    /// Mean training loss for this epoch.
    pub train_loss: f64,
    /// Validation loss (only when early stopping is enabled).
    pub val_loss: Option<f64>,
    /// Training accuracy (classification) or R² (regression).
    pub train_metric: Option<f64>,
    /// Validation accuracy / R² (only when early stopping is enabled).
    pub val_metric: Option<f64>,
    /// Current learning rate.
    pub learning_rate: f64,
    /// L2 norm of all parameter gradients (detects vanishing/exploding).
    pub grad_norm: f64,
    /// Wall-clock milliseconds for this epoch.
    pub elapsed_ms: u64,
}

/// Accumulated history of training metrics — returned after `fit()`.
///
/// Access via `model.history()` on any iterative model.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TrainingHistory {
    /// Per-epoch snapshots.
    pub epochs: Vec<EpochMetrics>,
}

impl TrainingHistory {
    /// Create a new empty history.
    pub fn new() -> Self {
        Self { epochs: Vec::new() }
    }

    /// Push a new epoch snapshot.
    pub fn push(&mut self, metrics: EpochMetrics) {
        self.epochs.push(metrics);
    }

    /// Number of recorded epochs.
    pub fn len(&self) -> usize {
        self.epochs.len()
    }

    /// Whether the history is empty.
    pub fn is_empty(&self) -> bool {
        self.epochs.is_empty()
    }

    /// Training loss per epoch.
    pub fn train_losses(&self) -> Vec<f64> {
        self.epochs.iter().map(|e| e.train_loss).collect()
    }

    /// Validation loss per epoch (only epochs that have it).
    pub fn val_losses(&self) -> Vec<f64> {
        self.epochs.iter().filter_map(|e| e.val_loss).collect()
    }

    /// Training metric (accuracy / R²) per epoch.
    pub fn train_metrics(&self) -> Vec<f64> {
        self.epochs.iter().filter_map(|e| e.train_metric).collect()
    }

    /// Validation metric per epoch.
    pub fn val_metrics(&self) -> Vec<f64> {
        self.epochs.iter().filter_map(|e| e.val_metric).collect()
    }

    /// Gradient L2 norm per epoch.
    pub fn grad_norms(&self) -> Vec<f64> {
        self.epochs.iter().map(|e| e.grad_norm).collect()
    }

    /// Learning rate per epoch.
    pub fn learning_rates(&self) -> Vec<f64> {
        self.epochs.iter().map(|e| e.learning_rate).collect()
    }

    /// Wall-clock milliseconds per epoch.
    pub fn epoch_times_ms(&self) -> Vec<u64> {
        self.epochs.iter().map(|e| e.elapsed_ms).collect()
    }
}

/// Action returned by a [`TrainingCallback`] to control training flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallbackAction {
    /// Continue training normally.
    Continue,
    /// Stop training immediately (user-driven early stop).
    Stop,
}

/// Trait for user-supplied training callbacks.
///
/// Implement this to add custom logging, checkpointing, or stopping
/// criteria during training.
///
/// # Example
///
/// ```ignore
/// struct PrintLogger;
///
/// impl TrainingCallback for PrintLogger {
///     fn on_epoch_end(&mut self, metrics: &EpochMetrics) -> CallbackAction {
///         println!("Epoch {}: loss={:.4}", metrics.epoch, metrics.train_loss);
///         CallbackAction::Continue
///     }
/// }
/// ```
pub trait TrainingCallback: Send + Sync {
    /// Called at the end of each training epoch.
    ///
    /// Return [`CallbackAction::Stop`] to halt training early.
    fn on_epoch_end(&mut self, metrics: &EpochMetrics) -> CallbackAction;

    /// Called when training finishes (after the last epoch).
    ///
    /// Default implementation does nothing.
    fn on_training_end(&mut self) {}
}

/// Compute the L2 norm of all gradients across layers.
///
/// `grads` is the list of `(weight_grads, bias_grads)` per layer.
pub(crate) fn compute_grad_norm(grads: &[(Vec<f64>, Vec<f64>)]) -> f64 {
    let mut sum_sq = 0.0;
    for (dw, db) in grads {
        for &g in dw {
            sum_sq += g * g;
        }
        for &g in db {
            sum_sq += g * g;
        }
    }
    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_accumulates() {
        let mut h = TrainingHistory::new();
        assert!(h.is_empty());

        h.push(EpochMetrics {
            epoch: 0,
            train_loss: 1.5,
            val_loss: Some(1.8),
            train_metric: Some(0.6),
            val_metric: Some(0.55),
            learning_rate: 0.001,
            grad_norm: 2.3,
            elapsed_ms: 42,
        });
        h.push(EpochMetrics {
            epoch: 1,
            train_loss: 1.2,
            val_loss: Some(1.4),
            train_metric: Some(0.7),
            val_metric: Some(0.65),
            learning_rate: 0.001,
            grad_norm: 1.8,
            elapsed_ms: 38,
        });

        assert_eq!(h.len(), 2);
        assert_eq!(h.train_losses(), vec![1.5, 1.2]);
        assert_eq!(h.val_losses(), vec![1.8, 1.4]);
        assert_eq!(h.train_metrics(), vec![0.6, 0.7]);
        assert_eq!(h.grad_norms(), vec![2.3, 1.8]);
    }

    #[test]
    fn history_without_validation() {
        let mut h = TrainingHistory::new();
        h.push(EpochMetrics {
            epoch: 0,
            train_loss: 1.0,
            val_loss: None,
            train_metric: None,
            val_metric: None,
            learning_rate: 0.01,
            grad_norm: 5.0,
            elapsed_ms: 10,
        });

        assert!(h.val_losses().is_empty());
        assert!(h.val_metrics().is_empty());
        assert_eq!(h.train_losses(), vec![1.0]);
    }

    #[test]
    fn grad_norm_basic() {
        let grads = vec![
            (vec![3.0, 4.0], vec![0.0]), // sqrt(9+16+0) = 5
        ];
        let norm = compute_grad_norm(&grads);
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn grad_norm_multi_layer() {
        let grads = vec![(vec![1.0, 0.0], vec![0.0]), (vec![0.0, 0.0], vec![2.0])];
        // sqrt(1 + 0 + 0 + 0 + 0 + 4) = sqrt(5)
        let norm = compute_grad_norm(&grads);
        assert!((norm - 5.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn callback_action() {
        struct StopAt3;
        impl TrainingCallback for StopAt3 {
            fn on_epoch_end(&mut self, m: &EpochMetrics) -> CallbackAction {
                if m.epoch >= 3 {
                    CallbackAction::Stop
                } else {
                    CallbackAction::Continue
                }
            }
        }

        let mut cb = StopAt3;
        let m = EpochMetrics {
            epoch: 2,
            train_loss: 0.0,
            val_loss: None,
            train_metric: None,
            val_metric: None,
            learning_rate: 0.0,
            grad_norm: 0.0,
            elapsed_ms: 0,
        };
        assert_eq!(cb.on_epoch_end(&m), CallbackAction::Continue);

        let m = EpochMetrics { epoch: 3, ..m };
        assert_eq!(cb.on_epoch_end(&m), CallbackAction::Stop);
    }
}
