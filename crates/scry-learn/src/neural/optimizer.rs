// SPDX-License-Identifier: MIT OR Apache-2.0
//! Optimizers for neural network training.
//!
//! Provides SGD with Nesterov momentum and Adam, matching sklearn defaults.

/// Available optimizer algorithms.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum OptimizerKind {
    /// Stochastic gradient descent with optional Nesterov momentum.
    Sgd {
        /// Momentum coefficient (0.0 = no momentum). Default: 0.9.
        momentum: f64,
        /// Use Nesterov accelerated gradient. Default: true.
        nesterov: bool,
    },
    /// Adaptive moment estimation (Adam).
    ///
    /// Defaults: β₁=0.9, β₂=0.999, ε=1e-8.
    Adam {
        /// Exponential decay rate for first moment. Default: 0.9.
        beta1: f64,
        /// Exponential decay rate for second moment. Default: 0.999.
        beta2: f64,
        /// Small constant for numerical stability. Default: 1e-8.
        epsilon: f64,
    },
}

impl Default for OptimizerKind {
    fn default() -> Self {
        Self::Adam {
            beta1: crate::constants::ADAM_BETA1,
            beta2: crate::constants::ADAM_BETA2,
            epsilon: crate::constants::ADAM_EPSILON,
        }
    }
}

impl OptimizerKind {
    /// SGD with default momentum (0.9, Nesterov).
    pub fn sgd() -> Self {
        Self::Sgd {
            momentum: crate::constants::SGD_MOMENTUM,
            nesterov: true,
        }
    }
}

/// Learning rate schedule for neural network training.
///
/// Controls how the learning rate changes over epochs.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum LearningRateSchedule {
    /// Fixed learning rate throughout training. Default.
    #[default]
    Constant,
    /// Reduce learning rate by `factor` when loss plateaus for `patience` epochs.
    ///
    /// Matches sklearn's `learning_rate='adaptive'` behavior.
    Adaptive {
        /// Multiplicative factor to reduce LR (default: 0.2).
        factor: f64,
        /// Number of plateau epochs before reducing (default: 10).
        patience: usize,
    },
    /// Inverse scaling: `lr(t) = initial_lr / t^power`.
    InvScaling {
        /// Exponent for inverse scaling (default: 0.5).
        power: f64,
    },
}



impl LearningRateSchedule {
    /// Adaptive schedule with sklearn-like defaults (factor=0.2, patience=10).
    pub fn adaptive() -> Self {
        Self::Adaptive {
            factor: 0.2,
            patience: 10,
        }
    }
}

/// Per-parameter optimizer state.
///
/// Tracks the moving averages needed by each optimizer algorithm.
pub(crate) struct OptimizerState {
    kind: OptimizerKind,
    lr: f64,
    initial_lr: f64,
    t: u64,
    // SGD momentum buffers (one per parameter group)
    velocity: Vec<Vec<f64>>,
    // Adam first moment (mean)
    m: Vec<Vec<f64>>,
    // Adam second moment (variance)
    v: Vec<Vec<f64>>,
    // ── Learning rate schedule state ──
    schedule: LearningRateSchedule,
    best_loss: f64,
    plateau_count: usize,
    epoch_count: usize,
}

impl OptimizerState {
    /// Create a new optimizer state for `n_groups` parameter groups,
    /// each with the given sizes.
    pub fn new(kind: OptimizerKind, lr: f64, group_sizes: &[usize]) -> Self {
        Self::new_with_schedule(kind, lr, group_sizes, LearningRateSchedule::Constant)
    }

    /// Create a new optimizer state with a learning rate schedule.
    pub fn new_with_schedule(
        kind: OptimizerKind,
        lr: f64,
        group_sizes: &[usize],
        schedule: LearningRateSchedule,
    ) -> Self {
        let n = group_sizes.len();
        let zeros =
            |sizes: &[usize]| -> Vec<Vec<f64>> { sizes.iter().map(|&s| vec![0.0; s]).collect() };

        Self {
            kind,
            lr,
            initial_lr: lr,
            t: 0,
            velocity: zeros(group_sizes),
            m: if matches!(kind, OptimizerKind::Adam { .. }) {
                zeros(group_sizes)
            } else {
                Vec::with_capacity(n)
            },
            v: if matches!(kind, OptimizerKind::Adam { .. }) {
                zeros(group_sizes)
            } else {
                Vec::with_capacity(n)
            },
            schedule,
            best_loss: f64::INFINITY,
            plateau_count: 0,
            epoch_count: 0,
        }
    }

    /// Apply one optimization step to parameter group `idx`.
    ///
    /// `params` are modified in-place. `grads` are the computed gradients.
    pub fn step(&mut self, idx: usize, params: &mut [f64], grads: &[f64]) {
        debug_assert_eq!(params.len(), grads.len());
        debug_assert!(idx < self.velocity.len());

        match self.kind {
            OptimizerKind::Sgd { momentum, nesterov } => {
                self.step_sgd(idx, params, grads, momentum, nesterov);
            }
            OptimizerKind::Adam {
                beta1,
                beta2,
                epsilon,
            } => {
                self.step_adam(idx, params, grads, beta1, beta2, epsilon);
            }
        }
    }

    /// Increment the global step counter. Call once per mini-batch.
    pub fn tick(&mut self) {
        self.t += 1;
    }

    /// Current learning rate (may differ from initial after scheduling).
    pub fn current_lr(&self) -> f64 {
        self.lr
    }

    /// Adjust learning rate based on the schedule after each epoch.
    ///
    /// Call this at the end of each epoch with the epoch's average loss.
    pub fn adjust_lr(&mut self, epoch_loss: f64) {
        self.epoch_count += 1;

        match self.schedule {
            LearningRateSchedule::Constant => {}
            LearningRateSchedule::Adaptive { factor, patience } => {
                if epoch_loss < self.best_loss - 1e-10 {
                    self.best_loss = epoch_loss;
                    self.plateau_count = 0;
                } else {
                    self.plateau_count += 1;
                    if self.plateau_count >= patience {
                        self.lr *= factor;
                        self.plateau_count = 0;
                        self.best_loss = epoch_loss;
                    }
                }
            }
            LearningRateSchedule::InvScaling { power } => {
                self.lr = self.initial_lr / (self.epoch_count as f64).powf(power);
            }
        }
    }

    fn step_sgd(
        &mut self,
        idx: usize,
        params: &mut [f64],
        grads: &[f64],
        momentum: f64,
        nesterov: bool,
    ) {
        let vel = &mut self.velocity[idx];
        let lr = self.lr;

        if momentum == 0.0 {
            for (p, g) in params.iter_mut().zip(grads.iter()) {
                *p -= lr * g;
            }
        } else if nesterov {
            for i in 0..params.len() {
                vel[i] = momentum * vel[i] + grads[i];
                params[i] -= lr * (grads[i] + momentum * vel[i]);
            }
        } else {
            for i in 0..params.len() {
                vel[i] = momentum * vel[i] + grads[i];
                params[i] -= lr * vel[i];
            }
        }
    }

    fn step_adam(
        &mut self,
        idx: usize,
        params: &mut [f64],
        grads: &[f64],
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) {
        let lr = self.lr;
        let t = self.t.max(1) as f64;
        let m = &mut self.m[idx];
        let v = &mut self.v[idx];

        // Bias correction
        let bc1 = 1.0 - beta1.powf(t);
        let bc2 = 1.0 - beta2.powf(t);

        for i in 0..params.len() {
            // Update biased first moment
            m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
            // Update biased second moment
            v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];
            // Bias-corrected estimates
            let m_hat = m[i] / bc1;
            let v_hat = v[i] / bc2;
            // Parameter update
            params[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgd_no_momentum() {
        let kind = OptimizerKind::Sgd {
            momentum: 0.0,
            nesterov: false,
        };
        let mut opt = OptimizerState::new(kind, 0.1, &[3]);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.5, -0.5, 1.0];
        opt.tick();
        opt.step(0, &mut params, &grads);
        assert!((params[0] - 0.95).abs() < 1e-10);
        assert!((params[1] - 2.05).abs() < 1e-10);
        assert!((params[2] - 2.9).abs() < 1e-10);
    }

    #[test]
    fn sgd_with_momentum() {
        let kind = OptimizerKind::Sgd {
            momentum: 0.9,
            nesterov: false,
        };
        let mut opt = OptimizerState::new(kind, 0.01, &[2]);
        let mut params = vec![1.0, 2.0];
        let grads = vec![1.0, -1.0];
        opt.tick();
        opt.step(0, &mut params, &grads);
        // velocity = 0.9*0 + 1.0 = 1.0, param = 1.0 - 0.01*1.0 = 0.99
        assert!((params[0] - 0.99).abs() < 1e-10);
        assert!((params[1] - 2.01).abs() < 1e-10);
    }

    #[test]
    fn adam_basic() {
        let kind = OptimizerKind::default(); // Adam
        let mut opt = OptimizerState::new(kind, 0.001, &[2]);
        let mut params = vec![1.0, 2.0];
        let grads = vec![0.5, -0.5];
        opt.tick();
        opt.step(0, &mut params, &grads);
        // After one step, params should have moved toward zero gradient
        assert!(params[0] < 1.0);
        assert!(params[1] > 2.0);
    }

    #[test]
    fn adam_converges_toward_minimum() {
        // Minimize f(x) = x^2, gradient = 2x
        let kind = OptimizerKind::default();
        let mut opt = OptimizerState::new(kind, 0.1, &[1]);
        let mut params = vec![5.0];

        for _ in 0..500 {
            let grads = vec![2.0 * params[0]];
            opt.tick();
            opt.step(0, &mut params, &grads);
        }
        assert!(
            params[0].abs() < 0.1,
            "should converge near 0, got {}",
            params[0]
        );
    }

    #[test]
    fn multiple_groups() {
        let kind = OptimizerKind::default();
        let mut opt = OptimizerState::new(kind, 0.001, &[3, 2]);
        let mut p1 = vec![1.0, 2.0, 3.0];
        let mut p2 = vec![4.0, 5.0];
        let g1 = vec![0.1, 0.2, 0.3];
        let g2 = vec![0.4, 0.5];
        opt.tick();
        opt.step(0, &mut p1, &g1);
        opt.step(1, &mut p2, &g2);
        // Just verify no panic and params changed
        assert!(p1[0] < 1.0);
        assert!(p2[0] < 4.0);
    }
}
