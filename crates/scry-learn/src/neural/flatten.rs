// SPDX-License-Identifier: MIT OR Apache-2.0
//! Flatten layer — reshapes multi-dimensional input to 1D per sample.
//!
//! Used between convolutional/pooling layers and dense layers.
//! No trainable parameters; forward/backward are identity operations
//! (the data is already stored flat in memory).

use crate::neural::traits::{BackwardOutput, Layer};

/// Flatten layer.
///
/// Reshapes input from `[batch, C, H, W]` to `[batch, C * H * W]`.
/// Since data is already stored flat, this is a no-op in practice,
/// but it tracks the dimensional change for the network graph.
pub struct Flatten {
    /// Total elements per sample (set on first forward).
    pub(crate) dim: usize,
    pub(crate) cache_batch: usize,
}

impl Flatten {
    /// Create a new Flatten layer.
    pub fn new() -> Self {
        Self {
            dim: 0,
            cache_batch: 0,
        }
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Flatten {
    fn forward(&mut self, input: &[f64], batch: usize, training: bool) -> Vec<f64> {
        self.dim = input.len() / batch;
        if training {
            self.cache_batch = batch;
        }
        // Data is already flat — just pass through.
        input.to_vec()
    }

    fn backward(&self, grad_output: &[f64]) -> BackwardOutput {
        // Identity reshape — gradient passes through unchanged.
        (grad_output.to_vec(), vec![])
    }

    fn n_param_groups(&self) -> usize {
        0
    }

    fn params_mut(&mut self) -> Vec<(&mut Vec<f64>, &mut Vec<f64>)> {
        vec![]
    }

    fn save_params(&self) -> Vec<(Vec<f64>, Vec<f64>)> {
        vec![]
    }

    fn restore_params(&mut self, _saved: &[(Vec<f64>, Vec<f64>)]) {}

    fn in_size(&self) -> usize {
        self.dim
    }

    fn out_size(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &'static str {
        "Flatten"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_passthrough() {
        let mut flat = Flatten::new();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let output = flat.forward(&input, 2, true);
        assert_eq!(output, input);
        assert_eq!(flat.dim, 3); // 6 / 2
    }

    #[test]
    fn flatten_backward_passthrough() {
        let mut flat = Flatten::new();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        flat.forward(&input, 1, true);

        let grad = vec![0.1, 0.2, 0.3, 0.4];
        let (grad_input, params) = flat.backward(&grad);
        assert_eq!(grad_input, grad);
        assert!(params.is_empty());
    }
}
