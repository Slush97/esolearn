// SPDX-License-Identifier: MIT OR Apache-2.0
//! Dropout layer for neural network regularization.
//!
//! Randomly zeros elements during training to prevent co-adaptation of
//! neurons.  Uses inverted dropout scaling so that the expected output
//! magnitude is preserved and no rescaling is needed at inference time.
//!
//! # Example
//!
//! ```ignore
//! use scry_learn::neural::{MLPClassifier, Activation};
//!
//! let clf = MLPClassifier::new()
//!     .hidden_layers(&[128, 64])
//!     .dropout(0.5)        // 50% dropout between hidden layers
//!     .activation(Activation::Relu)
//!     .seed(42);
//! ```

use super::layer::FastRng;
use super::traits::{BackwardOutput, Layer};

/// A dropout layer that randomly zeros a fraction of inputs during training.
///
/// Uses inverted dropout: surviving activations are scaled by `1 / (1 - p)`
/// during training so that inference requires no adjustment.
pub struct DropoutLayer {
    /// Dropout probability (fraction of elements to zero out).
    p: f64,
    /// Inverse keep probability: `1 / (1 - p)`.
    inv_keep: f64,
    /// Input/output dimension per sample.
    size: usize,
    /// Cached binary mask × inv_keep from the last forward pass.
    mask: Vec<f64>,
    /// Seed for the RNG.
    seed: u64,
    /// Monotonically increasing call counter to vary the mask each forward pass.
    call_count: u64,
}

impl DropoutLayer {
    /// Create a new dropout layer.
    ///
    /// `p` is the probability of zeroing each element (e.g. 0.5).
    /// `size` is the number of elements per sample (matching the preceding
    /// layer's output dimension).
    pub fn new(p: f64, size: usize, seed: u64) -> Self {
        let p_clamped = p.clamp(0.0, 1.0);
        let keep = 1.0 - p_clamped;
        Self {
            p: p_clamped,
            inv_keep: if keep > 0.0 { 1.0 / keep } else { 0.0 },
            size,
            mask: Vec::new(),
            seed,
            call_count: 0,
        }
    }

    /// Returns the dropout probability.
    pub fn dropout_rate(&self) -> f64 {
        self.p
    }
}

impl Layer for DropoutLayer {
    fn forward(&mut self, input: &[f64], batch: usize, training: bool) -> Vec<f64> {
        let total = batch * self.size;
        debug_assert_eq!(input.len(), total);

        if !training || self.p == 0.0 {
            // Inference or no dropout: pass through unchanged.
            self.mask.clear();
            return input.to_vec();
        }

        if self.p >= 1.0 {
            // Zero everything (edge case).
            self.mask = vec![0.0; total];
            return vec![0.0; total];
        }

        // Generate a fresh mask for this forward call.
        let mut rng = FastRng::new(self.seed.wrapping_add(self.call_count));
        self.call_count += 1;

        self.mask.resize(total, 0.0);
        let mut output = Vec::with_capacity(total);

        for i in 0..total {
            // rng.f64() returns [0, 1).  Keep when random >= p.
            let keep = if rng.f64() >= self.p {
                self.inv_keep
            } else {
                0.0
            };
            self.mask[i] = keep;
            output.push(input[i] * keep);
        }

        output
    }

    fn backward(&self, grad_output: &[f64]) -> BackwardOutput {
        // Gradient flows through the same mask used in forward.
        let grad_input: Vec<f64> = if self.mask.is_empty() {
            // No mask means forward was in inference mode — identity gradient.
            grad_output.to_vec()
        } else {
            grad_output
                .iter()
                .zip(self.mask.iter())
                .map(|(&g, &m)| g * m)
                .collect()
        };
        // No trainable parameters → empty param_grads.
        (grad_input, Vec::new())
    }

    fn n_param_groups(&self) -> usize {
        0
    }

    fn params_mut(&mut self) -> Vec<(&mut Vec<f64>, &mut Vec<f64>)> {
        Vec::new()
    }

    fn save_params(&self) -> Vec<(Vec<f64>, Vec<f64>)> {
        Vec::new()
    }

    fn restore_params(&mut self, _saved: &[(Vec<f64>, Vec<f64>)]) {}

    fn in_size(&self) -> usize {
        self.size
    }

    fn out_size(&self) -> usize {
        self.size
    }

    fn name(&self) -> &'static str {
        "Dropout"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dropout_identity_at_inference() {
        let mut layer = DropoutLayer::new(0.5, 4, 42);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = layer.forward(&input, 1, false);
        assert_eq!(input, output, "inference should be identity");
    }

    #[test]
    fn dropout_zeros_some_elements() {
        let mut layer = DropoutLayer::new(0.5, 100, 42);
        let input = vec![1.0; 100];
        let output = layer.forward(&input, 1, true);

        let n_zeros = output.iter().filter(|&&v| v == 0.0).count();
        // With p=0.5 and 100 elements, expect roughly 50 zeros ± 20.
        assert!(
            n_zeros > 20 && n_zeros < 80,
            "expected ~50 zeros at p=0.5, got {n_zeros}"
        );
    }

    #[test]
    fn dropout_inverted_scaling() {
        let mut layer = DropoutLayer::new(0.5, 100, 42);
        let input = vec![1.0; 100];
        let output = layer.forward(&input, 1, true);

        // Non-zero elements should be scaled by 1/(1-p) = 2.0.
        for &v in &output {
            assert!(
                (v - 0.0).abs() < 1e-10 || (v - 2.0).abs() < 1e-10,
                "expected 0.0 or 2.0, got {v}"
            );
        }
    }

    #[test]
    fn dropout_backward_preserves_mask() {
        let mut layer = DropoutLayer::new(0.5, 10, 42);
        let input = vec![1.0; 10];
        let output = layer.forward(&input, 1, true);

        let grad_out = vec![1.0; 10];
        let (grad_input, param_grads) = layer.backward(&grad_out);

        // Grad should match the forward mask pattern.
        for i in 0..10 {
            if output[i] == 0.0 {
                assert_eq!(grad_input[i], 0.0, "zeroed element should have zero grad");
            } else {
                assert!(
                    (grad_input[i] - 2.0).abs() < 1e-10,
                    "kept element grad should be 2.0"
                );
            }
        }
        assert!(param_grads.is_empty(), "dropout has no parameters");
    }

    #[test]
    fn dropout_zero_rate_is_passthrough() {
        let mut layer = DropoutLayer::new(0.0, 5, 42);
        let input = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let output = layer.forward(&input, 1, true);
        assert_eq!(input, output, "p=0 should be identity even during training");
    }

    #[test]
    fn dropout_full_rate_zeros_everything() {
        let mut layer = DropoutLayer::new(1.0, 5, 42);
        let input = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let output = layer.forward(&input, 1, true);
        assert_eq!(output, vec![0.0; 5], "p=1.0 should zero everything");
    }

    #[test]
    fn dropout_batched() {
        let mut layer = DropoutLayer::new(0.5, 4, 42);
        let input = vec![1.0; 8]; // batch=2, size=4
        let output = layer.forward(&input, 2, true);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn dropout_different_masks_per_call() {
        let mut layer = DropoutLayer::new(0.5, 20, 42);
        let input = vec![1.0; 20];

        let out1 = layer.forward(&input, 1, true);
        let out2 = layer.forward(&input, 1, true);

        // Very unlikely (< 1e-6) that two independent masks produce identical output.
        assert_ne!(out1, out2, "successive calls should produce different masks");
    }
}
