// SPDX-License-Identifier: MIT OR Apache-2.0
//! Dense (fully-connected) neural network layer.
//!
//! Each layer stores weights, biases, and caches for the forward/backward pass.
//! Weight initialization follows He (ReLU) or Xavier (sigmoid/tanh) schemes.

use crate::neural::activation::Activation;
pub(crate) use crate::rng::FastRng;

/// A single dense (fully-connected) layer.
///
/// Computes `z = x·W^T + b`, then `a = activation(z)`.
/// During training, caches `z` and `a` for backpropagation.
pub(crate) struct DenseLayer {
    /// Weight matrix: `[out_size, in_size]` row-major.
    pub weights: Vec<f64>,
    /// Bias vector: `[out_size]`.
    pub biases: Vec<f64>,
    /// Input dimension.
    pub in_size: usize,
    /// Output dimension.
    pub out_size: usize,
    /// Activation function for this layer.
    pub activation: Activation,

    // ── Training caches (populated during forward, consumed during backward) ──
    /// Cached input to this layer: `[batch, in_size]` row-major.
    pub cache_input: Vec<f64>,
    /// Cached pre-activation values: `[batch, out_size]` row-major.
    pub cache_z: Vec<f64>,
    /// Cached post-activation values: `[batch, out_size]` row-major.
    pub cache_a: Vec<f64>,
    /// Cached batch size from the last forward pass.
    pub cache_batch: usize,
}

impl DenseLayer {
    /// Create a new dense layer with the given dimensions.
    ///
    /// Weights are initialized using He or Xavier initialization depending
    /// on the activation function. Biases are initialized to zero.
    pub fn new(in_size: usize, out_size: usize, activation: Activation, rng: &mut FastRng) -> Self {
        let scale = if activation.uses_he_init() {
            // He initialization: sqrt(2 / fan_in)
            (2.0 / in_size as f64).sqrt()
        } else {
            // Xavier/Glorot initialization: sqrt(2 / (fan_in + fan_out))
            (2.0 / (in_size + out_size) as f64).sqrt()
        };

        let n = in_size * out_size;
        let mut weights = Vec::with_capacity(n);
        for _ in 0..n {
            weights.push(rng.normal() * scale);
        }

        Self {
            weights,
            biases: vec![0.0; out_size],
            in_size,
            out_size,
            activation,
            cache_input: Vec::new(),
            cache_z: Vec::new(),
            cache_a: Vec::new(),
            cache_batch: 0,
        }
    }

    /// Forward pass: compute `a = activation(x · W^T + b)`.
    ///
    /// `input` is row-major `[batch, in_size]`.
    /// Returns post-activation output `[batch, out_size]` row-major.
    ///
    /// If `training` is true, caches are stored for backprop.
    pub fn forward(&mut self, input: &[f64], batch: usize, training: bool) -> Vec<f64> {
        debug_assert_eq!(input.len(), batch * self.in_size);

        // z = input · W^T + b
        // W is [out, in], so W^T is [in, out].
        // Result z is [batch, out].
        let mut z = vec![0.0; batch * self.out_size];

        for i in 0..batch {
            for j in 0..self.out_size {
                let mut sum = self.biases[j];
                let in_row = i * self.in_size;
                let w_row = j * self.in_size;
                for k in 0..self.in_size {
                    sum += input[in_row + k] * self.weights[w_row + k];
                }
                z[i * self.out_size + j] = sum;
            }
        }

        // Apply activation
        let mut a = z.clone();
        self.activation.forward(&mut a);

        if training {
            self.cache_input = input.to_vec();
            self.cache_z = z;
            self.cache_a.clone_from(&a);
            self.cache_batch = batch;
        }

        a
    }

    /// Forward pass using a `ComputeBackend` for the matrix multiply.
    ///
    /// `input` is row-major `[batch, in_size]`.
    pub fn forward_with_backend(
        &mut self,
        input: &[f64],
        batch: usize,
        training: bool,
        backend: &dyn crate::accel::ComputeBackend,
    ) -> Vec<f64> {
        debug_assert_eq!(input.len(), batch * self.in_size);

        // z = input · W^T
        // input: [batch, in_size], W^T: [in_size, out_size]
        // We need to transpose W ([out, in]) to get [in, out].
        let wt = transpose(&self.weights, self.out_size, self.in_size);
        let mut z = backend.matmul(input, &wt, batch, self.in_size, self.out_size);

        // Add bias
        for i in 0..batch {
            for j in 0..self.out_size {
                z[i * self.out_size + j] += self.biases[j];
            }
        }

        let mut a = z.clone();
        self.activation.forward(&mut a);

        if training {
            self.cache_input = input.to_vec();
            self.cache_z = z;
            self.cache_a.clone_from(&a);
            self.cache_batch = batch;
        }

        a
    }

    /// Backward pass: compute gradients for weights, biases, and input.
    ///
    /// `grad_output` is `[batch, out_size]` — the gradient of the loss
    /// with respect to this layer's output.
    ///
    /// Returns:
    /// - `grad_input`: `[batch, in_size]` — gradient to pass to previous layer
    /// - Weight and bias gradients are written to `dw` and `db`.
    pub fn backward(&self, grad_output: &[f64], dw: &mut [f64], db: &mut [f64]) -> Vec<f64> {
        let batch = self.cache_batch;
        debug_assert_eq!(grad_output.len(), batch * self.out_size);
        debug_assert_eq!(dw.len(), self.out_size * self.in_size);
        debug_assert_eq!(db.len(), self.out_size);

        // Apply activation derivative: delta = grad_output ⊙ f'(z)
        let mut delta = grad_output.to_vec();
        self.activation
            .backward_from_activated(&self.cache_z, &self.cache_a, &mut delta);

        // Bias gradient: db[j] = sum_i(delta[i,j]) / batch
        let batch_f = batch as f64;
        for j in 0..self.out_size {
            let mut sum = 0.0;
            for i in 0..batch {
                sum += delta[i * self.out_size + j];
            }
            db[j] = sum / batch_f;
        }

        // Weight gradient: dW = delta^T · input / batch
        // delta: [batch, out], input: [batch, in] → dW: [out, in]
        for o in 0..self.out_size {
            for k in 0..self.in_size {
                let mut sum = 0.0;
                for i in 0..batch {
                    sum += delta[i * self.out_size + o] * self.cache_input[i * self.in_size + k];
                }
                dw[o * self.in_size + k] = sum / batch_f;
            }
        }

        // Input gradient: grad_input = delta · W
        // delta: [batch, out], W: [out, in] → grad_input: [batch, in]
        let mut grad_input = vec![0.0; batch * self.in_size];
        for i in 0..batch {
            for k in 0..self.in_size {
                let mut sum = 0.0;
                for o in 0..self.out_size {
                    sum += delta[i * self.out_size + o] * self.weights[o * self.in_size + k];
                }
                grad_input[i * self.in_size + k] = sum;
            }
        }

        grad_input
    }

    /// Number of trainable parameters (weights + biases).
    #[allow(dead_code)]
    pub fn n_params(&self) -> usize {
        self.weights.len() + self.biases.len()
    }
}

// ── Layer trait implementation ──

impl crate::neural::traits::Layer for DenseLayer {
    fn forward(&mut self, input: &[f64], batch: usize, training: bool) -> Vec<f64> {
        self.forward(input, batch, training)
    }

    fn backward(&self, grad_output: &[f64]) -> crate::neural::traits::BackwardOutput {
        let mut dw = vec![0.0; self.out_size * self.in_size];
        let mut db = vec![0.0; self.out_size];
        let grad_input = self.backward(grad_output, &mut dw, &mut db);
        (grad_input, vec![(dw, db)])
    }

    fn n_param_groups(&self) -> usize {
        1
    }

    fn params_mut(&mut self) -> Vec<(&mut Vec<f64>, &mut Vec<f64>)> {
        vec![(&mut self.weights, &mut self.biases)]
    }

    fn save_params(&self) -> Vec<(Vec<f64>, Vec<f64>)> {
        vec![(self.weights.clone(), self.biases.clone())]
    }

    fn restore_params(&mut self, saved: &[(Vec<f64>, Vec<f64>)]) {
        if let Some((w, b)) = saved.first() {
            self.weights.clone_from(w);
            self.biases.clone_from(b);
        }
    }

    fn in_size(&self) -> usize {
        self.in_size
    }

    fn out_size(&self) -> usize {
        self.out_size
    }

    fn name(&self) -> &'static str {
        "Dense"
    }
}

/// Transpose a row-major `[rows, cols]` matrix to `[cols, rows]`.
fn transpose(m: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    debug_assert_eq!(m.len(), rows * cols);
    let mut t = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = m[i * cols + j];
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_forward_shape() {
        let mut rng = FastRng::new(42);
        let mut layer = DenseLayer::new(3, 5, Activation::Relu, &mut rng);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // batch=2, in=3
        let output = layer.forward(&input, 2, true);
        assert_eq!(output.len(), 2 * 5);
    }

    #[test]
    fn dense_backward_shape() {
        let mut rng = FastRng::new(42);
        let mut layer = DenseLayer::new(3, 5, Activation::Relu, &mut rng);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // batch=2
        layer.forward(&input, 2, true);

        let grad_out = vec![1.0; 2 * 5];
        let mut dw = vec![0.0; 5 * 3];
        let mut db = vec![0.0; 5];
        let grad_input = layer.backward(&grad_out, &mut dw, &mut db);
        assert_eq!(grad_input.len(), 2 * 3);
    }

    #[test]
    fn identity_layer_passes_through() {
        let mut rng = FastRng::new(0);
        let mut layer = DenseLayer::new(2, 2, Activation::Identity, &mut rng);
        // Set weights to identity, biases to zero
        layer.weights = vec![1.0, 0.0, 0.0, 1.0];
        layer.biases = vec![0.0, 0.0];

        let input = vec![3.0, 7.0];
        let output = layer.forward(&input, 1, false);
        assert!((output[0] - 3.0).abs() < 1e-10);
        assert!((output[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn numerical_gradient_check() {
        let mut rng = FastRng::new(123);
        let mut layer = DenseLayer::new(3, 2, Activation::Tanh, &mut rng);
        let input = vec![0.5, -0.3, 0.8];
        let batch = 1;

        // Forward
        layer.forward(&input, batch, true);
        let grad_out = vec![1.0, 1.0]; // dL/da = 1

        let mut dw = vec![0.0; 2 * 3];
        let mut db = vec![0.0; 2];
        layer.backward(&grad_out, &mut dw, &mut db);

        // Numerical gradient for weights
        let eps = 1e-5;
        #[allow(clippy::needless_range_loop)]
        for idx in 0..layer.weights.len() {
            let orig = layer.weights[idx];

            layer.weights[idx] = orig + eps;
            let out_plus = layer.forward(&input, batch, false);

            layer.weights[idx] = orig - eps;
            let out_minus = layer.forward(&input, batch, false);

            layer.weights[idx] = orig;

            let numerical: f64 = out_plus
                .iter()
                .zip(out_minus.iter())
                .map(|(p, m)| (p - m) / (2.0 * eps))
                .sum::<f64>()
                / batch as f64;

            let analytic = dw[idx];
            let diff = (analytic - numerical).abs();
            let denom = analytic.abs().max(numerical.abs()).max(1e-8);
            assert!(
                diff / denom < 1e-4,
                "weight gradient mismatch at {idx}: analytic={analytic}, numerical={numerical}"
            );
        }
    }
}
