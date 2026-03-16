// SPDX-License-Identifier: MIT OR Apache-2.0
//! Layer trait for composable neural network architectures.
//!
//! All layer types (Dense, Conv2D, MaxPool2D, Flatten) implement [`Layer`],
//! enabling heterogeneous layer stacks in `Network`.

/// Gradient output from [`Layer::backward`]: `(grad_input, param_grads)`.
///
/// - `grad_input`: gradient to pass to the previous layer.
/// - `param_grads`: list of `(weight_grads, bias_grads)` per parameter group
///   (empty for parameter-free layers like MaxPool/Flatten).
pub type BackwardOutput = (Vec<f64>, Vec<(Vec<f64>, Vec<f64>)>);

/// A neural network layer that supports forward and backward passes.
///
/// Layers are stateful: `forward()` caches activations for backprop,
/// and `backward()` produces gradients for all trainable parameters.
pub trait Layer: Send {
    /// Forward pass.
    ///
    /// `input` is a flat buffer of shape `[batch, ...]` determined by the layer.
    /// `batch` is the batch dimension.
    /// If `training` is true, caches are stored for backpropagation.
    ///
    /// Returns the output as a flat buffer.
    fn forward(&mut self, input: &[f64], batch: usize, training: bool) -> Vec<f64>;

    /// Backward pass.
    ///
    /// `grad_output` is the gradient of the loss w.r.t. this layer's output.
    /// Returns [`BackwardOutput`] containing gradient for the previous layer
    /// and parameter gradients for optimizer updates.
    fn backward(&self, grad_output: &[f64]) -> BackwardOutput;

    /// Number of trainable parameter groups.
    ///
    /// Dense: 1 (weights + biases). Conv2D: 1 (filters + biases).
    /// MaxPool/Flatten: 0.
    fn n_param_groups(&self) -> usize;

    /// Mutable access to parameters for optimizer updates.
    ///
    /// Returns `(weights, biases)` pairs, one per parameter group.
    fn params_mut(&mut self) -> Vec<(&mut Vec<f64>, &mut Vec<f64>)>;

    /// Read-only parameter snapshot for saving/restoring.
    fn save_params(&self) -> Vec<(Vec<f64>, Vec<f64>)>;

    /// Restore parameters from a snapshot.
    fn restore_params(&mut self, saved: &[(Vec<f64>, Vec<f64>)]);

    /// Input dimension (total elements per sample, excluding batch).
    fn in_size(&self) -> usize;

    /// Output dimension (total elements per sample, excluding batch).
    fn out_size(&self) -> usize;

    /// Descriptive name for debugging.
    fn name(&self) -> &'static str;
}
