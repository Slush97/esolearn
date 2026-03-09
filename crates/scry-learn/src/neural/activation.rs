// SPDX-License-Identifier: MIT OR Apache-2.0
//! Activation functions for neural network layers.
//!
//! Each activation provides element-wise `forward()` and `backward()` methods.
//! The backward pass computes the derivative with respect to the pre-activation
//! input, used during backpropagation.

/// Available activation functions.
///
/// Choose based on layer position:
/// - [`Relu`](Activation::Relu) — default for hidden layers (He init)
/// - [`Sigmoid`](Activation::Sigmoid) — binary output or shallow nets
/// - [`Tanh`](Activation::Tanh) — zero-centered alternative to sigmoid
/// - [`Identity`](Activation::Identity) — output layer for regression
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Activation {
    /// Rectified Linear Unit: max(0, x).
    #[default]
    Relu,
    /// Logistic sigmoid: 1 / (1 + exp(-x)).
    Sigmoid,
    /// Hyperbolic tangent: tanh(x).
    Tanh,
    /// Identity (pass-through): x.
    Identity,
}

impl Activation {
    /// Apply the activation element-wise (in-place).
    pub fn forward(&self, z: &mut [f64]) {
        match self {
            Self::Relu => {
                for v in z.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
            Self::Sigmoid => {
                for v in z.iter_mut() {
                    *v = sigmoid(*v);
                }
            }
            Self::Tanh => {
                for v in z.iter_mut() {
                    *v = v.tanh();
                }
            }
            Self::Identity => {}
        }
    }

    /// Compute the element-wise derivative of the activation with respect to
    /// the pre-activation value `z`.
    ///
    /// For ReLU this uses the pre-activation `z` (not the activated value).
    /// For Sigmoid/Tanh, `activated` is the post-activation value `a = f(z)`.
    pub fn backward_from_activated(&self, z: &[f64], activated: &[f64], grad_out: &mut [f64]) {
        match self {
            Self::Relu => {
                for i in 0..grad_out.len() {
                    if z[i] <= 0.0 {
                        grad_out[i] = 0.0;
                    }
                }
            }
            Self::Sigmoid => {
                for i in 0..grad_out.len() {
                    let a = activated[i];
                    grad_out[i] *= a * (1.0 - a);
                }
            }
            Self::Tanh => {
                for i in 0..grad_out.len() {
                    let a = activated[i];
                    grad_out[i] *= 1.0 - a * a;
                }
            }
            Self::Identity => {}
        }
    }

    /// Whether this activation uses He initialization (ReLU family).
    pub(crate) fn uses_he_init(self) -> bool {
        matches!(self, Self::Relu)
    }
}

/// Numerically stable sigmoid.
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let ex = (-x).exp();
        1.0 / (1.0 + ex)
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_forward() {
        let mut z = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        Activation::Relu.forward(&mut z);
        assert_eq!(z, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn relu_backward() {
        let z = vec![-2.0, 0.0, 1.0, 3.0];
        let activated = vec![0.0, 0.0, 1.0, 3.0];
        let mut grad = vec![1.0, 1.0, 1.0, 1.0];
        Activation::Relu.backward_from_activated(&z, &activated, &mut grad);
        assert_eq!(grad, vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn sigmoid_forward() {
        let mut z = vec![0.0];
        Activation::Sigmoid.forward(&mut z);
        assert!((z[0] - 0.5).abs() < 1e-10);

        let mut z = vec![100.0];
        Activation::Sigmoid.forward(&mut z);
        assert!((z[0] - 1.0).abs() < 1e-10);

        let mut z = vec![-100.0];
        Activation::Sigmoid.forward(&mut z);
        assert!(z[0].abs() < 1e-10);
    }

    #[test]
    fn sigmoid_backward() {
        // At z=0, sigmoid=0.5, derivative = 0.5 * 0.5 = 0.25
        let z = vec![0.0];
        let activated = vec![0.5];
        let mut grad = vec![1.0];
        Activation::Sigmoid.backward_from_activated(&z, &activated, &mut grad);
        assert!((grad[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn tanh_forward() {
        let mut z = vec![0.0];
        Activation::Tanh.forward(&mut z);
        assert!(z[0].abs() < 1e-10);
    }

    #[test]
    fn tanh_backward() {
        // At z=0, tanh=0, derivative = 1 - 0 = 1
        let z = vec![0.0];
        let activated = vec![0.0];
        let mut grad = vec![1.0];
        Activation::Tanh.backward_from_activated(&z, &activated, &mut grad);
        assert!((grad[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn identity_is_noop() {
        let mut z = vec![1.0, -2.0, 3.0];
        let original = z.clone();
        Activation::Identity.forward(&mut z);
        assert_eq!(z, original);

        let mut grad = vec![1.0, 2.0, 3.0];
        let original_grad = grad.clone();
        Activation::Identity.backward_from_activated(&z, &z, &mut grad);
        assert_eq!(grad, original_grad);
    }

    #[test]
    fn sigmoid_numerical_stability() {
        // Very negative input should not NaN/Inf
        let mut z = vec![-750.0];
        Activation::Sigmoid.forward(&mut z);
        assert!(z[0].is_finite());
        assert!(z[0] >= 0.0);
    }
}
