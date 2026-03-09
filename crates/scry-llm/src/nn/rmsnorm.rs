use crate::backend::MathBackend;
use crate::nn::Module;
use crate::ops;
use crate::tensor::shape::Shape;
use crate::tensor::Tensor;

/// RMS normalization module (used by Llama).
///
/// Unlike `LayerNorm`, `RMSNorm` has no bias and no mean subtraction:
/// `out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]`
pub struct RMSNorm<B: MathBackend> {
    pub weight: Tensor<B>,
    pub eps: f32,
}

impl<B: MathBackend> RMSNorm<B> {
    pub fn new(dim: usize) -> Self {
        Self {
            weight: Tensor::from_vec(vec![1.0; dim], Shape::new(&[dim])),
            eps: 1e-5,
        }
    }

    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        ops::rmsnorm(input, &self.weight, self.eps)
    }
}

impl<B: MathBackend> Module<B> for RMSNorm<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.weight]
    }
}
