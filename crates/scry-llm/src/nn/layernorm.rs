use crate::backend::MathBackend;
use crate::nn::Module;
use crate::ops;
use crate::tensor::shape::Shape;
use crate::tensor::Tensor;

/// Layer normalization module.
pub struct LayerNormModule<B: MathBackend> {
    pub gamma: Tensor<B>,
    pub beta: Tensor<B>,
    pub eps: f32,
}

impl<B: MathBackend> LayerNormModule<B> {
    pub fn new(d_model: usize) -> Self {
        Self {
            gamma: Tensor::from_vec(vec![1.0; d_model], Shape::new(&[d_model])),
            beta: Tensor::from_vec(vec![0.0; d_model], Shape::new(&[d_model])),
            eps: 1e-5,
        }
    }

    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        ops::layernorm_inference(input, &self.gamma, &self.beta, self.eps)
    }
}

impl<B: MathBackend> Module<B> for LayerNormModule<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.gamma, &self.beta]
    }
}
