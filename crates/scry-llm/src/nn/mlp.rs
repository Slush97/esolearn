use crate::backend::MathBackend;
use crate::nn::linear::Linear;
use crate::nn::Module;
use crate::ops;
use crate::tensor::Tensor;

/// Feed-forward network: `fc1 -> gelu -> fc2`.
pub struct Mlp<B: MathBackend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

impl<B: MathBackend> Mlp<B> {
    pub fn new(d_model: usize, d_ff: usize, rng: &mut fastrand::Rng) -> Self {
        Self {
            fc1: Linear::new(d_model, d_ff, rng),
            fc2: Linear::new(d_ff, d_model, rng),
        }
    }

    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let h = self.fc1.forward(input);
        let h = ops::gelu(&h);
        self.fc2.forward(&h)
    }
}

impl<B: MathBackend> Module<B> for Mlp<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }
}
