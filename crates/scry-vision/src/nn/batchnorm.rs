// SPDX-License-Identifier: MIT OR Apache-2.0
//! Batch normalization layer for 2D inputs (inference-only).
//!
//! Applies `y = (x - mean) / sqrt(var + eps) * weight + bias` per channel
//! using stored running statistics.
//!
//! Input shape: `[C, H, W]`
//! Output shape: `[C, H, W]`

use scry_llm::backend::MathBackend;
use scry_llm::nn::Module;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

/// Batch normalization with running statistics (inference-only).
///
/// All parameter tensors have shape `[num_features]` (one value per channel).
pub struct BatchNorm2d<B: MathBackend> {
    /// Scale parameter (gamma).
    pub weight: Tensor<B>,
    /// Shift parameter (beta).
    pub bias: Tensor<B>,
    /// Running mean accumulated during training.
    pub running_mean: Tensor<B>,
    /// Running variance accumulated during training.
    pub running_var: Tensor<B>,
    pub num_features: usize,
    pub eps: f32,
}

impl<B: MathBackend> BatchNorm2d<B> {
    /// Create a BatchNorm2d with identity initialization.
    ///
    /// weight=1, bias=0, running_mean=0, running_var=1 → identity transform.
    pub fn new(num_features: usize, eps: f32) -> Self {
        Self {
            weight: Tensor::from_vec(vec![1.0; num_features], Shape::new(&[num_features])),
            bias: Tensor::from_vec(vec![0.0; num_features], Shape::new(&[num_features])),
            running_mean: Tensor::from_vec(vec![0.0; num_features], Shape::new(&[num_features])),
            running_var: Tensor::from_vec(vec![1.0; num_features], Shape::new(&[num_features])),
            num_features,
            eps,
        }
    }

    /// Forward pass: `[C, H, W]` → `[C, H, W]`.
    ///
    /// Applies `(x - mean) / sqrt(var + eps) * weight + bias` per channel.
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let dims = input.shape.dims();
        let c = dims[0];
        let spatial = dims[1] * dims[2];

        let input_vec = input.to_vec();
        let weight = self.weight.to_vec();
        let bias = self.bias.to_vec();
        let mean = self.running_mean.to_vec();
        let var = self.running_var.to_vec();

        let mut output = vec![0.0f32; c * spatial];

        for ch in 0..c {
            let scale = weight[ch] / (var[ch] + self.eps).sqrt();
            let shift = bias[ch] - mean[ch] * scale;
            let offset = ch * spatial;
            for i in 0..spatial {
                output[offset + i] = input_vec[offset + i] * scale + shift;
            }
        }

        Tensor::from_vec(output, input.shape.clone())
    }
}

impl<B: MathBackend> Module<B> for BatchNorm2d<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![
            &self.weight,
            &self.bias,
            &self.running_mean,
            &self.running_var,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scry_llm::backend::cpu::CpuBackend;

    #[test]
    fn batchnorm_identity_init() {
        // Default init should be identity transform
        let bn = BatchNorm2d::<CpuBackend>::new(3, 1e-5);
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            Shape::new(&[3, 2, 2]),
        );
        let output = bn.forward(&input);
        let out = output.to_vec();
        let inp = input.to_vec();
        for i in 0..out.len() {
            assert!(
                (out[i] - inp[i]).abs() < 1e-4,
                "mismatch at {i}: {} vs {}",
                out[i],
                inp[i]
            );
        }
    }

    #[test]
    fn batchnorm_output_shape() {
        let bn = BatchNorm2d::<CpuBackend>::new(16, 1e-5);
        let input = Tensor::from_vec(vec![0.0; 16 * 8 * 8], Shape::new(&[16, 8, 8]));
        let output = bn.forward(&input);
        assert_eq!(output.shape.dims(), &[16, 8, 8]);
    }

    #[test]
    fn batchnorm_known_values() {
        // Channel 0: mean=2, var=4, weight=1, bias=0, eps=0
        // x=4 → (4-2)/sqrt(4) = 1.0
        let mut bn = BatchNorm2d::<CpuBackend>::new(1, 0.0);
        bn.running_mean = Tensor::from_vec(vec![2.0], Shape::new(&[1]));
        bn.running_var = Tensor::from_vec(vec![4.0], Shape::new(&[1]));

        let input = Tensor::from_vec(vec![4.0], Shape::new(&[1, 1, 1]));
        let output = bn.forward(&input);
        assert!((output.to_vec()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn batchnorm_with_scale_and_shift() {
        // mean=0, var=1, weight=2, bias=3
        // x=1 → (1-0)/sqrt(1) * 2 + 3 = 5.0
        let mut bn = BatchNorm2d::<CpuBackend>::new(1, 0.0);
        bn.weight = Tensor::from_vec(vec![2.0], Shape::new(&[1]));
        bn.bias = Tensor::from_vec(vec![3.0], Shape::new(&[1]));

        let input = Tensor::from_vec(vec![1.0], Shape::new(&[1, 1, 1]));
        let output = bn.forward(&input);
        assert!((output.to_vec()[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn batchnorm_multi_channel() {
        let mut bn = BatchNorm2d::<CpuBackend>::new(2, 0.0);
        bn.running_mean = Tensor::from_vec(vec![1.0, 10.0], Shape::new(&[2]));
        bn.running_var = Tensor::from_vec(vec![1.0, 4.0], Shape::new(&[2]));

        // Channel 0: x=3, (3-1)/1 = 2.0
        // Channel 1: x=14, (14-10)/2 = 2.0
        let input = Tensor::from_vec(vec![3.0, 14.0], Shape::new(&[2, 1, 1]));
        let output = bn.forward(&input);
        let data = output.to_vec();
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn batchnorm_parameters() {
        let bn = BatchNorm2d::<CpuBackend>::new(16, 1e-5);
        let params = bn.parameters();
        assert_eq!(params.len(), 4); // weight, bias, running_mean, running_var
        for p in &params {
            assert_eq!(p.numel(), 16);
        }
    }
}
