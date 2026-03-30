// SPDX-License-Identifier: MIT OR Apache-2.0
//! Layer normalization for 2D feature maps (inference-only).
//!
//! Unlike [`super::BatchNorm2d`] which uses stored running statistics, this
//! computes per-sample mean and variance over the spatial dimensions of each
//! channel.
//!
//! Input shape: `[C, H, W]`
//! Output shape: `[C, H, W]`

use scry_llm::backend::MathBackend;
use scry_llm::nn::Module;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

/// Per-channel layer normalization over spatial dimensions.
///
/// For each channel, computes mean and variance over the `H * W` spatial
/// elements, then applies `(x - mean) / sqrt(var + eps) * weight + bias`.
pub struct LayerNorm2d<B: MathBackend> {
    /// Scale parameter (gamma): `[num_channels]`.
    pub weight: Tensor<B>,
    /// Shift parameter (beta): `[num_channels]`.
    pub bias: Tensor<B>,
    pub num_channels: usize,
    pub eps: f32,
}

impl<B: MathBackend> LayerNorm2d<B> {
    /// Create a `LayerNorm2d` with identity initialization (weight=1, bias=0).
    pub fn new(num_channels: usize, eps: f32) -> Self {
        Self {
            weight: Tensor::from_vec(vec![1.0; num_channels], Shape::new(&[num_channels])),
            bias: Tensor::from_vec(vec![0.0; num_channels], Shape::new(&[num_channels])),
            num_channels,
            eps,
        }
    }

    /// Forward pass: `[C, H, W]` -> `[C, H, W]`.
    ///
    /// Normalizes each channel independently using per-sample spatial statistics.
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let dims = input.shape.dims();
        let c = dims[0];
        let spatial = dims[1] * dims[2];

        let input_vec = input.to_vec();
        let weight = self.weight.to_vec();
        let bias = self.bias.to_vec();

        let mut output = vec![0.0f32; c * spatial];
        let inv_spatial = 1.0 / spatial as f32;

        for ch in 0..c {
            let offset = ch * spatial;
            let slice = &input_vec[offset..offset + spatial];

            // Compute mean
            let mean: f32 = slice.iter().sum::<f32>() * inv_spatial;

            // Compute variance
            let var: f32 =
                slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() * inv_spatial;

            let inv_std = 1.0 / (var + self.eps).sqrt();
            let scale = weight[ch] * inv_std;
            let shift = bias[ch] - mean * scale;

            for i in 0..spatial {
                output[offset + i] = input_vec[offset + i] * scale + shift;
            }
        }

        Tensor::from_vec(output, input.shape.clone())
    }
}

impl<B: MathBackend> Module<B> for LayerNorm2d<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.weight, &self.bias]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scry_llm::backend::cpu::CpuBackend;

    #[test]
    fn layernorm2d_identity_init() {
        // Identity init + constant input per channel → should still be identity
        // (constant normalizes to 0, then bias=0 → all zeros)
        let ln = LayerNorm2d::<CpuBackend>::new(2, 1e-5);
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            Shape::new(&[2, 2, 2]),
        );
        let output = ln.forward(&input);
        let data = output.to_vec();

        // Channel 0: mean=2.5, var=1.25
        // (1 - 2.5) / sqrt(1.25) = -1.3416...
        assert!((data[0] - (-1.3416)).abs() < 1e-3);
    }

    #[test]
    fn layernorm2d_output_shape() {
        let ln = LayerNorm2d::<CpuBackend>::new(16, 1e-5);
        let input = Tensor::from_vec(vec![0.0; 16 * 8 * 8], Shape::new(&[16, 8, 8]));
        let output = ln.forward(&input);
        assert_eq!(output.shape.dims(), &[16, 8, 8]);
    }

    #[test]
    fn layernorm2d_known_values() {
        // Single channel, 4 spatial elements: [0, 2, 4, 6]
        // mean = 3, var = 5, std = sqrt(5)
        // normalized: [-3/sqrt(5), -1/sqrt(5), 1/sqrt(5), 3/sqrt(5)]
        let ln = LayerNorm2d::<CpuBackend>::new(1, 0.0);
        let input = Tensor::from_vec(vec![0.0, 2.0, 4.0, 6.0], Shape::new(&[1, 2, 2]));
        let output = ln.forward(&input);
        let data = output.to_vec();

        let std = 5.0f32.sqrt();
        assert!((data[0] - (-3.0 / std)).abs() < 1e-5);
        assert!((data[1] - (-1.0 / std)).abs() < 1e-5);
        assert!((data[2] - (1.0 / std)).abs() < 1e-5);
        assert!((data[3] - (3.0 / std)).abs() < 1e-5);
    }

    #[test]
    fn layernorm2d_multi_channel_independent() {
        // Two channels with different distributions — verify they normalize independently
        let ln = LayerNorm2d::<CpuBackend>::new(2, 0.0);
        let input = Tensor::from_vec(
            vec![
                0.0, 10.0, // channel 0: mean=5, var=25
                100.0, 200.0, // channel 1: mean=150, var=2500
            ],
            Shape::new(&[2, 1, 2]),
        );
        let output = ln.forward(&input);
        let data = output.to_vec();

        // Both channels should produce [-1, 1] after normalization
        assert!((data[0] - (-1.0)).abs() < 1e-5);
        assert!((data[1] - 1.0).abs() < 1e-5);
        assert!((data[2] - (-1.0)).abs() < 1e-5);
        assert!((data[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn layernorm2d_with_scale_and_shift() {
        // weight=2, bias=1 on normalized [0,2,4,6] (mean=3, var=5)
        let mut ln = LayerNorm2d::<CpuBackend>::new(1, 0.0);
        ln.weight = Tensor::from_vec(vec![2.0], Shape::new(&[1]));
        ln.bias = Tensor::from_vec(vec![1.0], Shape::new(&[1]));

        let input = Tensor::from_vec(vec![0.0, 2.0, 4.0, 6.0], Shape::new(&[1, 2, 2]));
        let output = ln.forward(&input);
        let data = output.to_vec();

        let std = 5.0f32.sqrt();
        // 2 * (-3/sqrt(5)) + 1
        assert!((data[0] - (2.0 * (-3.0 / std) + 1.0)).abs() < 1e-5);
    }

    #[test]
    fn layernorm2d_parameters() {
        let ln = LayerNorm2d::<CpuBackend>::new(32, 1e-5);
        let params = ln.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].numel(), 32); // weight
        assert_eq!(params[1].numel(), 32); // bias
    }
}
