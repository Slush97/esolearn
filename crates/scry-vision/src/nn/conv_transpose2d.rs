// SPDX-License-Identifier: MIT OR Apache-2.0
//! Transposed 2D convolution (deconvolution) layer (inference-only).
//!
//! Used for learnable upsampling in decoder architectures.
//! Implements the col2im-based transposed convolution.
//!
//! Input shape: `[C_in, H, W]`
//! Output shape: `[C_out, H_out, W_out]`
//!
//! where `H_out = (H - 1) * stride - 2 * padding + kernel_h + output_padding`.

use scry_llm::backend::MathBackend;
use scry_llm::nn::Module;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

/// Transposed 2D convolution (learnable upsampling).
///
/// Weight shape: `[in_channels, out_channels, kernel_h, kernel_w]`
/// Bias shape: `[out_channels]`
///
/// Note: weight layout is `[in, out, kH, kW]` (transposed relative to [`super::Conv2d`]).
pub struct ConvTranspose2d<B: MathBackend> {
    pub weight: Tensor<B>,
    pub bias: Tensor<B>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
}

impl<B: MathBackend> ConvTranspose2d<B> {
    /// Create a zero-initialized `ConvTranspose2d`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
    ) -> Self {
        let w_len = in_channels * out_channels * kernel_h * kernel_w;
        Self {
            weight: Tensor::from_vec(
                vec![0.0f32; w_len],
                Shape::new(&[in_channels, out_channels, kernel_h, kernel_w]),
            ),
            bias: Tensor::from_vec(vec![0.0f32; out_channels], Shape::new(&[out_channels])),
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
            output_padding,
        }
    }

    /// Convenience constructor for square kernels.
    pub fn square(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        Self::new(
            in_channels,
            out_channels,
            kernel_size,
            kernel_size,
            stride,
            padding,
            0,
        )
    }

    /// Forward pass: `[C_in, H, W]` -> `[C_out, H_out, W_out]`.
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let dims = input.shape.dims();
        let h_in = dims[1];
        let w_in = dims[2];

        let h_out =
            (h_in - 1) * self.stride - 2 * self.padding + self.kernel_h + self.output_padding;
        let w_out =
            (w_in - 1) * self.stride - 2 * self.padding + self.kernel_w + self.output_padding;

        let input_vec = input.to_vec();
        let weight_vec = self.weight.to_vec();
        let bias_vec = self.bias.to_vec();

        let mut output = vec![0.0f32; self.out_channels * h_out * w_out];

        // Scatter-based transposed convolution.
        // For each input position (ic, ih, iw) and kernel position (kh, kw),
        // accumulate weight[ic, oc, kh, kw] * input[ic, ih, iw] into
        // output[oc, oh, ow] where oh = ih*stride + kh - padding, ow = iw*stride + kw - padding.
        for ic in 0..self.in_channels {
            for ih in 0..h_in {
                for iw in 0..w_in {
                    let x = input_vec[ic * h_in * w_in + ih * w_in + iw];
                    if x == 0.0 {
                        continue;
                    }
                    for kh in 0..self.kernel_h {
                        let oh_raw = ih * self.stride + kh;
                        if oh_raw < self.padding || oh_raw - self.padding >= h_out {
                            continue;
                        }
                        let oh = oh_raw - self.padding;
                        for kw in 0..self.kernel_w {
                            let ow_raw = iw * self.stride + kw;
                            if ow_raw < self.padding || ow_raw - self.padding >= w_out {
                                continue;
                            }
                            let ow = ow_raw - self.padding;
                            let w_base = (ic * self.out_channels) * self.kernel_h * self.kernel_w
                                + kh * self.kernel_w
                                + kw;
                            for oc in 0..self.out_channels {
                                let w_idx = w_base + oc * self.kernel_h * self.kernel_w;
                                output[oc * h_out * w_out + oh * w_out + ow] +=
                                    x * weight_vec[w_idx];
                            }
                        }
                    }
                }
            }
        }

        // Add bias
        for (oc, &b) in bias_vec.iter().enumerate() {
            let offset = oc * h_out * w_out;
            for i in 0..h_out * w_out {
                output[offset + i] += b;
            }
        }

        Tensor::from_vec(output, Shape::new(&[self.out_channels, h_out, w_out]))
    }
}

impl<B: MathBackend> Module<B> for ConvTranspose2d<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.weight, &self.bias]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scry_llm::backend::cpu::CpuBackend;

    #[test]
    fn conv_transpose2d_output_shape_stride2_kernel2() {
        // SAM's exact config: kernel=2, stride=2, padding=0
        // Input: [1, 4, 4] -> Output: [1, 8, 8]
        let conv = ConvTranspose2d::<CpuBackend>::square(1, 1, 2, 2, 0);
        let input = Tensor::from_vec(vec![0.0f32; 1 * 4 * 4], Shape::new(&[1, 4, 4]));
        let output = conv.forward(&input);
        // H_out = (4-1)*2 - 0 + 2 = 8
        assert_eq!(output.shape.dims(), &[1, 8, 8]);
    }

    #[test]
    fn conv_transpose2d_output_shape_multi_channel() {
        // SAM decoder: 256->64, kernel=2, stride=2
        let conv = ConvTranspose2d::<CpuBackend>::square(256, 64, 2, 2, 0);
        let input = Tensor::from_vec(vec![0.0f32; 256 * 64 * 64], Shape::new(&[256, 64, 64]));
        let output = conv.forward(&input);
        assert_eq!(output.shape.dims(), &[64, 128, 128]);
    }

    #[test]
    fn conv_transpose2d_stride2_kernel2_known_values() {
        // 1 in_ch, 1 out_ch, kernel=2, stride=2
        // Weight: [[1, 2], [3, 4]]
        // Input 2x2: [[1, 0], [0, 0]]
        // Each input pixel maps to a non-overlapping 2x2 block in output.
        // Input (0,0)=1 → output block (0:2, 0:2) = [[1,2],[3,4]] * 1
        let mut conv = ConvTranspose2d::<CpuBackend>::square(1, 1, 2, 2, 0);
        conv.weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(&[1, 1, 2, 2]));

        let input = Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0], Shape::new(&[1, 2, 2]));
        let output = conv.forward(&input);
        assert_eq!(output.shape.dims(), &[1, 4, 4]);

        let data = output.to_vec();
        // Top-left 2x2 block should be [1, 2, 3, 4]
        assert!((data[0] - 1.0).abs() < 1e-6, "got {}", data[0]);
        assert!((data[1] - 2.0).abs() < 1e-6, "got {}", data[1]);
        assert!((data[4] - 3.0).abs() < 1e-6, "got {}", data[4]);
        assert!((data[5] - 4.0).abs() < 1e-6, "got {}", data[5]);
        // All other positions should be zero (input was zero there)
        assert!((data[2]).abs() < 1e-6);
        assert!((data[3]).abs() < 1e-6);
    }

    #[test]
    fn conv_transpose2d_multi_input_accumulation() {
        // Two adjacent input pixels both contribute to output
        let mut conv = ConvTranspose2d::<CpuBackend>::square(1, 1, 2, 2, 0);
        conv.weight = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], Shape::new(&[1, 1, 2, 2]));

        let input = Tensor::from_vec(vec![2.0, 3.0, 0.0, 0.0], Shape::new(&[1, 2, 2]));
        let output = conv.forward(&input);
        let data = output.to_vec();

        // Input (0,0)=2 fills output[0:2, 0:2] with 2
        // Input (0,1)=3 fills output[0:2, 2:4] with 3
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
        assert!((data[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn conv_transpose2d_with_bias() {
        let mut conv = ConvTranspose2d::<CpuBackend>::square(1, 1, 2, 2, 0);
        conv.weight = Tensor::from_vec(vec![1.0; 4], Shape::new(&[1, 1, 2, 2]));
        conv.bias = Tensor::from_vec(vec![10.0], Shape::new(&[1]));

        let input = Tensor::from_vec(vec![1.0], Shape::new(&[1, 1, 1]));
        let output = conv.forward(&input);
        let data = output.to_vec();

        // 1*1 + 10 = 11 for all 4 output positions
        assert_eq!(output.shape.dims(), &[1, 2, 2]);
        for &v in &data {
            assert!((v - 11.0).abs() < 1e-6);
        }
    }

    #[test]
    fn conv_transpose2d_multi_channel() {
        // 2 in_channels, 1 out_channel
        let mut conv = ConvTranspose2d::<CpuBackend>::new(2, 1, 1, 1, 1, 0, 0);
        // weight: [2, 1, 1, 1] — each input channel contributes to single output channel
        conv.weight = Tensor::from_vec(vec![1.0, 2.0], Shape::new(&[2, 1, 1, 1]));

        let input = Tensor::from_vec(vec![3.0, 4.0], Shape::new(&[2, 1, 1]));
        let output = conv.forward(&input);
        let data = output.to_vec();

        // 3*1 + 4*2 = 11
        assert_eq!(output.shape.dims(), &[1, 1, 1]);
        assert!((data[0] - 11.0).abs() < 1e-6);
    }

    #[test]
    fn conv_transpose2d_parameters() {
        let conv = ConvTranspose2d::<CpuBackend>::square(256, 64, 2, 2, 0);
        let params = conv.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].numel(), 256 * 64 * 2 * 2); // weight
        assert_eq!(params[1].numel(), 64); // bias
    }

    #[test]
    fn conv_transpose2d_stride1_kernel3_padding1() {
        // stride=1, kernel=3, pad=1 → same spatial size (like regular conv)
        let conv = ConvTranspose2d::<CpuBackend>::square(1, 1, 3, 1, 1);
        let input = Tensor::from_vec(vec![0.0; 4 * 4], Shape::new(&[1, 4, 4]));
        let output = conv.forward(&input);
        // H_out = (4-1)*1 - 2 + 3 = 4
        assert_eq!(output.shape.dims(), &[1, 4, 4]);
    }
}
