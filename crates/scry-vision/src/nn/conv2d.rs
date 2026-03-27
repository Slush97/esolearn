// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2D convolution layer (inference-only).
//!
//! Uses im2col + matmul, following the same pattern as scry-stt's `Conv1d`.
//!
//! Input shape: `[C_in, H, W]`
//! Output shape: `[C_out, H_out, W_out]`

use std::cell::RefCell;

use scry_llm::backend::MathBackend;
use scry_llm::nn::Module;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

/// 2D convolution with im2col + matmul.
///
/// Weight shape: `[out_channels, in_channels, kernel_h, kernel_w]`
/// Bias shape: `[out_channels]`
///
/// Stride and padding are symmetric (same value for height and width).
pub struct Conv2d<B: MathBackend> {
    pub weight: Tensor<B>,
    pub bias: Tensor<B>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride: usize,
    pub padding: usize,
    /// Reusable im2col workspace to avoid per-forward allocation.
    pub workspace: RefCell<Vec<f32>>,
}

impl<B: MathBackend> Conv2d<B> {
    /// Create a zero-initialized Conv2d (for testing; real usage loads from checkpoint).
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let w_len = out_channels * in_channels * kernel_h * kernel_w;
        Self {
            weight: Tensor::from_vec(
                vec![0.0f32; w_len],
                Shape::new(&[out_channels, in_channels, kernel_h, kernel_w]),
            ),
            bias: Tensor::from_vec(vec![0.0f32; out_channels], Shape::new(&[out_channels])),
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
            workspace: RefCell::new(Vec::new()),
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
        Self::new(in_channels, out_channels, kernel_size, kernel_size, stride, padding)
    }

    /// Forward pass: `[C_in, H, W]` → `[C_out, H_out, W_out]`.
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let dims = input.shape.dims();
        let h_in = dims[1];
        let w_in = dims[2];

        let h_out = (h_in + 2 * self.padding - self.kernel_h) / self.stride + 1;
        let w_out = (w_in + 2 * self.padding - self.kernel_w) / self.stride + 1;
        let spatial_out = h_out * w_out;

        let input_vec = input.to_vec();

        // im2col: [C_in * kH * kW, H_out * W_out]
        let col_rows = self.in_channels * self.kernel_h * self.kernel_w;
        let needed = col_rows * spatial_out;
        let mut col = self.workspace.borrow_mut();
        if col.len() < needed {
            col.resize(needed, 0.0);
        }

        for oh in 0..h_out {
            for ow in 0..w_out {
                let out_col = oh * w_out + ow;
                for c in 0..self.in_channels {
                    for kh in 0..self.kernel_h {
                        for kw in 0..self.kernel_w {
                            let ih = oh * self.stride + kh;
                            let iw = ow * self.stride + kw;
                            let val = if ih >= self.padding
                                && ih < self.padding + h_in
                                && iw >= self.padding
                                && iw < self.padding + w_in
                            {
                                let h_idx = ih - self.padding;
                                let w_idx = iw - self.padding;
                                input_vec[c * h_in * w_in + h_idx * w_in + w_idx]
                            } else {
                                0.0 // zero padding
                            };
                            let row = c * self.kernel_h * self.kernel_w + kh * self.kernel_w + kw;
                            col[row * spatial_out + out_col] = val;
                        }
                    }
                }
            }
        }

        let col_storage =
            B::from_vec(col[..needed].to_vec(), &Shape::new(&[col_rows, spatial_out]));

        // Weight [out_ch, in_ch, kH, kW] has same flat layout as [out_ch, in_ch*kH*kW]
        let out_data = B::matmul(
            &self.weight.data,
            &col_storage,
            self.out_channels,
            col_rows,
            spatial_out,
            false,
            false,
        );

        // Bias add: bias[out_ch] broadcast across spatial dims
        let bias_col = B::from_vec(self.bias.to_vec(), &Shape::new(&[self.out_channels, 1]));
        let out_shape = Shape::new(&[self.out_channels, spatial_out]);
        let result = B::add(
            &out_data,
            &bias_col,
            &out_shape,
            &Shape::new(&[self.out_channels, 1]),
            &out_shape,
        );

        // Reshape to [C_out, H_out, W_out]
        Tensor::<B>::new(result, Shape::new(&[self.out_channels, h_out, w_out]))
    }
}

impl<B: MathBackend> Module<B> for Conv2d<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.weight, &self.bias]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scry_llm::backend::cpu::CpuBackend;

    #[test]
    fn conv2d_output_shape() {
        let conv = Conv2d::<CpuBackend>::square(3, 16, 3, 1, 1);
        let input = Tensor::from_vec(vec![0.0f32; 3 * 8 * 8], Shape::new(&[3, 8, 8]));
        let output = conv.forward(&input);
        // padding=1, kernel=3, stride=1 → same spatial size
        assert_eq!(output.shape.dims(), &[16, 8, 8]);
    }

    #[test]
    fn conv2d_output_shape_stride2() {
        let conv = Conv2d::<CpuBackend>::square(16, 32, 3, 2, 1);
        let input = Tensor::from_vec(vec![0.0f32; 16 * 8 * 8], Shape::new(&[16, 8, 8]));
        let output = conv.forward(&input);
        // (8 + 2 - 3) / 2 + 1 = 4
        assert_eq!(output.shape.dims(), &[32, 4, 4]);
    }

    #[test]
    fn conv2d_identity_kernel() {
        // 1×1 conv with identity-like weight should pass through center channel
        let mut conv = Conv2d::<CpuBackend>::square(1, 1, 1, 1, 0);
        conv.weight = Tensor::from_vec(vec![1.0], Shape::new(&[1, 1, 1, 1]));

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(&[1, 2, 2]));
        let output = conv.forward(&input);
        assert_eq!(output.shape.dims(), &[1, 2, 2]);
        let data = output.to_vec();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
        assert!((data[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn conv2d_with_bias() {
        let mut conv = Conv2d::<CpuBackend>::square(1, 1, 1, 1, 0);
        conv.weight = Tensor::from_vec(vec![1.0], Shape::new(&[1, 1, 1, 1]));
        conv.bias = Tensor::from_vec(vec![10.0], Shape::new(&[1]));

        let input = Tensor::from_vec(vec![5.0], Shape::new(&[1, 1, 1]));
        let output = conv.forward(&input);
        let data = output.to_vec();
        assert!((data[0] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn conv2d_3x3_manual() {
        // 1 input channel, 1 output channel, 3×3 kernel, no padding
        // Input: 3×3, all ones
        // Weight: 3×3, all ones → each output pixel = 9.0 (sum of 3×3 window)
        // Output: 1×1
        let mut conv = Conv2d::<CpuBackend>::square(1, 1, 3, 1, 0);
        conv.weight = Tensor::from_vec(vec![1.0; 9], Shape::new(&[1, 1, 3, 3]));

        let input = Tensor::from_vec(vec![1.0; 9], Shape::new(&[1, 3, 3]));
        let output = conv.forward(&input);
        assert_eq!(output.shape.dims(), &[1, 1, 1]);
        assert!((output.to_vec()[0] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn conv2d_padding_preserves_size() {
        // 3×3 conv with padding=1 on 5×5 input → 5×5 output
        let conv = Conv2d::<CpuBackend>::square(1, 1, 3, 1, 1);
        let input = Tensor::from_vec(vec![1.0; 25], Shape::new(&[1, 5, 5]));
        let output = conv.forward(&input);
        assert_eq!(output.shape.dims(), &[1, 5, 5]);
    }

    #[test]
    fn conv2d_parameters() {
        let conv = Conv2d::<CpuBackend>::square(3, 16, 3, 1, 1);
        let params = conv.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].numel(), 16 * 3 * 3 * 3); // weight
        assert_eq!(params[1].numel(), 16); // bias
    }
}
