// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pooling layers for 2D inputs.
//!
//! - [`MaxPool2d`] — max pooling with fixed kernel/stride/padding
//! - [`AdaptiveAvgPool2d`] — adaptive average pooling to a fixed output size

use scry_llm::backend::MathBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

/// Adaptive average pooling that produces a fixed output size.
///
/// Automatically computes pooling regions to map any `[C, H, W]` input
/// to `[C, output_h, output_w]`. No learnable parameters.
pub struct AdaptiveAvgPool2d {
    pub output_h: usize,
    pub output_w: usize,
}

impl AdaptiveAvgPool2d {
    pub fn new(output_h: usize, output_w: usize) -> Self {
        Self { output_h, output_w }
    }

    /// Global average pooling: each channel → single scalar.
    pub fn global() -> Self {
        Self::new(1, 1)
    }

    /// Forward pass: `[C, H, W]` → `[C, output_h, output_w]`.
    pub fn forward<B: MathBackend>(&self, input: &Tensor<B>) -> Tensor<B> {
        let dims = input.shape.dims();
        let c = dims[0];
        let h_in = dims[1];
        let w_in = dims[2];

        let input_vec = input.to_vec();
        let mut output = vec![0.0f32; c * self.output_h * self.output_w];

        for ch in 0..c {
            let ch_offset = ch * h_in * w_in;
            let out_ch_offset = ch * self.output_h * self.output_w;

            for oh in 0..self.output_h {
                let h_start = oh * h_in / self.output_h;
                let h_end = (oh + 1) * h_in / self.output_h;
                for ow in 0..self.output_w {
                    let w_start = ow * w_in / self.output_w;
                    let w_end = (ow + 1) * w_in / self.output_w;

                    let mut sum = 0.0f32;
                    let count = (h_end - h_start) * (w_end - w_start);
                    for h in h_start..h_end {
                        for w in w_start..w_end {
                            sum += input_vec[ch_offset + h * w_in + w];
                        }
                    }
                    output[out_ch_offset + oh * self.output_w + ow] = sum / count as f32;
                }
            }
        }

        Tensor::from_vec(output, Shape::new(&[c, self.output_h, self.output_w]))
    }
}

/// Max pooling with fixed kernel size, stride, and padding.
///
/// Input shape: `[C, H, W]`
/// Output shape: `[C, H_out, W_out]`
///
/// `H_out = (H + 2*padding - kernel_size) / stride + 1`
pub struct MaxPool2d {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl MaxPool2d {
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    /// Forward pass: `[C, H, W]` → `[C, H_out, W_out]`.
    pub fn forward<B: MathBackend>(&self, input: &Tensor<B>) -> Tensor<B> {
        let dims = input.shape.dims();
        let c = dims[0];
        let h_in = dims[1];
        let w_in = dims[2];

        let h_out = (h_in + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let w_out = (w_in + 2 * self.padding - self.kernel_size) / self.stride + 1;

        let input_vec = input.to_vec();
        let mut output = vec![f32::NEG_INFINITY; c * h_out * w_out];

        for ch in 0..c {
            let ch_offset = ch * h_in * w_in;
            let out_ch_offset = ch * h_out * w_out;
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut max_val = f32::NEG_INFINITY;
                    for kh in 0..self.kernel_size {
                        for kw in 0..self.kernel_size {
                            let ih = oh * self.stride + kh;
                            let iw = ow * self.stride + kw;
                            if ih >= self.padding
                                && ih < self.padding + h_in
                                && iw >= self.padding
                                && iw < self.padding + w_in
                            {
                                let h_idx = ih - self.padding;
                                let w_idx = iw - self.padding;
                                let val = input_vec[ch_offset + h_idx * w_in + w_idx];
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }
                    }
                    // If all positions were padding (shouldn't happen in practice),
                    // use 0.0 instead of -inf
                    if max_val == f32::NEG_INFINITY {
                        max_val = 0.0;
                    }
                    output[out_ch_offset + oh * w_out + ow] = max_val;
                }
            }
        }

        Tensor::from_vec(output, Shape::new(&[c, h_out, w_out]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scry_llm::backend::cpu::CpuBackend;

    #[test]
    fn global_avg_pool() {
        let pool = AdaptiveAvgPool2d::global();
        // 1 channel, 2×2, values [1, 2, 3, 4] → mean = 2.5
        let input =
            Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(&[1, 2, 2]));
        let output = pool.forward(&input);
        assert_eq!(output.shape.dims(), &[1, 1, 1]);
        assert!((output.to_vec()[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn global_avg_pool_multi_channel() {
        let pool = AdaptiveAvgPool2d::global();
        // 2 channels, each 2×2
        #[rustfmt::skip]
        let input = Tensor::<CpuBackend>::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0,   // channel 0, mean=2.5
                10.0, 20.0, 30.0, 40.0, // channel 1, mean=25.0
            ],
            Shape::new(&[2, 2, 2]),
        );
        let output = pool.forward(&input);
        assert_eq!(output.shape.dims(), &[2, 1, 1]);
        let data = output.to_vec();
        assert!((data[0] - 2.5).abs() < 1e-6);
        assert!((data[1] - 25.0).abs() < 1e-6);
    }

    #[test]
    fn adaptive_pool_downsample() {
        let pool = AdaptiveAvgPool2d::new(2, 2);
        // 1 channel, 4×4 → 2×2
        #[rustfmt::skip]
        let input = Tensor::<CpuBackend>::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            Shape::new(&[1, 4, 4]),
        );
        let output = pool.forward(&input);
        assert_eq!(output.shape.dims(), &[1, 2, 2]);
        let data = output.to_vec();
        // Top-left quadrant: mean(1,2,5,6) = 3.5
        assert!((data[0] - 3.5).abs() < 1e-6);
        // Top-right quadrant: mean(3,4,7,8) = 5.5
        assert!((data[1] - 5.5).abs() < 1e-6);
        // Bottom-left quadrant: mean(9,10,13,14) = 11.5
        assert!((data[2] - 11.5).abs() < 1e-6);
        // Bottom-right quadrant: mean(11,12,15,16) = 13.5
        assert!((data[3] - 13.5).abs() < 1e-6);
    }

    #[test]
    fn adaptive_pool_identity() {
        // Same input and output size → identity
        let pool = AdaptiveAvgPool2d::new(3, 3);
        let input = Tensor::<CpuBackend>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Shape::new(&[1, 3, 3]),
        );
        let output = pool.forward(&input);
        assert_eq!(output.shape.dims(), &[1, 3, 3]);
        let data = output.to_vec();
        for (i, &v) in data.iter().enumerate() {
            assert!(
                (v - (i as f32 + 1.0)).abs() < 1e-6,
                "mismatch at {i}: {v}"
            );
        }
    }

    #[test]
    fn maxpool_basic() {
        let pool = MaxPool2d::new(2, 2, 0);
        #[rustfmt::skip]
        let input = Tensor::<CpuBackend>::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            Shape::new(&[1, 4, 4]),
        );
        let output = pool.forward(&input);
        assert_eq!(output.shape.dims(), &[1, 2, 2]);
        let data = output.to_vec();
        assert!((data[0] - 6.0).abs() < 1e-6); // max(1,2,5,6)
        assert!((data[1] - 8.0).abs() < 1e-6); // max(3,4,7,8)
        assert!((data[2] - 14.0).abs() < 1e-6); // max(9,10,13,14)
        assert!((data[3] - 16.0).abs() < 1e-6); // max(11,12,15,16)
    }

    #[test]
    fn maxpool_with_padding() {
        // 3×3 kernel, stride 2, padding 1 on 4×4 → 2×2
        let pool = MaxPool2d::new(3, 2, 1);
        let input = Tensor::<CpuBackend>::from_vec(
            vec![1.0; 16],
            Shape::new(&[1, 4, 4]),
        );
        let output = pool.forward(&input);
        assert_eq!(output.shape.dims(), &[1, 2, 2]);
    }

    #[test]
    fn maxpool_resnet_stem() {
        // ResNet uses MaxPool2d(3, stride=2, padding=1) after the initial conv
        // On 112×112 input → 56×56 output
        let pool = MaxPool2d::new(3, 2, 1);
        let input = Tensor::<CpuBackend>::from_vec(
            vec![0.0; 64 * 112 * 112],
            Shape::new(&[64, 112, 112]),
        );
        let output = pool.forward(&input);
        assert_eq!(output.shape.dims(), &[64, 56, 56]);
    }
}
