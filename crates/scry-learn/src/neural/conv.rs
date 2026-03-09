// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2D convolutional layer using im2col + GEMM.
//!
//! Input shape: `[batch, C_in, H, W]` (channels-first, flattened).
//! Output shape: `[batch, C_out, H_out, W_out]` (flattened).
//!
//! Supports configurable kernel size, stride, and padding.

use crate::neural::activation::Activation;
use crate::neural::traits::{BackwardOutput, Layer};

/// 2D convolutional layer.
///
/// Implements convolution via the im2col transformation: each input patch
/// is unrolled into a column, then the convolution becomes a matrix multiply.
///
/// `filters` is `[C_out, C_in * kH * kW]` row-major.
/// `biases` is `[C_out]`.
pub struct Conv2D {
    /// Number of input channels.
    pub(crate) c_in: usize,
    /// Number of output channels (filters).
    pub(crate) c_out: usize,
    /// Kernel height.
    pub(crate) kh: usize,
    /// Kernel width.
    pub(crate) kw: usize,
    /// Stride.
    pub(crate) stride: usize,
    /// Zero-padding on each side.
    pub(crate) padding: usize,
    /// Input spatial height (set on first forward).
    pub(crate) h_in: usize,
    /// Input spatial width (set on first forward).
    pub(crate) w_in: usize,
    /// Activation function.
    pub(crate) activation: Activation,
    /// Filter weights: `[C_out, C_in * kH * kW]`.
    pub(crate) filters: Vec<f64>,
    /// Bias: `[C_out]`.
    pub(crate) biases: Vec<f64>,

    // Caches for backward pass.
    pub(crate) cache_col: Vec<f64>, // im2col output
    pub(crate) cache_z: Vec<f64>,   // pre-activation
    pub(crate) cache_a: Vec<f64>,   // post-activation
    pub(crate) cache_batch: usize,
    pub(crate) h_out: usize,
    pub(crate) w_out: usize,
}

impl Conv2D {
    /// Create a new Conv2D layer.
    ///
    /// # Arguments
    /// - `c_in`: number of input channels
    /// - `c_out`: number of output channels (filters)
    /// - `kernel_size`: spatial size of the square kernel
    /// - `stride`: convolution stride (default: 1)
    /// - `padding`: zero-padding on each side (default: 0)
    /// - `activation`: activation function
    /// - `seed`: random seed for weight initialization
    pub fn new(
        c_in: usize,
        c_out: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        activation: Activation,
        seed: u64,
    ) -> Self {
        let mut rng = crate::rng::FastRng::new(seed);
        let fan_in = c_in * kernel_size * kernel_size;
        let scale = if activation.uses_he_init() {
            (2.0 / fan_in as f64).sqrt()
        } else {
            let fan_out = c_out * kernel_size * kernel_size;
            (2.0 / (fan_in + fan_out) as f64).sqrt()
        };

        let n_weights = c_out * fan_in;
        let mut filters = Vec::with_capacity(n_weights);
        for _ in 0..n_weights {
            filters.push(rng.normal() * scale);
        }

        Self {
            c_in,
            c_out,
            kh: kernel_size,
            kw: kernel_size,
            stride,
            padding,
            h_in: 0,
            w_in: 0,
            activation,
            filters,
            biases: vec![0.0; c_out],
            cache_col: Vec::new(),
            cache_z: Vec::new(),
            cache_a: Vec::new(),
            cache_batch: 0,
            h_out: 0,
            w_out: 0,
        }
    }

    /// Compute output spatial dimensions.
    fn output_dims(&self, h: usize, w: usize) -> (usize, usize) {
        let h_out = (h + 2 * self.padding - self.kh) / self.stride + 1;
        let w_out = (w + 2 * self.padding - self.kw) / self.stride + 1;
        (h_out, w_out)
    }

    /// Total input elements per sample: `C_in * H * W`.
    fn input_size(&self) -> usize {
        self.c_in * self.h_in * self.w_in
    }

    /// Total output elements per sample: `C_out * H_out * W_out`.
    fn output_size(&self) -> usize {
        self.c_out * self.h_out * self.w_out
    }
}

/// Convolution geometry parameters, shared by [`im2col`] and [`col2im`].
struct ConvParams {
    c_in: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
}

/// im2col: unroll input patches into columns for GEMM-based convolution.
///
/// Input shape: `[C_in, H, W]` (single sample, flattened).
/// Output shape: `[C_in * kH * kW, H_out * W_out]` (flattened, col-major patches).
#[allow(clippy::cast_possible_wrap)]
fn im2col(input: &[f64], p: &ConvParams) -> (Vec<f64>, usize, usize) {
    let h_out = (p.h + 2 * p.padding - p.kh) / p.stride + 1;
    let w_out = (p.w + 2 * p.padding - p.kw) / p.stride + 1;
    let col_h = p.c_in * p.kh * p.kw;
    let col_w = h_out * w_out;

    let mut col = vec![0.0; col_h * col_w];

    for c in 0..p.c_in {
        for ky in 0..p.kh {
            for kx in 0..p.kw {
                let col_row = c * p.kh * p.kw + ky * p.kw + kx;
                for oy in 0..h_out {
                    for ox in 0..w_out {
                        let iy = oy * p.stride + ky;
                        let ix = ox * p.stride + kx;
                        let iy_orig = iy as isize - p.padding as isize;
                        let ix_orig = ix as isize - p.padding as isize;

                        let val = if iy_orig >= 0
                            && iy_orig < p.h as isize
                            && ix_orig >= 0
                            && ix_orig < p.w as isize
                        {
                            input[c * p.h * p.w + iy_orig as usize * p.w + ix_orig as usize]
                        } else {
                            0.0 // zero-padding
                        };

                        col[col_row * col_w + oy * w_out + ox] = val;
                    }
                }
            }
        }
    }

    (col, col_h, col_w)
}

/// col2im: accumulate gradients back from column format to input format.
///
/// Inverse of im2col (with accumulation for overlapping patches).
/// `h_out` and `w_out` are the output spatial dims of the convolution.
#[allow(clippy::cast_possible_wrap)]
fn col2im(col: &[f64], p: &ConvParams, h_out: usize, w_out: usize) -> Vec<f64> {
    let col_w = h_out * w_out;
    let mut input_grad = vec![0.0; p.c_in * p.h * p.w];

    for c in 0..p.c_in {
        for ky in 0..p.kh {
            for kx in 0..p.kw {
                let col_row = c * p.kh * p.kw + ky * p.kw + kx;
                for oy in 0..h_out {
                    for ox in 0..w_out {
                        let iy = oy * p.stride + ky;
                        let ix = ox * p.stride + kx;
                        let iy_orig = iy as isize - p.padding as isize;
                        let ix_orig = ix as isize - p.padding as isize;

                        if iy_orig >= 0
                            && iy_orig < p.h as isize
                            && ix_orig >= 0
                            && ix_orig < p.w as isize
                        {
                            input_grad
                                [c * p.h * p.w + iy_orig as usize * p.w + ix_orig as usize] +=
                                col[col_row * col_w + oy * w_out + ox];
                        }
                    }
                }
            }
        }
    }

    input_grad
}

/// Simple GEMM: C = A × B.
///
/// A: `[m, k]`, B: `[k, n]` → C: `[m, n]`, all row-major.
fn gemm(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// GEMM with A transposed: C = A^T × B.
///
/// A: `[k, m]` (transposed to `[m, k]`), B: `[k, n]` → C: `[m, n]`.
fn gemm_at(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[p * m + i] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// GEMM with B transposed: C = A × B^T.
///
/// A: `[m, k]`, B: `[n, k]` (transposed to `[k, n]`) → C: `[m, n]`.
fn gemm_bt(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

impl Conv2D {
    /// Build [`ConvParams`] for the current layer geometry.
    fn conv_params(&self) -> ConvParams {
        ConvParams {
            c_in: self.c_in,
            h: self.h_in,
            w: self.w_in,
            kh: self.kh,
            kw: self.kw,
            stride: self.stride,
            padding: self.padding,
        }
    }
}

impl Layer for Conv2D {
    fn forward(&mut self, input: &[f64], batch: usize, training: bool) -> Vec<f64> {
        // Infer spatial dims from input size.
        let per_sample = input.len() / batch;
        let spatial = per_sample / self.c_in;
        let side = (spatial as f64).sqrt() as usize;
        debug_assert_eq!(side * side, spatial, "Conv2D: input must be square spatial");
        self.h_in = side;
        self.w_in = side;

        let (h_out, w_out) = self.output_dims(self.h_in, self.w_in);
        self.h_out = h_out;
        self.w_out = w_out;

        let cp = self.conv_params();
        let col_h = self.c_in * self.kh * self.kw; // rows of im2col output
        let col_w = h_out * w_out; // cols of im2col output
        let out_per_sample = self.c_out * h_out * w_out;

        let mut all_col = Vec::with_capacity(batch * col_h * col_w);
        let mut output = Vec::with_capacity(batch * out_per_sample);

        for b in 0..batch {
            let sample = &input[b * per_sample..(b + 1) * per_sample];
            let (col, _, _) = im2col(sample, &cp);

            // out = filters × col: [C_out, col_h] × [col_h, col_w] → [C_out, col_w]
            let mut z = gemm(&self.filters, &col, self.c_out, col_h, col_w);

            // Add bias.
            for f in 0..self.c_out {
                for j in 0..col_w {
                    z[f * col_w + j] += self.biases[f];
                }
            }

            // Apply activation.
            let mut a = z.clone();
            self.activation.forward(&mut a);

            if training {
                all_col.extend_from_slice(&col);
            }

            output.extend_from_slice(&a);
        }

        if training {
            // Also cache pre-activation for backward.
            // Recompute z without activation to get exact pre-activation values.
            let mut all_z = Vec::with_capacity(batch * out_per_sample);
            for b in 0..batch {
                let col_start = b * col_h * col_w;
                let col_slice = &all_col[col_start..col_start + col_h * col_w];
                let mut z = gemm(&self.filters, col_slice, self.c_out, col_h, col_w);
                for f in 0..self.c_out {
                    for j in 0..col_w {
                        z[f * col_w + j] += self.biases[f];
                    }
                }
                all_z.extend_from_slice(&z);
            }

            self.cache_col = all_col;
            self.cache_z = all_z;
            self.cache_a.clone_from(&output);
            self.cache_batch = batch;
        }

        output
    }

    fn backward(&self, grad_output: &[f64]) -> BackwardOutput {
        let batch = self.cache_batch;
        let col_h = self.c_in * self.kh * self.kw;
        let col_w = self.h_out * self.w_out;
        let out_per_sample = self.c_out * col_w;
        let in_per_sample = self.c_in * self.h_in * self.w_in;
        let batch_f = batch as f64;
        let cp = self.conv_params();

        // Apply activation derivative.
        let mut delta = grad_output.to_vec();
        self.activation
            .backward_from_activated(&self.cache_z, &self.cache_a, &mut delta);

        let mut d_filters = vec![0.0; self.c_out * col_h];
        let mut d_biases = vec![0.0; self.c_out];
        let mut grad_input = vec![0.0; batch * in_per_sample];

        for b in 0..batch {
            let delta_b = &delta[b * out_per_sample..(b + 1) * out_per_sample];
            let col_b = &self.cache_col[b * col_h * col_w..(b + 1) * col_h * col_w];

            // dFilters += delta_b × col_b^T: [C_out, col_w] × [col_w, col_h] → [C_out, col_h]
            let df = gemm_bt(delta_b, col_b, self.c_out, col_w, col_h);
            for (i, v) in df.iter().enumerate() {
                d_filters[i] += v / batch_f;
            }

            // dBiases += sum over spatial dims.
            for f in 0..self.c_out {
                let mut sum = 0.0;
                for j in 0..col_w {
                    sum += delta_b[f * col_w + j];
                }
                d_biases[f] += sum / batch_f;
            }

            // dCol = filters^T × delta_b: [col_h, C_out] × [C_out, col_w] → [col_h, col_w]
            let d_col = gemm_at(&self.filters, delta_b, col_h, self.c_out, col_w);

            // col2im to get input gradient.
            let gi = col2im(&d_col, &cp, self.h_out, self.w_out);

            grad_input[b * in_per_sample..(b + 1) * in_per_sample].copy_from_slice(&gi);
        }

        (grad_input, vec![(d_filters, d_biases)])
    }

    fn n_param_groups(&self) -> usize {
        1
    }

    fn params_mut(&mut self) -> Vec<(&mut Vec<f64>, &mut Vec<f64>)> {
        vec![(&mut self.filters, &mut self.biases)]
    }

    fn save_params(&self) -> Vec<(Vec<f64>, Vec<f64>)> {
        vec![(self.filters.clone(), self.biases.clone())]
    }

    fn restore_params(&mut self, saved: &[(Vec<f64>, Vec<f64>)]) {
        if let Some((w, b)) = saved.first() {
            self.filters.clone_from(w);
            self.biases.clone_from(b);
        }
    }

    fn in_size(&self) -> usize {
        self.input_size()
    }

    fn out_size(&self) -> usize {
        self.output_size()
    }

    fn name(&self) -> &'static str {
        "Conv2D"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::float_cmp)]
    #[test]
    fn im2col_basic() {
        // 1 channel, 3×3 input, 2×2 kernel, stride 1, no padding.
        // Input:
        // 1 2 3
        // 4 5 6
        // 7 8 9
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let p = ConvParams {
            c_in: 1,
            h: 3,
            w: 3,
            kh: 2,
            kw: 2,
            stride: 1,
            padding: 0,
        };
        let (col, col_h, col_w) = im2col(&input, &p);

        assert_eq!(col_h, 4); // 1 * 2 * 2
        assert_eq!(col_w, 4); // 2 * 2 output spatial
        assert_eq!(col.len(), 16);

        // First patch (top-left): [1, 2, 4, 5]
        // col layout: [col_h=4, col_w=4], row i starts at i*col_w
        assert_eq!(col[0], 1.0); // row 0, col 0
        assert_eq!(col[4], 2.0); // row 1, col 0
        assert_eq!(col[8], 4.0); // row 2, col 0
        assert_eq!(col[12], 5.0); // row 3, col 0
    }

    #[test]
    fn conv2d_output_shape() {
        let mut conv = Conv2D::new(1, 4, 3, 1, 0, Activation::Relu, 42);

        // Input: batch=2, C_in=1, 5×5 → 50 elements
        let input = vec![1.0; 50];
        let output = conv.forward(&input, 2, false);

        // Output: 5 - 3 + 1 = 3, so [2, 4, 3, 3] = 72
        assert_eq!(output.len(), 72);
    }

    #[test]
    fn conv2d_with_padding() {
        let mut conv = Conv2D::new(1, 2, 3, 1, 1, Activation::Identity, 42);

        // Input: batch=1, C_in=1, 4×4 → 16 elements; output: 4×4 (same padding)
        let input = vec![1.0; 16];
        let output = conv.forward(&input, 1, false);

        // [1, 2, 4, 4] = 32
        assert_eq!(output.len(), 32);
    }

    #[test]
    fn conv2d_backward_shape() {
        let mut conv = Conv2D::new(1, 4, 3, 1, 0, Activation::Relu, 42);

        // batch=2, C_in=1, 5×5 → 50 elements
        let input = vec![0.5; 50];
        let output = conv.forward(&input, 2, true);

        let grad_out = vec![1.0; output.len()];
        let (grad_input, param_grads) = conv.backward(&grad_out);

        assert_eq!(grad_input.len(), 50); // 2 * 1 * 5 * 5
        assert_eq!(param_grads.len(), 1);
        assert_eq!(param_grads[0].0.len(), 36); // 4 * 1 * 3 * 3 filter grads
        assert_eq!(param_grads[0].1.len(), 4); // bias grads
    }

    #[test]
    fn conv2d_numerical_gradient() {
        let mut conv = Conv2D::new(1, 2, 2, 1, 0, Activation::Identity, 42);

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // 1ch, 3×3
        let batch = 1;

        let output = conv.forward(&input, batch, true);
        let grad_out = vec![1.0; output.len()];
        let (_, param_grads) = conv.backward(&grad_out);

        // Numerical gradient check for filter weights.
        let eps = 1e-5;
        for idx in 0..conv.filters.len().min(4) {
            let orig = conv.filters[idx];

            conv.filters[idx] = orig + eps;
            let out_p = conv.forward(&input, batch, false);

            conv.filters[idx] = orig - eps;
            let out_m = conv.forward(&input, batch, false);

            conv.filters[idx] = orig;

            let numerical: f64 = out_p
                .iter()
                .zip(out_m.iter())
                .map(|(p, m)| (p - m) / (2.0 * eps))
                .sum::<f64>()
                / batch as f64;

            let analytic = param_grads[0].0[idx];
            let diff = (analytic - numerical).abs();
            let denom = analytic.abs().max(numerical.abs()).max(1e-8);
            assert!(
                diff / denom < 1e-3,
                "filter weight {idx}: analytic={analytic:.6}, numerical={numerical:.6}",
            );
        }
    }
}
