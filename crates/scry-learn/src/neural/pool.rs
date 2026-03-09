// SPDX-License-Identifier: MIT OR Apache-2.0
//! Max pooling layer.
//!
//! Downsamples spatial dimensions by taking the maximum value in each
//! pooling window. No trainable parameters.

use crate::neural::traits::{BackwardOutput, Layer};

/// 2D max pooling layer.
///
/// Input shape: `[batch, C, H, W]` (channels-first, flattened).
/// Output shape: `[batch, C, H_out, W_out]` where
/// `H_out = (H - pool_size) / stride + 1`.
pub struct MaxPool2D {
    /// Pooling window size.
    pub(crate) pool_size: usize,
    /// Stride (defaults to pool_size).
    pub(crate) stride: usize,
    /// Number of channels (set on first forward).
    pub(crate) channels: usize,
    /// Input spatial height.
    pub(crate) h_in: usize,
    /// Input spatial width.
    pub(crate) w_in: usize,
    /// Output spatial height.
    pub(crate) h_out: usize,
    /// Output spatial width.
    pub(crate) w_out: usize,
    /// Index of the max element in each pooling window (for backward).
    pub(crate) cache_max_indices: Vec<usize>,
    pub(crate) cache_batch: usize,
}

impl MaxPool2D {
    /// Create a new MaxPool2D layer.
    ///
    /// `pool_size` is the spatial extent of the pooling window.
    /// `stride` defaults to `pool_size` if `None`.
    pub fn new(pool_size: usize, stride: Option<usize>) -> Self {
        let stride = stride.unwrap_or(pool_size);
        Self {
            pool_size,
            stride,
            channels: 0,
            h_in: 0,
            w_in: 0,
            h_out: 0,
            w_out: 0,
            cache_max_indices: Vec::new(),
            cache_batch: 0,
        }
    }
}

impl Layer for MaxPool2D {
    fn forward(&mut self, input: &[f64], batch: usize, training: bool) -> Vec<f64> {
        // Infer dimensions from input.
        let per_sample = input.len() / batch;
        // We need to know the number of channels. Assume it was set via the
        // previous layer's output. If channels is 0, try to infer: assume
        // square spatial dims.
        if self.channels == 0 {
            // Can't infer channels without additional info.
            // Default: assume channels = 1 if spatial is square.
            // Otherwise the caller must set channels beforehand.
            let side_sq = (per_sample as f64).sqrt() as usize;
            if side_sq * side_sq == per_sample {
                self.channels = 1;
                self.h_in = side_sq;
                self.w_in = side_sq;
            } else {
                // Try to find c such that per_sample / c is a perfect square.
                for c in (1..=per_sample).rev() {
                    if per_sample % c == 0 {
                        let spatial = per_sample / c;
                        let side = (spatial as f64).sqrt() as usize;
                        if side * side == spatial {
                            self.channels = c;
                            self.h_in = side;
                            self.w_in = side;
                            break;
                        }
                    }
                }
            }
        } else {
            let spatial = per_sample / self.channels;
            let side = (spatial as f64).sqrt() as usize;
            debug_assert_eq!(side * side, spatial);
            self.h_in = side;
            self.w_in = side;
        }

        self.h_out = (self.h_in - self.pool_size) / self.stride + 1;
        self.w_out = (self.w_in - self.pool_size) / self.stride + 1;

        let out_per_sample = self.channels * self.h_out * self.w_out;
        let mut output = Vec::with_capacity(batch * out_per_sample);
        let mut max_indices = if training {
            Vec::with_capacity(batch * out_per_sample)
        } else {
            Vec::new()
        };

        for b in 0..batch {
            let sample = &input[b * per_sample..];

            for c in 0..self.channels {
                let ch_offset = c * self.h_in * self.w_in;
                for oy in 0..self.h_out {
                    for ox in 0..self.w_out {
                        let mut max_val = f64::NEG_INFINITY;
                        let mut max_idx = 0;

                        for py in 0..self.pool_size {
                            for px in 0..self.pool_size {
                                let iy = oy * self.stride + py;
                                let ix = ox * self.stride + px;
                                let idx = ch_offset + iy * self.w_in + ix;
                                let val = sample[idx];
                                if val > max_val {
                                    max_val = val;
                                    max_idx = idx;
                                }
                            }
                        }

                        output.push(max_val);
                        if training {
                            max_indices.push(b * per_sample + max_idx);
                        }
                    }
                }
            }
        }

        if training {
            self.cache_max_indices = max_indices;
            self.cache_batch = batch;
        }

        output
    }

    fn backward(&self, grad_output: &[f64]) -> BackwardOutput {
        let batch = self.cache_batch;
        let in_per_sample = self.channels * self.h_in * self.w_in;
        let mut grad_input = vec![0.0; batch * in_per_sample];

        for (i, &max_idx) in self.cache_max_indices.iter().enumerate() {
            grad_input[max_idx] += grad_output[i];
        }

        (grad_input, vec![]) // no parameters
    }

    fn n_param_groups(&self) -> usize {
        0
    }

    fn params_mut(&mut self) -> Vec<(&mut Vec<f64>, &mut Vec<f64>)> {
        vec![]
    }

    fn save_params(&self) -> Vec<(Vec<f64>, Vec<f64>)> {
        vec![]
    }

    fn restore_params(&mut self, _saved: &[(Vec<f64>, Vec<f64>)]) {}

    fn in_size(&self) -> usize {
        self.channels * self.h_in * self.w_in
    }

    fn out_size(&self) -> usize {
        self.channels * self.h_out * self.w_out
    }

    fn name(&self) -> &'static str {
        "MaxPool2D"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::float_cmp)]
    #[test]
    fn maxpool_basic() {
        let mut pool = MaxPool2D::new(2, None);
        pool.channels = 1;

        // 1 channel, 4×4 input:
        // 1  2  3  4
        // 5  6  7  8
        // 9  10 11 12
        // 13 14 15 16
        let input: Vec<f64> = (1..=16).map(|x| x as f64).collect();
        let output = pool.forward(&input, 1, false);

        // 2×2 pool, stride 2 → 2×2 output
        assert_eq!(output.len(), 4);
        assert_eq!(output[0], 6.0); // max(1,2,5,6)
        assert_eq!(output[1], 8.0); // max(3,4,7,8)
        assert_eq!(output[2], 14.0); // max(9,10,13,14)
        assert_eq!(output[3], 16.0); // max(11,12,15,16)
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn maxpool_backward() {
        let mut pool = MaxPool2D::new(2, None);
        pool.channels = 1;

        let input: Vec<f64> = (1..=16).map(|x| x as f64).collect();
        pool.forward(&input, 1, true);

        let grad_out = vec![1.0, 2.0, 3.0, 4.0];
        let (grad_input, param_grads) = pool.backward(&grad_out);

        assert_eq!(grad_input.len(), 16);
        assert!(param_grads.is_empty());

        // Gradient should be routed to the max positions.
        assert_eq!(grad_input[5], 1.0); // position of 6 (max of patch 0)
        assert_eq!(grad_input[7], 2.0); // position of 8 (max of patch 1)
        assert_eq!(grad_input[13], 3.0); // position of 14 (max of patch 2)
        assert_eq!(grad_input[15], 4.0); // position of 16 (max of patch 3)

        // Non-max positions should be zero.
        assert_eq!(grad_input[0], 0.0);
        assert_eq!(grad_input[4], 0.0);
    }

    #[test]
    fn maxpool_multi_channel() {
        let mut pool = MaxPool2D::new(2, None);
        pool.channels = 2;

        // 2 channels, 4×4 each → total 32 elements.
        let mut input = Vec::with_capacity(32);
        for c in 0..2 {
            for i in 0..16 {
                input.push((c * 16 + i + 1) as f64);
            }
        }

        let output = pool.forward(&input, 1, false);

        // Each channel produces 2×2 output → 2 * 4 = 8
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn maxpool_batched() {
        let mut pool = MaxPool2D::new(2, None);
        pool.channels = 1;

        let input: Vec<f64> = (0..32).map(|x| x as f64).collect(); // 2 samples, 4×4
        let output = pool.forward(&input, 2, false);

        assert_eq!(output.len(), 8); // 2 * 1 * 2 * 2
    }
}
