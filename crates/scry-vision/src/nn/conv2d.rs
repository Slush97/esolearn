// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2D convolution layer (inference-only).
//!
//! Uses im2col + matmul as the default path, with Winograd F(2×2, 3×3) fast
//! convolution auto-dispatched for 3×3 stride-1 padding-1 kernels.
//!
//! Input shape: `[C_in, H, W]`
//! Output shape: `[C_out, H_out, W_out]`

use std::cell::RefCell;

use scry_llm::backend::MathBackend;
use scry_llm::nn::Module;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

/// 2D convolution with im2col + matmul and Winograd F(2×2, 3×3) fast path.
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
    /// Reusable workspace to avoid per-forward allocation.
    pub workspace: RefCell<Vec<f32>>,
    /// Cached Winograd F(2×2, 3×3) transformed filter weights as backend storage.
    /// 16 elements, each a `[C_out, C_in]` matrix in `B::Storage` form.
    /// Lazily computed on first forward pass for eligible convolutions.
    pub winograd_weight: RefCell<Option<Vec<B::Storage>>>,
}

/// Precompute Winograd F(2×2, 3×3) transformed filter weights.
///
/// Transforms each 3×3 kernel to 4×4 via U = G · g · Gᵀ.
/// Output layout: `[16, out_channels, in_channels]` (position-major),
/// so each of the 16 positions is a contiguous `[C_out, C_in]` matrix.
fn precompute_winograd_weights(
    weight: &[f32],
    out_channels: usize,
    in_channels: usize,
) -> Vec<f32> {
    let oc_ic = out_channels * in_channels;
    let mut u = vec![0.0f32; 16 * oc_ic];
    for oc in 0..out_channels {
        for ic in 0..in_channels {
            let base = (oc * in_channels + ic) * 9;
            let g = |r: usize, c: usize| weight[base + r * 3 + c];

            // temp = g · Gᵀ  (3×4)
            // Gᵀ = [[1, 1/2, 1/2, 0], [0, 1/2, -1/2, 0], [0, 1/2, 1/2, 1]]
            let mut t = [0.0f32; 12];
            for i in 0..3 {
                t[i * 4] = g(i, 0);
                t[i * 4 + 1] = (g(i, 0) + g(i, 1) + g(i, 2)) * 0.5;
                t[i * 4 + 2] = (g(i, 0) - g(i, 1) + g(i, 2)) * 0.5;
                t[i * 4 + 3] = g(i, 2);
            }

            // U = G · temp  (4×4)
            // G = [[1,0,0], [1/2,1/2,1/2], [1/2,-1/2,1/2], [0,0,1]]
            let mut u_tile = [0.0f32; 16];
            for j in 0..4 {
                u_tile[j] = t[j]; // row 0 = temp row 0
                u_tile[4 + j] = (t[j] + t[4 + j] + t[8 + j]) * 0.5;
                u_tile[8 + j] = (t[j] - t[4 + j] + t[8 + j]) * 0.5;
                u_tile[12 + j] = t[8 + j]; // row 3 = temp row 2
            }

            // Scatter to position-major layout
            let idx = oc * in_channels + ic;
            for p in 0..16 {
                u[p * oc_ic + idx] = u_tile[p];
            }
        }
    }
    u
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
            winograd_weight: RefCell::new(None),
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

    /// Whether this conv qualifies for Winograd F(2×2, 3×3).
    fn use_winograd(&self) -> bool {
        self.kernel_h == 3 && self.kernel_w == 3 && self.stride == 1 && self.padding == 1
    }

    /// Forward pass: `[C_in, H, W]` → `[C_out, H_out, W_out]`.
    ///
    /// Auto-dispatches to Winograd for 3×3/stride-1/pad-1 kernels.
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        if self.use_winograd() {
            return self.forward_winograd(input);
        }
        self.forward_im2col(input)
    }

    /// Im2col + matmul convolution (general case, also used as test reference).
    pub fn forward_im2col(&self, input: &Tensor<B>) -> Tensor<B> {
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

    /// Winograd F(2×2, 3×3) convolution for 3×3 stride-1 padding-1 kernels.
    ///
    /// Reduces arithmetic cost by ~2.25× through domain transforms:
    ///   1. Transform 3×3 filters to 4×4: U = G · g · Gᵀ  (cached as `B::Storage`)
    ///   2. Extract 4×4 input tiles and transform: V = Bᵀ · d · B
    ///   3. Batched matmul per transform position: M_p = U_p · V_p
    ///   4. Inverse-transform to 2×2 output tiles: Y = Aᵀ · M · A
    fn forward_winograd(&self, input: &Tensor<B>) -> Tensor<B> {
        let dims = input.shape.dims();
        let (c_in, h_in, w_in) = (dims[0], dims[1], dims[2]);
        debug_assert_eq!(c_in, self.in_channels);
        let c_out = self.out_channels;

        // stride=1, pad=1, kernel=3 → output = input size
        let h_out = h_in;
        let w_out = w_in;
        let tiles_h = (h_out + 1) / 2;
        let tiles_w = (w_out + 1) / 2;
        let num_tiles = tiles_h * tiles_w;

        // ── Filter transform (cached as B::Storage, computed once) ──
        {
            let mut cache = self.winograd_weight.borrow_mut();
            if cache.is_none() {
                let u_flat = precompute_winograd_weights(
                    &self.weight.to_vec(),
                    c_out,
                    c_in,
                );
                let oc_ic = c_out * c_in;
                let u_shape = Shape::new(&[c_out, c_in]);
                let storages: Vec<B::Storage> = (0..16)
                    .map(|p| {
                        B::from_vec(u_flat[p * oc_ic..(p + 1) * oc_ic].to_vec(), &u_shape)
                    })
                    .collect();
                *cache = Some(storages);
            }
        }
        let cache = self.winograd_weight.borrow();
        let u_storages = cache.as_ref().unwrap();

        // ── Padded input ──
        let input_vec = input.to_vec();
        let pad_h = tiles_h * 2 + 2;
        let pad_w = tiles_w * 2 + 2;
        let mut padded = vec![0.0f32; c_in * pad_h * pad_w];
        for c in 0..c_in {
            let src_base = c * h_in * w_in;
            let dst_base = c * pad_h * pad_w;
            for h in 0..h_in {
                let src_off = src_base + h * w_in;
                let dst_off = dst_base + (h + 1) * pad_w + 1;
                padded[dst_off..dst_off + w_in]
                    .copy_from_slice(&input_vec[src_off..src_off + w_in]);
            }
        }

        // ── Input tile transform → 16 separate V buffers (zero-copy into BLAS) ──
        let ic_tiles = c_in * num_tiles;
        let mut v_bufs: Vec<Vec<f32>> = (0..16).map(|_| vec![0.0f32; ic_tiles]).collect();

        for ic in 0..c_in {
            let ch_base = ic * pad_h * pad_w;
            for ty in 0..tiles_h {
                let row0 = ty * 2;
                for tx in 0..tiles_w {
                    let col0 = tx * 2;
                    let tile_idx = ty * tiles_w + tx;

                    // Load 4×4 tile from padded input
                    let mut d = [0.0f32; 16];
                    for i in 0..4 {
                        let off = ch_base + (row0 + i) * pad_w + col0;
                        d[i * 4..i * 4 + 4].copy_from_slice(&padded[off..off + 4]);
                    }

                    // temp = d · B
                    // B = [[1,0,0,0],[0,1,-1,1],[-1,1,1,0],[0,0,0,-1]]
                    let mut tmp = [0.0f32; 16];
                    for i in 0..4 {
                        let (d0, d1, d2, d3) =
                            (d[i * 4], d[i * 4 + 1], d[i * 4 + 2], d[i * 4 + 3]);
                        tmp[i * 4] = d0 - d2;
                        tmp[i * 4 + 1] = d1 + d2;
                        tmp[i * 4 + 2] = d2 - d1;
                        tmp[i * 4 + 3] = d1 - d3;
                    }

                    // V = Bᵀ · tmp  →  scatter to v_bufs[pos][ic * num_tiles + tile]
                    let v_base = ic * num_tiles + tile_idx;
                    for j in 0..4 {
                        let (t0, t1, t2, t3) =
                            (tmp[j], tmp[4 + j], tmp[8 + j], tmp[12 + j]);
                        v_bufs[0 * 4 + j][v_base] = t0 - t2;
                        v_bufs[1 * 4 + j][v_base] = t1 + t2;
                        v_bufs[2 * 4 + j][v_base] = t2 - t1;
                        v_bufs[3 * 4 + j][v_base] = t1 - t3;
                    }
                }
            }
        }

        // ── Batched GEMM: M[p] = U[p] · V[p] ──
        // U[p] cached as B::Storage (zero-copy), V[p] moved in (zero-copy on CPU)
        let v_shape = Shape::new(&[c_in, num_tiles]);
        let mut m_bufs: Vec<Vec<f32>> = Vec::with_capacity(16);
        for p in 0..16 {
            let v_vec = std::mem::take(&mut v_bufs[p]);
            let v_s = B::from_vec(v_vec, &v_shape);
            let result = B::matmul(
                &u_storages[p],
                &v_s,
                c_out,
                c_in,
                num_tiles,
                false,
                false,
            );
            m_bufs.push(B::into_vec(result));
        }

        // ── Output inverse transform + bias + scatter ──
        let mut output = vec![0.0f32; c_out * h_out * w_out];
        let bias = self.bias.to_vec();

        for oc in 0..c_out {
            let b = bias[oc];
            let out_base = oc * h_out * w_out;
            for ty in 0..tiles_h {
                for tx in 0..tiles_w {
                    let tile_idx = ty * tiles_w + tx;
                    let m_base = oc * num_tiles + tile_idx;

                    // Gather m_tile[4][4] from m_bufs[p][oc * num_tiles + tile]
                    let mut mt = [0.0f32; 16];
                    for p in 0..16 {
                        mt[p] = m_bufs[p][m_base];
                    }

                    // temp = m · A  (4×2)
                    // A = [[1,0],[1,1],[1,-1],[0,-1]]
                    let mut tmp = [0.0f32; 8];
                    for i in 0..4 {
                        tmp[i * 2] = mt[i * 4] + mt[i * 4 + 1] + mt[i * 4 + 2];
                        tmp[i * 2 + 1] = mt[i * 4 + 1] - mt[i * 4 + 2] - mt[i * 4 + 3];
                    }

                    // Y = Aᵀ · tmp  (2×2) + bias
                    let y00 = tmp[0] + tmp[2] + tmp[4] + b;
                    let y01 = tmp[1] + tmp[3] + tmp[5] + b;
                    let y10 = tmp[2] - tmp[4] - tmp[6] + b;
                    let y11 = tmp[3] - tmp[5] - tmp[7] + b;

                    let oh = ty * 2;
                    let ow = tx * 2;
                    output[out_base + oh * w_out + ow] = y00;
                    if ow + 1 < w_out {
                        output[out_base + oh * w_out + ow + 1] = y01;
                    }
                    if oh + 1 < h_out {
                        output[out_base + (oh + 1) * w_out + ow] = y10;
                        if ow + 1 < w_out {
                            output[out_base + (oh + 1) * w_out + ow + 1] = y11;
                        }
                    }
                }
            }
        }

        Tensor::<B>::from_vec(output, Shape::new(&[c_out, h_out, w_out]))
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

    // ── Winograd tests ──

    /// Helper: create a Conv2d with deterministic pseudo-random weights.
    fn random_conv(in_ch: usize, out_ch: usize) -> Conv2d<CpuBackend> {
        let n = out_ch * in_ch * 9;
        let w: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7 + 0.3).sin()).collect();
        let b: Vec<f32> = (0..out_ch).map(|i| (i as f32 * 0.13).cos() * 0.5).collect();
        let mut conv = Conv2d::<CpuBackend>::square(in_ch, out_ch, 3, 1, 1);
        conv.weight = Tensor::from_vec(w, Shape::new(&[out_ch, in_ch, 3, 3]));
        conv.bias = Tensor::from_vec(b, Shape::new(&[out_ch]));
        conv
    }

    /// Helper: create a deterministic pseudo-random input.
    fn random_input(c: usize, h: usize, w: usize) -> Tensor<CpuBackend> {
        let n = c * h * w;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 1.3 + 0.7).sin()).collect();
        Tensor::from_vec(data, Shape::new(&[c, h, w]))
    }

    /// Assert two f32 slices are close within tolerance.
    fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        let max_err = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < tol,
            "{label}: max error {max_err:.2e} exceeds tolerance {tol:.2e}"
        );
    }

    #[test]
    fn winograd_matches_im2col_1ch_even() {
        let conv = random_conv(1, 1);
        let input = random_input(1, 8, 8);
        let wino = conv.forward_winograd(&input);
        let ref_out = conv.forward_im2col(&input);
        assert_eq!(wino.shape.dims(), ref_out.shape.dims());
        assert_close(&wino.to_vec(), &ref_out.to_vec(), 1e-4, "1ch 8x8");
    }

    #[test]
    fn winograd_matches_im2col_1ch_odd() {
        let conv = random_conv(1, 1);
        let input = random_input(1, 7, 7);
        let wino = conv.forward_winograd(&input);
        let ref_out = conv.forward_im2col(&input);
        assert_eq!(wino.shape.dims(), ref_out.shape.dims());
        assert_close(&wino.to_vec(), &ref_out.to_vec(), 1e-4, "1ch 7x7");
    }

    #[test]
    fn winograd_matches_im2col_multi_channel() {
        let conv = random_conv(3, 16);
        let input = random_input(3, 14, 14);
        let wino = conv.forward_winograd(&input);
        let ref_out = conv.forward_im2col(&input);
        assert_eq!(wino.shape.dims(), ref_out.shape.dims());
        assert_close(&wino.to_vec(), &ref_out.to_vec(), 1e-4, "3→16 14x14");
    }

    #[test]
    fn winograd_matches_im2col_large() {
        let conv = random_conv(64, 64);
        let input = random_input(64, 56, 56);
        let wino = conv.forward_winograd(&input);
        let ref_out = conv.forward_im2col(&input);
        assert_eq!(wino.shape.dims(), ref_out.shape.dims());
        assert_close(&wino.to_vec(), &ref_out.to_vec(), 1e-3, "64→64 56x56");
    }

    #[test]
    fn winograd_matches_im2col_small_spatial() {
        // 3×3 input with padding=1 → 3×3 output, tiles_h=2 tiles_w=2
        let conv = random_conv(4, 8);
        let input = random_input(4, 3, 3);
        let wino = conv.forward_winograd(&input);
        let ref_out = conv.forward_im2col(&input);
        assert_eq!(wino.shape.dims(), ref_out.shape.dims());
        assert_close(&wino.to_vec(), &ref_out.to_vec(), 1e-4, "4→8 3x3");
    }

    #[test]
    fn winograd_dispatch_auto() {
        // 3×3/stride-1/pad-1 should use Winograd (forward dispatches)
        let conv = random_conv(3, 16);
        assert!(conv.use_winograd());
        let input = random_input(3, 8, 8);
        let out = conv.forward(&input);
        let ref_out = conv.forward_im2col(&input);
        assert_close(&out.to_vec(), &ref_out.to_vec(), 1e-4, "dispatch");
    }

    #[test]
    fn winograd_not_used_for_stride2() {
        let conv = Conv2d::<CpuBackend>::square(3, 16, 3, 2, 1);
        assert!(!conv.use_winograd());
    }

    #[test]
    fn winograd_not_used_for_1x1() {
        let conv = Conv2d::<CpuBackend>::square(3, 16, 1, 1, 0);
        assert!(!conv.use_winograd());
    }
}
