pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod kernels;
#[cfg(feature = "scry-gpu")]
pub mod scry_gpu;

use crate::tensor::shape::Shape;

/// Abramowitz–Stegun 7.1.26 approximation to `erf`, accurate to ~1.5e-7
/// in f64. Shared by the CPU default for `MathBackend::gelu_exact`.
#[inline]
fn erf64_for_gelu(x: f64) -> f64 {
    const P: f64 = 0.327_591_1;
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    let y = 1.0 - poly * (-ax * ax).exp();
    sign * y
}

/// Memory management trait — allocate, copy, transfer between host/device.
pub trait DeviceBackend: Sized {
    type Storage: Clone;
    type Stream;
    #[cfg(feature = "quantize")]
    type I8Storage: Clone;

    #[cfg(feature = "quantize")]
    fn i8_from_vec(data: Vec<i8>) -> Self::I8Storage;
    #[cfg(feature = "quantize")]
    fn i8_to_vec(storage: &[i8]) -> Vec<i8>;

    fn zeros(shape: &Shape) -> Self::Storage;
    fn ones(shape: &Shape) -> Self::Storage;
    fn from_vec(data: Vec<f32>, shape: &Shape) -> Self::Storage;
    /// Create storage from f32 data with optional raw bf16 bytes for direct upload.
    /// Default ignores the bf16 bytes and falls back to `from_vec`.
    #[cfg(feature = "bf16")]
    fn from_vec_with_bf16(data: Vec<f32>, bf16_bytes: &[u8], shape: &Shape) -> Self::Storage {
        let _ = bf16_bytes;
        Self::from_vec(data, shape)
    }
    fn to_vec(storage: &Self::Storage) -> Vec<f32>;
    /// Consume storage and return the underlying `Vec<f32>`.
    /// Default clones via `to_vec`; backends with `Storage = Vec<f32>` override to move.
    fn into_vec(storage: Self::Storage) -> Vec<f32> {
        Self::to_vec(&storage)
    }
    /// Borrow storage as a `&[f32]` slice without cloning.
    /// Default clones via `to_vec` — backends with `Storage = Vec<f32>` should override.
    fn as_slice(storage: &Self::Storage) -> std::borrow::Cow<'_, [f32]> {
        std::borrow::Cow::Owned(Self::to_vec(storage))
    }
    fn clone_storage(storage: &Self::Storage) -> Self::Storage;

    /// Move a storage cell to the backend's "device-resident" form, in place.
    ///
    /// For `CpuBackend` this is a no-op. For `ScryGpuBackend` it uploads a
    /// CPU-resident `Vec<f32>` to a `Gpu` buffer; subsequent reads via the
    /// matmul / pool / etc. kernels see the device buffer directly instead
    /// of paying a fresh upload on every forward pass.
    ///
    /// Idempotent: safe to call on an already-device-resident storage.
    /// Models pre-upload all parameters by walking the parameter tree once
    /// after construction (e.g. `ResNet::to_device`).
    fn to_device_in_place(_storage: &mut Self::Storage) {
        // CPU default: nothing to move.
    }
}

/// Math operations trait — all operations needed for forward passes.
pub trait MathBackend: DeviceBackend {
    // ---- Routing hints ----

    /// Whether this backend's `im2col_2d` + `matmul` chain stays on-device,
    /// so consumers (e.g. `Conv2d`) should prefer im2col over Winograd
    /// F(2×2, 3×3) for 3×3-stride-1-padding-1 kernels.
    ///
    /// Winograd reduces arithmetic count by ~2.25× but its standard
    /// implementation does CPU-side filter and tile transforms. Backends
    /// where that forces host↔device round-trips (e.g. `ScryGpuBackend`
    /// on CUDA) lose more to transfer overhead than they gain in FLOPs.
    /// CPU backends keep the default `false` and benefit from the lower
    /// FLOP count.
    ///
    /// Read once at `Conv2d` construction via
    /// `Conv2dStrategy::default_for::<B>()`, not branched per forward.
    const PREFERS_IM2COL_OVER_WINOGRAD: bool = false;

    // ---- Core forward ops ----

    /// General matrix multiply: `C = alpha * op(A) * op(B) + beta * C`
    fn matmul(
        a: &Self::Storage,
        b: &Self::Storage,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Self::Storage;

    /// Fused matrix multiply + bias add: `C = A @ B + bias` (bias broadcast along rows).
    /// Avoids a separate allocation for the matmul result.
    fn matmul_bias(
        a: &Self::Storage,
        b: &Self::Storage,
        bias: &Self::Storage,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Self::Storage {
        // Default: matmul then add bias in-place
        let mut c = Self::matmul(a, b, m, k, n, trans_a, trans_b);
        let c_vec = Self::to_vec(&c);
        let bias_vec = Self::to_vec(bias);
        let mut out = c_vec;
        for row in 0..m {
            for col in 0..n {
                out[row * n + col] += bias_vec[col];
            }
        }
        c = Self::from_vec(out, &Shape::new(&[m, n]));
        c
    }

    /// Elementwise add with broadcasting.
    fn add(
        a: &Self::Storage,
        b: &Self::Storage,
        a_shape: &Shape,
        b_shape: &Shape,
        out_shape: &Shape,
    ) -> Self::Storage;

    /// In-place elementwise add: `dst[i] += src[i]` (same shape, no broadcast).
    /// Eliminates an allocation compared to `add` + reassign.
    fn add_inplace(dst: &mut Self::Storage, src: &Self::Storage) {
        let mut d = Self::to_vec(dst);
        let s = Self::to_vec(src);
        for (di, si) in d.iter_mut().zip(s.iter()) {
            *di += si;
        }
        let len = d.len();
        *dst = Self::from_vec(d, &Shape::new(&[len]));
    }

    /// Softmax along the last axis.
    fn softmax(input: &Self::Storage, shape: &Shape) -> Self::Storage;

    /// Fused scale + softmax: applies `x * scale` then softmax along the last axis.
    /// Eliminates the separate `scale()` allocation.
    fn scaled_softmax(input: &Self::Storage, scale: f32, shape: &Shape) -> Self::Storage {
        let scaled = Self::scale(input, scale);
        Self::softmax(&scaled, shape)
    }

    /// Layer normalization along the last axis.
    /// Returns `(output, mean, rstd)`.
    fn layernorm(
        input: &Self::Storage,
        gamma: &Self::Storage,
        beta: &Self::Storage,
        shape: &Shape,
        eps: f32,
    ) -> (Self::Storage, Self::Storage, Self::Storage);

    /// Inference-only layer normalization — returns only the output, no mean/rstd.
    fn layernorm_inference(
        input: &Self::Storage,
        gamma: &Self::Storage,
        beta: &Self::Storage,
        shape: &Shape,
        eps: f32,
    ) -> Self::Storage {
        let (output, _, _) = Self::layernorm(input, gamma, beta, shape, eps);
        output
    }

    /// Group normalization, inference path with per-channel affine.
    ///
    /// For input shaped `[batch, channels, spatial]` (where `spatial = H*W`,
    /// flattened to `batch * channels * spatial` elements), splits channels
    /// into `num_groups` groups of `channels / num_groups` channels each,
    /// computes mean/variance per `(batch, group)`, normalizes, then applies
    /// per-channel affine `out = norm * weight[c] + bias[c]`. `weight` and
    /// `bias` are length `channels`. `channels` must be evenly divisible by
    /// `num_groups`.
    ///
    /// Stable Diffusion uses `num_groups = 32` everywhere it appears (UNet
    /// ResBlocks + VAE decoder).
    ///
    /// The default impl runs on host via `to_vec`/`from_vec`; GPU backends
    /// override to keep tensors device-resident.
    fn group_norm(
        input: &Self::Storage,
        weight: &Self::Storage,
        bias: &Self::Storage,
        num_groups: usize,
        channels: usize,
        spatial: usize,
        eps: f32,
    ) -> Self::Storage {
        assert!(
            num_groups > 0 && channels % num_groups == 0,
            "group_norm: channels={channels} must be divisible by num_groups={num_groups}"
        );
        let cpg = channels / num_groups;
        let input_v = Self::to_vec(input);
        let weight_v = Self::to_vec(weight);
        let bias_v = Self::to_vec(bias);
        assert!(weight_v.len() == channels && bias_v.len() == channels);

        let total = input_v.len();
        let plane = channels * spatial;
        let n_batch = total.checked_div(plane).unwrap_or(0);
        let mut out = vec![0.0f32; total];
        let gsize = cpg * spatial;

        for n in 0..n_batch {
            for g in 0..num_groups {
                let c_start = g * cpg;
                let base = (n * channels + c_start) * spatial;

                let mut sum = 0.0f64;
                for j in 0..gsize {
                    sum += f64::from(input_v[base + j]);
                }
                let mean = (sum / gsize as f64) as f32;

                let mut sq = 0.0f64;
                for j in 0..gsize {
                    let diff = f64::from(input_v[base + j]) - f64::from(mean);
                    sq += diff * diff;
                }
                let var = (sq / gsize as f64) as f32;
                let rstd = 1.0f32 / (var + eps).sqrt();

                for j in 0..gsize {
                    let local_c = j / spatial;
                    let c = c_start + local_c;
                    let norm = (input_v[base + j] - mean) * rstd;
                    out[base + j] = norm * weight_v[c] + bias_v[c];
                }
            }
        }

        Self::from_vec(out, &Shape::new(&[total]))
    }

    /// 2D batch normalization, inference-only with stored running statistics.
    ///
    /// For input shaped `[channels, spatial]` (or `[batch, channels, spatial]`
    /// flattened to `batch * channels * spatial` elements), computes
    /// `out[c, i] = (in[c, i] - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]`.
    /// `weight`, `bias`, `running_mean`, and `running_var` are all length `channels`.
    /// `spatial` is `H * W`. The default impl runs on host via `to_vec`/`from_vec`;
    /// GPU backends override to keep tensors device-resident.
    fn batchnorm_2d_inference(
        input: &Self::Storage,
        weight: &Self::Storage,
        bias: &Self::Storage,
        running_mean: &Self::Storage,
        running_var: &Self::Storage,
        channels: usize,
        spatial: usize,
        eps: f32,
    ) -> Self::Storage {
        let input_v = Self::to_vec(input);
        let weight_v = Self::to_vec(weight);
        let bias_v = Self::to_vec(bias);
        let mean_v = Self::to_vec(running_mean);
        let var_v = Self::to_vec(running_var);
        let total = input_v.len();
        let plane = channels * spatial;
        let mut out = vec![0.0f32; total];
        let n_batch = total.checked_div(plane).unwrap_or(0);
        for n in 0..n_batch {
            for c in 0..channels {
                let scale = weight_v[c] / (var_v[c] + eps).sqrt();
                let shift = bias_v[c] - mean_v[c] * scale;
                let off = (n * channels + c) * spatial;
                for i in 0..spatial {
                    out[off + i] = input_v[off + i] * scale + shift;
                }
            }
        }
        Self::from_vec(out, &Shape::new(&[total]))
    }

    /// GELU activation (tanh approximation).
    fn gelu(input: &Self::Storage) -> Self::Storage;

    /// Exact (erf-based) GELU: `0.5 * x * (1 + erf(x / sqrt(2)))`.
    ///
    /// Distinct from [`Self::gelu`] which is the tanh approximation HF
    /// uses for `approximate="tanh"`. PyTorch's `F.gelu(approximate="none")`
    /// is this erf form, and the SD UNet's GeGLU MLP uses exactly that —
    /// the tanh approximation drifts ~3e-4 per block and accumulates over
    /// SD 1.5's 16 transformer blocks to break the M6 1e-3 parity gate.
    ///
    /// CPU default uses Abramowitz–Stegun 7.1.26 in f64 (max error
    /// ≈ 1.5e-7). GPU backends should override with a CUDA / WGSL shader
    /// to avoid the host round-trip — at SD 1.5's deepest stage the gate
    /// tensor is `[4096, 5120]` = 80 MB, and the GeGLU MLP is the
    /// single largest cost in the UNet forward (47% at 512×512 / bf16).
    fn gelu_exact(input: &Self::Storage) -> Self::Storage {
        const INV_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
        let v = Self::to_vec(input);
        let n = v.len();
        let out: Vec<f32> = v
            .into_iter()
            .map(|x| {
                let xd = f64::from(x);
                #[allow(clippy::cast_possible_truncation)]
                let y = (0.5 * xd * (1.0 + erf64_for_gelu(xd * INV_SQRT_2))) as f32;
                y
            })
            .collect();
        Self::from_vec(out, &Shape::new(&[n]))
    }

    /// `ReLU` activation: `out[i] = max(0, in[i])`. Default impl runs on
    /// host via `to_vec`/`from_vec`; GPU backends override to keep tensors
    /// device-resident.
    fn relu(input: &Self::Storage) -> Self::Storage {
        let v = Self::to_vec(input);
        let n = v.len();
        let out: Vec<f32> = v.into_iter().map(|x| x.max(0.0)).collect();
        Self::from_vec(out, &Shape::new(&[n]))
    }

    /// SiLU / Swish activation: `out[i] = x * sigmoid(x) = x / (1 + exp(-x))`.
    ///
    /// Used in every Stable-Diffusion UNet ResBlock and across the SD family.
    /// Default impl runs on host via `to_vec`/`from_vec`; GPU backends
    /// override to keep tensors device-resident.
    fn silu(input: &Self::Storage) -> Self::Storage {
        let v = Self::to_vec(input);
        let n = v.len();
        let out: Vec<f32> = v.into_iter().map(|x| x / (1.0 + (-x).exp())).collect();
        Self::from_vec(out, &Shape::new(&[n]))
    }

    /// Forward 2D convolution: lowers + multiplies in one call.
    ///
    /// Input is `[in_channels, h_in, w_in]`, weight is `[out_channels,
    /// in_channels, kernel_h, kernel_w]` (flattened to one row-major
    /// tensor). Output is `[out_channels, h_out*w_out]` row-major where
    /// `h_out = (h_in + 2*padding - kernel_h) / stride + 1` (and likewise
    /// for w). **No bias** — the caller adds bias separately.
    ///
    /// The default impl is `im2col_2d` + `matmul` (same composition the call
    /// site used to do explicitly). GPU backends override to use a fused
    /// path — `ScryGpuBackend` with the `scry-gpu-cudnn` feature routes
    /// through cuDNN's implicit-GEMM, skipping the lowering kernel + its
    /// HBM round-trip.
    #[allow(clippy::too_many_arguments)]
    fn conv2d_forward(
        input: &Self::Storage,
        weight: &Self::Storage,
        in_channels: usize,
        h_in: usize,
        w_in: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
    ) -> Self::Storage {
        let h_out = (h_in + 2 * padding - kernel_h) / stride + 1;
        let w_out = (w_in + 2 * padding - kernel_w) / stride + 1;
        let lowered = Self::im2col_2d(
            input,
            in_channels,
            h_in,
            w_in,
            kernel_h,
            kernel_w,
            stride,
            padding,
        );
        Self::matmul(
            weight,
            &lowered,
            out_channels,
            in_channels * kernel_h * kernel_w,
            h_out * w_out,
            false,
            false,
        )
    }

    /// Im2col lowering for a 2D convolution.
    ///
    /// Input is `[in_channels, h_in, w_in]`; output is
    /// `[in_channels*kernel_h*kernel_w, h_out*w_out]` in row-major layout
    /// suitable for direct multiplication by a `[out_channels, in_channels*kernel_h*kernel_w]`
    /// weight matrix. Zero-pads outside the input. Stride and padding are
    /// symmetric (same value for height and width).
    ///
    /// `h_out = (h_in + 2*padding - kernel_h) / stride + 1` (and likewise
    /// for `w_out`); callers compute these themselves.
    ///
    /// The default impl runs on host via `to_vec`/`from_vec`; GPU backends
    /// override to keep tensors device-resident.
    #[allow(clippy::too_many_arguments)]
    fn im2col_2d(
        input: &Self::Storage,
        in_channels: usize,
        h_in: usize,
        w_in: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
    ) -> Self::Storage {
        let input_v = Self::to_vec(input);
        let h_out = (h_in + 2 * padding - kernel_h) / stride + 1;
        let w_out = (w_in + 2 * padding - kernel_w) / stride + 1;
        let spatial_out = h_out * w_out;
        let col_rows = in_channels * kernel_h * kernel_w;
        let mut col = vec![0.0f32; col_rows * spatial_out];

        for oh in 0..h_out {
            for ow in 0..w_out {
                let out_col = oh * w_out + ow;
                for c in 0..in_channels {
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let ih = oh * stride + kh;
                            let iw = ow * stride + kw;
                            let val = if ih >= padding
                                && ih < padding + h_in
                                && iw >= padding
                                && iw < padding + w_in
                            {
                                let h_idx = ih - padding;
                                let w_idx = iw - padding;
                                input_v[c * h_in * w_in + h_idx * w_in + w_idx]
                            } else {
                                0.0
                            };
                            let row = c * kernel_h * kernel_w + kh * kernel_w + kw;
                            col[row * spatial_out + out_col] = val;
                        }
                    }
                }
            }
        }

        Self::from_vec(col, &Shape::new(&[col_rows, spatial_out]))
    }

    /// 2D max-pooling over a `[channels, h_in, w_in]` input.
    ///
    /// Output is `[channels, h_out, w_out]` flattened to `channels*h_out*w_out`
    /// elements where `h_out = (h_in + 2*padding - kernel) / stride + 1` (and
    /// likewise for `w_out`). Stride and padding are symmetric. Padded
    /// positions don't contribute to the max; if every position in a window
    /// is padding the output is `0.0` (matches PyTorch).
    ///
    /// The default impl runs on host via `to_vec`/`from_vec`; GPU backends
    /// override to keep tensors device-resident.
    fn max_pool_2d(
        input: &Self::Storage,
        channels: usize,
        h_in: usize,
        w_in: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Self::Storage {
        let h_out = (h_in + 2 * padding - kernel) / stride + 1;
        let w_out = (w_in + 2 * padding - kernel) / stride + 1;
        let input_v = Self::to_vec(input);
        let mut out = vec![0.0f32; channels * h_out * w_out];
        for c in 0..channels {
            let plane = c * h_in * w_in;
            let out_plane = c * h_out * w_out;
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut m = f32::NEG_INFINITY;
                    let mut any = false;
                    for kh in 0..kernel {
                        for kw in 0..kernel {
                            let ih = oh * stride + kh;
                            let iw = ow * stride + kw;
                            if ih >= padding
                                && ih < padding + h_in
                                && iw >= padding
                                && iw < padding + w_in
                            {
                                let v = input_v[plane + (ih - padding) * w_in + (iw - padding)];
                                if !any || v > m {
                                    m = v;
                                }
                                any = true;
                            }
                        }
                    }
                    out[out_plane + oh * w_out + ow] = if any { m } else { 0.0 };
                }
            }
        }
        Self::from_vec(out, &Shape::new(&[channels, h_out, w_out]))
    }

    /// 2D zero-padding, NCHW layout, with **asymmetric pad support**.
    ///
    /// Input is `[channels, h_in, w_in]`; output is
    /// `[channels, h_in + pad_top + pad_bottom, w_in + pad_left + pad_right]`
    /// with the input copied into the `(pad_top.., pad_left..)` region and
    /// zeros everywhere else. Matches `torch.nn.functional.pad(x, (pad_left,
    /// pad_right, pad_top, pad_bottom), mode="constant", value=0)` —
    /// note PyTorch's pad order is `(W_left, W_right, H_top, H_bottom)`.
    ///
    /// HF's `Downsample2D` in the VAE encoder uses `pad(0, 1, 0, 1)`
    /// (i.e. `pad_right=1, pad_bottom=1`) before a stride-2 padding-0
    /// conv — `scry_vision::nn::Conv2d` only supports symmetric padding,
    /// so the asymmetric pad lives here.
    ///
    /// The default impl runs on host via `to_vec`/`from_vec`; GPU backends
    /// can override to keep tensors device-resident.
    fn pad_2d_zero(
        input: &Self::Storage,
        channels: usize,
        h_in: usize,
        w_in: usize,
        pad_top: usize,
        pad_bottom: usize,
        pad_left: usize,
        pad_right: usize,
    ) -> Self::Storage {
        let h_out = h_in + pad_top + pad_bottom;
        let w_out = w_in + pad_left + pad_right;
        let input_v = Self::to_vec(input);
        let mut out = vec![0.0f32; channels * h_out * w_out];
        for c in 0..channels {
            let in_plane = c * h_in * w_in;
            let out_plane = c * h_out * w_out;
            for ih in 0..h_in {
                let oh = ih + pad_top;
                let in_row = in_plane + ih * w_in;
                let out_row = out_plane + oh * w_out + pad_left;
                out[out_row..out_row + w_in].copy_from_slice(&input_v[in_row..in_row + w_in]);
            }
        }
        Self::from_vec(out, &Shape::new(&[channels, h_out, w_out]))
    }

    /// 2D nearest-neighbor upsample by an integer factor, NCHW layout.
    ///
    /// Input is `[channels, h_in, w_in]`; output is
    /// `[channels, h_in*scale, w_in*scale]` with
    /// `out[c, oh, ow] = in[c, oh/scale, ow/scale]` (integer divide). Used in
    /// SD UNet UpBlocks and the VAE decoder. Matches PyTorch's
    /// `F.interpolate(mode="nearest")` and `nn.Upsample(mode="nearest")`
    /// byte-for-byte.
    ///
    /// The default impl runs on host via `to_vec`/`from_vec`; GPU backends
    /// override to keep tensors device-resident.
    fn upsample_2d_nearest(
        input: &Self::Storage,
        channels: usize,
        h_in: usize,
        w_in: usize,
        scale: usize,
    ) -> Self::Storage {
        let h_out = h_in * scale;
        let w_out = w_in * scale;
        let input_v = Self::to_vec(input);
        let mut out = vec![0.0f32; channels * h_out * w_out];
        for c in 0..channels {
            let in_plane = c * h_in * w_in;
            let out_plane = c * h_out * w_out;
            for oh in 0..h_out {
                let ih = oh / scale;
                for ow in 0..w_out {
                    let iw = ow / scale;
                    out[out_plane + oh * w_out + ow] = input_v[in_plane + ih * w_in + iw];
                }
            }
        }
        Self::from_vec(out, &Shape::new(&[channels, h_out, w_out]))
    }

    /// Adaptive 2D average pooling: `[channels, h_in, w_in]` →
    /// `[channels, h_out, w_out]` with per-output regions
    /// `h_start = oh*h_in/h_out`, `h_end = (oh+1)*h_in/h_out` (matches
    /// PyTorch's `AdaptiveAvgPool2d`). Global average pooling is the
    /// `h_out=w_out=1` special case.
    ///
    /// The default impl runs on host via `to_vec`/`from_vec`; GPU backends
    /// override to keep tensors device-resident.
    fn adaptive_avg_pool_2d(
        input: &Self::Storage,
        channels: usize,
        h_in: usize,
        w_in: usize,
        h_out: usize,
        w_out: usize,
    ) -> Self::Storage {
        let input_v = Self::to_vec(input);
        let mut out = vec![0.0f32; channels * h_out * w_out];
        for c in 0..channels {
            let plane = c * h_in * w_in;
            let out_plane = c * h_out * w_out;
            for oh in 0..h_out {
                let h_start = oh * h_in / h_out;
                let h_end = (oh + 1) * h_in / h_out;
                for ow in 0..w_out {
                    let w_start = ow * w_in / w_out;
                    let w_end = (ow + 1) * w_in / w_out;
                    let mut sum = 0.0f32;
                    for h in h_start..h_end {
                        for w in w_start..w_end {
                            sum += input_v[plane + h * w_in + w];
                        }
                    }
                    let count = (h_end - h_start) * (w_end - w_start);
                    out[out_plane + oh * w_out + ow] = sum / count as f32;
                }
            }
        }
        Self::from_vec(out, &Shape::new(&[channels, h_out, w_out]))
    }

    /// Embedding lookup: gather rows by indices.
    fn embedding(
        weight: &Self::Storage,
        indices: &[usize],
        vocab: usize,
        dim: usize,
    ) -> Self::Storage;

    /// Sum all elements to scalar.
    fn sum(input: &Self::Storage) -> f32;

    /// Elementwise multiply: `a * b` (same shape, no broadcast).
    fn mul_elementwise(a: &Self::Storage, b: &Self::Storage) -> Self::Storage;

    /// Scale all elements: `a * scalar`. Must allocate fresh storage —
    /// callers (notably `vae::blocks::clone_tensor` and the analogous
    /// `unet/common.rs` helper) rely on `scale(_, 1.0)` producing an
    /// independent copy so a subsequent in-place op on the result cannot
    /// alias `a`. Do not add an in-place fast path for `scalar == 1.0`.
    fn scale(a: &Self::Storage, scalar: f32) -> Self::Storage;

    /// 2D transpose: input `[rows, cols]` (row-major) → output `[cols, rows]`
    /// (row-major). CPU default uses a scalar permute through host memory;
    /// GPU backends should override with a kernel to avoid the round-trip.
    ///
    /// Used inside the SD UNet's `SpatialTransformer` to permute
    /// `[C, H*W] ↔ [H*W, C]` around the transformer-block stack — at SD
    /// shapes that's tens of MB per call × 32 calls per UNet forward.
    fn transpose_2d(input: &Self::Storage, rows: usize, cols: usize) -> Self::Storage {
        let v = Self::to_vec(input);
        debug_assert_eq!(v.len(), rows * cols);
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = v[r * cols + c];
            }
        }
        Self::from_vec(out, &Shape::new(&[cols, rows]))
    }

    /// Concatenate two row-major matrices along rows (axis 0).
    fn concat_rows(
        a: &Self::Storage,
        b: &Self::Storage,
        a_rows: usize,
        b_rows: usize,
        cols: usize,
    ) -> Self::Storage;

    // ---- Llama-specific ops ----

    /// RMS normalization: `out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]`
    fn rmsnorm(
        input: &Self::Storage,
        weight: &Self::Storage,
        shape: &Shape,
        eps: f32,
    ) -> Self::Storage;

    /// Fused RMS normalization + bf16 cast.
    /// Returns f32 output with bf16 shadow pre-populated (avoids a separate f32→bf16 kernel).
    /// Default falls back to `rmsnorm` (no bf16 fusion).
    #[cfg(feature = "bf16")]
    fn rmsnorm_with_bf16(
        input: &Self::Storage,
        weight: &Self::Storage,
        shape: &Shape,
        eps: f32,
    ) -> Self::Storage {
        Self::rmsnorm(input, weight, shape, eps)
    }

    /// Rotary position embeddings.
    fn rope(
        input: &Self::Storage,
        shape: &Shape,
        pos: usize,
        head_dim: usize,
        theta: f32,
    ) -> Self::Storage;

    /// Rotary position embeddings with pre-computed frequencies (GPU-resident).
    ///
    /// The `freqs` storage is already on-device (uploaded once at model load),
    /// eliminating per-token host→device transfers.
    fn rope_with_freqs_preloaded(
        input: &Self::Storage,
        seq: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
        freqs: &Self::Storage,
    ) -> Self::Storage;

    /// Rotary position embeddings with pre-computed frequencies and output scaling (GPU-resident).
    ///
    /// Fuses RoPE and scalar multiply into a single kernel: `out = RoPE(input) * scale`.
    /// Default implementation calls `rope_with_freqs_preloaded` then `scale`.
    fn rope_with_freqs_scaled_preloaded(
        input: &Self::Storage,
        seq: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
        freqs: &Self::Storage,
        scale: f32,
    ) -> Self::Storage {
        let out = Self::rope_with_freqs_preloaded(input, seq, n_heads, head_dim, start_pos, freqs);
        Self::scale(&out, scale)
    }

    /// Rotary position embeddings with pre-computed frequencies (host-side).
    ///
    /// Converts `&[f64]` freqs to `Self::Storage` then delegates to `rope_with_freqs_preloaded`.
    /// Prefer `_preloaded` variant in hot paths to avoid per-call uploads.
    fn rope_with_freqs(
        input: &Self::Storage,
        seq: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
        freqs: &[f64],
    ) -> Self::Storage {
        let freqs_f32: Vec<f32> = freqs.iter().map(|&f| f as f32).collect();
        let freqs_storage = Self::from_vec(freqs_f32, &Shape::new(&[head_dim / 2]));
        Self::rope_with_freqs_preloaded(input, seq, n_heads, head_dim, start_pos, &freqs_storage)
    }

    /// Rotary position embeddings with pre-computed frequencies and output scaling (host-side).
    ///
    /// Converts `&[f64]` freqs to `Self::Storage` then delegates to `rope_with_freqs_scaled_preloaded`.
    fn rope_with_freqs_scaled(
        input: &Self::Storage,
        seq: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
        freqs: &[f64],
        scale: f32,
    ) -> Self::Storage {
        let freqs_f32: Vec<f32> = freqs.iter().map(|&f| f as f32).collect();
        let freqs_storage = Self::from_vec(freqs_f32, &Shape::new(&[head_dim / 2]));
        Self::rope_with_freqs_scaled_preloaded(
            input,
            seq,
            n_heads,
            head_dim,
            start_pos,
            &freqs_storage,
            scale,
        )
    }

    /// `SwiGLU`: `silu(gate) * up` where `silu(x) = x / (1 + exp(-x))`
    fn swiglu(gate: &Self::Storage, up: &Self::Storage) -> Self::Storage;

    /// Fused SwiGLU + bf16 cast.
    /// Returns f32 output with bf16 shadow pre-populated.
    /// Default falls back to `swiglu` (no bf16 fusion).
    #[cfg(feature = "bf16")]
    fn swiglu_with_bf16(gate: &Self::Storage, up: &Self::Storage) -> Self::Storage {
        Self::swiglu(gate, up)
    }

    /// Repeat KV heads for GQA: expand `[n_kv_heads, seq, d_head]` to `[n_q_heads, seq, d_head]`.
    fn repeat_kv(
        input: &Self::Storage,
        n_kv_heads: usize,
        n_q_heads: usize,
        seq: usize,
        d_head: usize,
    ) -> Self::Storage;

    /// Fused gather + reshape + repeat_kv: read directly from pre-allocated KV cache
    /// `[max_seq, n_kv_heads * head_dim]` and produce `[n_q_heads, cached_len, head_dim]`
    /// in a single pass. Eliminates separate gather_rows, reshape_for_heads, and repeat_kv.
    fn gather_reshape_repeat_kv(
        cache: &Self::Storage,
        max_seq: usize,
        cached_len: usize,
        n_kv_heads: usize,
        n_q_heads: usize,
        head_dim: usize,
    ) -> Self::Storage {
        // Default: fall back to the 3-step path
        let _ = max_seq;
        let kv_dim = n_kv_heads * head_dim;
        let gathered = Self::gather_rows(cache, max_seq, kv_dim, 0, cached_len);
        let reshaped = Self::reshape_for_heads(&gathered, 1, cached_len, n_kv_heads, head_dim);
        Self::repeat_kv(&reshaped, n_kv_heads, n_q_heads, cached_len, head_dim)
    }

    // ---- Attention helpers ----

    /// Extract columns `[col_start..col_start+col_count)` from a `[rows, total_cols]` matrix.
    fn gather_columns(
        storage: &Self::Storage,
        rows: usize,
        total_cols: usize,
        col_start: usize,
        col_count: usize,
    ) -> Self::Storage {
        let data = Self::to_vec(storage);
        let mut out = vec![0.0f32; rows * col_count];
        for r in 0..rows {
            for c in 0..col_count {
                out[r * col_count + c] = data[r * total_cols + col_start + c];
            }
        }
        Self::from_vec(out, &Shape::new(&[rows, col_count]))
    }

    /// Scatter (additive) a `[rows, col_count]` source into columns of a destination.
    fn scatter_columns(
        dst: &mut Self::Storage,
        src: &Self::Storage,
        rows: usize,
        total_cols: usize,
        col_start: usize,
        col_count: usize,
    ) {
        let mut dst_vec = Self::to_vec(dst);
        let src_vec = Self::to_vec(src);
        for r in 0..rows {
            for c in 0..col_count {
                dst_vec[r * total_cols + col_start + c] += src_vec[r * col_count + c];
            }
        }
        *dst = Self::from_vec(dst_vec, &Shape::new(&[rows, total_cols]));
    }

    /// Extract rows `[row_start..row_start+row_count)` from a `[total_rows, cols]` matrix.
    fn gather_rows(
        storage: &Self::Storage,
        total_rows: usize,
        cols: usize,
        row_start: usize,
        row_count: usize,
    ) -> Self::Storage {
        let _ = total_rows;
        let data = Self::to_vec(storage);
        let start = row_start * cols;
        let end = start + row_count * cols;
        Self::from_vec(data[start..end].to_vec(), &Shape::new(&[row_count, cols]))
    }

    /// Write rows into a destination matrix (overwrite).
    fn scatter_rows(
        dst: &mut Self::Storage,
        src: &Self::Storage,
        total_rows: usize,
        cols: usize,
        row_start: usize,
        row_count: usize,
    ) {
        let mut dst_vec = Self::to_vec(dst);
        let src_vec = Self::to_vec(src);
        let start = row_start * cols;
        dst_vec[start..start + row_count * cols].copy_from_slice(&src_vec[..row_count * cols]);
        *dst = Self::from_vec(dst_vec, &Shape::new(&[total_rows, cols]));
    }

    /// Apply causal mask and scale to a `[seq_len, seq_len]` matrix in-place.
    fn apply_causal_mask_and_scale(
        scores: &mut Self::Storage,
        seq_len: usize,
        scale: f32,
        mask_value: f32,
    ) {
        let mut data = Self::to_vec(scores);
        for s in 0..seq_len {
            for t in 0..seq_len {
                if t > s {
                    data[s * seq_len + t] = mask_value;
                } else {
                    data[s * seq_len + t] *= scale;
                }
            }
        }
        *scores = Self::from_vec(data, &Shape::new(&[seq_len, seq_len]));
    }

    /// Fused QKV split + reshape for multi-head attention.
    ///
    /// Input: `[seq, 3*d_model]` containing concatenated Q, K, V.
    /// Returns `(Q_heads, K_heads, V_heads)` each shaped `[n_heads, seq, d_head]`.
    /// Eliminates separate split + 3x from_vec + 3x reshape_for_heads.
    fn split_qkv_reshape_heads(
        qkv: &Self::Storage,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> (Self::Storage, Self::Storage, Self::Storage) {
        let d_model = n_heads * d_head;
        let head_len = n_heads * seq * d_head;
        let data = Self::to_vec(qkv);
        let mut q = vec![0.0f32; head_len];
        let mut k = vec![0.0f32; head_len];
        let mut v = vec![0.0f32; head_len];
        for s in 0..seq {
            let row = s * 3 * d_model;
            for h in 0..n_heads {
                for d in 0..d_head {
                    let dst = (h * seq + s) * d_head + d;
                    let src_col = h * d_head + d;
                    q[dst] = data[row + src_col];
                    k[dst] = data[row + d_model + src_col];
                    v[dst] = data[row + 2 * d_model + src_col];
                }
            }
        }
        let shape = Shape::new(&[n_heads, seq, d_head]);
        (
            Self::from_vec(q, &shape),
            Self::from_vec(k, &shape),
            Self::from_vec(v, &shape),
        )
    }

    /// Reshape `[B*S, H*d_head]` → `[B*H, S, d_head]` for batched multi-head attention.
    fn reshape_for_heads(
        storage: &Self::Storage,
        batch: usize,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> Self::Storage {
        // When B=1, S=1 the reshape is an identity permutation — skip the transpose.
        if batch == 1 && seq == 1 {
            return Self::clone_storage(storage);
        }
        let data = Self::to_vec(storage);
        let d_model = n_heads * d_head;
        let total = batch * n_heads * seq * d_head;
        let mut out = vec![0.0f32; total];
        for b in 0..batch {
            for h in 0..n_heads {
                for s in 0..seq {
                    for d in 0..d_head {
                        out[(b * n_heads + h) * seq * d_head + s * d_head + d] =
                            data[(b * seq + s) * d_model + h * d_head + d];
                    }
                }
            }
        }
        Self::from_vec(out, &Shape::new(&[batch * n_heads, seq, d_head]))
    }

    /// Reshape host `&[f32]` data `[B*S, H*d_head]` → `[B*H, S, d_head]` for batched attention.
    /// Avoids clone + from_vec for CpuBackend where Storage = Vec<f32>.
    fn reshape_for_heads_from_host(
        data: &[f32],
        batch: usize,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> Self::Storage {
        let v = data.to_vec();
        let storage = Self::from_vec(v, &Shape::new(&[batch * seq, n_heads * d_head]));
        Self::reshape_for_heads(&storage, batch, seq, n_heads, d_head)
    }

    /// Reshape `[B*H, S, d_head]` → `[B*S, H*d_head]` (reverse of `reshape_for_heads`).
    fn reshape_from_heads(
        storage: &Self::Storage,
        batch: usize,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> Self::Storage {
        // When B=1, S=1 the reshape is an identity permutation — skip the transpose.
        if batch == 1 && seq == 1 {
            return Self::clone_storage(storage);
        }
        let data = Self::to_vec(storage);
        let d_model = n_heads * d_head;
        let total = batch * seq * d_model;
        let mut out = vec![0.0f32; total];
        for b in 0..batch {
            for h in 0..n_heads {
                for s in 0..seq {
                    for d in 0..d_head {
                        out[(b * seq + s) * d_model + h * d_head + d] =
                            data[(b * n_heads + h) * seq * d_head + s * d_head + d];
                    }
                }
            }
        }
        Self::from_vec(out, &Shape::new(&[batch * seq, d_model]))
    }

    // ---- INT8 quantized ops ----

    /// Matrix multiply with i8 weights: `C = A_f32 @ dequant(B_i8)`.
    ///
    /// `B_i8` is `[k, n]` in i8, `scale` converts i8→f32.
    /// Result is `[m, n]` in f32.
    ///
    /// Default: dequantize to f32 buffer then call `matmul`.
    #[cfg(feature = "quantize")]
    fn matmul_i8_f32(
        a: &Self::Storage,
        b_q: &Self::I8Storage,
        scale: f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> Self::Storage {
        let b_i8 = Self::i8_to_vec(b_q);
        let b_f32: Vec<f32> = b_i8.iter().map(|&q| f32::from(q) * scale).collect();
        let b_storage = Self::from_vec(b_f32, &Shape::new(&[k, n]));
        Self::matmul(a, &b_storage, m, k, n, false, false)
    }

    /// Fused i8 matmul + bias: `C = A_f32 @ dequant(B_i8) + bias`.
    #[cfg(feature = "quantize")]
    fn matmul_i8_f32_bias(
        a: &Self::Storage,
        b_q: &Self::I8Storage,
        scale: f32,
        bias: &Self::Storage,
        m: usize,
        k: usize,
        n: usize,
    ) -> Self::Storage {
        let mut c = Self::matmul_i8_f32(a, b_q, scale, m, k, n);
        let c_vec = Self::to_vec(&c);
        let bias_vec = Self::to_vec(bias);
        let mut out = c_vec;
        for row in 0..m {
            for col in 0..n {
                out[row * n + col] += bias_vec[col];
            }
        }
        c = Self::from_vec(out, &Shape::new(&[m, n]));
        c
    }

    /// Strided batched matrix multiply.
    #[allow(clippy::too_many_arguments)]
    fn matmul_strided_batched(
        a: &Self::Storage,
        b: &Self::Storage,
        batch_count: usize,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Self::Storage {
        let a_stride = m * k;
        let b_stride = k * n;
        let c_stride = m * n;
        let total = batch_count * c_stride;
        let a_vec = Self::to_vec(a);
        let b_vec = Self::to_vec(b);
        let mut c_vec = vec![0.0f32; total];
        for i in 0..batch_count {
            let a_slice = Self::from_vec(
                a_vec[i * a_stride..(i + 1) * a_stride].to_vec(),
                &Shape::new(&[m, k]),
            );
            let b_slice = Self::from_vec(
                b_vec[i * b_stride..(i + 1) * b_stride].to_vec(),
                &Shape::new(&[k, n]),
            );
            let c_slice = Self::matmul(&a_slice, &b_slice, m, k, n, trans_a, trans_b);
            let c_data = Self::to_vec(&c_slice);
            c_vec[i * c_stride..(i + 1) * c_stride].copy_from_slice(&c_data);
        }
        Self::from_vec(c_vec, &Shape::new(&[batch_count * m, n]))
    }

    /// Multi-head scaled dot-product attention in one trait call.
    ///
    /// Inputs are post-`reshape_for_heads`:
    ///   - `q`: `[num_heads, n_q, head_dim]`
    ///   - `k`: `[num_heads, n_kv, head_dim]`
    ///   - `v`: `[num_heads, n_kv, head_dim]`
    ///
    /// Returns `[num_heads, n_q, head_dim]`. `scale` is the pre-softmax
    /// scale (typically `1 / sqrt(head_dim)`).
    ///
    /// The default impl composes the existing
    /// `matmul_strided_batched` (scores) → `scaled_softmax` →
    /// `matmul_strided_batched` (values) cascade. GPU backends override
    /// this with a fused kernel that avoids materializing the
    /// `[num_heads · n_q, n_kv]` softmax intermediate — at SD 1.5's
    /// deepest self-attn stage that intermediate is ~537 MB of
    /// device traffic per dispatch.
    ///
    /// Why a single trait method instead of leaving the cascade to the
    /// caller: the fused kernel reads K/V tiles from shared memory and
    /// updates a running softmax max/denom + accumulator without ever
    /// writing the full attention matrix to global memory. That fusion
    /// has to happen across the three sub-operations, so it cannot be
    /// expressed as an override of any one of them.
    #[allow(clippy::too_many_arguments)]
    fn fused_attention(
        q: &Self::Storage,
        k: &Self::Storage,
        v: &Self::Storage,
        num_heads: usize,
        n_q: usize,
        n_kv: usize,
        head_dim: usize,
        scale: f32,
    ) -> Self::Storage {
        let scores =
            Self::matmul_strided_batched(q, k, num_heads, n_q, head_dim, n_kv, false, true);
        let attn = Self::scaled_softmax(&scores, scale, &Shape::new(&[num_heads * n_q, n_kv]));
        Self::matmul_strided_batched(&attn, v, num_heads, n_q, n_kv, head_dim, false, false)
    }

    /// Apply causal mask and scale to batched matrices.
    fn apply_batched_causal_mask_and_scale(
        scores: &mut Self::Storage,
        num_matrices: usize,
        seq_len: usize,
        scale: f32,
        mask_value: f32,
    ) {
        let _ = num_matrices;
        let mut data = Self::to_vec(scores);
        let mat_size = seq_len * seq_len;
        let total = data.len() / mat_size;
        for m in 0..total {
            let off = m * mat_size;
            for s in 0..seq_len {
                for t in 0..seq_len {
                    let idx = off + s * seq_len + t;
                    if t > s {
                        data[idx] = mask_value;
                    } else {
                        data[idx] *= scale;
                    }
                }
            }
        }
        let n = data.len();
        *scores = Self::from_vec(data, &Shape::new(&[n]));
    }
}
