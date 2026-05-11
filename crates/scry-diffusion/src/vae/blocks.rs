// SPDX-License-Identifier: MIT OR Apache-2.0
//! Building blocks shared by the VAE encoder and decoder.
//!
//! Both networks use the same ResNet flavor (`silu → conv → silu → conv`
//! with a pre-norm and a `conv_shortcut` for channel changes), the same
//! mid-block single-head self-attention, and the same tiny tensor helpers
//! (NCHW↔HWC transpose, biased matmul, residual add). Keeping them here
//! avoids drift between the encoder and decoder.

use scry_llm::backend::MathBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;
use scry_vision::nn::conv2d::Conv2d;

#[cfg(feature = "safetensors")]
use crate::error::{Error, Result};

pub(crate) const NORM_EPS: f32 = 1e-6;

// -----------------------------------------------------------------------
// GroupNorm
// -----------------------------------------------------------------------

/// 1D GroupNorm parameters. The compute path goes through
/// `MathBackend::group_norm` directly; this struct just holds the affine.
pub(crate) struct GroupNormParams<B: MathBackend> {
    pub(crate) weight: Tensor<B>,
    pub(crate) bias: Tensor<B>,
    pub(crate) num_groups: usize,
    pub(crate) channels: usize,
}

impl<B: MathBackend> GroupNormParams<B> {
    pub(crate) fn to_device(&mut self) {
        B::to_device_in_place(&mut self.weight.data);
        B::to_device_in_place(&mut self.bias.data);
    }

    /// Apply group norm to `[C, H, W]` (batch=1 implicit). Returns same shape.
    pub(crate) fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let dims = input.shape.dims();
        debug_assert_eq!(dims[0], self.channels);
        let spatial = dims[1] * dims[2];
        let out = B::group_norm(
            &input.data,
            &self.weight.data,
            &self.bias.data,
            self.num_groups,
            self.channels,
            spatial,
            NORM_EPS,
        );
        Tensor::new(out, input.shape.clone())
    }
}

// -----------------------------------------------------------------------
// ResNet block
// -----------------------------------------------------------------------

/// VAE residual block (no timestep — that's the UNet flavor).
///
/// `silu → conv1 → silu → conv2`, sandwiched between two GroupNorms,
/// with a `conv_shortcut` 1×1 conv on the residual when channel count
/// changes. HF's `ResnetBlock2D` runs the activations *before* the
/// convs, not after — see the diffusers source.
pub(crate) struct VaeResnetBlock<B: MathBackend> {
    norm1: GroupNormParams<B>,
    conv1: Conv2d<B>,
    norm2: GroupNormParams<B>,
    conv2: Conv2d<B>,
    /// 1×1 channel-matching conv for the residual when `in != out`.
    /// Identity skip when `None`.
    conv_shortcut: Option<Conv2d<B>>,
}

impl<B: MathBackend> VaeResnetBlock<B> {
    pub(crate) fn to_device(&mut self) {
        self.norm1.to_device();
        self.conv1.to_device();
        self.norm2.to_device();
        self.conv2.to_device();
        if let Some(c) = self.conv_shortcut.as_mut() {
            c.to_device();
        }
    }

    pub(crate) fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let h = self.norm1.forward(input);
        let h = silu_inplace::<B>(&h);
        let h = self.conv1.forward(&h);
        let h = self.norm2.forward(&h);
        let h = silu_inplace::<B>(&h);
        let h = self.conv2.forward(&h);

        let shortcut = match &self.conv_shortcut {
            Some(c) => c.forward(input),
            None => clone_tensor::<B>(input),
        };
        add_same_shape::<B>(&shortcut, &h)
    }
}

// -----------------------------------------------------------------------
// Mid-block self-attention
// -----------------------------------------------------------------------

/// VAE mid-block self-attention. Single-head, channels-as-features —
/// HF's `AttnProcessor` with `heads=1`. Used by both encoder and decoder
/// at their respective `mid_block.attentions.0` slot.
pub(crate) struct VaeMidAttention<B: MathBackend> {
    group_norm: GroupNormParams<B>,
    q_weight: Tensor<B>, // [C, C], scry-llm [in, out] convention
    q_bias: Tensor<B>,
    k_weight: Tensor<B>,
    k_bias: Tensor<B>,
    v_weight: Tensor<B>,
    v_bias: Tensor<B>,
    proj_weight: Tensor<B>,
    proj_bias: Tensor<B>,
    channels: usize,
}

impl<B: MathBackend> VaeMidAttention<B> {
    pub(crate) fn to_device(&mut self) {
        self.group_norm.to_device();
        B::to_device_in_place(&mut self.q_weight.data);
        B::to_device_in_place(&mut self.q_bias.data);
        B::to_device_in_place(&mut self.k_weight.data);
        B::to_device_in_place(&mut self.k_bias.data);
        B::to_device_in_place(&mut self.v_weight.data);
        B::to_device_in_place(&mut self.v_bias.data);
        B::to_device_in_place(&mut self.proj_weight.data);
        B::to_device_in_place(&mut self.proj_bias.data);
    }

    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let dims = input.shape.dims();
        let (c, h, w) = (dims[0], dims[1], dims[2]);
        debug_assert_eq!(c, self.channels);
        let n = h * w;

        // Pre-norm.
        let normed = self.group_norm.forward(input);

        // Reshape [C, H, W] -> [H*W, C] (transpose).
        let flat = transpose_chw_to_hwc::<B>(&normed, c, n);

        // Q, K, V projections in [H*W, C] layout.
        let q = matmul_bias_2d::<B>(&flat, &self.q_weight, &self.q_bias, n, c, c);
        let k = matmul_bias_2d::<B>(&flat, &self.k_weight, &self.k_bias, n, c, c);
        let v = matmul_bias_2d::<B>(&flat, &self.v_weight, &self.v_bias, n, c, c);

        // Single-head scaled dot-product attention.
        // scores = Q @ K^T → [H*W, H*W], scaled.
        let scores_raw = B::matmul(&q.data, &k.data, n, c, n, false, true);
        let scale = 1.0 / (c as f32).sqrt();
        let attn = B::scaled_softmax(&scores_raw, scale, &Shape::new(&[n, n]));
        // out = attn @ V → [H*W, C]
        let out_data = B::matmul(&attn, &v.data, n, n, c, false, false);
        let out = Tensor::new(out_data, Shape::new(&[n, c]));

        // Output projection.
        let proj = matmul_bias_2d::<B>(&out, &self.proj_weight, &self.proj_bias, n, c, c);

        // Reshape [H*W, C] -> [C, H, W] and add residual.
        let proj_chw = transpose_hwc_to_chw::<B>(&proj, n, c, h, w);
        add_same_shape::<B>(input, &proj_chw)
    }
}

// -----------------------------------------------------------------------
// Safetensors loaders
// -----------------------------------------------------------------------

#[cfg(feature = "safetensors")]
pub(crate) fn load_resnet_block<B: MathBackend>(
    view: &safetensors::SafeTensors<'_>,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    num_groups: usize,
    consume: &mut impl FnMut(&str),
) -> Result<VaeResnetBlock<B>> {
    use scry_vision::checkpoint::{load_conv2d_with_bias, load_tensor};

    let norm1 = GroupNormParams {
        weight: load_tensor::<B>(view, &format!("{prefix}.norm1.weight"), &[in_channels])
            .map_err(|e| Error::Llm(format!("load {prefix}.norm1.weight: {e}")))?,
        bias: load_tensor::<B>(view, &format!("{prefix}.norm1.bias"), &[in_channels])
            .map_err(|e| Error::Llm(format!("load {prefix}.norm1.bias: {e}")))?,
        num_groups,
        channels: in_channels,
    };
    consume(&format!("{prefix}.norm1.weight"));
    consume(&format!("{prefix}.norm1.bias"));

    let conv1 = load_conv2d_with_bias::<B>(
        view,
        &format!("{prefix}.conv1"),
        in_channels,
        out_channels,
        3,
        3,
        1,
        1,
    )
    .map_err(|e| Error::Llm(format!("load {prefix}.conv1: {e}")))?;
    consume(&format!("{prefix}.conv1.weight"));
    consume(&format!("{prefix}.conv1.bias"));

    let norm2 = GroupNormParams {
        weight: load_tensor::<B>(view, &format!("{prefix}.norm2.weight"), &[out_channels])
            .map_err(|e| Error::Llm(format!("load {prefix}.norm2.weight: {e}")))?,
        bias: load_tensor::<B>(view, &format!("{prefix}.norm2.bias"), &[out_channels])
            .map_err(|e| Error::Llm(format!("load {prefix}.norm2.bias: {e}")))?,
        num_groups,
        channels: out_channels,
    };
    consume(&format!("{prefix}.norm2.weight"));
    consume(&format!("{prefix}.norm2.bias"));

    let conv2 = load_conv2d_with_bias::<B>(
        view,
        &format!("{prefix}.conv2"),
        out_channels,
        out_channels,
        3,
        3,
        1,
        1,
    )
    .map_err(|e| Error::Llm(format!("load {prefix}.conv2: {e}")))?;
    consume(&format!("{prefix}.conv2.weight"));
    consume(&format!("{prefix}.conv2.bias"));

    let conv_shortcut = if in_channels == out_channels {
        None
    } else {
        let key = format!("{prefix}.conv_shortcut");
        let conv = load_conv2d_with_bias::<B>(view, &key, in_channels, out_channels, 1, 1, 1, 0)
            .map_err(|e| Error::Llm(format!("load {key}: {e}")))?;
        consume(&format!("{key}.weight"));
        consume(&format!("{key}.bias"));
        Some(conv)
    };

    Ok(VaeResnetBlock {
        norm1,
        conv1,
        norm2,
        conv2,
        conv_shortcut,
    })
}

#[cfg(feature = "safetensors")]
pub(crate) fn load_mid_attention<B: MathBackend>(
    view: &safetensors::SafeTensors<'_>,
    prefix: &str,
    channels: usize,
    num_groups: usize,
    consume: &mut impl FnMut(&str),
) -> Result<VaeMidAttention<B>> {
    use scry_vision::checkpoint::{load_f32, load_tensor};

    let group_norm = GroupNormParams {
        weight: load_tensor::<B>(view, &format!("{prefix}.group_norm.weight"), &[channels])
            .map_err(|e| Error::Llm(format!("load {prefix}.group_norm.weight: {e}")))?,
        bias: load_tensor::<B>(view, &format!("{prefix}.group_norm.bias"), &[channels])
            .map_err(|e| Error::Llm(format!("load {prefix}.group_norm.bias: {e}")))?,
        num_groups,
        channels,
    };
    consume(&format!("{prefix}.group_norm.weight"));
    consume(&format!("{prefix}.group_norm.bias"));

    // q/k/v/proj_attn: HF stores [out=C, in=C] (PyTorch nn.Linear). We
    // need scry-llm's [in, out] convention, so transpose-on-load.
    let load_lin = |stem: &str, consume: &mut dyn FnMut(&str)| -> Result<(Tensor<B>, Tensor<B>)> {
        let w_key = format!("{prefix}.{stem}.weight");
        let b_key = format!("{prefix}.{stem}.bias");
        let raw = load_f32(view, &w_key).map_err(|e| Error::Llm(format!("load {w_key}: {e}")))?;
        if raw.len() != channels * channels {
            return Err(Error::Llm(format!(
                "{w_key}: expected {} elems, got {}",
                channels * channels,
                raw.len()
            )));
        }
        let mut t = vec![0.0f32; channels * channels];
        for in_i in 0..channels {
            for out_i in 0..channels {
                t[in_i * channels + out_i] = raw[out_i * channels + in_i];
            }
        }
        let weight = Tensor::from_vec(t, Shape::new(&[channels, channels]));
        let bias = load_tensor::<B>(view, &b_key, &[channels])
            .map_err(|e| Error::Llm(format!("load {b_key}: {e}")))?;
        consume(&w_key);
        consume(&b_key);
        Ok((weight, bias))
    };

    let (q_weight, q_bias) = load_lin("query", &mut |k| consume(k))?;
    let (k_weight, k_bias) = load_lin("key", &mut |k| consume(k))?;
    let (v_weight, v_bias) = load_lin("value", &mut |k| consume(k))?;
    let (proj_weight, proj_bias) = load_lin("proj_attn", &mut |k| consume(k))?;

    Ok(VaeMidAttention {
        group_norm,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        proj_weight,
        proj_bias,
        channels,
    })
}

// -----------------------------------------------------------------------
// Tiny tensor helpers — shared by encoder, decoder, and the blocks above.
// -----------------------------------------------------------------------

pub(crate) fn silu_inplace<B: MathBackend>(t: &Tensor<B>) -> Tensor<B> {
    Tensor::new(B::silu(&t.data), t.shape.clone())
}

/// Materialize a fresh tensor without round-tripping through host. Mirrors
/// `unet/common.rs::clone_tensor` — `B::scale(_, 1.0)` is a no-op multiply
/// that produces a new device-resident storage on `ScryGpuBackend`.
pub(crate) fn clone_tensor<B: MathBackend>(t: &Tensor<B>) -> Tensor<B> {
    let storage = B::scale(&t.data, 1.0);
    Tensor::new(storage, t.shape.clone())
}

pub(crate) fn add_same_shape<B: MathBackend>(a: &Tensor<B>, b: &Tensor<B>) -> Tensor<B> {
    debug_assert_eq!(a.shape.dims(), b.shape.dims());
    let out = B::add(&a.data, &b.data, &a.shape, &b.shape, &a.shape);
    Tensor::new(out, a.shape.clone())
}

pub(crate) fn matmul_bias_2d<B: MathBackend>(
    a: &Tensor<B>,
    weight: &Tensor<B>,
    bias: &Tensor<B>,
    m: usize,
    k: usize,
    n: usize,
) -> Tensor<B> {
    let out = B::matmul_bias(&a.data, &weight.data, &bias.data, m, k, n, false, false);
    Tensor::new(out, Shape::new(&[m, n]))
}

/// `[C, H, W]` (NCHW flat) → `[H*W, C]`. Mirrors
/// `unet/common.rs::transpose_chw_to_hwc` — viewing the input as `[C, H*W]`
/// and routing through `B::transpose_2d` keeps the work on-device.
pub(crate) fn transpose_chw_to_hwc<B: MathBackend>(t: &Tensor<B>, c: usize, n: usize) -> Tensor<B> {
    debug_assert_eq!(t.shape.numel(), c * n);
    let storage = B::transpose_2d(&t.data, c, n);
    Tensor::new(storage, Shape::new(&[n, c]))
}

/// `[H*W, C]` → `[C, H, W]`. Same `B::transpose_2d` dispatch as
/// [`transpose_chw_to_hwc`]; the result-shape `[c, h, w]` is the
/// contiguous reinterpretation of `[c, h*w]`.
pub(crate) fn transpose_hwc_to_chw<B: MathBackend>(
    t: &Tensor<B>,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Tensor<B> {
    debug_assert_eq!(n, h * w);
    debug_assert_eq!(t.shape.numel(), c * n);
    let storage = B::transpose_2d(&t.data, n, c);
    Tensor::new(storage, Shape::new(&[c, h, w]))
}
