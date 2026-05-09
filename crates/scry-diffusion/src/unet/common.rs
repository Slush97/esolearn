// SPDX-License-Identifier: MIT OR Apache-2.0
//! Helpers shared across `unet/{resblock, attention, blocks, mod}.rs`:
//! [`GroupNormParams`] holding affine weights for `MathBackend::group_norm`,
//! and HF-`[out, in]` → scry-llm-`[in, out]` linear loaders that handle the
//! transpose-on-load while consuming the safetensors keys for the 100%
//! consumption check. Kept private to the crate because the UNet is the
//! only caller for these specific shapes (the VAE decoder rolls its own
//! local transpose helpers).

#[cfg(feature = "safetensors")]
use std::collections::HashSet;

use scry_llm::backend::MathBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

#[cfg(feature = "safetensors")]
use crate::error::{Error, Result};

/// GroupNorm eps used across the SD UNet. HF diffusers sets this to 1e-5
/// for the UNet (vs 1e-6 in the VAE) — same value across SD 1.5 / 2.x / SDXL.
const NORM_EPS: f32 = 1e-5;

/// 1D GroupNorm parameter pack. Compute goes through
/// `MathBackend::group_norm`; this struct just holds the affine weights
/// and shape metadata so callers don't have to plumb `num_groups` /
/// `channels` separately.
pub(crate) struct GroupNormParams<B: MathBackend> {
    pub(crate) weight: Tensor<B>,
    pub(crate) bias: Tensor<B>,
    pub(crate) num_groups: usize,
    pub(crate) channels: usize,
}

impl<B: MathBackend> GroupNormParams<B> {
    /// Apply GroupNorm to a `[C, H, W]` tensor (batch=1). Returns same shape.
    pub(crate) fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let dims = input.shape.dims();
        debug_assert_eq!(dims.len(), 3);
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
// Tensor helpers shared across UNet submodules.
//
// These mirror the ones VaeDecoder rolls locally — we keep a UNet copy
// rather than promoting them to a public API because the shapes/conventions
// are specific to the SD UNet's batch=1, NCHW path.
// -----------------------------------------------------------------------

/// SiLU on a single tensor (`x * sigmoid(x)`), shape-preserving.
pub(crate) fn silu<B: MathBackend>(t: &Tensor<B>) -> Tensor<B> {
    Tensor::new(B::silu(&t.data), t.shape.clone())
}

/// Elementwise add of two same-shape tensors.
pub(crate) fn add_same<B: MathBackend>(a: &Tensor<B>, b: &Tensor<B>) -> Tensor<B> {
    debug_assert_eq!(a.shape.dims(), b.shape.dims());
    let out = B::add(&a.data, &b.data, &a.shape, &b.shape, &a.shape);
    Tensor::new(out, a.shape.clone())
}

/// Round-trip clone via `to_vec`/`from_vec`. Used where we need to keep
/// an intermediate alive across an op that consumes the tensor.
pub(crate) fn clone_tensor<B: MathBackend>(t: &Tensor<B>) -> Tensor<B> {
    let v = B::to_vec(&t.data);
    Tensor::from_vec(v, t.shape.clone())
}

/// `C = A @ W + b`, with `A: [m, k]`, `W: [k, n]` (scry-llm `[in, out]`),
/// `b: [n]`. Output shape `[m, n]`.
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

/// Concatenate two `[C_i, H, W]` tensors along the channel axis, host-side.
/// Both inputs must share spatial dims; output is `[C1+C2, H, W]`. The
/// SD UNet's UpBlock skip-concat is the only caller.
pub(crate) fn concat_channels<B: MathBackend>(a: &Tensor<B>, b: &Tensor<B>) -> Tensor<B> {
    let ad = a.shape.dims();
    let bd = b.shape.dims();
    debug_assert_eq!(ad.len(), 3);
    debug_assert_eq!(bd.len(), 3);
    debug_assert_eq!(ad[1], bd[1]);
    debug_assert_eq!(ad[2], bd[2]);
    let mut va = B::to_vec(&a.data);
    let vb = B::to_vec(&b.data);
    va.extend_from_slice(&vb);
    Tensor::from_vec(va, Shape::new(&[ad[0] + bd[0], ad[1], ad[2]]))
}

/// Reshape `[C, H, W]` (NCHW row-major) into `[H*W, C]` (HWC row-major).
/// This is a transpose, not a reshape — channel data is interleaved per
/// spatial location in the output.
pub(crate) fn transpose_chw_to_hwc<B: MathBackend>(t: &Tensor<B>, c: usize, n: usize) -> Tensor<B> {
    let v = B::to_vec(&t.data);
    debug_assert_eq!(v.len(), c * n);
    let mut out = vec![0.0f32; v.len()];
    for ci in 0..c {
        for ni in 0..n {
            out[ni * c + ci] = v[ci * n + ni];
        }
    }
    Tensor::from_vec(out, Shape::new(&[n, c]))
}

/// Reshape `[H*W, C]` back to `[C, H, W]`.
pub(crate) fn transpose_hwc_to_chw<B: MathBackend>(
    t: &Tensor<B>,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Tensor<B> {
    debug_assert_eq!(n, h * w);
    let v = B::to_vec(&t.data);
    debug_assert_eq!(v.len(), c * n);
    let mut out = vec![0.0f32; v.len()];
    for ni in 0..n {
        for ci in 0..c {
            out[ci * n + ni] = v[ni * c + ci];
        }
    }
    Tensor::from_vec(out, Shape::new(&[c, h, w]))
}

/// Exact GELU (erf-based) on a tensor. Used by the UNet's GeGLU
/// feed-forward, which HF computes with `F.gelu(approximate="none")` —
/// the tanh approximation in `MathBackend::gelu` drifts ~3e-4 at the
/// peak of the curve and cumulatively pushes the M6 parity gate just
/// over 1e-3 across SD 1.5's 16 transformer blocks.
///
/// Delegates to [`MathBackend::gelu_exact`], so on `ScryGpuBackend` the
/// computation stays on the device. Profiling at 512×512 / bf16 found
/// this op was the single largest cost in the UNet forward (~18.5%) when
/// it round-tripped through host; routing through the trait keeps the
/// gate tensor (up to `[4096, 5120]` = 80 MB at the deepest stage) on
/// the GPU through the dispatch.
pub(crate) fn exact_gelu<B: MathBackend>(t: &Tensor<B>) -> Tensor<B> {
    let out = B::gelu_exact(&t.data);
    Tensor::new(out, t.shape.clone())
}

/// Add a per-channel bias `[C]` into a `[C, H, W]` tensor (broadcast over
/// the spatial axes). Used by ResBlock to inject the timestep projection.
pub(crate) fn add_channel_bias<B: MathBackend>(t: &Tensor<B>, bias: &Tensor<B>) -> Tensor<B> {
    let dims = t.shape.dims();
    debug_assert_eq!(dims.len(), 3);
    debug_assert_eq!(bias.shape.numel(), dims[0]);
    let c = dims[0];
    let spatial = dims[1] * dims[2];
    let lhs_shape = Shape::new(&[c, spatial]);
    let bias_shape = Shape::new(&[c, 1]);
    let out = B::add(&t.data, &bias.data, &lhs_shape, &bias_shape, &lhs_shape);
    Tensor::new(out, t.shape.clone())
}

/// Load `{prefix}.weight` + `{prefix}.bias` as a GroupNorm affine pair and
/// register both keys for the consumption check.
#[cfg(feature = "safetensors")]
pub(crate) fn load_group_norm<B: MathBackend>(
    view: &safetensors::SafeTensors<'_>,
    prefix: &str,
    channels: usize,
    num_groups: usize,
    consume: &mut impl FnMut(&str),
) -> Result<GroupNormParams<B>> {
    use scry_vision::checkpoint::load_tensor;

    let w_key = format!("{prefix}.weight");
    let b_key = format!("{prefix}.bias");
    let weight = load_tensor::<B>(view, &w_key, &[channels])
        .map_err(|e| Error::Llm(format!("load {w_key}: {e}")))?;
    let bias = load_tensor::<B>(view, &b_key, &[channels])
        .map_err(|e| Error::Llm(format!("load {b_key}: {e}")))?;
    consume(&w_key);
    consume(&b_key);
    Ok(GroupNormParams {
        weight,
        bias,
        num_groups,
        channels,
    })
}

/// Load a HF Linear (`{prefix}.weight: [out, in]`, optional `bias: [out]`)
/// into scry-llm's `[in, out]` weight layout, transposing the weight
/// once at load time. `has_bias = false` is used by the SD attention
/// `to_q` / `to_k` / `to_v` projections (HF `nn.Linear(bias=False)`).
///
/// Returns `(weight [in, out], bias_or_zeros [out])`. When `has_bias`
/// is true, both `{prefix}.weight` and `{prefix}.bias` are consumed;
/// when false, only `{prefix}.weight` is consumed and the bias tensor
/// is filled with zeros so downstream `matmul_bias` calls can stay
/// unconditional.
#[cfg(feature = "safetensors")]
pub(crate) fn load_linear<B: MathBackend>(
    view: &safetensors::SafeTensors<'_>,
    prefix: &str,
    in_features: usize,
    out_features: usize,
    has_bias: bool,
    consume: &mut impl FnMut(&str),
) -> Result<(Tensor<B>, Tensor<B>)> {
    use scry_vision::checkpoint::{load_f32, load_tensor};

    let w_key = format!("{prefix}.weight");
    let raw = load_f32(view, &w_key).map_err(|e| Error::Llm(format!("load {w_key}: {e}")))?;
    let expected = out_features * in_features;
    if raw.len() != expected {
        return Err(Error::Llm(format!(
            "{w_key}: expected {expected} elements ({out_features}×{in_features}), got {}",
            raw.len()
        )));
    }
    let mut transposed = vec![0.0f32; expected];
    for in_i in 0..in_features {
        for out_i in 0..out_features {
            transposed[in_i * out_features + out_i] = raw[out_i * in_features + in_i];
        }
    }
    let weight = Tensor::from_vec(transposed, Shape::new(&[in_features, out_features]));
    consume(&w_key);

    let bias = if has_bias {
        let b_key = format!("{prefix}.bias");
        let b = load_tensor::<B>(view, &b_key, &[out_features])
            .map_err(|e| Error::Llm(format!("load {b_key}: {e}")))?;
        consume(&b_key);
        b
    } else {
        Tensor::from_vec(vec![0.0; out_features], Shape::new(&[out_features]))
    };
    Ok((weight, bias))
}

/// Load one HF linear projection of HF shape `[2·M, in]` and split it
/// column-wise into **two** `[in, M]` weights (in scry-llm convention).
/// Bias is similarly split into two `[M]` halves.
///
/// Used to convert the HF GeGLU `ff.net.0.proj` (HF
/// `[2·d_ff, d_model]` → scry-llm `[d_model, 2·d_ff]`) into two
/// independent `[d_model, d_ff]` matmuls. Empirically, two narrower
/// matmuls hit a faster cuBLAS algorithm than one wide one at SD
/// shapes — and the post-projection `gather_columns` step disappears
/// because each matmul produces exactly the right half directly.
///
/// The first returned `(weight, bias)` pair corresponds to HF rows
/// `[0, M)`; the second to rows `[M, 2·M)`. For HF GeGLU that means
/// **first = values, second = gate**, matching `tensor.chunk(2, dim=-1)`.
#[cfg(feature = "safetensors")]
pub(crate) fn load_split2_linear<B: MathBackend>(
    view: &safetensors::SafeTensors<'_>,
    prefix: &str,
    in_features: usize,
    out_features: usize,
    has_bias: bool,
    consume: &mut impl FnMut(&str),
) -> Result<((Tensor<B>, Tensor<B>), (Tensor<B>, Tensor<B>))> {
    use scry_vision::checkpoint::load_f32;

    let two_m = 2 * out_features;
    let w_key = format!("{prefix}.weight");
    let raw = load_f32(view, &w_key).map_err(|e| Error::Llm(format!("load {w_key}: {e}")))?;
    let expected = two_m * in_features;
    if raw.len() != expected {
        return Err(Error::Llm(format!(
            "{w_key}: expected {expected} elements ({two_m}×{in_features}), got {}",
            raw.len()
        )));
    }
    // Transpose-and-split in one pass. HF row r holds output unit r;
    // we want our `[in, M]` matrices indexed by [in_i * M + out_i].
    // First half (values): HF rows [0, M); second half (gate): rows [M, 2M).
    let total = in_features * out_features;
    let mut w_a = vec![0.0f32; total];
    let mut w_b = vec![0.0f32; total];
    for in_i in 0..in_features {
        let dst_row = in_i * out_features;
        for out_i in 0..out_features {
            w_a[dst_row + out_i] = raw[out_i * in_features + in_i];
            w_b[dst_row + out_i] = raw[(out_features + out_i) * in_features + in_i];
        }
    }
    let weight_a = Tensor::from_vec(w_a, Shape::new(&[in_features, out_features]));
    let weight_b = Tensor::from_vec(w_b, Shape::new(&[in_features, out_features]));
    consume(&w_key);

    let (bias_a, bias_b) = if has_bias {
        let b_key = format!("{prefix}.bias");
        let raw_bias = load_f32(view, &b_key)
            .map_err(|e| Error::Llm(format!("load {b_key}: {e}")))?;
        if raw_bias.len() != two_m {
            return Err(Error::Llm(format!(
                "{b_key}: expected {two_m} elements, got {}",
                raw_bias.len()
            )));
        }
        let ba: Vec<f32> = raw_bias[..out_features].to_vec();
        let bb: Vec<f32> = raw_bias[out_features..].to_vec();
        consume(&b_key);
        (
            Tensor::from_vec(ba, Shape::new(&[out_features])),
            Tensor::from_vec(bb, Shape::new(&[out_features])),
        )
    } else {
        (
            Tensor::from_vec(vec![0.0; out_features], Shape::new(&[out_features])),
            Tensor::from_vec(vec![0.0; out_features], Shape::new(&[out_features])),
        )
    };

    Ok(((weight_a, bias_a), (weight_b, bias_b)))
}

/// Sort + truncate a missing-key set into a `cargo test`-readable error.
/// Used at the tail of every from_safetensors loader.
#[cfg(feature = "safetensors")]
pub(crate) fn missing_keys_error(
    label: &str,
    relevant: &HashSet<String>,
    consumed: &HashSet<String>,
) -> Option<Error> {
    let missing: Vec<String> = relevant.difference(consumed).cloned().collect();
    if missing.is_empty() {
        return None;
    }
    let mut sorted = missing;
    sorted.sort();
    let head = sorted
        .iter()
        .take(8)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    let tail = if sorted.len() > 8 { ", ..." } else { "" };
    Some(Error::Llm(format!(
        "{label}: {} keys not consumed by loader: {head}{tail}",
        sorted.len()
    )))
}
