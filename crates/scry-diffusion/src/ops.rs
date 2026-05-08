// SPDX-License-Identifier: MIT OR Apache-2.0
//! Small helper ops shared by the UNet and VAE.
//!
//! Thin wrappers over the `MathBackend` trait methods that the scaffold's
//! UNet / VAE / scheduler call sites use. The kernels themselves live in
//! `scry-gpu` (CUDA) and `scry-llm` (trait + CPU default) — see HANDOFF M1.
//! Shape parsing here keeps the model code from having to thread (channels,
//! spatial) into every call.

use scry_llm::backend::MathBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use crate::error::{Error, Result};

/// Parse an NCHW input shape into `(batch, channels, h, w)`.
///
/// Accepts either a 3-D `[C, H, W]` (treated as `batch=1`) or 4-D
/// `[N, C, H, W]` shape. Returns an error for any other rank.
fn parse_nchw(shape: &Shape) -> Result<(usize, usize, usize, usize)> {
    let dims = shape.dims();
    match dims {
        [c, h, w] => Ok((1, *c, *h, *w)),
        [n, c, h, w] => Ok((*n, *c, *h, *w)),
        _ => Err(Error::Llm(format!(
            "expected 3-D [C,H,W] or 4-D [N,C,H,W] tensor, got rank-{} {:?}",
            dims.len(),
            dims
        ))),
    }
}

/// Sinusoidal timestep embedding, identical to the one in HF diffusers'
/// `Timesteps` module.
///
/// Produces a `[d_model]` embedding for a single integer timestep. Half the
/// dims encode `cos(t * freq)` and half encode `sin(t * freq)` with
/// log-spaced frequencies. CPU-side: the work is `d_model` floats per call,
/// dwarfed by the immediately-following Linear projection — no GPU kernel
/// needed.
///
/// Reference: `diffusers/src/diffusers/models/embeddings.py::Timesteps`,
/// which uses the order `[cos(...), sin(...)]` (not `[sin, cos]`).
pub fn sinusoidal_timestep_embedding<B: MathBackend>(
    timestep: f32,
    d_model: usize,
    max_period: f32,
) -> Result<Tensor<B>> {
    if d_model == 0 {
        return Err(Error::Llm(
            "sinusoidal_timestep_embedding: d_model = 0".into(),
        ));
    }
    let half = d_model / 2;
    let mut emb = vec![0.0f32; d_model];
    if half > 0 {
        let log_max = max_period.ln();
        for i in 0..half {
            let freq = (-log_max * (i as f32) / (half as f32)).exp();
            let arg = timestep * freq;
            // diffusers order: [cos, sin] then concat — matches their
            // `get_timestep_embedding` with flip_sin_to_cos=True.
            emb[i] = arg.cos();
            emb[half + i] = arg.sin();
        }
    }
    // Odd `d_model`: trailing slot stays 0 — matches diffusers' zero-pad
    // when `embedding_dim` is odd.
    Ok(Tensor::from_vec(emb, Shape::new(&[d_model])))
}

/// Nearest-neighbor 2D upsample by an integer factor, NCHW layout.
///
/// SD UNet UpBlocks call this between resolution stages. PyTorch's default is
/// nearest-neighbor (no interpolation), and we match. Accepts 3-D `[C, H, W]`
/// or 4-D `[N, C, H, W]` input; output preserves the rank.
pub fn upsample_2d_nearest<B: MathBackend>(
    input: &Tensor<B>,
    scale_factor: u32,
) -> Result<Tensor<B>> {
    if scale_factor == 0 {
        return Err(Error::Llm("upsample_2d_nearest: scale_factor = 0".into()));
    }
    let (batch, channels, h_in, w_in) = parse_nchw(&input.shape)?;
    let scale = scale_factor as usize;
    // Spatial-only op: batched input flattens to (batch*channels) planes
    // without any cross-channel mixing.
    let merged_channels = batch * channels;
    let out_storage = B::upsample_2d_nearest(&input.data, merged_channels, h_in, w_in, scale);
    let h_out = h_in * scale;
    let w_out = w_in * scale;
    let shape = if input.shape.dims().len() == 3 {
        Shape::new(&[channels, h_out, w_out])
    } else {
        Shape::new(&[batch, channels, h_out, w_out])
    };
    Ok(Tensor::new(out_storage, shape))
}

/// Apply SiLU / Swish activation: `x * sigmoid(x)`.
///
/// Used everywhere in SD UNet ResBlocks. Shape-preserving thin wrapper over
/// [`MathBackend::silu`].
pub fn silu<B: MathBackend>(input: &Tensor<B>) -> Result<Tensor<B>> {
    let out = B::silu(&input.data);
    Ok(Tensor::new(out, input.shape.clone()))
}

/// Apply GroupNorm with `num_groups` groups along the channel dim.
///
/// SD UNet uses `num_groups = 32` everywhere. Accepts 3-D `[C, H, W]` or 4-D
/// `[N, C, H, W]` input; the same shape is preserved on the output.
/// `weight` and `bias` must each be length `channels`.
pub fn group_norm<B: MathBackend>(
    input: &Tensor<B>,
    num_groups: usize,
    weight: &Tensor<B>,
    bias: &Tensor<B>,
    eps: f32,
) -> Result<Tensor<B>> {
    if num_groups == 0 {
        return Err(Error::Llm("group_norm: num_groups = 0".into()));
    }
    let (_batch, channels, h, w) = parse_nchw(&input.shape)?;
    if channels % num_groups != 0 {
        return Err(Error::Llm(format!(
            "group_norm: channels={channels} not divisible by num_groups={num_groups}"
        )));
    }
    if weight.shape.numel() != channels || bias.shape.numel() != channels {
        return Err(Error::Llm(format!(
            "group_norm: weight/bias must be length channels={}, got weight={} bias={}",
            channels,
            weight.shape.numel(),
            bias.shape.numel()
        )));
    }
    let spatial = h * w;
    let out = B::group_norm(
        &input.data,
        &weight.data,
        &bias.data,
        num_groups,
        channels,
        spatial,
        eps,
    );
    Ok(Tensor::new(out, input.shape.clone()))
}
