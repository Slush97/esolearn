// SPDX-License-Identifier: MIT OR Apache-2.0
//! Small helper ops shared by the UNet and VAE.
//!
//! Operations that aren't yet first-class on `MathBackend` but are small
//! enough to live as crate-local helpers. The intent is to *promote* these
//! to `MathBackend` methods (with CUDA kernels) once usage is settled — see
//! `HANDOFF.md` M1 for the kernels that need to land in scry-llm/scry-gpu.

use scry_llm::backend::MathBackend;
use scry_llm::tensor::Tensor;

use crate::error::Result;

/// Sinusoidal timestep embedding, identical to the one in HF diffusers'
/// `Timesteps` module.
///
/// Produces a `[d_model]` embedding for a single integer timestep. Half the
/// dims encode `sin(t * freq)` and half encode `cos(t * freq)` with
/// log-spaced frequencies in `[1, max_period]`. CPU-side: the work is
/// `d_model` floats per call, dwarfed by the immediately-following Linear
/// projection — no GPU kernel needed.
///
/// Reference: `diffusers/src/diffusers/models/embeddings.py::Timesteps`.
pub fn sinusoidal_timestep_embedding<B: MathBackend>(
    timestep: f32,
    d_model: usize,
    max_period: f32,
) -> Result<Tensor<B>> {
    let _ = (timestep, d_model, max_period);
    todo!("M1: sinusoidal embedding helper (CPU-side; ~hundreds of bytes per call)")
}

/// Nearest-neighbor 2D upsample by an integer factor, NCHW layout.
///
/// SD UNet UpBlocks call this between resolution stages. PyTorch's default is
/// nearest-neighbor (no interpolation), and we match. Once a kernel lands
/// (`UPSAMPLE_2D_NN_CUDA`, see HANDOFF M1), this helper becomes a thin
/// `MathBackend::upsample_2d_nearest` wrapper.
pub fn upsample_2d_nearest<B: MathBackend>(
    input: &Tensor<B>,
    scale_factor: u32,
) -> Result<Tensor<B>> {
    let _ = (input, scale_factor);
    todo!("M1: upsample_2d_nearest — kernel goes in scry-gpu, MathBackend method in scry-llm")
}

/// Apply SiLU / Swish activation: `x * sigmoid(x)`.
///
/// Used everywhere in SD UNet ResBlocks. We have GELU as a backend op but
/// not SiLU yet — see HANDOFF M1 for the kernel.
pub fn silu<B: MathBackend>(input: &Tensor<B>) -> Result<Tensor<B>> {
    let _ = input;
    todo!("M1: SiLU kernel + MathBackend method")
}

/// Apply GroupNorm with `num_groups` groups along the channel dim.
///
/// SD UNet uses `num_groups = 32` everywhere. Math: split channels into
/// `num_groups` groups, compute mean/variance per `(batch, group)`, normalize,
/// then apply per-channel affine. See HANDOFF M1 for the kernel design (one
/// block per `(batch, group)`, two block-wide reductions like layernorm).
pub fn group_norm<B: MathBackend>(
    input: &Tensor<B>,
    num_groups: usize,
    weight: &Tensor<B>,
    bias: &Tensor<B>,
    eps: f32,
) -> Result<Tensor<B>> {
    let _ = (input, num_groups, weight, bias, eps);
    todo!("M1: GroupNorm kernel + MathBackend method")
}
