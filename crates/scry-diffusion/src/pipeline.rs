// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end txt2img pipeline.
//!
//! Wires tokenizer → text encoder → scheduler → UNet → VAE decoder. Scope is
//! text-to-image today; img2img and inpainting reuse the same UNet + VAE
//! plumbing once a VAE encoder and an init-latent path are added (M10+).

use scry_llm::backend::MathBackend;
use scry_llm::tensor::Tensor;

use crate::error::Result;
use crate::scheduler::Scheduler;
use crate::text_encoder::TextEncoder;
use crate::tokenizer::Tokenizer;
use crate::unet::Unet;
use crate::vae::VaeDecoder;

/// Inputs to a single generation.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    /// User-provided text prompt.
    pub prompt: String,
    /// Negative prompt for classifier-free guidance. Empty disables.
    pub negative_prompt: String,
    /// Number of denoising steps.
    pub num_inference_steps: u32,
    /// CFG scale. 1.0 disables CFG (single forward); 7-9 is typical for SD 1.5.
    pub guidance_scale: f32,
    /// PRNG seed for the initial latent noise.
    pub seed: u64,
    /// Output `(width, height)`. Must be multiples of 8 (VAE downsample factor).
    pub size: (u32, u32),
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            num_inference_steps: 30,
            guidance_scale: 7.5,
            seed: 0,
            size: (512, 512),
        }
    }
}

/// Top-level txt2img pipeline.
pub struct Txt2ImgPipeline<B, T, S>
where
    B: MathBackend,
    T: TextEncoder<B>,
    S: Scheduler,
{
    /// CLIP BPE tokenizer.
    pub tokenizer: Tokenizer,
    /// Text encoder producing token-level conditioning.
    pub text_encoder: T,
    /// UNet predicting noise per step.
    pub unet: Unet<B>,
    /// VAE decoder lifting latents back to RGB.
    pub vae: VaeDecoder<B>,
    /// Scheduler controlling the denoising trajectory.
    pub scheduler: S,
}

impl<B, T, S> Txt2ImgPipeline<B, T, S>
where
    B: MathBackend,
    T: TextEncoder<B>,
    S: Scheduler,
{
    /// Run a single generation and return an `[3, H, W]` RGB tensor in `[0, 1]`.
    ///
    /// Pipeline order:
    ///   1. Tokenize `prompt` and `negative_prompt`.
    ///   2. Encode both via the text encoder. Stack the two conditionings
    ///      so a single UNet forward serves both branches under CFG.
    ///   3. Initialize latents with `seed`-deterministic Gaussian noise.
    ///   4. For each scheduler timestep:
    ///       - Stack latent twice (CFG: uncond + cond), call UNet.
    ///       - Combine: `noise = uncond + guidance_scale * (cond - uncond)`.
    ///       - `scheduler.step(noise, t, latent)` → next latent.
    ///   5. VAE decode the final latent. Clamp to `[-1, 1]`, rescale to `[0, 1]`.
    pub fn generate(&mut self, params: &GenerationParams) -> Result<Tensor<B>> {
        let _ = params;
        todo!("M9: end-to-end txt2img — tokenize, encode, sample, decode (see HANDOFF.md)")
    }
}
