// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end SD pipeline — txt2img and img2img share the same struct.
//!
//! Wires tokenizer → text encoder → scheduler → UNet → VAE decoder, with an
//! optional VAE encoder for img2img / inpainting. The two entry points are
//! [`SdPipeline::generate`] (txt2img) and [`SdPipeline::img2img`]; both
//! reuse the same denoise loop after the latent is initialized.

use scry_llm::backend::MathBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use crate::error::{Error, Result};
use crate::scheduler::Scheduler;
use crate::text_encoder::TextEncoder;
use crate::tokenizer::Tokenizer;
use crate::unet::Unet;
use crate::vae::{VaeDecoder, VaeEncoder};

/// VAE downsample factor — latents are 1/8 of the output resolution.
const VAE_SCALE: u32 = 8;

/// SD's noise-prediction head emits 4-channel latents.
const LATENT_CHANNELS: usize = 4;

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

/// Inputs to a single img2img generation. The init image is passed
/// separately to [`SdPipeline::img2img`] because it is backend-typed.
#[derive(Debug, Clone)]
pub struct Img2ImgParams {
    /// User-provided text prompt.
    pub prompt: String,
    /// Negative prompt for classifier-free guidance. Empty disables.
    pub negative_prompt: String,
    /// Total denoising steps in the schedule. The actual number of UNet
    /// forwards equals `round(num_inference_steps * strength)` — img2img
    /// skips the early (most-noisy) steps proportional to `1 - strength`.
    pub num_inference_steps: u32,
    /// CFG scale. 1.0 disables CFG (single forward); 7-9 is typical for SD 1.5.
    pub guidance_scale: f32,
    /// PRNG seed for the reparameterization and add-noise tensors.
    pub seed: u64,
    /// How much of the schedule to run, in `[0, 1]`. `0.0` returns the
    /// VAE-decoded init image (no denoising); `1.0` runs the full
    /// schedule (≈ txt2img). HF's default is `0.8`.
    pub strength: f32,
}

impl Default for Img2ImgParams {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            num_inference_steps: 30,
            guidance_scale: 7.5,
            seed: 0,
            strength: 0.8,
        }
    }
}

/// Inputs to a single inpainting generation. Image and mask are passed
/// separately to [`SdPipeline::inpaint`] because they are backend-typed.
///
/// Unlike img2img, inpainting starts from pure noise (no strength
/// truncation) — the masked-image latent and downsampled mask are
/// concatenated to the noisy latent at every UNet forward, anchoring
/// the unmasked region throughout the trajectory.
#[derive(Debug, Clone)]
pub struct InpaintParams {
    /// User-provided text prompt.
    pub prompt: String,
    /// Negative prompt for classifier-free guidance. Empty disables.
    pub negative_prompt: String,
    /// Number of denoising steps.
    pub num_inference_steps: u32,
    /// CFG scale. 1.0 disables CFG (single forward); 7-9 is typical.
    pub guidance_scale: f32,
    /// PRNG seed for the initial latent noise.
    pub seed: u64,
}

impl Default for InpaintParams {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            num_inference_steps: 30,
            guidance_scale: 7.5,
            seed: 0,
        }
    }
}

/// Top-level Stable-Diffusion pipeline. Hosts both txt2img
/// ([`Self::generate`]) and img2img ([`Self::img2img`]); the img2img path
/// is only available when [`Self::vae_encoder`] is `Some`.
pub struct SdPipeline<B, T, S>
where
    B: MathBackend,
    T: TextEncoder<B>,
    S: Scheduler<B>,
{
    /// CLIP BPE tokenizer.
    pub tokenizer: Tokenizer,
    /// Text encoder producing token-level conditioning.
    pub text_encoder: T,
    /// UNet predicting noise per step.
    pub unet: Unet<B>,
    /// VAE decoder lifting latents back to RGB.
    pub vae: VaeDecoder<B>,
    /// VAE encoder for img2img. Optional — txt2img-only callers leave
    /// this `None`; [`Self::img2img`] errors when it is `None`.
    pub vae_encoder: Option<VaeEncoder<B>>,
    /// Scheduler controlling the denoising trajectory.
    pub scheduler: S,
    /// Optional progress callback fired before each denoising step.
    /// Receives `(step_index, total_steps, timestep)`.
    pub progress: Option<ProgressCallback>,
}

/// Callback type for [`SdPipeline::progress`]. Invoked once per
/// denoising step with `(step_index, total_steps, timestep)`.
pub type ProgressCallback = Box<dyn FnMut(u32, u32, f32) + Send>;

impl<B, T, S> SdPipeline<B, T, S>
where
    B: MathBackend,
    T: TextEncoder<B>,
    S: Scheduler<B>,
{
    /// Run a single generation and return an `[3, H, W]` RGB tensor in `[0, 1]`.
    ///
    /// Pipeline order:
    ///   1. Tokenize `prompt` and `negative_prompt`.
    ///   2. Encode both via the text encoder.
    ///   3. Initialize latents with `seed`-deterministic Gaussian noise,
    ///      scaled by `scheduler.init_noise_sigma()`.
    ///   4. For each scheduler timestep:
    ///       - Run UNet on the uncond branch and the cond branch separately.
    ///       - Combine: `noise = uncond + guidance_scale * (cond - uncond)`.
    ///       - `scheduler.step(noise, t, latent)` → next latent.
    ///   5. VAE decode the final latent. Clamp to `[-1, 1]`, rescale to `[0, 1]`.
    ///
    /// # Errors
    /// - Invalid params (size not divisible by 8, zero steps).
    /// - Tokenizer / text encoder / UNet / VAE / scheduler propagated failures.
    #[allow(clippy::too_many_lines)]
    pub fn generate(&mut self, params: &GenerationParams) -> Result<Tensor<B>> {
        let (width, height) = params.size;
        if width == 0 || height == 0 || width % VAE_SCALE != 0 || height % VAE_SCALE != 0 {
            return Err(Error::Llm(format!(
                "size {width}x{height} must be non-zero multiples of {VAE_SCALE}"
            )));
        }
        if params.num_inference_steps == 0 {
            return Err(Error::Scheduler("num_inference_steps must be > 0".into()));
        }

        // Enable the bf16 GemmEx fast-path globally on ScryGpuBackend so users
        // don't have to set `SCRY_GPU_MATMUL_BF16=1` to get the perf-target
        // path. Idempotent and best-effort: no-op when scry-gpu is unavailable
        // (e.g. CpuBackend pipeline) or the feature isn't compiled in. The
        // toggle is process-wide on `ScryGpuBackend` — call
        // `set_bf16_matmul(false)` after generate if you need fp32 matmul
        // for some other workload in the same process.
        #[cfg(feature = "scry-gpu-bf16")]
        {
            let _ = scry_llm::backend::scry_gpu::ScryGpuBackend::set_bf16_matmul(true);
        }
        let latent_w = (width / VAE_SCALE) as usize;
        let latent_h = (height / VAE_SCALE) as usize;
        let elements = LATENT_CHANNELS * latent_h * latent_w;

        // ---- 1-2. Tokenize + encode (cond + uncond). ----
        let cond_tokens = self.tokenizer.encode(&params.prompt)?;
        let uncond_tokens = self.tokenizer.encode(&params.negative_prompt)?;
        let cond_embed = self.text_encoder.encode(&cond_tokens)?;
        let uncond_embed = self.text_encoder.encode(&uncond_tokens)?;

        // ---- 3. Init latent. ----
        let init_sigma = self.scheduler.init_noise_sigma();
        let mut latent_host = sample_standard_normal(elements, params.seed);
        if (init_sigma - 1.0).abs() > f32::EPSILON {
            for v in &mut latent_host {
                *v *= init_sigma;
            }
        }
        let latent_shape = Shape::new(&[LATENT_CHANNELS, latent_h, latent_w]);
        let mut latent: Tensor<B> = Tensor::from_vec(latent_host, latent_shape.clone());

        // ---- 4. Denoising loop. ----
        // Latent stays on the device the backend lives on across all steps:
        // `scale_model_input`, the UNet forwards, the CFG combine, and the
        // scheduler step are all tensor-typed. CFG combine is rewritten as
        // `out = s · cond + (1 − s) · uncond` (algebraically equivalent to
        // `uncond + s · (cond − uncond)`) so it composes from existing
        // `MathBackend::scale` + `add` without a dedicated kernel.
        self.scheduler.set_timesteps(params.num_inference_steps)?;
        let timesteps: Vec<f32> = self.scheduler.timesteps().to_vec();
        let total_steps = u32::try_from(timesteps.len()).unwrap_or(u32::MAX);
        let do_cfg = params.guidance_scale > 1.0 + f32::EPSILON;
        let s = params.guidance_scale;

        for (i, &t) in timesteps.iter().enumerate() {
            if let Some(cb) = self.progress.as_mut() {
                cb(u32::try_from(i).unwrap_or(u32::MAX), total_steps, t);
            }
            let model_input = self.scheduler.scale_model_input(&latent, t)?;

            let cond_eps = self.unet.forward(&model_input, t, &cond_embed)?;
            let combined = if do_cfg {
                let uncond_eps = self.unet.forward(&model_input, t, &uncond_embed)?;
                let scaled_cond = B::scale(&cond_eps.data, s);
                let scaled_uncond = B::scale(&uncond_eps.data, 1.0 - s);
                let storage = B::add(
                    &scaled_cond,
                    &scaled_uncond,
                    &cond_eps.shape,
                    &uncond_eps.shape,
                    &cond_eps.shape,
                );
                Tensor::new(storage, cond_eps.shape.clone())
            } else {
                cond_eps
            };

            latent = self.scheduler.step(&combined, t, &latent)?;
        }

        // ---- 5. VAE decode + range remap. ----
        let decoded = self.vae.decode(&latent)?;
        let mut pixels = decoded.to_vec();
        for v in &mut pixels {
            // Clamp to (-1, 1) then rescale to [0, 1].
            *v = v.clamp(-1.0, 1.0) * 0.5 + 0.5;
        }
        let dims = decoded.shape.dims().to_vec();
        Ok(Tensor::from_vec(pixels, Shape::new(&dims)))
    }

    /// Run an img2img generation: VAE-encode `init_image` to a latent,
    /// mix with seed-deterministic noise at the strength-derived starting
    /// timestep, then run the denoise loop and decode.
    ///
    /// `init_image` must be a `[3, H, W]` tensor with values **already
    /// normalized to `[-1, 1]`** — HF's `image_processor.preprocess`
    /// does `2 * x - 1` on `[0, 1]` inputs and we keep that step out of
    /// the pipeline so the caller controls cropping / letterboxing.
    /// `H` and `W` must be multiples of 8.
    ///
    /// # Errors
    /// - [`Self::vae_encoder`] is `None`.
    /// - `strength` is outside `[0, 1]`, or `num_inference_steps == 0`.
    /// - `init_image` is not `[3, H, W]` with `H % 8 == 0`, `W % 8 == 0`.
    /// - Tokenizer / encoder / VAE / UNet / scheduler propagated failures.
    #[allow(clippy::too_many_lines)]
    pub fn img2img(&mut self, params: &Img2ImgParams, init_image: &Tensor<B>) -> Result<Tensor<B>> {
        let encoder = self.vae_encoder.as_ref().ok_or_else(|| {
            Error::Llm("img2img requires vae_encoder; pipeline was built without one".into())
        })?;
        if !(0.0..=1.0).contains(&params.strength) {
            return Err(Error::Llm(format!(
                "strength {} must be in [0, 1]",
                params.strength
            )));
        }
        if params.num_inference_steps == 0 {
            return Err(Error::Scheduler("num_inference_steps must be > 0".into()));
        }
        let img_dims = init_image.shape.dims();
        if img_dims.len() != 3 || img_dims[0] != 3 {
            return Err(Error::Llm(format!(
                "init_image must be [3, H, W], got {img_dims:?}"
            )));
        }
        let (height, width) = (img_dims[1], img_dims[2]);
        let vae_scale = VAE_SCALE as usize;
        if height == 0 || width == 0 || height % vae_scale != 0 || width % vae_scale != 0 {
            return Err(Error::Llm(format!(
                "init_image size {width}x{height} must be non-zero multiples of {VAE_SCALE}"
            )));
        }

        // Same bf16 toggle as generate() so users don't have to set
        // SCRY_GPU_MATMUL_BF16=1 to get the perf-target path.
        #[cfg(feature = "scry-gpu-bf16")]
        {
            let _ = scry_llm::backend::scry_gpu::ScryGpuBackend::set_bf16_matmul(true);
        }

        let latent_h = height / vae_scale;
        let latent_w = width / vae_scale;
        let elements = LATENT_CHANNELS * latent_h * latent_w;
        let latent_shape = Shape::new(&[LATENT_CHANNELS, latent_h, latent_w]);

        // ---- Text conditioning (cond + uncond). ----
        let cond_tokens = self.tokenizer.encode(&params.prompt)?;
        let uncond_tokens = self.tokenizer.encode(&params.negative_prompt)?;
        let cond_embed = self.text_encoder.encode(&cond_tokens)?;
        let uncond_embed = self.text_encoder.encode(&uncond_tokens)?;

        // ---- VAE encode + reparameterization. ----
        // latent = (mean + exp(0.5 · logvar) · noise) · scaling_factor.
        // logvar is small (16k–64k floats at SD shapes); exp() on host is
        // cheaper than adding a backend op for the once-per-call work.
        let (mean, logvar) = encoder.encode(init_image)?;
        let std_host: Vec<f32> = logvar
            .to_vec()
            .into_iter()
            .map(|v| (0.5 * v).exp())
            .collect();
        let std_tensor: Tensor<B> = Tensor::from_vec(std_host, logvar.shape.clone());
        let noise_enc_host = sample_standard_normal(elements, params.seed);
        let noise_enc: Tensor<B> = Tensor::from_vec(noise_enc_host, latent_shape.clone());
        let scaled_noise = B::mul_elementwise(&std_tensor.data, &noise_enc.data);
        let sampled = B::add(
            &mean.data,
            &scaled_noise,
            &mean.shape,
            &latent_shape,
            &latent_shape,
        );
        let scaling = encoder.config.scaling_factor;
        let latent_storage = B::scale(&sampled, scaling);
        let mut latent: Tensor<B> = Tensor::new(latent_storage, latent_shape.clone());

        // ---- Strength-truncated trajectory. ----
        self.scheduler.set_timesteps(params.num_inference_steps)?;
        let timesteps: Vec<f32> = self.scheduler.timesteps().to_vec();
        let n_steps = params.num_inference_steps;
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let init_timestep = (((n_steps as f32) * params.strength).round() as u32).min(n_steps);
        let t_start_idx = (n_steps - init_timestep) as usize;

        // strength == 0 → empty denoise tail; skip add_noise too.
        if t_start_idx < timesteps.len() {
            let noise_add_host = sample_standard_normal(elements, params.seed.wrapping_add(1));
            let noise_add: Tensor<B> = Tensor::from_vec(noise_add_host, latent_shape.clone());
            let t_start = timesteps[t_start_idx];
            latent = self.scheduler.add_noise(&latent, &noise_add, t_start)?;
        }

        // ---- Denoise loop over the truncated tail. ----
        let total_steps = u32::try_from(timesteps.len() - t_start_idx).unwrap_or(u32::MAX);
        let do_cfg = params.guidance_scale > 1.0 + f32::EPSILON;
        let s = params.guidance_scale;
        for (i, &t) in timesteps[t_start_idx..].iter().enumerate() {
            if let Some(cb) = self.progress.as_mut() {
                cb(u32::try_from(i).unwrap_or(u32::MAX), total_steps, t);
            }
            let model_input = self.scheduler.scale_model_input(&latent, t)?;

            let cond_eps = self.unet.forward(&model_input, t, &cond_embed)?;
            let combined = if do_cfg {
                let uncond_eps = self.unet.forward(&model_input, t, &uncond_embed)?;
                let scaled_cond = B::scale(&cond_eps.data, s);
                let scaled_uncond = B::scale(&uncond_eps.data, 1.0 - s);
                let storage = B::add(
                    &scaled_cond,
                    &scaled_uncond,
                    &cond_eps.shape,
                    &uncond_eps.shape,
                    &cond_eps.shape,
                );
                Tensor::new(storage, cond_eps.shape.clone())
            } else {
                cond_eps
            };

            latent = self.scheduler.step(&combined, t, &latent)?;
        }

        // ---- VAE decode + range remap, identical to generate(). ----
        let decoded = self.vae.decode(&latent)?;
        let mut pixels = decoded.to_vec();
        for v in &mut pixels {
            *v = v.clamp(-1.0, 1.0) * 0.5 + 0.5;
        }
        let out_dims = decoded.shape.dims().to_vec();
        Ok(Tensor::from_vec(pixels, Shape::new(&out_dims)))
    }

    /// Run an inpaint generation: VAE-encode `image * (1 - mask)` to a
    /// fixed conditioning latent, concatenate it (plus the latent-resolution
    /// mask) to the noisy latent at every UNet forward, and denoise from
    /// pure noise. The unmasked region of `image` stays anchored throughout
    /// the schedule; the masked region is re-synthesized to match `prompt`.
    ///
    /// `image` must be `[3, H, W]` already normalized to `[-1, 1]`. `mask`
    /// must be `[1, H, W]` in `{0, 1}` (or `[0, 1]`) — `1.0` marks pixels
    /// to inpaint. `H` and `W` must be multiples of 8. Requires the
    /// pipeline to be built with a [`Self::vae_encoder`].
    ///
    /// The 9-channel UNet input is built per-step as
    /// `concat([noisy_latent, mask_latent, masked_latent], dim=0)` —
    /// the channel order HF's `StableDiffusionInpaintPipeline` uses.
    /// Pair this with a UNet loaded from `UnetConfig::sd_1_5_inpainting()`.
    ///
    /// # Errors
    /// - [`Self::vae_encoder`] is `None`.
    /// - `image` is not `[3, H, W]`, `mask` is not `[1, H, W]`, or shapes
    ///   disagree.
    /// - `H` / `W` not multiples of 8.
    /// - `num_inference_steps == 0`.
    /// - Tokenizer / encoder / VAE / UNet / scheduler propagated failures.
    #[allow(clippy::too_many_lines)]
    pub fn inpaint(
        &mut self,
        params: &InpaintParams,
        image: &Tensor<B>,
        mask: &Tensor<B>,
    ) -> Result<Tensor<B>> {
        let encoder = self.vae_encoder.as_ref().ok_or_else(|| {
            Error::Llm("inpaint requires vae_encoder; pipeline was built without one".into())
        })?;
        if params.num_inference_steps == 0 {
            return Err(Error::Scheduler("num_inference_steps must be > 0".into()));
        }
        let img_dims = image.shape.dims();
        if img_dims.len() != 3 || img_dims[0] != 3 {
            return Err(Error::Llm(format!(
                "image must be [3, H, W], got {img_dims:?}"
            )));
        }
        let mask_dims = mask.shape.dims();
        if mask_dims.len() != 3 || mask_dims[0] != 1 {
            return Err(Error::Llm(format!(
                "mask must be [1, H, W], got {mask_dims:?}"
            )));
        }
        let (height, width) = (img_dims[1], img_dims[2]);
        if mask_dims[1] != height || mask_dims[2] != width {
            return Err(Error::Llm(format!(
                "mask shape [1, {}, {}] does not match image [3, {height}, {width}]",
                mask_dims[1], mask_dims[2]
            )));
        }
        let vae_scale = VAE_SCALE as usize;
        if height == 0 || width == 0 || height % vae_scale != 0 || width % vae_scale != 0 {
            return Err(Error::Llm(format!(
                "image size {width}x{height} must be non-zero multiples of {VAE_SCALE}"
            )));
        }

        #[cfg(feature = "scry-gpu-bf16")]
        {
            let _ = scry_llm::backend::scry_gpu::ScryGpuBackend::set_bf16_matmul(true);
        }

        let latent_h = height / vae_scale;
        let latent_w = width / vae_scale;
        let plane = height * width;
        let latent_shape = Shape::new(&[LATENT_CHANNELS, latent_h, latent_w]);
        let input_shape = Shape::new(&[9, latent_h, latent_w]);

        // ---- Text conditioning (cond + uncond). ----
        let cond_tokens = self.tokenizer.encode(&params.prompt)?;
        let uncond_tokens = self.tokenizer.encode(&params.negative_prompt)?;
        let cond_embed = self.text_encoder.encode(&cond_tokens)?;
        let uncond_embed = self.text_encoder.encode(&uncond_tokens)?;

        // ---- masked_image = image * (1 - mask), broadcast across channels. ----
        let image_host = image.to_vec();
        let mask_host = mask.to_vec();
        let mut masked_image_host = vec![0.0f32; image_host.len()];
        for c in 0..3 {
            for p in 0..plane {
                masked_image_host[c * plane + p] = image_host[c * plane + p] * (1.0 - mask_host[p]);
            }
        }
        let masked_image_t: Tensor<B> = Tensor::from_vec(masked_image_host, image.shape.clone());

        // ---- masked_latent = vae.encode(masked_image).mean * scaling_factor. ----
        // Inpainting uses the mode of the latent distribution (= mean for
        // a Gaussian), not a reparameterized sample. The result conditions
        // every step deterministically.
        let (mean, _logvar) = encoder.encode(&masked_image_t)?;
        let scaling = encoder.config.scaling_factor;
        let masked_latent_storage = B::scale(&mean.data, scaling);
        let masked_latent_host =
            Tensor::<B>::new(masked_latent_storage, mean.shape.clone()).to_vec();

        // ---- mask_latent: nearest-downsample to latent res (stride VAE_SCALE). ----
        // Matches HF's `F.interpolate(mask, size=(h, w), mode='nearest')` for
        // integer-stride downsampling.
        let mut mask_latent_host = vec![0.0f32; latent_h * latent_w];
        for y in 0..latent_h {
            for x in 0..latent_w {
                mask_latent_host[y * latent_w + x] =
                    mask_host[(y * vae_scale) * width + (x * vae_scale)];
            }
        }

        // ---- Initial noisy latent. ----
        let init_sigma = self.scheduler.init_noise_sigma();
        let elements = LATENT_CHANNELS * latent_h * latent_w;
        let mut latent_host = sample_standard_normal(elements, params.seed);
        if (init_sigma - 1.0).abs() > f32::EPSILON {
            for v in &mut latent_host {
                *v *= init_sigma;
            }
        }
        let mut latent: Tensor<B> = Tensor::from_vec(latent_host, latent_shape.clone());

        // ---- Denoise loop. ----
        // Per step: scale noisy latent, build 9-ch UNet input by concatenating
        // (noisy, mask_latent, masked_latent) along the channel dim, forward
        // through the cond + uncond branches, CFG combine, scheduler step.
        // The concat is host-side: scaled.to_vec() rounds the noisy latent
        // through host memory each step. Pre-uploading the static mask /
        // masked-latent channels into device storage and using a
        // backend-side channel-concat op would skip that round-trip; left
        // for follow-up.
        self.scheduler.set_timesteps(params.num_inference_steps)?;
        let timesteps: Vec<f32> = self.scheduler.timesteps().to_vec();
        let total_steps = u32::try_from(timesteps.len()).unwrap_or(u32::MAX);
        let do_cfg = params.guidance_scale > 1.0 + f32::EPSILON;
        let s = params.guidance_scale;

        for (i, &t) in timesteps.iter().enumerate() {
            if let Some(cb) = self.progress.as_mut() {
                cb(u32::try_from(i).unwrap_or(u32::MAX), total_steps, t);
            }
            let scaled = self.scheduler.scale_model_input(&latent, t)?;
            let scaled_host = scaled.to_vec();
            let mut input_host = Vec::with_capacity(9 * latent_h * latent_w);
            input_host.extend_from_slice(&scaled_host);
            input_host.extend_from_slice(&mask_latent_host);
            input_host.extend_from_slice(&masked_latent_host);
            let unet_input: Tensor<B> = Tensor::from_vec(input_host, input_shape.clone());

            let cond_eps = self.unet.forward(&unet_input, t, &cond_embed)?;
            let combined = if do_cfg {
                let uncond_eps = self.unet.forward(&unet_input, t, &uncond_embed)?;
                let scaled_cond = B::scale(&cond_eps.data, s);
                let scaled_uncond = B::scale(&uncond_eps.data, 1.0 - s);
                let storage = B::add(
                    &scaled_cond,
                    &scaled_uncond,
                    &cond_eps.shape,
                    &uncond_eps.shape,
                    &cond_eps.shape,
                );
                Tensor::new(storage, cond_eps.shape.clone())
            } else {
                cond_eps
            };

            latent = self.scheduler.step(&combined, t, &latent)?;
        }

        // ---- VAE decode + range remap. ----
        let decoded = self.vae.decode(&latent)?;
        let mut pixels = decoded.to_vec();
        for v in &mut pixels {
            *v = v.clamp(-1.0, 1.0) * 0.5 + 0.5;
        }
        let out_dims = decoded.shape.dims().to_vec();
        Ok(Tensor::from_vec(pixels, Shape::new(&out_dims)))
    }

    /// Install a progress callback fired before each denoising step.
    /// Returns `self` for builder-style chaining.
    pub fn with_progress<F>(mut self, callback: F) -> Self
    where
        F: FnMut(u32, u32, f32) + Send + 'static,
    {
        self.progress = Some(Box::new(callback));
        self
    }

    /// Pre-upload every parameter tensor in the text encoder, UNet, and VAE
    /// to the backend's device-resident form. No-op on `CpuBackend`;
    /// idempotent on any backend.
    ///
    /// Without this, weights stay CPU-resident and `MathBackend::matmul` /
    /// `conv2d` re-uploads on every kernel dispatch — for SD 1.5 that's
    /// 3.4 GB × 60 forwards per image. Also unblocks the bf16 fast path,
    /// which short-circuits to `None` on CPU-resident storage.
    pub fn to_device(&mut self) {
        self.text_encoder.to_device();
        self.unet.to_device();
        self.vae.to_device();
        if let Some(enc) = self.vae_encoder.as_mut() {
            enc.to_device();
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic Gaussian sampler.
// ---------------------------------------------------------------------------

/// Sample `n` standard-normal floats using a `seed`-keyed SplitMix64 RNG
/// and Box-Muller. Deterministic across runs and architectures.
///
/// We do **not** try to match `torch.randn(seed)` byte-for-byte — PyTorch
/// uses a Mersenne Twister + Box-Muller variant whose state isn't
/// portable to a pure-Rust pipeline without pulling in `rand_distr`.
/// Cross-language seed parity is M9 territory; this is enough for
/// "same seed produces same image on this stack" determinism.
fn sample_standard_normal(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next_u64 = || -> u64 {
        // SplitMix64 — 64 bits of state, no warmup required.
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut next_uniform = || -> f32 {
        // 24 high bits → [0, 1) with full mantissa precision.
        #[allow(clippy::cast_precision_loss)]
        let r = (next_u64() >> 40) as f32;
        // Shift away from exactly 0 so ln() is finite.
        (r + 0.5) / 16_777_216.0
    };

    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        // Box-Muller: each call burns one pair of uniforms for two normals.
        let u1 = next_uniform();
        let u2 = next_uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        out.push(r * theta.cos());
        if out.len() < n {
            out.push(r * theta.sin());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_seed_is_deterministic() {
        let a = sample_standard_normal(64, 42);
        let b = sample_standard_normal(64, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn gaussian_seeds_diverge() {
        let a = sample_standard_normal(64, 42);
        let b = sample_standard_normal(64, 43);
        assert_ne!(a, b);
    }

    #[test]
    fn gaussian_mean_and_variance_are_close_to_zero_one() {
        let n = 32_768;
        let xs = sample_standard_normal(n, 7);
        #[allow(clippy::cast_precision_loss)]
        let mean = xs.iter().copied().sum::<f32>() / n as f32;
        #[allow(clippy::cast_precision_loss)]
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        // Loose envelope; a Box-Muller sample of 32k should sit well inside.
        assert!(mean.abs() < 0.05, "mean={mean}");
        assert!((var - 1.0).abs() < 0.05, "var={var}");
    }
}
