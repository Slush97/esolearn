// SPDX-License-Identifier: MIT OR Apache-2.0
//! Schedulers — control the noise schedule and the per-step latent update.
//!
//! A scheduler owns the noise schedule (β / α / α-cumulative-product
//! arrays) and exposes two operations:
//!   1. [`Scheduler::set_timesteps`] — pick the inference-time subset of the
//!      training schedule (the "denoising trajectory").
//!   2. [`Scheduler::step`] — given the UNet's prediction at `timestep`,
//!      return the previous latent for the next iteration.
//!
//! The trait is generic over a [`MathBackend`] so the latent stays on
//! whichever device the rest of the pipeline uses — no `to_vec()` round
//! trips inside the denoise loop. DDIM's step math is just two scalar
//! coefficients applied element-wise + one add, so it composes from the
//! existing `MathBackend::scale` and `MathBackend::add` ops without new
//! kernels.
//!
//! [`ddim::DdimScheduler`] is the simplest deterministic sampler — good
//! first cut. Other schedulers (Euler, DPM-Solver++, PNDM, EulerAncestral)
//! have different `step` math but the same trait.

pub mod common;
pub mod ddim;
pub mod dpm_solver_pp;

pub use common::BetaSchedule;
pub use ddim::DdimScheduler;
pub use dpm_solver_pp::DpmSolverPpScheduler;

use scry_llm::backend::MathBackend;
use scry_llm::tensor::Tensor;

use crate::error::Result;

/// Scheduler controlling the denoising trajectory.
///
/// Generic over the math backend so the latent never has to leave the
/// device it lives on. All op work runs through [`MathBackend`].
pub trait Scheduler<B: MathBackend> {
    /// Initial sigma the latent should be scaled by. Default 1.0.
    fn init_noise_sigma(&self) -> f32 {
        1.0
    }

    /// Pick the inference-time timestep subset. Called once before the loop.
    ///
    /// # Errors
    /// Backend- and scheduler-specific; see implementations.
    fn set_timesteps(&mut self, num_inference_steps: u32) -> Result<()>;

    /// The full timestep trajectory in iteration order (typically descending
    /// from ~999 down to 0).
    fn timesteps(&self) -> &[f32];

    /// Pre-condition the model input. Most samplers are no-op (return the
    /// latent as-is); Euler-family samplers divide by `sqrt(sigma^2 + 1)`.
    ///
    /// The default uses `B::scale(_, 1.0)` to materialize a fresh storage
    /// without requiring `B: Clone` — a single dispatch on GPU, a memcpy
    /// of the latent on CPU. Either is negligible relative to the UNet
    /// forward that follows, but Euler-family schedulers should override
    /// to do the divide in one pass anyway.
    ///
    /// # Errors
    /// Implementation-defined.
    fn scale_model_input(&self, latent: &Tensor<B>, _timestep: f32) -> Result<Tensor<B>> {
        let storage = B::scale(&latent.data, 1.0);
        Ok(Tensor::new(storage, latent.shape.clone()))
    }

    /// Per-step update: compute the previous latent given the model's
    /// prediction at `timestep`.
    ///
    /// # Errors
    /// - `set_timesteps` not yet called.
    /// - shape / length mismatch between `model_output` and `latent`.
    /// - timestep out of bounds for the schedule.
    fn step(
        &mut self,
        model_output: &Tensor<B>,
        timestep: f32,
        latent: &Tensor<B>,
    ) -> Result<Tensor<B>>;

    /// Mix `original` with `noise` at `timestep` per the forward-diffusion
    /// formula `latent = √(ᾱ_t) · original + √(1 − ᾱ_t) · noise`. Used by
    /// img2img to lift a clean VAE-encoded latent up to the noise level
    /// the truncated denoise loop starts at. Stateless — does not require
    /// `set_timesteps`.
    ///
    /// # Errors
    /// - shape / length mismatch between `original` and `noise`.
    /// - timestep out of bounds for the training schedule.
    fn add_noise(
        &self,
        original: &Tensor<B>,
        noise: &Tensor<B>,
        timestep: f32,
    ) -> Result<Tensor<B>>;
}
