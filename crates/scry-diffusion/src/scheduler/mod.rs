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
//! [`ddim::DdimScheduler`] is the simplest deterministic sampler — good
//! first cut. Other schedulers (Euler, DPM-Solver++, PNDM, EulerAncestral)
//! have different `step` math but the same trait.

pub mod ddim;

pub use ddim::DdimScheduler;

use crate::error::Result;

/// Scheduler controlling the denoising trajectory.
pub trait Scheduler {
    /// Initial sigma the latent should be scaled by. Default 1.0.
    fn init_noise_sigma(&self) -> f32 {
        1.0
    }

    /// Pick the inference-time timestep subset. Called once before the loop.
    fn set_timesteps(&mut self, num_inference_steps: u32) -> Result<()>;

    /// The full timestep trajectory in iteration order (typically descending
    /// from ~999 down to 0).
    fn timesteps(&self) -> &[f32];

    /// Pre-condition the model input. Most samplers are no-op (return the
    /// latent as-is); Euler-family samplers divide by `sqrt(sigma^2 + 1)`.
    fn scale_model_input(&self, latent: Vec<f32>, _timestep: f32) -> Vec<f32> {
        latent
    }

    /// Per-step update: compute the previous latent given the model's
    /// prediction at `timestep`. Returns the new latent in row-major
    /// element order matching the input.
    fn step(&mut self, model_output: &[f32], timestep: f32, latent: &[f32]) -> Result<Vec<f32>>;
}
