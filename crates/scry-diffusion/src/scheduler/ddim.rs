// SPDX-License-Identifier: MIT OR Apache-2.0
//! DDIM (Denoising Diffusion Implicit Models) sampler.
//!
//! Deterministic, simple, no extra noise injection — the canonical first
//! sampler to wire up. Reference:
//!   Song, Meng, Ermon. "Denoising Diffusion Implicit Models." ICLR 2021.
//!   `diffusers/src/diffusers/schedulers/scheduling_ddim.py`.
//!
//! Math (in α-cumulative-product form):
//! ```text
//!   x_0 ≈ (x_t − √(1 − ᾱ_t) · ε_θ(x_t, t)) / √(ᾱ_t)
//!   x_{t-1} = √(ᾱ_{t-1}) · x_0 + √(1 − ᾱ_{t-1}) · ε_θ(x_t, t)
//! ```
//! where `ε_θ` is the UNet's predicted noise. With η=0 (DDIM default) the
//! sampler is fully deterministic.

use super::Scheduler;
use crate::error::Result;

/// Beta schedule shape. `ScaledLinear` is what SD 1.5 uses.
#[derive(Debug, Clone, Copy)]
pub enum BetaSchedule {
    /// `linspace(beta_start^0.5, beta_end^0.5, T)` then squared. SD 1.5 default.
    ScaledLinear,
    /// Plain linear `linspace(beta_start, beta_end, T)`.
    Linear,
}

/// DDIM scheduler configuration.
#[derive(Debug, Clone)]
pub struct DdimConfig {
    /// Total number of training timesteps. SD = 1000.
    pub num_train_timesteps: u32,
    /// Beta start. SD = 0.00085.
    pub beta_start: f32,
    /// Beta end. SD = 0.012.
    pub beta_end: f32,
    /// Beta schedule shape.
    pub beta_schedule: BetaSchedule,
    /// Whether the model is parameterized as ε-prediction (`true`, SD 1.5)
    /// or v-prediction (`false`, SD 2.x with v-objective checkpoints).
    pub epsilon_prediction: bool,
}

impl DdimConfig {
    /// SD 1.5 defaults.
    pub fn sd_1_5() -> Self {
        Self {
            num_train_timesteps: 1000,
            beta_start: 0.000_85,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            epsilon_prediction: true,
        }
    }
}

/// DDIM scheduler.
pub struct DdimScheduler {
    /// Configuration.
    pub config: DdimConfig,
    /// Inference-time timestep trajectory (set by `set_timesteps`).
    pub timesteps: Vec<f32>,
    /// Cumulative α product table indexed by integer timestep.
    pub alphas_cumprod: Vec<f32>,
}

impl DdimScheduler {
    /// Construct a scheduler. `set_timesteps` must be called before `step`.
    pub fn new(config: DdimConfig) -> Result<Self> {
        let _ = config;
        todo!("M7: build betas (per beta_schedule), derive alphas + alphas_cumprod")
    }
}

impl Scheduler for DdimScheduler {
    fn set_timesteps(&mut self, num_inference_steps: u32) -> Result<()> {
        let _ = num_inference_steps;
        todo!("M7: pick num_inference_steps timesteps from [0, num_train_timesteps)")
    }

    fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    fn step(&mut self, model_output: &[f32], timestep: f32, latent: &[f32]) -> Result<Vec<f32>> {
        let _ = (model_output, timestep, latent);
        todo!(
            "M7: x0 = (x_t - sqrt(1-α_t̄) * ε) / sqrt(α_t̄); \
             x_{{t-1}} = sqrt(α_{{t-1}}̄) * x0 + sqrt(1 - α_{{t-1}}̄) * ε"
        )
    }
}
