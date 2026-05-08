// SPDX-License-Identifier: MIT OR Apache-2.0
//! DDIM (Denoising Diffusion Implicit Models) sampler.
//!
//! Deterministic, simple, no extra noise injection — the canonical first
//! sampler to wire up. Reference:
//!   Song, Meng, Ermon. "Denoising Diffusion Implicit Models." ICLR 2021.
//!   `diffusers/src/diffusers/schedulers/scheduling_ddim.py`.
//!
//! Math (in α-cumulative-product form, ε-prediction, η=0):
//! ```text
//!   x_0 ≈ (x_t − √(1 − ᾱ_t) · ε_θ(x_t, t)) / √(ᾱ_t)
//!   x_{t-1} = √(ᾱ_{t-1}) · x_0 + √(1 − ᾱ_{t-1}) · ε_θ(x_t, t)
//! ```
//! where `ε_θ` is the UNet's predicted noise. With η=0 (DDIM default) the
//! sampler is fully deterministic.

use super::Scheduler;
use crate::error::{Error, Result};

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
    /// Offset added to every timestep produced by [`DdimScheduler::set_timesteps`].
    /// SD 1.5 ships with `steps_offset = 1` so `(t = 999, 967, …, 1)`
    /// instead of `(998, 966, …, 0)`. Matches diffusers'
    /// `DDIMScheduler.steps_offset`.
    pub steps_offset: u32,
    /// If `true`, the "previous" α-cumprod for the final step is `1.0`
    /// (the "exact" theoretical limit). If `false`, it is `alphas_cumprod[0]`
    /// — diffusers' default and what SD 1.5 ships with. Matches
    /// `DDIMScheduler.set_alpha_to_one`.
    pub set_alpha_to_one: bool,
}

impl DdimConfig {
    /// SD 1.5 defaults — match the `scheduler/scheduler_config.json` shipped
    /// with `runwayml/stable-diffusion-v1-5`.
    #[must_use]
    pub fn sd_1_5() -> Self {
        Self {
            num_train_timesteps: 1000,
            beta_start: 0.000_85,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            epsilon_prediction: true,
            steps_offset: 1,
            set_alpha_to_one: false,
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
    /// Step size between successive inference timesteps
    /// (`num_train_timesteps / num_inference_steps`). Captured at
    /// `set_timesteps` so `step` can locate the previous α-cumprod.
    step_size: u32,
}

impl DdimScheduler {
    /// Construct a scheduler. `set_timesteps` must be called before `step`.
    ///
    /// # Errors
    /// Returns [`Error::Scheduler`] if `num_train_timesteps == 0`.
    pub fn new(config: DdimConfig) -> Result<Self> {
        let t = config.num_train_timesteps as usize;
        if t == 0 {
            return Err(Error::Scheduler(
                "num_train_timesteps must be > 0".into(),
            ));
        }
        let betas = build_betas(&config);
        let mut alphas_cumprod = Vec::with_capacity(t);
        let mut acc = 1.0_f32;
        for &b in &betas {
            acc *= 1.0 - b;
            alphas_cumprod.push(acc);
        }
        Ok(Self {
            config,
            timesteps: Vec::new(),
            alphas_cumprod,
            step_size: 0,
        })
    }
}

/// Build the per-step β schedule for the configured shape.
fn build_betas(config: &DdimConfig) -> Vec<f32> {
    let t = config.num_train_timesteps as usize;
    let mut betas = Vec::with_capacity(t);
    match config.beta_schedule {
        BetaSchedule::ScaledLinear => {
            let start = config.beta_start.sqrt();
            let end = config.beta_end.sqrt();
            for i in 0..t {
                let v = lerp(start, end, i, t);
                betas.push(v * v);
            }
        }
        BetaSchedule::Linear => {
            for i in 0..t {
                betas.push(lerp(config.beta_start, config.beta_end, i, t));
            }
        }
    }
    betas
}

/// `linspace(a, b, n)[i]`. Matches NumPy / PyTorch's formula for `n >= 2`;
/// returns `a` when `n == 1`.
#[allow(clippy::cast_precision_loss)]
fn lerp(a: f32, b: f32, i: usize, n: usize) -> f32 {
    if n <= 1 {
        a
    } else {
        a + (b - a) * (i as f32) / ((n - 1) as f32)
    }
}

impl Scheduler for DdimScheduler {
    fn set_timesteps(&mut self, num_inference_steps: u32) -> Result<()> {
        if num_inference_steps == 0 {
            return Err(Error::Scheduler("num_inference_steps must be > 0".into()));
        }
        if num_inference_steps > self.config.num_train_timesteps {
            return Err(Error::Scheduler(format!(
                "num_inference_steps ({}) must be <= num_train_timesteps ({})",
                num_inference_steps, self.config.num_train_timesteps
            )));
        }
        // Diffusers' "leading" spacing (the SD 1.5 default):
        //   step_ratio = num_train_timesteps // num_inference_steps
        //   timesteps  = arange(0, num_inference_steps) * step_ratio
        //   reverse, then add steps_offset.
        let step_ratio = self.config.num_train_timesteps / num_inference_steps;
        let offset = self.config.steps_offset;
        #[allow(clippy::cast_precision_loss)]
        let ts: Vec<f32> = (0..num_inference_steps)
            .rev()
            .map(|i| (i * step_ratio + offset) as f32)
            .collect();
        self.timesteps = ts;
        self.step_size = step_ratio;
        Ok(())
    }

    fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    fn step(
        &mut self,
        model_output: &[f32],
        timestep: f32,
        latent: &[f32],
    ) -> Result<Vec<f32>> {
        if self.step_size == 0 {
            return Err(Error::Scheduler(
                "set_timesteps must be called before step".into(),
            ));
        }
        if model_output.len() != latent.len() {
            return Err(Error::Scheduler(format!(
                "model_output ({}) and latent ({}) length mismatch",
                model_output.len(),
                latent.len()
            )));
        }
        let t_idx = timestep_to_index(timestep, self.alphas_cumprod.len())?;
        let prev_signed = i64::from(t_idx) - i64::from(self.step_size);
        let alpha_prod_t = self.alphas_cumprod[t_idx as usize];
        let alpha_prod_t_prev = if prev_signed >= 0 {
            self.alphas_cumprod[prev_signed as usize]
        } else if self.config.set_alpha_to_one {
            1.0
        } else {
            self.alphas_cumprod[0]
        };

        let sqrt_alpha_t = alpha_prod_t.sqrt();
        let sqrt_one_minus_t = (1.0 - alpha_prod_t).sqrt();
        let sqrt_alpha_prev = alpha_prod_t_prev.sqrt();
        let sqrt_one_minus_prev = (1.0 - alpha_prod_t_prev).sqrt();

        let mut out = Vec::with_capacity(latent.len());
        if self.config.epsilon_prediction {
            for (&x, &eps) in latent.iter().zip(model_output) {
                let pred_x0 = (x - sqrt_one_minus_t * eps) / sqrt_alpha_t;
                let dir_xt = sqrt_one_minus_prev * eps;
                out.push(sqrt_alpha_prev * pred_x0 + dir_xt);
            }
        } else {
            // v-prediction (SD 2.x with v-objective): convert v -> (x0, eps)
            // then apply the same DDIM update. Matches diffusers
            // `DDIMScheduler.step` with `prediction_type="v_prediction"`.
            for (&x, &v) in latent.iter().zip(model_output) {
                let pred_x0 = sqrt_alpha_t * x - sqrt_one_minus_t * v;
                let eps = sqrt_alpha_t * v + sqrt_one_minus_t * x;
                let dir_xt = sqrt_one_minus_prev * eps;
                out.push(sqrt_alpha_prev * pred_x0 + dir_xt);
            }
        }
        Ok(out)
    }
}

/// Round a floating-point timestep to its integer α-cumprod index, validating bounds.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn timestep_to_index(timestep: f32, table_len: usize) -> Result<u32> {
    if !timestep.is_finite() || timestep < 0.0 {
        return Err(Error::Scheduler(format!(
            "timestep {timestep} must be finite and non-negative"
        )));
    }
    let rounded = timestep.round();
    if rounded as usize >= table_len {
        return Err(Error::Scheduler(format!(
            "timestep {timestep} out of range for {table_len}-step schedule"
        )));
    }
    Ok(rounded as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_linear_first_and_last_betas_match_hf() {
        // HF computes betas as `linspace(beta_start^0.5, beta_end^0.5, T) ** 2`.
        // For SD 1.5, the first beta equals beta_start exactly and the last
        // equals beta_end exactly (within fp32).
        let s = DdimScheduler::new(DdimConfig::sd_1_5()).unwrap();
        let betas_first = (0.000_85_f32.sqrt()).powi(2);
        let betas_last = (0.012_f32.sqrt()).powi(2);
        // alphas_cumprod[0] = 1 - betas[0]
        assert!((s.alphas_cumprod[0] - (1.0 - betas_first)).abs() < 1e-7);
        // alphas_cumprod monotonically decreasing
        for w in s.alphas_cumprod.windows(2) {
            assert!(w[0] >= w[1], "alphas_cumprod not monotonic");
        }
        // alphas_cumprod[T-1] should equal product of (1 - beta_i); we just
        // sanity-check it's finite and within the expected SD range.
        // HF's alphas_cumprod[-1] for SD 1.5 is ~0.00466 — small but not
        // microscopic; the schedule keeps a tiny floor so the model can still
        // recover signal at t=999.
        let last = *s.alphas_cumprod.last().unwrap();
        assert!(last > 0.0 && last < 0.01, "got {last}");
        // Ensure the very last beta we consume to build that value matches HF.
        let betas = build_betas(&DdimConfig::sd_1_5());
        assert!((betas[999] - betas_last).abs() < 1e-7);
    }

    #[test]
    fn set_timesteps_30_matches_hf_leading_spacing() {
        // HF DDIMScheduler with steps_offset=1, num_train=1000, num_inf=30:
        // [958, 925, 892, 859, 826, 793, 760, 727, 694, 661,
        //  628, 595, 562, 529, 496, 463, 430, 397, 364, 331,
        //  298, 265, 232, 199, 166, 133, 100,  67,  34,   1]
        let mut s = DdimScheduler::new(DdimConfig::sd_1_5()).unwrap();
        s.set_timesteps(30).unwrap();
        let expected: [f32; 30] = [
            958.0, 925.0, 892.0, 859.0, 826.0, 793.0, 760.0, 727.0, 694.0, 661.0, 628.0, 595.0,
            562.0, 529.0, 496.0, 463.0, 430.0, 397.0, 364.0, 331.0, 298.0, 265.0, 232.0, 199.0,
            166.0, 133.0, 100.0, 67.0, 34.0, 1.0,
        ];
        assert_eq!(s.timesteps(), &expected);
    }

    #[test]
    fn step_chain_30_steps_is_finite_and_changes_latent() {
        // Smoke test independent of HF — proves the chain runs end to end
        // and produces a different latent than it started with.
        let mut s = DdimScheduler::new(DdimConfig::sd_1_5()).unwrap();
        s.set_timesteps(30).unwrap();
        let n = 4 * 16 * 16;
        // Deterministic pseudo-noise (Park-Miller LCG) — keeps the test
        // self-contained and reproducible.
        let mut rng_state = 1_u64;
        let mut next = || -> f32 {
            rng_state = rng_state.wrapping_mul(48_271).wrapping_rem(2_147_483_647);
            #[allow(clippy::cast_precision_loss)]
            let u = rng_state as f32 / 2_147_483_647_f32;
            u * 2.0 - 1.0
        };
        let mut latent: Vec<f32> = (0..n).map(|_| next()).collect();
        let starting: Vec<f32> = latent.clone();
        let timesteps = s.timesteps().to_vec();
        for &t in &timesteps {
            let eps: Vec<f32> = (0..n).map(|_| next()).collect();
            latent = s.step(&eps, t, &latent).unwrap();
        }
        let max_diff = starting
            .iter()
            .zip(&latent)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_diff > 1e-3, "latent did not change: {max_diff}");
        for v in &latent {
            assert!(v.is_finite(), "non-finite latent: {v}");
        }
    }

    #[test]
    fn step_size_mismatch_errors() {
        let mut s = DdimScheduler::new(DdimConfig::sd_1_5()).unwrap();
        s.set_timesteps(30).unwrap();
        let err = s.step(&[0.0; 4], 958.0, &[0.0; 8]);
        assert!(err.is_err());
    }

    #[test]
    fn step_without_set_timesteps_errors() {
        let mut s = DdimScheduler::new(DdimConfig::sd_1_5()).unwrap();
        let err = s.step(&[0.0; 4], 958.0, &[0.0; 4]);
        assert!(err.is_err());
    }
}
