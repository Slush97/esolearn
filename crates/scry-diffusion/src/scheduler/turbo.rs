// SPDX-License-Identifier: MIT OR Apache-2.0
//! SDXL-Turbo / SD-Turbo single-step (and few-step) sampler.
//!
//! Distilled-model sampler shipped with `stabilityai/sd-turbo` and
//! `stabilityai/sdxl-turbo`. The distillation is trained to denoise
//! directly to the data manifold in 1 step at the noisiest timestep, with
//! 2- and 4-step trajectories also producing useful samples. This crate's
//! Turbo support is wired here as a **placeholder** at the scheduler
//! level — the actual SDXL UNet + dual text encoders land in M13, at
//! which point this scheduler gets exercised against a real checkpoint.
//!
//! ## Math (ε-prediction, deterministic Euler)
//!
//! `EulerDiscreteScheduler` parameterizes the trajectory in σ-space:
//!
//! ```text
//!   σ_t  = √((1 − ᾱ_t) / ᾱ_t)
//!   x_0  = x_t − σ_t · ε
//!   x_{t-1} = x_t + (σ_{t-1} − σ_t) · (x_t − x_0) / σ_t
//!          = x_t + (σ_{t-1} − σ_t) · ε
//!   x_T_input_to_unet = x_t / √(σ_t² + 1)   (scale_model_input)
//!   init_noise_sigma  = max(σ)              (latent at t=T scaled up)
//! ```
//!
//! With `final_sigmas_type = "zero"` (Turbo's setting) the trajectory's
//! final σ is `0`, so `x_0 = x_{T_final}` exactly after the last step.
//!
//! ## Timestep spacing
//!
//! Turbo uses **trailing** spacing (`np.arange(num_train, 0, −step)`),
//! not the leading-spacing default DDIM/DPM++ use. With
//! `num_train_timesteps = 1000` and `num_inference_steps = 4` the
//! integer timesteps are `[999, 749, 499, 249]`; `[999]` for 1-step.

use scry_llm::backend::MathBackend;
use scry_llm::tensor::Tensor;

use super::common::{build_betas, timestep_to_index, BetaSchedule};
use super::Scheduler;
use crate::error::{Error, Result};

/// SDXL-Turbo scheduler configuration.
#[derive(Debug, Clone)]
pub struct TurboConfig {
    /// Total number of training timesteps. SDXL-Turbo = 1000.
    pub num_train_timesteps: u32,
    /// Beta start. SDXL-Turbo = 0.00085.
    pub beta_start: f32,
    /// Beta end. SDXL-Turbo = 0.012.
    pub beta_end: f32,
    /// Beta schedule shape.
    pub beta_schedule: BetaSchedule,
    /// `true` for ε-prediction (SDXL-Turbo). v-prediction isn't wired here.
    pub epsilon_prediction: bool,
    /// Maximum supported number of inference steps. SDXL-Turbo is trained
    /// for 1–4 steps; values above that are rejected by `set_timesteps`.
    pub max_inference_steps: u32,
}

impl TurboConfig {
    /// SDXL-Turbo defaults — match
    /// `stabilityai/sdxl-turbo/scheduler/scheduler_config.json`
    /// (`EulerDiscreteScheduler`, `timestep_spacing="trailing"`,
    /// `final_sigmas_type="zero"`).
    #[must_use]
    pub fn sdxl_turbo() -> Self {
        Self {
            num_train_timesteps: 1000,
            beta_start: 0.000_85,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            epsilon_prediction: true,
            max_inference_steps: 4,
        }
    }

    /// SD-Turbo defaults — same as SDXL-Turbo at the scheduler level
    /// (the distinction lives in the UNet / text encoder, not the
    /// noise schedule).
    #[must_use]
    pub fn sd_turbo() -> Self {
        Self::sdxl_turbo()
    }
}

/// Turbo (Euler-discrete, trailing-spacing, zero-terminal-σ) scheduler.
///
/// Holds two parallel f32 tables: integer-timestep trajectory (returned
/// by [`Scheduler::timesteps`]) and the corresponding σ ladder used by
/// `step` and `scale_model_input`. The final σ is `0`.
pub struct TurboScheduler {
    /// Configuration.
    pub config: TurboConfig,
    /// Inference-time timestep trajectory (set by `set_timesteps`).
    pub timesteps: Vec<f32>,
    /// σ ladder, length `timesteps.len() + 1`. `sigmas[i]` is σ at
    /// `timesteps[i]`; `sigmas[len]` is the post-final σ (0 with
    /// `final_sigmas_type = "zero"`).
    pub sigmas: Vec<f32>,
    /// Cumulative α product table, indexed by integer timestep. Used to
    /// map a timestep onto a σ.
    pub alphas_cumprod: Vec<f32>,
    /// 0-indexed counter across `step` calls within one inference run.
    step_index: usize,
}

impl TurboScheduler {
    /// Construct a scheduler. `set_timesteps` must be called before `step`.
    ///
    /// # Errors
    /// Returns [`Error::Scheduler`] if `num_train_timesteps == 0`.
    pub fn new(config: TurboConfig) -> Result<Self> {
        let t = config.num_train_timesteps as usize;
        if t == 0 {
            return Err(Error::Scheduler("num_train_timesteps must be > 0".into()));
        }
        let betas = build_betas(
            config.num_train_timesteps,
            config.beta_start,
            config.beta_end,
            config.beta_schedule,
        );
        let mut alphas_cumprod = Vec::with_capacity(t);
        let mut acc = 1.0_f32;
        for &b in &betas {
            acc *= 1.0 - b;
            alphas_cumprod.push(acc);
        }
        Ok(Self {
            config,
            timesteps: Vec::new(),
            sigmas: Vec::new(),
            alphas_cumprod,
            step_index: 0,
        })
    }

    /// σ at integer timestep, derived from ᾱ via
    /// `σ = √((1 − ᾱ) / ᾱ)`.
    fn sigma_at(&self, t_idx: u32) -> f32 {
        let alpha = self.alphas_cumprod[t_idx as usize];
        ((1.0 - alpha) / alpha).sqrt()
    }
}

impl<B: MathBackend> Scheduler<B> for TurboScheduler {
    fn init_noise_sigma(&self) -> f32 {
        // EulerDiscreteScheduler scales the initial latent up by the
        // maximum σ on the trajectory so `x_T = ε · σ_max`. With
        // trailing spacing on SD-Turbo's 1000-step training schedule
        // the max σ lives at t=999, which is what the pipeline starts
        // every run from regardless of `num_inference_steps`.
        self.sigmas.first().copied().unwrap_or(1.0)
    }

    fn set_timesteps(&mut self, num_inference_steps: u32) -> Result<()> {
        if num_inference_steps == 0 {
            return Err(Error::Scheduler("num_inference_steps must be > 0".into()));
        }
        if num_inference_steps > self.config.max_inference_steps {
            return Err(Error::Scheduler(format!(
                "num_inference_steps ({}) exceeds max_inference_steps ({}) — Turbo is distilled for 1–{} steps",
                num_inference_steps, self.config.max_inference_steps, self.config.max_inference_steps
            )));
        }
        // Trailing spacing:
        //   step_ratio = num_train / num_inference
        //   timesteps  = arange(num_train, 0, -step_ratio) - 1
        //              = [num_train-1, num_train-step-1, ..., step_ratio-1]
        // Equivalent integer-only form below; avoids casting through f32.
        let step_ratio = self.config.num_train_timesteps / num_inference_steps;
        let mut ts: Vec<f32> = Vec::with_capacity(num_inference_steps as usize);
        for i in 0..num_inference_steps {
            // Trailing: T, T-step, T-2*step, ... each minus 1.
            let raw = self.config.num_train_timesteps - i * step_ratio;
            let t_int = raw.saturating_sub(1);
            #[allow(clippy::cast_precision_loss)]
            ts.push(t_int as f32);
        }

        let mut sigmas: Vec<f32> = ts
            .iter()
            .map(|&t| {
                let idx = timestep_to_index(t, self.alphas_cumprod.len())
                    .expect("trailing-spacing timestep is in-range by construction");
                self.sigma_at(idx)
            })
            .collect();
        // `final_sigmas_type = "zero"`: append a 0 σ so the last step
        // lands exactly noise-free without extrapolating to t < 0.
        sigmas.push(0.0);

        self.timesteps = ts;
        self.sigmas = sigmas;
        self.step_index = 0;
        Ok(())
    }

    fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    fn scale_model_input(&self, latent: &Tensor<B>, _timestep: f32) -> Result<Tensor<B>> {
        // EulerDiscreteScheduler divides by √(σ_t² + 1) per step. The
        // pipeline calls this with `t == timesteps[step_index]`, so we
        // pull σ off the ladder by step_index. Falls back to the default
        // (identity) materialize if the scheduler hasn't been advanced.
        if self.step_index >= self.sigmas.len().saturating_sub(1) {
            let storage = B::scale(&latent.data, 1.0);
            return Ok(Tensor::new(storage, latent.shape.clone()));
        }
        let sigma = self.sigmas[self.step_index];
        let scale = 1.0 / sigma.mul_add(sigma, 1.0).sqrt();
        let storage = B::scale(&latent.data, scale);
        Ok(Tensor::new(storage, latent.shape.clone()))
    }

    fn step(
        &mut self,
        model_output: &Tensor<B>,
        _timestep: f32,
        latent: &Tensor<B>,
    ) -> Result<Tensor<B>> {
        if self.sigmas.is_empty() {
            return Err(Error::Scheduler(
                "set_timesteps must be called before step".into(),
            ));
        }
        if model_output.shape.numel() != latent.shape.numel() {
            return Err(Error::Scheduler(format!(
                "model_output ({}) and latent ({}) length mismatch",
                model_output.shape.numel(),
                latent.shape.numel()
            )));
        }
        if !self.config.epsilon_prediction {
            return Err(Error::Scheduler(
                "Turbo v-prediction is not yet implemented".into(),
            ));
        }
        if self.step_index >= self.timesteps.len() {
            return Err(Error::Scheduler(
                "step called past end of trajectory".into(),
            ));
        }

        let sigma_t = self.sigmas[self.step_index];
        let sigma_next = self.sigmas[self.step_index + 1];
        let dt = sigma_next - sigma_t;

        // Euler update with ε-prediction:
        //   x_{t-1} = x_t + dt · ε
        let scaled_eps = B::scale(&model_output.data, dt);
        let out_shape = latent.shape.clone();
        let storage = B::add(
            &latent.data,
            &scaled_eps,
            &out_shape,
            &out_shape,
            &out_shape,
        );

        self.step_index += 1;
        Ok(Tensor::new(storage, out_shape))
    }

    fn add_noise(
        &self,
        original: &Tensor<B>,
        noise: &Tensor<B>,
        timestep: f32,
    ) -> Result<Tensor<B>> {
        // Forward-diffusion identity in α-cumprod form, same as DDIM/LCM.
        // Used by img2img-style flows; Turbo's primary mode is pure txt2img.
        if original.shape.numel() != noise.shape.numel() {
            return Err(Error::Scheduler(format!(
                "original ({}) and noise ({}) length mismatch",
                original.shape.numel(),
                noise.shape.numel()
            )));
        }
        let t_idx = timestep_to_index(timestep, self.alphas_cumprod.len())?;
        let alpha_prod_t = self.alphas_cumprod[t_idx as usize];
        let sqrt_alpha = alpha_prod_t.sqrt();
        let sqrt_one_minus = (1.0 - alpha_prod_t).sqrt();
        let scaled_orig = B::scale(&original.data, sqrt_alpha);
        let scaled_noise = B::scale(&noise.data, sqrt_one_minus);
        let out_shape = original.shape.clone();
        let storage = B::add(
            &scaled_orig,
            &scaled_noise,
            &out_shape,
            &noise.shape,
            &out_shape,
        );
        Ok(Tensor::new(storage, out_shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scry_llm::backend::cpu::CpuBackend;
    use scry_llm::tensor::shape::Shape;

    type CpuTensor = Tensor<CpuBackend>;

    fn cpu_tensor(data: Vec<f32>) -> CpuTensor {
        let len = data.len();
        Tensor::from_vec(data, Shape::new(&[len]))
    }

    #[test]
    fn set_timesteps_1_yields_single_step_at_top() {
        let mut s = TurboScheduler::new(TurboConfig::sdxl_turbo()).unwrap();
        <TurboScheduler as Scheduler<CpuBackend>>::set_timesteps(&mut s, 1).unwrap();
        let ts = <TurboScheduler as Scheduler<CpuBackend>>::timesteps(&s);
        assert_eq!(ts, &[999.0_f32]);
        // sigmas has the trajectory + the appended 0.0 boundary.
        assert_eq!(s.sigmas.len(), 2);
        assert!(s.sigmas[0] > 0.0);
        assert!(s.sigmas[1].abs() < f32::EPSILON, "got {}", s.sigmas[1]);
    }

    #[test]
    fn set_timesteps_4_matches_trailing_spacing() {
        let mut s = TurboScheduler::new(TurboConfig::sdxl_turbo()).unwrap();
        <TurboScheduler as Scheduler<CpuBackend>>::set_timesteps(&mut s, 4).unwrap();
        let ts = <TurboScheduler as Scheduler<CpuBackend>>::timesteps(&s);
        assert_eq!(ts, &[999.0_f32, 749.0, 499.0, 249.0]);
    }

    #[test]
    fn init_noise_sigma_is_max_sigma() {
        let mut s = TurboScheduler::new(TurboConfig::sdxl_turbo()).unwrap();
        <TurboScheduler as Scheduler<CpuBackend>>::set_timesteps(&mut s, 4).unwrap();
        // σ is monotonically decreasing with t, so σ[0] (at t=999) is max.
        let init = <TurboScheduler as Scheduler<CpuBackend>>::init_noise_sigma(&s);
        assert!(init > 1.0, "got {init}");
        for w in s.sigmas.windows(2) {
            assert!(w[0] >= w[1], "sigmas not monotonic: {:?}", s.sigmas);
        }
    }

    #[test]
    fn one_step_chain_lands_with_dt_full_sigma_jump() {
        // 1-step: dt = σ_next − σ_t = 0 − σ_max = −σ_max, so the update
        // x' = x + (−σ_max) · ε. Verify the formula end-to-end on a
        // simple constant tensor.
        let mut s = TurboScheduler::new(TurboConfig::sdxl_turbo()).unwrap();
        <TurboScheduler as Scheduler<CpuBackend>>::set_timesteps(&mut s, 1).unwrap();
        let sigma_max = s.sigmas[0];
        let latent = cpu_tensor(vec![1.0; 16]);
        let eps = cpu_tensor(vec![0.25; 16]);
        let out =
            <TurboScheduler as Scheduler<CpuBackend>>::step(&mut s, &eps, 999.0, &latent).unwrap();
        let v = out.to_vec();
        let want = 1.0 - sigma_max * 0.25;
        for x in &v {
            assert!((x - want).abs() < 1e-4, "got {x}, want {want}");
        }
    }

    #[test]
    fn step_chain_4_steps_finite() {
        let mut s = TurboScheduler::new(TurboConfig::sdxl_turbo()).unwrap();
        <TurboScheduler as Scheduler<CpuBackend>>::set_timesteps(&mut s, 4).unwrap();
        let n = 4 * 8 * 8;
        let mut latent: CpuTensor = cpu_tensor(vec![0.5; n]);
        let timesteps = <TurboScheduler as Scheduler<CpuBackend>>::timesteps(&s).to_vec();
        for &t in &timesteps {
            let eps: CpuTensor = cpu_tensor(vec![0.1; n]);
            latent =
                <TurboScheduler as Scheduler<CpuBackend>>::step(&mut s, &eps, t, &latent).unwrap();
        }
        for v in latent.to_vec() {
            assert!(v.is_finite(), "non-finite: {v}");
        }
    }

    #[test]
    fn step_without_set_timesteps_errors() {
        let mut s = TurboScheduler::new(TurboConfig::sdxl_turbo()).unwrap();
        let eps = cpu_tensor(vec![0.0; 4]);
        let latent = cpu_tensor(vec![0.0; 4]);
        let err = <TurboScheduler as Scheduler<CpuBackend>>::step(&mut s, &eps, 999.0, &latent);
        assert!(err.is_err());
    }

    #[test]
    fn unsupported_inference_steps_errors() {
        let mut s = TurboScheduler::new(TurboConfig::sdxl_turbo()).unwrap();
        let err = <TurboScheduler as Scheduler<CpuBackend>>::set_timesteps(&mut s, 5);
        assert!(err.is_err());
    }
}
