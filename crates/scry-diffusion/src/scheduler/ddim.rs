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
//!
//! The per-step update collapses to `out = c_x · x_t + c_eps · ε` with
//! two host-side scalar coefficients, so the on-device work is just two
//! [`MathBackend::scale`] dispatches plus one [`MathBackend::add`] —
//! latents stay on the device they came in on.

use scry_llm::backend::MathBackend;
use scry_llm::tensor::Tensor;

use super::common::{build_betas, timestep_to_index};
use super::Scheduler;
use crate::error::{Error, Result};

pub use super::common::BetaSchedule;

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
///
/// All scheduler state is host-side scalars (no per-step tensors), which
/// keeps the type backend-agnostic — the same `DdimScheduler` instance
/// satisfies `Scheduler<CpuBackend>` and `Scheduler<ScryGpuBackend>`.
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
            alphas_cumprod,
            step_size: 0,
        })
    }

    /// Resolve the two scalar coefficients `(c_in, c_pred)` so that the
    /// next latent equals `c_in · x_t + c_pred · model_output` in either
    /// the ε-prediction (`model_output == ε`) or v-prediction
    /// (`model_output == v`) parameterization.
    fn step_coefficients(&self, timestep: f32) -> Result<(f32, f32)> {
        if self.step_size == 0 {
            return Err(Error::Scheduler(
                "set_timesteps must be called before step".into(),
            ));
        }
        let t_idx = timestep_to_index(timestep, self.alphas_cumprod.len())?;
        let prev_signed = i64::from(t_idx) - i64::from(self.step_size);
        let alpha_prod_t = self.alphas_cumprod[t_idx as usize];
        let alpha_prod_t_prev = if let Ok(prev_idx) = usize::try_from(prev_signed) {
            self.alphas_cumprod[prev_idx]
        } else if self.config.set_alpha_to_one {
            1.0
        } else {
            self.alphas_cumprod[0]
        };

        let sqrt_alpha_t = alpha_prod_t.sqrt();
        let sqrt_one_minus_t = (1.0 - alpha_prod_t).sqrt();
        let sqrt_alpha_prev = alpha_prod_t_prev.sqrt();
        let sqrt_one_minus_prev = (1.0 - alpha_prod_t_prev).sqrt();

        if self.config.epsilon_prediction {
            // pred_x0 = (x - sqrt_one_minus_t * eps) / sqrt_alpha_t
            // out     = sqrt_alpha_prev * pred_x0 + sqrt_one_minus_prev * eps
            //         = (sqrt_alpha_prev / sqrt_alpha_t) * x
            //           + (sqrt_one_minus_prev
            //              - sqrt_alpha_prev * sqrt_one_minus_t / sqrt_alpha_t) * eps
            let c_in = sqrt_alpha_prev / sqrt_alpha_t;
            let c_eps = sqrt_one_minus_prev - c_in * sqrt_one_minus_t;
            Ok((c_in, c_eps))
        } else {
            // v-prediction: pred_x0 = sqrt_alpha_t * x - sqrt_one_minus_t * v
            //               eps     = sqrt_alpha_t * v + sqrt_one_minus_t * x
            //               out     = sqrt_alpha_prev * pred_x0
            //                       + sqrt_one_minus_prev * eps
            let c_in = sqrt_alpha_prev * sqrt_alpha_t + sqrt_one_minus_prev * sqrt_one_minus_t;
            let c_v = sqrt_one_minus_prev * sqrt_alpha_t - sqrt_alpha_prev * sqrt_one_minus_t;
            Ok((c_in, c_v))
        }
    }
}

impl<B: MathBackend> Scheduler<B> for DdimScheduler {
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
        model_output: &Tensor<B>,
        timestep: f32,
        latent: &Tensor<B>,
    ) -> Result<Tensor<B>> {
        if model_output.shape.numel() != latent.shape.numel() {
            return Err(Error::Scheduler(format!(
                "model_output ({}) and latent ({}) length mismatch",
                model_output.shape.numel(),
                latent.shape.numel()
            )));
        }
        let (c_in, c_pred) = self.step_coefficients(timestep)?;
        let scaled_x = B::scale(&latent.data, c_in);
        let scaled_pred = B::scale(&model_output.data, c_pred);
        let out_shape = latent.shape.clone();
        let storage = B::add(&scaled_x, &scaled_pred, &out_shape, &out_shape, &out_shape);
        Ok(Tensor::new(storage, out_shape))
    }

    fn add_noise(
        &self,
        original: &Tensor<B>,
        noise: &Tensor<B>,
        timestep: f32,
    ) -> Result<Tensor<B>> {
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
        let last = *s.alphas_cumprod.last().unwrap();
        assert!(last > 0.0 && last < 0.01, "got {last}");
        let cfg = DdimConfig::sd_1_5();
        let betas = build_betas(
            cfg.num_train_timesteps,
            cfg.beta_start,
            cfg.beta_end,
            cfg.beta_schedule,
        );
        assert!((betas[999] - betas_last).abs() < 1e-7);
    }

    #[test]
    fn set_timesteps_30_matches_hf_leading_spacing() {
        let mut s = DdimScheduler::new(DdimConfig::sd_1_5()).unwrap();
        <DdimScheduler as Scheduler<CpuBackend>>::set_timesteps(&mut s, 30).unwrap();
        let expected: [f32; 30] = [
            958.0, 925.0, 892.0, 859.0, 826.0, 793.0, 760.0, 727.0, 694.0, 661.0, 628.0, 595.0,
            562.0, 529.0, 496.0, 463.0, 430.0, 397.0, 364.0, 331.0, 298.0, 265.0, 232.0, 199.0,
            166.0, 133.0, 100.0, 67.0, 34.0, 1.0,
        ];
        assert_eq!(
            <DdimScheduler as Scheduler<CpuBackend>>::timesteps(&s),
            &expected
        );
    }

    #[test]
    fn step_chain_30_steps_is_finite_and_changes_latent() {
        // Smoke test independent of HF — proves the chain runs end to end
        // and produces a different latent than it started with.
        let mut s = DdimScheduler::new(DdimConfig::sd_1_5()).unwrap();
        <DdimScheduler as Scheduler<CpuBackend>>::set_timesteps(&mut s, 30).unwrap();
        let n = 4 * 16 * 16;
        let mut rng_state = 1_u64;
        let mut next = || -> f32 {
            rng_state = rng_state.wrapping_mul(48_271).wrapping_rem(2_147_483_647);
            #[allow(clippy::cast_precision_loss)]
            let u = rng_state as f32 / 2_147_483_647_f32;
            u * 2.0 - 1.0
        };
        let mut latent: CpuTensor = cpu_tensor((0..n).map(|_| next()).collect());
        let starting: Vec<f32> = latent.to_vec();
        let timesteps = <DdimScheduler as Scheduler<CpuBackend>>::timesteps(&s).to_vec();
        for &t in &timesteps {
            let eps: CpuTensor = cpu_tensor((0..n).map(|_| next()).collect());
            latent =
                <DdimScheduler as Scheduler<CpuBackend>>::step(&mut s, &eps, t, &latent).unwrap();
        }
        let final_v = latent.to_vec();
        let max_diff = starting
            .iter()
            .zip(&final_v)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_diff > 1e-3, "latent did not change: {max_diff}");
        for v in &final_v {
            assert!(v.is_finite(), "non-finite latent: {v}");
        }
    }

    #[test]
    fn step_size_mismatch_errors() {
        let mut s = DdimScheduler::new(DdimConfig::sd_1_5()).unwrap();
        <DdimScheduler as Scheduler<CpuBackend>>::set_timesteps(&mut s, 30).unwrap();
        let eps = cpu_tensor(vec![0.0; 4]);
        let latent = cpu_tensor(vec![0.0; 8]);
        let err = <DdimScheduler as Scheduler<CpuBackend>>::step(&mut s, &eps, 958.0, &latent);
        assert!(err.is_err());
    }

    #[test]
    fn step_without_set_timesteps_errors() {
        let mut s = DdimScheduler::new(DdimConfig::sd_1_5()).unwrap();
        let eps = cpu_tensor(vec![0.0; 4]);
        let latent = cpu_tensor(vec![0.0; 4]);
        let err = <DdimScheduler as Scheduler<CpuBackend>>::step(&mut s, &eps, 958.0, &latent);
        assert!(err.is_err());
    }

    #[test]
    fn add_noise_at_t0_returns_near_original() {
        // At the smallest timestep ᾱ_t is closest to 1, so the noise term
        // contributes ~√(1 − ᾱ_0) ≈ √0.00085 ≈ 0.029 of the noise. The
        // original side gets √ᾱ_0 ≈ 0.99957. We use t=0 here as a sanity
        // check that the coefficient lookup hits the head of the table.
        let s = DdimScheduler::new(DdimConfig::sd_1_5()).unwrap();
        let original = cpu_tensor(vec![1.0; 16]);
        let noise = cpu_tensor(vec![0.0; 16]);
        let mixed = <DdimScheduler as Scheduler<CpuBackend>>::add_noise(&s, &original, &noise, 0.0)
            .unwrap();
        let v = mixed.to_vec();
        let sqrt_alpha = s.alphas_cumprod[0].sqrt();
        for x in &v {
            assert!((x - sqrt_alpha).abs() < 1e-6, "got {x}, want {sqrt_alpha}");
        }
    }

    #[test]
    fn add_noise_at_last_timestep_is_mostly_noise() {
        // ᾱ_{T-1} ≈ 0.0047 for SD 1.5, so √(1 − ᾱ) ≈ 0.998 and √ᾱ ≈ 0.068.
        let s = DdimScheduler::new(DdimConfig::sd_1_5()).unwrap();
        let original = cpu_tensor(vec![0.0; 16]);
        let noise = cpu_tensor(vec![1.0; 16]);
        let mixed =
            <DdimScheduler as Scheduler<CpuBackend>>::add_noise(&s, &original, &noise, 999.0)
                .unwrap();
        let v = mixed.to_vec();
        let want = (1.0 - s.alphas_cumprod[999]).sqrt();
        for x in &v {
            assert!((x - want).abs() < 1e-6, "got {x}, want {want}");
        }
    }

    #[test]
    fn add_noise_shape_mismatch_errors() {
        let s = DdimScheduler::new(DdimConfig::sd_1_5()).unwrap();
        let original = cpu_tensor(vec![0.0; 16]);
        let noise = cpu_tensor(vec![0.0; 8]);
        let err = <DdimScheduler as Scheduler<CpuBackend>>::add_noise(&s, &original, &noise, 500.0);
        assert!(err.is_err());
    }

    #[test]
    fn add_noise_does_not_require_set_timesteps() {
        // add_noise is stateless — img2img calls it before the denoise
        // loop, and we want users to be able to wire that without an
        // off-the-trajectory `set_timesteps` first.
        let s = DdimScheduler::new(DdimConfig::sd_1_5()).unwrap();
        let original = cpu_tensor(vec![0.5; 16]);
        let noise = cpu_tensor(vec![0.5; 16]);
        let ok = <DdimScheduler as Scheduler<CpuBackend>>::add_noise(&s, &original, &noise, 500.0);
        assert!(ok.is_ok());
    }
}
