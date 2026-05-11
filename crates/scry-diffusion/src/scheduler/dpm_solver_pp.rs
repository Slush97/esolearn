// SPDX-License-Identifier: MIT OR Apache-2.0
//! DPM-Solver++ multistep sampler (2M variant).
//!
//! Higher-quality, fewer-step alternative to DDIM. 20 DPM-Solver++ steps
//! typically produce equal-or-better samples than 30 DDIM steps for SD 1.5,
//! so swapping schedulers gives ~33% wall-clock for free without touching
//! the UNet. References:
//!
//! - Lu, Zhou, Bao, Chen, Li, Zhu. "DPM-Solver++: Fast Solver for Guided
//!   Sampling of Diffusion Probabilistic Models." (2022).
//! - diffusers `schedulers/scheduling_dpmsolver_multistep.py` with
//!   `algorithm_type="dpmsolver++"`, `solver_order=2`,
//!   `lower_order_final=True` — the canonical configuration.
//!
//! ## Math (data-prediction form, ε-prediction model)
//!
//! Notation: at integer timestep `t`, `ᾱ_t` is the cumulative product of
//! `(1 - β_t)`. Define `α_t = √ᾱ_t`, `σ_t = √(1 − ᾱ_t)`,
//! `λ_t = log(α_t / σ_t)`.
//!
//! 1. Convert ε model output to x_0 prediction:
//!    `x_0 = (x_t − σ_t · ε) / α_t`.
//! 2. Compute step size in log-SNR space: `h_i = λ_{t_{i-1}} − λ_{t_i}`,
//!    where `t_{i-1}` is the next (less-noisy) timestep in the trajectory.
//! 3. **First step** (and last step when `lower_order_final = true`) —
//!    order-1 update, identical to data-prediction DDIM:
//!    `x_{t_{i-1}} = (σ_{t_{i-1}} / σ_{t_i}) · x_{t_i}
//!                 − α_{t_{i-1}} · (e^{−h_i} − 1) · x_0_i`.
//! 4. **Otherwise** — order-2 multistep update using the previous step's x_0:
//!    `r_i = h_{i-1} / h_i`,
//!    `D_i = (1 + 1/(2·r_i)) · x_0_i − (1/(2·r_i)) · x_0_{i-1}`,
//!    `x_{t_{i-1}} = (σ_{t_{i-1}} / σ_{t_i}) · x_{t_i}
//!                 − α_{t_{i-1}} · (e^{−h_i} − 1) · D_i`.
//!
//! On-device cost per step: 5 `MathBackend::scale` + 3 `MathBackend::add`
//! at order-2 (4 + 2 at order-1) — small relative to the UNet forward.

use scry_llm::backend::MathBackend;
use scry_llm::tensor::Tensor;

use super::common::{build_betas, timestep_to_index, BetaSchedule};
use super::Scheduler;
use crate::error::{Error, Result};

/// DPM-Solver++ scheduler configuration.
#[derive(Debug, Clone)]
pub struct DpmSolverPpConfig {
    /// Total number of training timesteps. SD = 1000.
    pub num_train_timesteps: u32,
    /// Beta start. SD = 0.00085.
    pub beta_start: f32,
    /// Beta end. SD = 0.012.
    pub beta_end: f32,
    /// Beta schedule shape.
    pub beta_schedule: BetaSchedule,
    /// `true` for ε-prediction (SD 1.5). v-prediction is not yet implemented
    /// here — would require a different `predict_x0` formula.
    pub epsilon_prediction: bool,
    /// Offset added to every timestep produced by [`DpmSolverPpScheduler::set_timesteps`].
    /// SD 1.5 = 1.
    pub steps_offset: u32,
    /// Solver order. Only `2` (DPM-Solver++(2M)) is implemented.
    pub solver_order: u32,
    /// If `true`, the last step uses an order-1 update for stability.
    /// Diffusers default and what SD 1.5 ships with.
    pub lower_order_final: bool,
    /// If `true`, the "previous" α-cumprod for the final step is `1.0`;
    /// otherwise it is `alphas_cumprod[0]`. SD 1.5 = false.
    pub set_alpha_to_one: bool,
}

impl DpmSolverPpConfig {
    /// SD 1.5 defaults — match `DPMSolverMultistepScheduler` with
    /// `algorithm_type="dpmsolver++"`, `solver_order=2`,
    /// `lower_order_final=True`, `steps_offset=1`,
    /// `timestep_spacing="leading"`, `final_sigmas_type="zero"`.
    ///
    /// `set_alpha_to_one=true` here corresponds to diffusers'
    /// `final_sigmas_type="zero"` — the boundary `σ_prev` is 0 (last step
    /// lands exactly noise-free), not σ at training-timestep 0.
    /// This intentionally differs from DDIM's SD 1.5 default
    /// (which keeps `set_alpha_to_one=false`) because the two scheduler
    /// families pick different boundary conventions in the diffusers config.
    #[must_use]
    pub fn sd_1_5() -> Self {
        Self {
            num_train_timesteps: 1000,
            beta_start: 0.000_85,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            epsilon_prediction: true,
            steps_offset: 1,
            solver_order: 2,
            lower_order_final: true,
            set_alpha_to_one: true,
        }
    }
}

/// DPM-Solver++ multistep scheduler.
///
/// Generic over the math backend. Multistep state (`prev_x0`) lives as a
/// `Tensor<B>`, so a single instance is bound to one backend — pick `B` at
/// construction. The pipeline already concretizes `S: Scheduler<B>`, so this
/// composes the same way as `DdimScheduler`.
pub struct DpmSolverPpScheduler<B: MathBackend> {
    /// Configuration.
    pub config: DpmSolverPpConfig,
    /// Inference-time timestep trajectory (set by `set_timesteps`).
    pub timesteps: Vec<f32>,
    /// Cumulative α product table indexed by integer timestep.
    pub alphas_cumprod: Vec<f32>,
    /// Step size between successive inference timesteps.
    step_size: u32,

    /// Previous step's x_0 prediction. `None` until after the first `step` call.
    prev_x0: Option<Tensor<B>>,
    /// Previous step's `λ_t` (log-SNR at the more-noisy side of the previous step).
    prev_lambda: Option<f32>,
    /// 0-indexed step counter across `step` calls within one inference run.
    step_index: usize,
}

impl<B: MathBackend> DpmSolverPpScheduler<B> {
    /// Construct a scheduler. `set_timesteps` must be called before `step`.
    ///
    /// # Errors
    /// - `num_train_timesteps == 0`.
    /// - `solver_order != 2` (only the 2M variant is implemented).
    pub fn new(config: DpmSolverPpConfig) -> Result<Self> {
        let t = config.num_train_timesteps as usize;
        if t == 0 {
            return Err(Error::Scheduler("num_train_timesteps must be > 0".into()));
        }
        if config.solver_order != 2 {
            return Err(Error::Scheduler(format!(
                "only solver_order = 2 (DPM-Solver++(2M)) is implemented, got {}",
                config.solver_order
            )));
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
            prev_x0: None,
            prev_lambda: None,
            step_index: 0,
        })
    }

    /// Resolve `(α_t, σ_t, λ_t)` at an integer timestep.
    fn alpha_sigma_lambda(&self, timestep: f32) -> Result<(f32, f32, f32)> {
        let idx = timestep_to_index(timestep, self.alphas_cumprod.len())?;
        Ok(self.alpha_sigma_lambda_at_index(idx as usize))
    }

    /// Resolve `(α_prev, σ_prev, λ_prev)` for the next-less-noisy step.
    /// Mirrors DDIM's `prev_signed = idx − step_size` logic.
    fn alpha_sigma_lambda_prev(&self, current_timestep: f32) -> Result<(f32, f32, f32)> {
        if self.step_size == 0 {
            return Err(Error::Scheduler(
                "set_timesteps must be called before step".into(),
            ));
        }
        let cur_idx = timestep_to_index(current_timestep, self.alphas_cumprod.len())?;
        let prev_signed = i64::from(cur_idx) - i64::from(self.step_size);
        let prev_idx = if let Ok(idx) = usize::try_from(prev_signed) {
            idx
        } else if self.config.set_alpha_to_one {
            // Sentinel: caller treats abar = 1.0 → α=1, σ=0, λ=+∞. We avoid
            // returning ±∞ by clamping σ slightly above 0; in practice this
            // path is gated by `set_alpha_to_one` which SD 1.5 disables.
            return Ok((1.0, 0.0, f32::MAX));
        } else {
            0
        };
        Ok(self.alpha_sigma_lambda_at_index(prev_idx))
    }

    fn alpha_sigma_lambda_at_index(&self, idx: usize) -> (f32, f32, f32) {
        let abar = self.alphas_cumprod[idx];
        let alpha = abar.sqrt();
        let sigma = (1.0 - abar).sqrt();
        let lambda = alpha.ln() - sigma.ln();
        (alpha, sigma, lambda)
    }

    /// Materialize the x_0 prediction tensor:
    /// `x_0 = (1/α_t) · x_t + (−σ_t/α_t) · ε`.
    fn predict_x0(
        latent: &Tensor<B>,
        model_output: &Tensor<B>,
        alpha_t: f32,
        sigma_t: f32,
    ) -> Tensor<B> {
        let inv_alpha = 1.0 / alpha_t;
        let coeff_eps = -sigma_t / alpha_t;
        let scaled_x = B::scale(&latent.data, inv_alpha);
        let scaled_eps = B::scale(&model_output.data, coeff_eps);
        let storage = B::add(
            &scaled_x,
            &scaled_eps,
            &latent.shape,
            &model_output.shape,
            &latent.shape,
        );
        Tensor::new(storage, latent.shape.clone())
    }
}

impl<B: MathBackend> Scheduler<B> for DpmSolverPpScheduler<B> {
    fn set_timesteps(&mut self, num_inference_steps: u32) -> Result<()> {
        if num_inference_steps == 0 {
            return Err(Error::Scheduler("num_inference_steps must be > 0".into()));
        }
        if num_inference_steps + 1 > self.config.num_train_timesteps {
            return Err(Error::Scheduler(format!(
                "num_inference_steps + 1 ({}) must be <= num_train_timesteps ({})",
                num_inference_steps + 1,
                self.config.num_train_timesteps
            )));
        }
        // Diffusers `DPMSolverMultistepScheduler` "leading" spacing — note
        // this differs from DDIM's leading by using `(N + 1)` in the
        // denominator, which leaves room for the trailing σ=0 boundary
        // (final_sigmas_type=zero). For SD 1.5 / 30 steps this gives
        // step_ratio = 1000 // 31 = 32 and trajectory
        // [961, 929, …, 33], not DDIM's [958, 925, …, 1].
        let step_ratio = self.config.num_train_timesteps / (num_inference_steps + 1);
        let offset = self.config.steps_offset;
        #[allow(clippy::cast_precision_loss)]
        let ts: Vec<f32> = (1..=num_inference_steps)
            .rev()
            .map(|i| (i * step_ratio + offset) as f32)
            .collect();
        self.timesteps = ts;
        self.step_size = step_ratio;
        // Reset multistep state — a fresh inference run starts at order 1.
        self.prev_x0 = None;
        self.prev_lambda = None;
        self.step_index = 0;
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
        if self.step_size == 0 {
            return Err(Error::Scheduler(
                "set_timesteps must be called before step".into(),
            ));
        }
        if !self.config.epsilon_prediction {
            return Err(Error::Scheduler(
                "only epsilon_prediction is implemented".into(),
            ));
        }

        let total = self.timesteps.len();
        let (alpha_t, sigma_t, lambda_t) = self.alpha_sigma_lambda(timestep)?;
        let is_first = self.step_index == 0;
        let is_last = self.step_index + 1 == total;

        // For DPM++'s "leading" trajectory, the last step's "previous"
        // timestep does not sit below 0 (so the indexed prev wouldn't fire
        // the boundary case via `alpha_sigma_lambda_prev`). Diffusers uses
        // `final_sigmas_type="zero"` to insert σ_prev = 0 explicitly at the
        // last step — we mirror that with `set_alpha_to_one = true` on the
        // SD 1.5 default. λ_prev = +∞ at the boundary; expm1(-h) → -1, so
        // c_d → α_prev (= 1), making the last step land on the x_0 / D
        // prediction directly.
        let (alpha_prev, sigma_prev, lambda_prev) = if is_last && self.config.set_alpha_to_one {
            (1.0, 0.0, f32::MAX)
        } else {
            self.alpha_sigma_lambda_prev(timestep)?
        };

        // Materialize x_0_i (always needed).
        let x0_t = Self::predict_x0(latent, model_output, alpha_t, sigma_t);

        // h_i = λ_{t_{i-1}} − λ_{t_i} > 0 for descending sampling.
        let h_t = lambda_prev - lambda_t;

        // Order selection: 1 for first step or last step (with lower_order_final).
        let use_first_order =
            is_first || (self.config.lower_order_final && is_last) || self.prev_x0.is_none();

        // Common update: x_{t-1} = c_x · x_t + c_d · D, where D is x0 (order-1)
        // or the linear blend (order-2). `expm1(-h)` is more accurate than
        // `(exp(-h) - 1)` for small h.
        let c_x = sigma_prev / sigma_t;
        let c_d = -alpha_prev * (-h_t).exp_m1();

        let next_latent = if use_first_order {
            let scaled_x = B::scale(&latent.data, c_x);
            let scaled_d = B::scale(&x0_t.data, c_d);
            let storage = B::add(
                &scaled_x,
                &scaled_d,
                &latent.shape,
                &x0_t.shape,
                &latent.shape,
            );
            Tensor::new(storage, latent.shape.clone())
        } else {
            // 2nd-order multistep:
            //   h_{i-1} = λ_{t_i} − λ_{t_{i-1}} (the previous λ_t we saved
            //   was at the more-noisy end of the previous step, which equals
            //   λ_{t_{i-1}} in the canonical indexing).
            let prev_lambda_saved = self
                .prev_lambda
                .expect("prev_lambda set when prev_x0 is set");
            let prev_x0 = self
                .prev_x0
                .as_ref()
                .expect("prev_x0 checked above by use_first_order");
            let h_prev = lambda_t - prev_lambda_saved;
            let r_i = h_prev / h_t;
            let blend_t = 1.0 + 0.5 / r_i;
            let blend_prev = -0.5 / r_i;

            // D_i = blend_t · x0_t + blend_prev · x0_prev
            let scaled_x0_t = B::scale(&x0_t.data, blend_t);
            let scaled_x0_prev = B::scale(&prev_x0.data, blend_prev);
            let d_storage = B::add(
                &scaled_x0_t,
                &scaled_x0_prev,
                &x0_t.shape,
                &prev_x0.shape,
                &x0_t.shape,
            );
            let d_tensor: Tensor<B> = Tensor::new(d_storage, x0_t.shape.clone());

            let scaled_x = B::scale(&latent.data, c_x);
            let scaled_d = B::scale(&d_tensor.data, c_d);
            let storage = B::add(
                &scaled_x,
                &scaled_d,
                &latent.shape,
                &d_tensor.shape,
                &latent.shape,
            );
            Tensor::new(storage, latent.shape.clone())
        };

        // Save state for the next step. We always overwrite — the last step's
        // saved state is harmlessly never read.
        self.prev_x0 = Some(x0_t);
        self.prev_lambda = Some(lambda_t);
        self.step_index += 1;

        Ok(next_latent)
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
    fn rejects_unsupported_solver_order() {
        let mut cfg = DpmSolverPpConfig::sd_1_5();
        cfg.solver_order = 3;
        let err = DpmSolverPpScheduler::<CpuBackend>::new(cfg);
        assert!(err.is_err());
    }

    #[test]
    fn alphas_cumprod_match_ddim_construction() {
        // Both schedulers build the same β table from the same config; the
        // α-cumprod table should byte-match.
        use crate::scheduler::ddim::{DdimConfig, DdimScheduler};
        let dpm = DpmSolverPpScheduler::<CpuBackend>::new(DpmSolverPpConfig::sd_1_5()).unwrap();
        let ddim = DdimScheduler::new(DdimConfig::sd_1_5()).unwrap();
        assert_eq!(dpm.alphas_cumprod.len(), ddim.alphas_cumprod.len());
        for (a, b) in dpm.alphas_cumprod.iter().zip(&ddim.alphas_cumprod) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn set_timesteps_30_matches_diffusers_dpmpp_leading_spacing() {
        // Diffusers DPM++ leading: step_ratio = num_train // (num_inf + 1) = 32.
        // Trajectory: [961, 929, …, 33] for 30 steps + steps_offset=1.
        // This intentionally differs from DDIM's [958, 925, …, 1].
        let mut dpm = DpmSolverPpScheduler::<CpuBackend>::new(DpmSolverPpConfig::sd_1_5()).unwrap();
        <DpmSolverPpScheduler<CpuBackend> as Scheduler<CpuBackend>>::set_timesteps(&mut dpm, 30)
            .unwrap();
        let expected: [f32; 30] = [
            961.0, 929.0, 897.0, 865.0, 833.0, 801.0, 769.0, 737.0, 705.0, 673.0, 641.0, 609.0,
            577.0, 545.0, 513.0, 481.0, 449.0, 417.0, 385.0, 353.0, 321.0, 289.0, 257.0, 225.0,
            193.0, 161.0, 129.0, 97.0, 65.0, 33.0,
        ];
        assert_eq!(
            <DpmSolverPpScheduler<CpuBackend> as Scheduler<CpuBackend>>::timesteps(&dpm),
            &expected
        );
    }

    #[test]
    fn step_chain_30_steps_is_finite_and_changes_latent() {
        let mut s = DpmSolverPpScheduler::<CpuBackend>::new(DpmSolverPpConfig::sd_1_5()).unwrap();
        <DpmSolverPpScheduler<CpuBackend> as Scheduler<CpuBackend>>::set_timesteps(&mut s, 30)
            .unwrap();
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
        let timesteps =
            <DpmSolverPpScheduler<CpuBackend> as Scheduler<CpuBackend>>::timesteps(&s).to_vec();
        for &t in &timesteps {
            let eps: CpuTensor = cpu_tensor((0..n).map(|_| next()).collect());
            latent = <DpmSolverPpScheduler<CpuBackend> as Scheduler<CpuBackend>>::step(
                &mut s, &eps, t, &latent,
            )
            .unwrap();
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
        let mut s = DpmSolverPpScheduler::<CpuBackend>::new(DpmSolverPpConfig::sd_1_5()).unwrap();
        <DpmSolverPpScheduler<CpuBackend> as Scheduler<CpuBackend>>::set_timesteps(&mut s, 30)
            .unwrap();
        let eps = cpu_tensor(vec![0.0; 4]);
        let latent = cpu_tensor(vec![0.0; 8]);
        let err = <DpmSolverPpScheduler<CpuBackend> as Scheduler<CpuBackend>>::step(
            &mut s, &eps, 958.0, &latent,
        );
        assert!(err.is_err());
    }

    #[test]
    fn step_without_set_timesteps_errors() {
        let mut s = DpmSolverPpScheduler::<CpuBackend>::new(DpmSolverPpConfig::sd_1_5()).unwrap();
        let eps = cpu_tensor(vec![0.0; 4]);
        let latent = cpu_tensor(vec![0.0; 4]);
        let err = <DpmSolverPpScheduler<CpuBackend> as Scheduler<CpuBackend>>::step(
            &mut s, &eps, 958.0, &latent,
        );
        assert!(err.is_err());
    }

    #[test]
    fn first_step_uses_order_1() {
        // The first step with no saved x_0 must use the order-1 formula.
        // Since we can't introspect the branch, verify by hand-computing.
        let mut s = DpmSolverPpScheduler::<CpuBackend>::new(DpmSolverPpConfig::sd_1_5()).unwrap();
        <DpmSolverPpScheduler<CpuBackend> as Scheduler<CpuBackend>>::set_timesteps(&mut s, 30)
            .unwrap();
        let timesteps =
            <DpmSolverPpScheduler<CpuBackend> as Scheduler<CpuBackend>>::timesteps(&s).to_vec();
        let t0 = timesteps[0];
        let n = 4;
        let latent_data: Vec<f32> = (0..n).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let eps_data: Vec<f32> = (0..n).map(|i| 0.05 * (i as f32 - 1.5)).collect();
        let latent = cpu_tensor(latent_data.clone());
        let eps = cpu_tensor(eps_data.clone());

        let (alpha_t, sigma_t, lambda_t) = s.alpha_sigma_lambda(t0).unwrap();
        let (alpha_prev, sigma_prev, lambda_prev) = s.alpha_sigma_lambda_prev(t0).unwrap();
        let h = lambda_prev - lambda_t;
        let c_x = sigma_prev / sigma_t;
        let c_d = -alpha_prev * (-h).exp_m1();

        // Hand-computed expected: x0 = (x − σ·ε)/α; out = c_x · x + c_d · x0.
        let expected: Vec<f32> = latent_data
            .iter()
            .zip(&eps_data)
            .map(|(&x, &e)| {
                let x0 = (x - sigma_t * e) / alpha_t;
                c_x * x + c_d * x0
            })
            .collect();

        let next = <DpmSolverPpScheduler<CpuBackend> as Scheduler<CpuBackend>>::step(
            &mut s, &eps, t0, &latent,
        )
        .unwrap();
        let got = next.to_vec();
        for (a, b) in expected.iter().zip(&got) {
            assert!((a - b).abs() < 1e-6, "expected {a}, got {b}");
        }
    }

    #[test]
    fn add_noise_matches_ddim_coefficients() {
        // DPM++ and DDIM share their training-time `alphas_cumprod` (the
        // existing `alphas_cumprod_match_ddim_construction` test asserts
        // bit-for-bit equality at construction), so add_noise must yield
        // the same √ᾱ_t · original + √(1 − ᾱ_t) · noise for both.
        use crate::scheduler::ddim::{DdimConfig, DdimScheduler};

        let dpm = DpmSolverPpScheduler::<CpuBackend>::new(DpmSolverPpConfig::sd_1_5()).unwrap();
        let ddim = DdimScheduler::new(DdimConfig::sd_1_5()).unwrap();
        let original = cpu_tensor((0..16).map(|i| 0.01 * (i as f32 + 1.0)).collect());
        let noise = cpu_tensor((0..16).map(|i| 0.02 * (i as f32 - 7.0)).collect());

        let want =
            <DdimScheduler as Scheduler<CpuBackend>>::add_noise(&ddim, &original, &noise, 500.0)
                .unwrap();
        let got = <DpmSolverPpScheduler<CpuBackend> as Scheduler<CpuBackend>>::add_noise(
            &dpm, &original, &noise, 500.0,
        )
        .unwrap();
        for (a, b) in want.to_vec().iter().zip(&got.to_vec()) {
            assert!((a - b).abs() < 1e-7, "want {a}, got {b}");
        }
    }

    #[test]
    fn add_noise_shape_mismatch_errors() {
        let s = DpmSolverPpScheduler::<CpuBackend>::new(DpmSolverPpConfig::sd_1_5()).unwrap();
        let original = cpu_tensor(vec![0.0; 16]);
        let noise = cpu_tensor(vec![0.0; 8]);
        let err = <DpmSolverPpScheduler<CpuBackend> as Scheduler<CpuBackend>>::add_noise(
            &s, &original, &noise, 500.0,
        );
        assert!(err.is_err());
    }
}
