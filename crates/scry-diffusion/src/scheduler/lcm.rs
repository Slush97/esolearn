// SPDX-License-Identifier: MIT OR Apache-2.0
//! LCM (Latent Consistency Model) sampler.
//!
//! Few-step sampler for SD-family models distilled into a consistency
//! function (e.g. LCM-LoRA on top of SD 1.5). Typical use: 4 or 8 inference
//! steps with `guidance_scale = 1.0` (no CFG). References:
//!
//! - Luo, Tan, Huang, Li, Zhao. "Latent Consistency Models: Synthesizing
//!   High-Resolution Images with Few-Step Inference." (2023).
//! - diffusers `schedulers/scheduling_lcm.py` (the canonical reference
//!   implementation; defaults below match its config).
//!
//! ## Math (ε-prediction model, boundary-condition consistency form)
//!
//! 1. Convert ε model output to an x_0 prediction in the standard way:
//!    `x_0_pred = (x_t − √(1 − ᾱ_t) · ε) / √(ᾱ_t)`.
//! 2. Apply the consistency function's boundary scalings, using
//!    `c_skip(t) + (s · c_out(t))² · (1/σ_data²) = 1` and
//!    `c_skip(0) = 1, c_out(0) = 0` so the consistency function reduces
//!    to identity at `t = 0`:
//!    ```text
//!      scaled_t = t · timestep_scaling                       (default 10.0)
//!      c_skip   = σ_data² / (scaled_t² + σ_data²)
//!      c_out    = scaled_t / √(scaled_t² + σ_data²)
//!      denoised = c_out · x_0_pred + c_skip · x_t
//!    ```
//! 3. **Non-final step:** lift `denoised` back to noise level `t_prev` so
//!    the next iteration starts from a forward-process-consistent latent:
//!    `x_{t_prev} = √(ᾱ_{t_prev}) · denoised + √(1 − ᾱ_{t_prev}) · z`
//!    with `z ~ N(0, I)` drawn from a host-side splitmix RNG seeded by
//!    `LcmConfig::noise_seed` and advanced per step.
//!    **Final step:** `x_0 = denoised`, no noise injection.
//!
//! ## Timestep schedule
//!
//! LCM samples its inference trajectory from a coarser "training" grid of
//! `original_inference_steps` points (default 50 over the 1000-step
//! training schedule), then sub-samples `num_inference_steps` of them with
//! `np.linspace(0, original_inference_steps, num, endpoint=False)` floor
//! indices. For SD 1.5 with `num_inference_steps = 4` and defaults this
//! yields `[999, 759, 499, 259]`. The "previous" α-cumprod inside `step`
//! uses `prev_t = max(t − num_train_timesteps/original_inference_steps, 0)`,
//! **not** the inter-step ratio — this is an LCM-specific quirk and is
//! independent of how many inference steps the caller picks.
//!
//! ## Backend ergonomics
//!
//! Per-step noise is sampled on the host (a few kB of f32 for a 512×512
//! SD 1.5 latent) and uploaded with `Tensor::from_vec`; on a CPU backend
//! that's a `Vec` move, on `ScryGpuBackend` a single staging copy. Both
//! are negligible relative to a UNet forward.

use scry_llm::backend::MathBackend;
use scry_llm::tensor::Tensor;

use super::common::{build_betas, timestep_to_index, BetaSchedule};
use super::Scheduler;
use crate::error::{Error, Result};

/// LCM scheduler configuration. Defaults match diffusers' `LCMScheduler`
/// shipped with the SD 1.5 LCM-LoRA distillation.
#[derive(Debug, Clone)]
pub struct LcmConfig {
    /// Total number of training timesteps. SD = 1000.
    pub num_train_timesteps: u32,
    /// Beta start. SD = 0.00085.
    pub beta_start: f32,
    /// Beta end. SD = 0.012.
    pub beta_end: f32,
    /// Beta schedule shape.
    pub beta_schedule: BetaSchedule,
    /// `true` for ε-prediction (SD 1.5). v-prediction is not yet wired here.
    pub epsilon_prediction: bool,
    /// Coarse "teacher" grid size from which the inference trajectory is
    /// sub-sampled. Diffusers default = 50.
    pub original_inference_steps: u32,
    /// Multiplier applied to `timestep` before computing the boundary
    /// scalings `c_skip` / `c_out`. Diffusers default = 10.0.
    pub timestep_scaling: f32,
    /// Standard deviation of the data distribution used in the consistency
    /// function. Diffusers default = 0.5.
    pub sigma_data: f32,
    /// If `true`, the "previous" α-cumprod for the final step is `1.0`;
    /// otherwise it is `alphas_cumprod[0]`. Diffusers default = true.
    pub set_alpha_to_one: bool,
    /// Seed for the per-step noise injection. Two LCM runs with the same
    /// `noise_seed` and same `num_inference_steps` produce identical
    /// trajectories on a given backend.
    pub noise_seed: u64,
}

impl LcmConfig {
    /// SD 1.5 LCM defaults — match diffusers' `LCMScheduler` config shipped
    /// with `latent-consistency/lcm-lora-sdv1-5`.
    #[must_use]
    pub fn sd_1_5() -> Self {
        Self {
            num_train_timesteps: 1000,
            beta_start: 0.000_85,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            epsilon_prediction: true,
            original_inference_steps: 50,
            timestep_scaling: 10.0,
            sigma_data: 0.5,
            set_alpha_to_one: true,
            noise_seed: 0,
        }
    }
}

/// LCM scheduler. Stateless across backends except for the host-side
/// step counter; safe to construct once and run on any `B: MathBackend`.
pub struct LcmScheduler {
    /// Configuration.
    pub config: LcmConfig,
    /// Inference-time timestep trajectory (set by `set_timesteps`).
    pub timesteps: Vec<f32>,
    /// Cumulative α product table indexed by integer timestep.
    pub alphas_cumprod: Vec<f32>,
    /// 0-indexed step counter across `step` calls within one inference run.
    /// Reset by `set_timesteps`. Used to (a) decide whether to inject
    /// noise (skip on final step) and (b) advance the per-step RNG seed.
    step_index: usize,
    /// Number of timesteps queued in the current trajectory; cached at
    /// `set_timesteps` so `step` can detect the final step.
    num_inference_steps: u32,
}

impl LcmScheduler {
    /// Construct a scheduler. `set_timesteps` must be called before `step`.
    ///
    /// # Errors
    /// Returns [`Error::Scheduler`] if `num_train_timesteps == 0` or
    /// `original_inference_steps == 0` or `original_inference_steps`
    /// does not divide `num_train_timesteps` evenly.
    pub fn new(config: LcmConfig) -> Result<Self> {
        let t = config.num_train_timesteps as usize;
        if t == 0 {
            return Err(Error::Scheduler("num_train_timesteps must be > 0".into()));
        }
        if config.original_inference_steps == 0 {
            return Err(Error::Scheduler(
                "original_inference_steps must be > 0".into(),
            ));
        }
        if config.num_train_timesteps % config.original_inference_steps != 0 {
            return Err(Error::Scheduler(format!(
                "original_inference_steps ({}) must divide num_train_timesteps ({})",
                config.original_inference_steps, config.num_train_timesteps
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
            step_index: 0,
            num_inference_steps: 0,
        })
    }

    /// Override the noise seed used by the per-step injection.
    /// Take effect on the next `set_timesteps`.
    pub fn set_noise_seed(&mut self, seed: u64) {
        self.config.noise_seed = seed;
    }

    /// Boundary scalings `(c_skip, c_out)` for the consistency function
    /// evaluated at integer `timestep`. See module docs for the formula.
    fn boundary_scalings(&self, timestep: f32) -> (f32, f32) {
        let scaled = timestep * self.config.timestep_scaling;
        let s2 = self.config.sigma_data * self.config.sigma_data;
        let denom = scaled.mul_add(scaled, s2);
        let c_skip = s2 / denom;
        let c_out = scaled / denom.sqrt();
        (c_skip, c_out)
    }

    /// α-cumprod at the "previous" (less-noisy) inference timestep for
    /// LCM's noise re-injection. Diffusers' `LCMScheduler.step` looks
    /// this up as `alphas_cumprod[timesteps[step_index + 1]]` — the
    /// NEXT entry in the inference trajectory, not a fixed offset from
    /// the current training timestep. For the final step the caller
    /// skips noise injection entirely, so this function only matters
    /// for `step_index < num_inference_steps - 1`.
    fn alpha_prod_t_prev_inference(&self) -> f32 {
        let next_idx = self.step_index + 1;
        if next_idx >= self.timesteps.len() {
            return if self.config.set_alpha_to_one {
                1.0
            } else {
                self.alphas_cumprod[0]
            };
        }
        let prev_t = self.timesteps[next_idx];
        // The trajectory is built from integer timesteps so this round
        // is exact, but timestep_to_index still guards the bound.
        let prev_t_idx = timestep_to_index(prev_t, self.alphas_cumprod.len())
            .expect("trajectory timestep is in-range by construction");
        self.alphas_cumprod[prev_t_idx as usize]
    }
}

impl<B: MathBackend> Scheduler<B> for LcmScheduler {
    fn set_timesteps(&mut self, num_inference_steps: u32) -> Result<()> {
        if num_inference_steps == 0 {
            return Err(Error::Scheduler("num_inference_steps must be > 0".into()));
        }
        if num_inference_steps > self.config.original_inference_steps {
            return Err(Error::Scheduler(format!(
                "num_inference_steps ({}) must be <= original_inference_steps ({})",
                num_inference_steps, self.config.original_inference_steps
            )));
        }
        // Diffusers' LCM trajectory (`scheduling_lcm.py::set_timesteps`):
        //   k_train  = num_train / original_inference                  # 1000/50 = 20
        //   origin   = [k_train-1, 2·k_train-1, ..., original_inference·k_train - 1]
        //   skipping = original_inference // num_inference              # 50/4 = 12
        //   timesteps = origin[::-1][::skipping][:num_inference]
        // The slice `[::skipping]` picks indices `i·skipping`, **not**
        // `floor(i · n_origin / n_inf)` — they differ by 1 at every
        // sub-multiple position. For 4-step LCM on SD 1.5 the correct
        // trajectory is `[999, 759, 519, 279]`, not `[..., 499, 259]`.
        let k_train = self.config.num_train_timesteps / self.config.original_inference_steps;
        let origin: Vec<u32> = (1..=self.config.original_inference_steps)
            .map(|i| i * k_train - 1)
            .collect();
        let n_origin = origin.len();
        let n_inf = num_inference_steps as usize;
        let skipping = n_origin / n_inf;
        let mut ts = Vec::with_capacity(n_inf);
        for i in 0..n_inf {
            // Position in the reversed-origin sequence: `i * skipping`.
            let reversed_idx = i * skipping;
            // Convert to forward-origin index.
            let origin_idx = n_origin - 1 - reversed_idx;
            #[allow(clippy::cast_precision_loss)]
            ts.push(origin[origin_idx] as f32);
        }
        self.timesteps = ts;
        self.step_index = 0;
        self.num_inference_steps = num_inference_steps;
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
        if self.num_inference_steps == 0 {
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
                "LCM v-prediction is not yet implemented".into(),
            ));
        }

        let t_idx = timestep_to_index(timestep, self.alphas_cumprod.len())?;
        let alpha_prod_t = self.alphas_cumprod[t_idx as usize];
        let sqrt_alpha_t = alpha_prod_t.sqrt();
        let sqrt_one_minus_t = (1.0 - alpha_prod_t).sqrt();

        // (1) Predicted x_0 from ε. Reshape as a `c_x · x_t + c_eps · ε`
        //     combine so the on-device cost is one `scale` + one `scale`
        //     + one `add` — same shape as the DDIM step.
        //       pred_x0 = (1/√ᾱ_t) · x_t + (−√(1−ᾱ_t)/√ᾱ_t) · ε
        let inv_sqrt_alpha = 1.0 / sqrt_alpha_t;
        let c_pred_x = inv_sqrt_alpha;
        let c_pred_eps = -sqrt_one_minus_t * inv_sqrt_alpha;

        // (2) Boundary scalings → consistency-function denoised output:
        //       denoised = c_out · pred_x0 + c_skip · x_t
        //                = (c_out · c_pred_x + c_skip) · x_t
        //                  + (c_out · c_pred_eps)       · ε
        let (c_skip, c_out) = self.boundary_scalings(timestep);
        let coeff_x = c_out.mul_add(c_pred_x, c_skip);
        let coeff_eps = c_out * c_pred_eps;

        let scaled_x = B::scale(&latent.data, coeff_x);
        let scaled_eps = B::scale(&model_output.data, coeff_eps);
        let out_shape = latent.shape.clone();
        let denoised_storage = B::add(&scaled_x, &scaled_eps, &out_shape, &out_shape, &out_shape);

        let is_final = self.step_index as u32 + 1 >= self.num_inference_steps;
        let result_storage = if is_final {
            denoised_storage
        } else {
            // (3) Lift back up to noise level `t_prev`:
            //       prev = √ᾱ_{t_prev} · denoised + √(1−ᾱ_{t_prev}) · z
            let alpha_prev = self.alpha_prod_t_prev_inference();
            let sqrt_alpha_prev = alpha_prev.sqrt();
            let sqrt_one_minus_prev = (1.0 - alpha_prev).sqrt();

            let noise_seed = self
                .config
                .noise_seed
                .wrapping_add((self.step_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let noise_host = sample_standard_normal(out_shape.numel(), noise_seed);
            let noise_tensor: Tensor<B> = Tensor::from_vec(noise_host, out_shape.clone());

            let scaled_denoised = B::scale(&denoised_storage, sqrt_alpha_prev);
            let scaled_noise = B::scale(&noise_tensor.data, sqrt_one_minus_prev);
            B::add(
                &scaled_denoised,
                &scaled_noise,
                &out_shape,
                &out_shape,
                &out_shape,
            )
        };

        self.step_index += 1;
        Ok(Tensor::new(result_storage, out_shape))
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

/// Host-side `N(0, 1)` sampler — splitmix64 → Box–Muller. Duplicated from
/// `pipeline::sample_standard_normal` to keep the scheduler module
/// self-contained (and so the bench/check examples can run LCM without
/// pulling in the full pipeline).
fn sample_standard_normal(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next_u64 = || -> u64 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut next_uniform = || -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let r = (next_u64() >> 40) as f32;
        (r + 0.5) / 16_777_216.0
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
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
    use scry_llm::backend::cpu::CpuBackend;
    use scry_llm::tensor::shape::Shape;

    type CpuTensor = Tensor<CpuBackend>;

    fn cpu_tensor(data: Vec<f32>) -> CpuTensor {
        let len = data.len();
        Tensor::from_vec(data, Shape::new(&[len]))
    }

    #[test]
    fn set_timesteps_4_matches_hf_trajectory() {
        // diffusers reference: 4-step LCM on SD 1.5 with original_inference_steps=50
        // → timesteps = [999, 759, 519, 279]. Cross-checked against
        // `LCMScheduler.set_timesteps(4, lcm_origin_steps=50)` on the
        // `SimianLuo/LCM_Dreamshaper_v7` reference pipeline.
        let mut s = LcmScheduler::new(LcmConfig::sd_1_5()).unwrap();
        <LcmScheduler as Scheduler<CpuBackend>>::set_timesteps(&mut s, 4).unwrap();
        let got = <LcmScheduler as Scheduler<CpuBackend>>::timesteps(&s);
        assert_eq!(got, &[999.0_f32, 759.0, 519.0, 279.0]);
    }

    #[test]
    fn set_timesteps_8_starts_at_top_and_descends() {
        let mut s = LcmScheduler::new(LcmConfig::sd_1_5()).unwrap();
        <LcmScheduler as Scheduler<CpuBackend>>::set_timesteps(&mut s, 8).unwrap();
        let ts = <LcmScheduler as Scheduler<CpuBackend>>::timesteps(&s).to_vec();
        assert_eq!(ts.len(), 8);
        assert!((ts[0] - 999.0).abs() < f32::EPSILON);
        for w in ts.windows(2) {
            assert!(w[0] > w[1], "trajectory not strictly descending: {ts:?}");
        }
    }

    #[test]
    fn boundary_scalings_match_paper_limits() {
        // c_skip(0) = 1, c_out(0) = 0 (identity at the boundary).
        let s = LcmScheduler::new(LcmConfig::sd_1_5()).unwrap();
        let (c_skip0, c_out0) = s.boundary_scalings(0.0);
        assert!((c_skip0 - 1.0).abs() < 1e-6, "c_skip(0) = {c_skip0}");
        assert!(c_out0.abs() < 1e-6, "c_out(0) = {c_out0}");

        // At the noise end c_skip → 0 and c_out → 1.
        let (c_skip_hi, c_out_hi) = s.boundary_scalings(999.0);
        assert!(c_skip_hi < 1e-6, "c_skip(999) = {c_skip_hi}");
        assert!((c_out_hi - 1.0).abs() < 1e-6, "c_out(999) = {c_out_hi}");
    }

    #[test]
    fn step_chain_4_steps_is_finite_and_changes_latent() {
        let mut s = LcmScheduler::new(LcmConfig::sd_1_5()).unwrap();
        <LcmScheduler as Scheduler<CpuBackend>>::set_timesteps(&mut s, 4).unwrap();
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
        let timesteps = <LcmScheduler as Scheduler<CpuBackend>>::timesteps(&s).to_vec();
        for &t in &timesteps {
            let eps: CpuTensor = cpu_tensor((0..n).map(|_| next()).collect());
            latent =
                <LcmScheduler as Scheduler<CpuBackend>>::step(&mut s, &eps, t, &latent).unwrap();
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
    fn same_noise_seed_is_deterministic() {
        // Two independent runs with the same `noise_seed` and same inputs
        // should produce bit-identical trajectories on CpuBackend.
        let mut cfg = LcmConfig::sd_1_5();
        cfg.noise_seed = 0xCAFE_BABE;
        let run = |seed_offset: u64| -> Vec<f32> {
            let mut s = LcmScheduler::new(cfg.clone()).unwrap();
            <LcmScheduler as Scheduler<CpuBackend>>::set_timesteps(&mut s, 4).unwrap();
            let n = 4 * 8 * 8;
            // Same model_output / latent across both runs.
            let _ = seed_offset;
            let latent_data: Vec<f32> = (0..n).map(|i| (i as f32) * 1e-3).collect();
            let eps_data: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
            let mut latent: CpuTensor = cpu_tensor(latent_data);
            let timesteps = <LcmScheduler as Scheduler<CpuBackend>>::timesteps(&s).to_vec();
            for &t in &timesteps {
                let eps: CpuTensor = cpu_tensor(eps_data.clone());
                latent = <LcmScheduler as Scheduler<CpuBackend>>::step(&mut s, &eps, t, &latent)
                    .unwrap();
            }
            latent.to_vec()
        };
        assert_eq!(run(0), run(0));
    }

    #[test]
    fn different_noise_seeds_diverge() {
        let mut cfg_a = LcmConfig::sd_1_5();
        cfg_a.noise_seed = 1;
        let mut cfg_b = LcmConfig::sd_1_5();
        cfg_b.noise_seed = 2;
        let run = |cfg: LcmConfig| -> Vec<f32> {
            let mut s = LcmScheduler::new(cfg).unwrap();
            <LcmScheduler as Scheduler<CpuBackend>>::set_timesteps(&mut s, 4).unwrap();
            let n = 4 * 8 * 8;
            let latent_data: Vec<f32> = (0..n).map(|i| (i as f32) * 1e-3).collect();
            let eps_data: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
            let mut latent: CpuTensor = cpu_tensor(latent_data);
            let timesteps = <LcmScheduler as Scheduler<CpuBackend>>::timesteps(&s).to_vec();
            for &t in &timesteps {
                let eps: CpuTensor = cpu_tensor(eps_data.clone());
                latent = <LcmScheduler as Scheduler<CpuBackend>>::step(&mut s, &eps, t, &latent)
                    .unwrap();
            }
            latent.to_vec()
        };
        let a = run(cfg_a);
        let b = run(cfg_b);
        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_diff > 1e-4, "seeds did not diverge: {max_diff}");
    }

    #[test]
    fn step_without_set_timesteps_errors() {
        let mut s = LcmScheduler::new(LcmConfig::sd_1_5()).unwrap();
        let eps = cpu_tensor(vec![0.0; 4]);
        let latent = cpu_tensor(vec![0.0; 4]);
        let err = <LcmScheduler as Scheduler<CpuBackend>>::step(&mut s, &eps, 999.0, &latent);
        assert!(err.is_err());
    }

    #[test]
    fn step_size_mismatch_errors() {
        let mut s = LcmScheduler::new(LcmConfig::sd_1_5()).unwrap();
        <LcmScheduler as Scheduler<CpuBackend>>::set_timesteps(&mut s, 4).unwrap();
        let eps = cpu_tensor(vec![0.0; 4]);
        let latent = cpu_tensor(vec![0.0; 8]);
        let err = <LcmScheduler as Scheduler<CpuBackend>>::step(&mut s, &eps, 999.0, &latent);
        assert!(err.is_err());
    }

    #[test]
    fn add_noise_matches_ddim_form() {
        // LCM's add_noise uses the same forward-diffusion identity as DDIM.
        let s = LcmScheduler::new(LcmConfig::sd_1_5()).unwrap();
        let original = cpu_tensor(vec![1.0; 16]);
        let noise = cpu_tensor(vec![0.0; 16]);
        let mixed =
            <LcmScheduler as Scheduler<CpuBackend>>::add_noise(&s, &original, &noise, 0.0).unwrap();
        let v = mixed.to_vec();
        let want = s.alphas_cumprod[0].sqrt();
        for x in &v {
            assert!((x - want).abs() < 1e-6, "got {x}, want {want}");
        }
    }

    #[test]
    fn unsupported_inference_steps_errors() {
        let mut s = LcmScheduler::new(LcmConfig::sd_1_5()).unwrap();
        // Diffusers caps `num_inference_steps` at `original_inference_steps`.
        let err = <LcmScheduler as Scheduler<CpuBackend>>::set_timesteps(&mut s, 51);
        assert!(err.is_err());
    }
}
