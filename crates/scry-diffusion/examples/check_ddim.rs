// SPDX-License-Identifier: MIT OR Apache-2.0
//! M7 numerical-equivalence gate for the DDIM scheduler.
//!
//! Loads a reference trajectory dumped by `python/dump_ddim_ref.py` (HF
//! `DDIMScheduler` with SD 1.5 config, 30 steps, η=0, ε-prediction) and
//! replays the same `step` chain through our [`DdimScheduler`]. Compares
//! the final latent at 1e-4 absolute. The chain runs 30 deterministic
//! ε-prediction updates; a 16×16×4 SD-1.5 trajectory can drift up to
//! ~5e-5 absolute from order-of-operation differences in cumprod / sqrt
//! between PyTorch and our pure-Rust path, which is ~1e-6 relative
//! against the ±40 scale of the final latent.
//!
//! Both sides see byte-identical inputs (init latent + per-step noise)
//! to keep cross-language RNG drift out of the test.
//!
//! Usage:
//!
//! ```text
//! crates/scry-vision/.venv/bin/python \
//!     crates/scry-diffusion/python/dump_ddim_ref.py
//! cargo run -p scry-diffusion --release \
//!     --features safetensors --example check_ddim
//! ```

use std::path::PathBuf;

use scry_diffusion::scheduler::ddim::{DdimConfig, DdimScheduler};
use scry_diffusion::scheduler::Scheduler;
use scry_diffusion::weights::SafetensorsCheckpoint;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

const TOL_ABS: f32 = 1e-4;
const ALPHAS_TOL: f32 = 5e-5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ref_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".assets/refs/ddim_seed42.safetensors");
    if !ref_path.exists() {
        return Err(format!(
            "reference dump not found at {}\n\
             generate it first:\n  \
             crates/scry-vision/.venv/bin/python \
             crates/scry-diffusion/python/dump_ddim_ref.py",
            ref_path.display()
        )
        .into());
    }

    let ckpt = SafetensorsCheckpoint::open(&ref_path)?;
    let init_latent = ckpt.tensor_f32("init_latent")?;
    let eps_seq = ckpt.tensor_f32("eps_seq")?;
    let timesteps = ckpt.tensor_f32("timesteps")?;
    let alphas_cumprod_ref = ckpt.tensor_f32("alphas_cumprod")?;
    let final_latent_ref = ckpt.tensor_f32("final_latent")?;

    let num_steps = timesteps.len();
    let elements = init_latent.len();
    if eps_seq.len() != num_steps * elements {
        return Err(format!(
            "eps_seq length {} != num_steps {} * elements {}",
            eps_seq.len(),
            num_steps,
            elements
        )
        .into());
    }

    let cfg = DdimConfig::sd_1_5();
    let mut sched = DdimScheduler::new(cfg)?;
    <DdimScheduler as Scheduler<CpuBackend>>::set_timesteps(&mut sched, u32::try_from(num_steps)?)?;

    // ---- Sanity: alphas_cumprod table matches HF ----
    if sched.alphas_cumprod.len() != alphas_cumprod_ref.len() {
        return Err(format!(
            "alphas_cumprod length: ours {} vs HF {}",
            sched.alphas_cumprod.len(),
            alphas_cumprod_ref.len()
        )
        .into());
    }
    let max_alpha_diff = sched
        .alphas_cumprod
        .iter()
        .zip(&alphas_cumprod_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("alphas_cumprod max abs diff vs HF: {max_alpha_diff:.3e}");
    if max_alpha_diff > ALPHAS_TOL {
        return Err(format!(
            "alphas_cumprod diff {max_alpha_diff:.3e} exceeds tolerance {ALPHAS_TOL:.0e}"
        )
        .into());
    }

    // ---- Sanity: timestep trajectory matches HF ----
    let our_timesteps = <DdimScheduler as Scheduler<CpuBackend>>::timesteps(&sched).to_vec();
    if our_timesteps != timesteps {
        return Err(format!(
            "timestep trajectory mismatch:\n  ours: {our_timesteps:?}\n  HF:   {timesteps:?}",
        )
        .into());
    }
    println!("timesteps match HF exactly ({num_steps} steps)");

    // ---- Replay the step chain ----
    let latent_shape = Shape::new(&[elements]);
    let mut latent: Tensor<CpuBackend> =
        Tensor::from_vec(init_latent.clone(), latent_shape.clone());
    for (i, &t) in timesteps.iter().enumerate() {
        let eps = eps_seq[i * elements..(i + 1) * elements].to_vec();
        let eps_tensor = Tensor::<CpuBackend>::from_vec(eps, latent_shape.clone());
        latent =
            <DdimScheduler as Scheduler<CpuBackend>>::step(&mut sched, &eps_tensor, t, &latent)?;
    }
    let latent_v = latent.to_vec();

    if latent_v.len() != final_latent_ref.len() {
        return Err(format!(
            "final latent length: ours {} vs HF {}",
            latent_v.len(),
            final_latent_ref.len()
        )
        .into());
    }
    let mut max_diff = 0.0_f32;
    let mut sum_diff = 0.0_f64;
    let mut max_pos = 0usize;
    for (i, (a, b)) in latent_v.iter().zip(&final_latent_ref).enumerate() {
        let d = (a - b).abs();
        sum_diff += f64::from(d);
        if d > max_diff {
            max_diff = d;
            max_pos = i;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let mean_diff = sum_diff / latent_v.len() as f64;
    println!(
        "final latent diff vs HF: max abs {max_diff:.3e} at index {max_pos}, mean abs {mean_diff:.3e}"
    );

    if max_diff > TOL_ABS {
        return Err(format!(
            "max diff {max_diff:.3e} exceeds tolerance {TOL_ABS:.0e} — DDIM fails M7 gate"
        )
        .into());
    }
    println!("\n✓ M7 numerical-equivalence gate PASSED");
    Ok(())
}
