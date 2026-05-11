// SPDX-License-Identifier: MIT OR Apache-2.0
//! M10 v2 numerical-equivalence gate for the img2img init-latent path.
//!
//! `SdPipeline::img2img` composes three primitives end-to-end:
//!
//!   1. `VaeEncoder::encode(image_norm) -> (mean, logvar)` — already
//!      byte-parity-gated by `check_vae_encoder` (M10 v1).
//!   2. Reparameterization + scaling:
//!      `latent = (mean + exp(0.5·logvar) * noise_enc) * scaling_factor`.
//!   3. `Scheduler::add_noise(latent, noise_add, t_start)`.
//!
//! This example loads `dump_img2img_ref.py`'s output — image, both noise
//! tensors, the `t_start` timestep, and HF's final `init_latent_post_noise`
//! — runs primitives 1 + 2 + 3 against the loaded noises (no RNG byte
//! parity needed between torch and our sampler) and asserts max-abs
//! diff < 1e-3 against HF.
//!
//! The full denoise loop is intentionally not exercised here: scheduler
//! `step` is already gated by `check_ddim` / `check_dpmpp`, the `UNet` by
//! `check_unet`, and the VAE decoder by `check_vae_decoder` — the
//! pipeline's `img2img()` is composition of validated pieces with the
//! init-latent path being the only new math.
//!
//! Usage:
//!
//! ```text
//! crates/scry-vision/.venv/bin/python \
//!     crates/scry-diffusion/python/dump_img2img_ref.py
//! cargo run -p scry-diffusion --release \
//!     --features safetensors --example check_img2img
//! ```

use std::path::PathBuf;
use std::time::Instant;

use scry_diffusion::scheduler::ddim::{DdimConfig, DdimScheduler};
use scry_diffusion::scheduler::Scheduler;
use scry_diffusion::vae::encoder::VaeEncoderConfig;
use scry_diffusion::vae::VaeEncoder;
use scry_diffusion::weights::SafetensorsCheckpoint;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::MathBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

const TOL_ABS: f32 = 1e-3;

#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seed: u32 = std::env::args()
        .nth(1)
        .map_or(42, |s| s.parse().expect("seed must be a u32"));

    let snapshot = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".assets/sd-1-5");
    let ref_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(".assets/refs")
        .join(format!("img2img_seed{seed}.safetensors"));

    if !ref_path.exists() {
        return Err(format!(
            "reference dump not found at {}\n\
             generate it first:\n  \
             crates/scry-vision/.venv/bin/python \
             crates/scry-diffusion/python/dump_img2img_ref.py --seed {seed}",
            ref_path.display()
        )
        .into());
    }

    // ---- Load reference dump --------------------------------------
    let ref_ckpt = SafetensorsCheckpoint::open(&ref_path)?;
    let image_flat = ref_ckpt.tensor_f32("image_norm")?;
    let noise_enc_flat = ref_ckpt.tensor_f32("noise_enc")?;
    let noise_add_flat = ref_ckpt.tensor_f32("noise_add")?;
    let hf_init_latent = ref_ckpt.tensor_f32("init_latent_post_noise")?;
    let t_start_vec = ref_ckpt.tensor_f32("t_start_timestep")?;
    if t_start_vec.len() != 1 {
        return Err(format!(
            "t_start_timestep must be 1 element, got {}",
            t_start_vec.len()
        )
        .into());
    }
    let t_start = t_start_vec[0];

    let image_size = ((image_flat.len() / 3) as f64).sqrt() as usize;
    if image_flat.len() != 3 * image_size * image_size {
        return Err(format!(
            "image has {} elements; expected 3*N*N, got N={image_size}",
            image_flat.len()
        )
        .into());
    }
    let latent_size = image_size / 8;
    let latent_numel = 4 * latent_size * latent_size;
    for (name, len) in [
        ("noise_enc", noise_enc_flat.len()),
        ("noise_add", noise_add_flat.len()),
        ("init_latent_post_noise", hf_init_latent.len()),
    ] {
        if len != latent_numel {
            return Err(format!(
                "{name} has {len} elements; expected 4*{latent_size}*{latent_size} = {latent_numel}",
            )
            .into());
        }
    }

    println!("seed: {seed}");
    println!("image shape:  [3, {image_size}, {image_size}]");
    println!("latent shape: [4, {latent_size}, {latent_size}]");
    println!("t_start_timestep: {t_start}");

    // ---- Load VAE encoder -----------------------------------------
    println!("\nloading VAE encoder weights...");
    let vae_path = snapshot.join("vae/diffusion_pytorch_model.safetensors");
    let vae_ckpt = SafetensorsCheckpoint::open(&vae_path)?;
    let cfg = VaeEncoderConfig::sd_1_5();
    let scaling_factor = cfg.scaling_factor;
    let t0 = Instant::now();
    let encoder = VaeEncoder::<CpuBackend>::from_safetensors(cfg, &vae_ckpt)?;
    println!("  loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- VAE encode -----------------------------------------------
    let image =
        Tensor::<CpuBackend>::from_vec(image_flat, Shape::new(&[3, image_size, image_size]));
    println!("\nrunning VAE encoder (CPU)...");
    let t0 = Instant::now();
    let (mean, logvar) = encoder.encode(&image)?;
    println!("  encode took {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Reparameterization + scaling -----------------------------
    //   latent = (mean + exp(0.5·logvar) * noise_enc) * scaling_factor
    let latent_shape = Shape::new(&[4, latent_size, latent_size]);
    let std_host: Vec<f32> = logvar
        .to_vec()
        .into_iter()
        .map(|v| (0.5 * v).exp())
        .collect();
    let std_t: Tensor<CpuBackend> = Tensor::from_vec(std_host, latent_shape.clone());
    let noise_enc: Tensor<CpuBackend> = Tensor::from_vec(noise_enc_flat, latent_shape.clone());
    let scaled_noise = CpuBackend::mul_elementwise(&std_t.data, &noise_enc.data);
    let sampled = CpuBackend::add(
        &mean.data,
        &scaled_noise,
        &mean.shape,
        &latent_shape,
        &latent_shape,
    );
    let latent_storage = CpuBackend::scale(&sampled, scaling_factor);
    let latent: Tensor<CpuBackend> = Tensor::new(latent_storage, latent_shape.clone());

    // ---- add_noise via scheduler ----------------------------------
    let scheduler = DdimScheduler::new(DdimConfig::sd_1_5())?;
    let noise_add: Tensor<CpuBackend> = Tensor::from_vec(noise_add_flat, latent_shape.clone());
    let after_add = <DdimScheduler as Scheduler<CpuBackend>>::add_noise(
        &scheduler, &latent, &noise_add, t_start,
    )?;
    let after_add_vec = after_add.to_vec();

    // ---- Compare --------------------------------------------------
    if after_add_vec.len() != hf_init_latent.len() {
        return Err(format!(
            "length mismatch: ours {} vs HF {}",
            after_add_vec.len(),
            hf_init_latent.len()
        )
        .into());
    }
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    let mut max_pos = 0usize;
    for (i, (a, b)) in after_add_vec.iter().zip(hf_init_latent.iter()).enumerate() {
        let d = (a - b).abs();
        sum_diff += f64::from(d);
        if d > max_diff {
            max_diff = d;
            max_pos = i;
        }
    }
    let mean_diff = sum_diff / after_add_vec.len() as f64;
    let plane = latent_size * latent_size;
    let (channel, pixel_idx) = (max_pos / plane, max_pos % plane);
    let (px_y, px_x) = (pixel_idx / latent_size, pixel_idx % latent_size);

    println!("\ninit_latent_post_noise diff vs HF reference:");
    println!("  max abs diff:  {max_diff:.3e} at channel {channel} pixel ({px_y},{px_x})");
    println!("  mean abs diff: {mean_diff:.3e}");
    println!("  tolerance:     {TOL_ABS:.0e}");

    if max_diff > TOL_ABS {
        println!("  ✗ FAIL");
        return Err(format!(
            "init-latent diff {max_diff:.3e} exceeds {TOL_ABS:.0e} — img2img fails M10 v2 gate"
        )
        .into());
    }
    println!("  ✓ pass");
    println!("\n✓ M10 v2 img2img init-latent gate PASSED");
    Ok(())
}
