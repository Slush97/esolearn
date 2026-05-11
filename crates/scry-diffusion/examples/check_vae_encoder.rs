// SPDX-License-Identifier: MIT OR Apache-2.0
//! M10 numerical-equivalence gate for the VAE encoder.
//!
//! Loads our `VaeEncoder` from the SD 1.5 checkpoint, encodes a fixed
//! random image (seed 42, `[1, 3, 64, 64]`), and byte-compares the
//! resulting `(mean, logvar)` — each `[1, 4, 8, 8]` — against a
//! reference dump from HF `diffusers.AutoencoderKL.encode(...).latent_dist`.
//! Tolerance is 1e-3 absolute (the standard "this conv stack matches HF
//! in fp32" gate, same as `check_vae_decoder`).
//!
//! The reference dump (from `python/dump_vae_encoder_ref.py`) ships the
//! input image plus HF's `mean` and `logvar` in one safetensors blob.
//! Our example reads the image from there too — that way both sides see
//! byte-identical input regardless of cross-language `torch.manual_seed`
//! RNG drift.
//!
//! Usage:
//!
//! ```text
//! crates/scry-vision/.venv/bin/python \
//!     crates/scry-diffusion/python/dump_vae_encoder_ref.py
//! cargo run -p scry-diffusion --release \
//!     --features safetensors --example check_vae_encoder
//! ```

use std::path::PathBuf;
use std::time::Instant;

use scry_diffusion::vae::encoder::VaeEncoderConfig;
use scry_diffusion::vae::VaeEncoder;
use scry_diffusion::weights::SafetensorsCheckpoint;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

const TOL_ABS: f32 = 1e-3;

#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seed: u32 = std::env::args()
        .nth(1)
        .map_or(42, |s| s.parse().expect("seed must be a u32"));

    let snapshot = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".assets/sd-1-5");
    let ref_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(".assets/refs")
        .join(format!("vae_encoder_seed{seed}.safetensors"));

    if !ref_path.exists() {
        return Err(format!(
            "reference dump not found at {}\n\
             generate it first:\n  \
             crates/scry-vision/.venv/bin/python \
             crates/scry-diffusion/python/dump_vae_encoder_ref.py --seed {seed}",
            ref_path.display()
        )
        .into());
    }

    // ---- Load image + reference mean/logvar -----------------------
    let ref_ckpt = SafetensorsCheckpoint::open(&ref_path)?;
    let image_flat = ref_ckpt.tensor_f32("image")?;
    let hf_mean = ref_ckpt.tensor_f32("mean")?;
    let hf_logvar = ref_ckpt.tensor_f32("logvar")?;
    // (image.len()/3).isqrt() — stable Rust lacks integer sqrt; via f64.
    let image_size = ((image_flat.len() / 3) as f64).sqrt() as usize;
    if image_flat.len() != 3 * image_size * image_size {
        return Err(format!(
            "image has {} elements; expected 3*N*N for some N, got N={image_size}",
            image_flat.len()
        )
        .into());
    }
    let latent_size = image_size / 8;
    let expected_param_len = 4 * latent_size * latent_size;
    for (name, len) in [("mean", hf_mean.len()), ("logvar", hf_logvar.len())] {
        if len != expected_param_len {
            return Err(format!(
                "{name} has {len} elements; expected 4*{latent_size}*{latent_size} = {expected_param_len}",
            )
            .into());
        }
    }
    println!("seed: {seed}");
    println!(
        "image shape:  [3, {image_size}, {image_size}] ({} elements)",
        image_flat.len()
    );
    println!(
        "expected mean/logvar shape: [4, {latent_size}, {latent_size}] ({expected_param_len} elements each)",
    );

    // ---- Load VAE encoder -----------------------------------------
    println!("\nloading VAE encoder weights...");
    let vae_path = snapshot.join("vae/diffusion_pytorch_model.safetensors");
    let vae_ckpt = SafetensorsCheckpoint::open(&vae_path)?;
    let t0 = Instant::now();
    let encoder =
        VaeEncoder::<CpuBackend>::from_safetensors(VaeEncoderConfig::sd_1_5(), &vae_ckpt)?;
    println!("  loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Forward --------------------------------------------------
    let image =
        Tensor::<CpuBackend>::from_vec(image_flat, Shape::new(&[3, image_size, image_size]));
    println!("\nrunning encoder forward (CPU)...");
    let t0 = Instant::now();
    let (mean, logvar) = encoder.encode(&image)?;
    let mean_vec = mean.to_vec();
    let logvar_vec = logvar.to_vec();
    println!(
        "  encode took {:.1}s, mean shape {:?}, logvar shape {:?}",
        t0.elapsed().as_secs_f32(),
        mean.shape.dims(),
        logvar.shape.dims()
    );

    // ---- Compare --------------------------------------------------
    let mut overall_pass = true;
    for (name, ours, theirs) in [
        ("mean", &mean_vec, &hf_mean),
        ("logvar", &logvar_vec, &hf_logvar),
    ] {
        if ours.len() != theirs.len() {
            return Err(format!(
                "{name} length mismatch: ours {} vs HF {}",
                ours.len(),
                theirs.len()
            )
            .into());
        }
        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f64;
        let mut max_pos = 0usize;
        for (i, (a, b)) in ours.iter().zip(theirs.iter()).enumerate() {
            let d = (a - b).abs();
            sum_diff += f64::from(d);
            if d > max_diff {
                max_diff = d;
                max_pos = i;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let mean_diff = sum_diff / ours.len() as f64;
        let plane = latent_size * latent_size;
        let (channel, pixel_idx) = (max_pos / plane, max_pos % plane);
        let (px_y, px_x) = (pixel_idx / latent_size, pixel_idx % latent_size);

        println!("\n{name} diff vs HF reference:");
        println!("  max abs diff:  {max_diff:.3e} at channel {channel} pixel ({px_y},{px_x})");
        println!("  mean abs diff: {mean_diff:.3e}");
        println!("  tolerance:     {TOL_ABS:.0e}");

        if max_diff > TOL_ABS {
            println!("  ✗ FAIL");
            overall_pass = false;
        } else {
            println!("  ✓ pass");
        }
    }

    if !overall_pass {
        return Err(
            format!("one or more diffs exceed {TOL_ABS:.0e} — VAE encoder fails M10 gate").into(),
        );
    }
    println!("\n✓ M10 numerical-equivalence gate PASSED");
    Ok(())
}
