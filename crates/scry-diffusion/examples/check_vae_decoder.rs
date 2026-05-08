// SPDX-License-Identifier: MIT OR Apache-2.0
//! M4 numerical-equivalence gate for the VAE decoder.
//!
//! Loads our `VaeDecoder` from the SD 1.5 checkpoint, decodes a fixed
//! noise latent (seed 42, `[1, 4, 64, 64]`), and byte-compares the
//! `[1, 3, 512, 512]` output against a reference dump from HF
//! `diffusers.AutoencoderKL.decode`. Tolerance is 1e-3 absolute, the
//! HANDOFF gate for "this conv stack matches HF in fp32".
//!
//! The reference dump (from `python/dump_vae_decoder_ref.py`) ships
//! both the input latent and HF's decoded output in the same
//! safetensors blob. Our example reads the latent from there too —
//! that way both sides see byte-identical input regardless of any
//! cross-language `torch.manual_seed` RNG drift.
//!
//! Usage:
//!
//! ```text
//! crates/scry-vision/.venv/bin/python \
//!     crates/scry-diffusion/python/dump_vae_decoder_ref.py
//! cargo run -p scry-diffusion --release \
//!     --features safetensors --example check_vae_decoder
//! ```

use std::path::PathBuf;
use std::time::Instant;

use scry_diffusion::vae::{decoder::VaeDecoderConfig, VaeDecoder};
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
        .join(format!("vae_decoder_seed{seed}.safetensors"));

    if !ref_path.exists() {
        return Err(format!(
            "reference dump not found at {}\n\
             generate it first:\n  \
             crates/scry-vision/.venv/bin/python \
             crates/scry-diffusion/python/dump_vae_decoder_ref.py --seed {seed}",
            ref_path.display()
        )
        .into());
    }

    // ---- Load latent + reference output ---------------------------
    let ref_ckpt = SafetensorsCheckpoint::open(&ref_path)?;
    let latent_flat = ref_ckpt.tensor_f32("latent")?;
    let hf_decoded = ref_ckpt.tensor_f32("decoded")?;
    // (latent.len()/4).isqrt() — but stable Rust doesn't have integer
    // sqrt yet, so go through f64.
    let latent_size = ((latent_flat.len() / 4) as f64).sqrt() as usize;
    if latent_flat.len() != 4 * latent_size * latent_size {
        return Err(format!(
            "latent has {} elements; expected 4*N*N for some N, got N={latent_size}",
            latent_flat.len()
        )
        .into());
    }
    let img_size = latent_size * 8;
    let expected_decoded_len = 3 * img_size * img_size;
    if hf_decoded.len() != expected_decoded_len {
        return Err(format!(
            "decoded has {} elements; expected 3*{img_size}*{img_size} = {expected_decoded_len}",
            hf_decoded.len()
        )
        .into());
    }
    println!("seed: {seed}");
    println!(
        "latent shape:  [4, {latent_size}, {latent_size}] ({} elements)",
        latent_flat.len()
    );
    println!(
        "expected output shape: [3, {img_size}, {img_size}] ({} elements)",
        hf_decoded.len()
    );

    // ---- Load VAE decoder -----------------------------------------
    println!("\nloading VAE decoder weights...");
    let vae_path = snapshot.join("vae/diffusion_pytorch_model.safetensors");
    let vae_ckpt = SafetensorsCheckpoint::open(&vae_path)?;
    let t0 = Instant::now();
    let decoder =
        VaeDecoder::<CpuBackend>::from_safetensors(VaeDecoderConfig::sd_1_5(), &vae_ckpt)?;
    println!("  loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Forward --------------------------------------------------
    let latent =
        Tensor::<CpuBackend>::from_vec(latent_flat, Shape::new(&[4, latent_size, latent_size]));
    println!("\nrunning decoder forward (CPU)...");
    let t0 = Instant::now();
    let our_out = decoder.decode(&latent)?;
    let our_vec = our_out.to_vec();
    println!(
        "  decode took {:.1}s, output shape {:?}, {} elements",
        t0.elapsed().as_secs_f32(),
        our_out.shape.dims(),
        our_vec.len()
    );

    // ---- Compare --------------------------------------------------
    if our_vec.len() != hf_decoded.len() {
        return Err(format!(
            "output length mismatch: ours {} vs HF {}",
            our_vec.len(),
            hf_decoded.len()
        )
        .into());
    }
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    let mut max_pos = 0usize;
    for (i, (a, b)) in our_vec.iter().zip(hf_decoded.iter()).enumerate() {
        let d = (a - b).abs();
        sum_diff += f64::from(d);
        if d > max_diff {
            max_diff = d;
            max_pos = i;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let mean_diff = sum_diff / our_vec.len() as f64;
    let plane = img_size * img_size;
    let (channel, pixel_idx) = (max_pos / plane, max_pos % plane);
    let (px_y, px_x) = (pixel_idx / img_size, pixel_idx % img_size);

    println!("\ndiff vs HF reference:");
    println!("  max abs diff:  {max_diff:.3e} at channel {channel} pixel ({px_y},{px_x})");
    println!("  mean abs diff: {mean_diff:.3e}");
    println!("  tolerance:     {TOL_ABS:.0e}");

    if max_diff > TOL_ABS {
        return Err(format!(
            "max diff {max_diff:.3e} exceeds tolerance {TOL_ABS:.0e} — VAE fails M4 gate"
        )
        .into());
    }
    println!("\n✓ M4 numerical-equivalence gate PASSED");
    Ok(())
}
