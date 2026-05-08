// SPDX-License-Identifier: MIT OR Apache-2.0
//! M6 numerical-equivalence gate for the `UNet` noise predictor.
//!
//! Loads our `Unet` from the SD 1.5 checkpoint, runs a single forward at
//! `t=981` against a fixed `[1, 4, N, N]` noise latent and a fixed
//! conditioning embedding from CLIP-L, and byte-compares the
//! `[1, 4, N, N]` output against a reference dump from HF
//! `diffusers.UNet2DConditionModel`. Tolerance is 1e-3 absolute, the
//! HANDOFF gate for "this stack matches HF in fp32".
//!
//! Both the latent and the conditioning embedding are read from the
//! reference dump so both sides see byte-identical inputs (avoiding any
//! cross-language RNG / CLIP drift in this test specifically — M3
//! already covered the CLIP side at 5.7e-6).
//!
//! Usage:
//!
//! ```text
//! crates/scry-vision/.venv/bin/python \
//!     crates/scry-diffusion/python/dump_unet_ref.py
//! cargo run -p scry-diffusion --release \
//!     --features safetensors --example check_unet
//! ```

use std::path::PathBuf;
use std::time::Instant;

use scry_diffusion::conditioning::Conditioning;
use scry_diffusion::unet::{Unet, UnetConfig};
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
        .join(format!("unet_seed{seed}.safetensors"));

    if !ref_path.exists() {
        return Err(format!(
            "reference dump not found at {}\n\
             generate it first:\n  \
             crates/scry-vision/.venv/bin/python \
             crates/scry-diffusion/python/dump_unet_ref.py --seed {seed}",
            ref_path.display()
        )
        .into());
    }

    // ---- Load latent + conditioning + reference output -------------
    let ref_ckpt = SafetensorsCheckpoint::open(&ref_path)?;
    let latent_flat = ref_ckpt.tensor_f32("latent")?;
    let cond_flat = ref_ckpt.tensor_f32("conditioning")?;
    let timestep_arr = ref_ckpt.tensor_f32("timestep")?;
    let hf_out = ref_ckpt.tensor_f32("predicted_noise")?;

    let cfg = UnetConfig::sd_1_5();
    let in_ch = cfg.in_channels;
    let out_ch = cfg.out_channels;
    let cross_dim = cfg.cross_attention_dim;
    let seq_len = 77usize;

    if cond_flat.len() != seq_len * cross_dim {
        return Err(format!(
            "conditioning length {} != {seq_len}*{cross_dim}",
            cond_flat.len()
        )
        .into());
    }
    if timestep_arr.len() != 1 {
        return Err(format!("timestep tensor must have 1 element, got {}", timestep_arr.len()).into());
    }
    let timestep = timestep_arr[0];

    let latent_size = ((latent_flat.len() / in_ch) as f64).sqrt() as usize;
    if latent_flat.len() != in_ch * latent_size * latent_size {
        return Err(format!(
            "latent has {} elements; expected {in_ch}*N*N for some N, got N={latent_size}",
            latent_flat.len()
        )
        .into());
    }
    let expected_out_len = out_ch * latent_size * latent_size;
    if hf_out.len() != expected_out_len {
        return Err(format!(
            "predicted_noise has {} elements; expected {out_ch}*{latent_size}*{latent_size} = {expected_out_len}",
            hf_out.len()
        )
        .into());
    }
    println!("seed: {seed}");
    println!("timestep: {timestep}");
    println!(
        "latent shape:          [{in_ch}, {latent_size}, {latent_size}] ({} elements)",
        latent_flat.len()
    );
    println!(
        "conditioning shape:    [{seq_len}, {cross_dim}] ({} elements)",
        cond_flat.len()
    );
    println!(
        "expected output shape: [{out_ch}, {latent_size}, {latent_size}] ({} elements)",
        hf_out.len()
    );

    // ---- Load UNet -------------------------------------------------
    println!("\nloading UNet weights...");
    let unet_path = snapshot.join("unet/diffusion_pytorch_model.safetensors");
    let unet_ckpt = SafetensorsCheckpoint::open(&unet_path)?;
    let t0 = Instant::now();
    let mut unet = Unet::<CpuBackend>::from_safetensors(cfg, &unet_ckpt)?;
    println!("  loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Forward ---------------------------------------------------
    let latent = Tensor::<CpuBackend>::from_vec(
        latent_flat,
        Shape::new(&[in_ch, latent_size, latent_size]),
    );
    let embeddings = Tensor::<CpuBackend>::from_vec(cond_flat, Shape::new(&[seq_len, cross_dim]));
    let cond = Conditioning {
        embeddings,
        extras: None,
    };

    println!("\nrunning UNet forward (CPU)...");
    let t0 = Instant::now();
    let our_out = unet.forward(&latent, timestep, &cond)?;
    let our_vec = our_out.to_vec();
    println!(
        "  forward took {:.1}s, output shape {:?}, {} elements",
        t0.elapsed().as_secs_f32(),
        our_out.shape.dims(),
        our_vec.len()
    );

    // ---- Compare ---------------------------------------------------
    if our_vec.len() != hf_out.len() {
        return Err(format!(
            "output length mismatch: ours {} vs HF {}",
            our_vec.len(),
            hf_out.len()
        )
        .into());
    }
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    let mut max_pos = 0usize;
    for (i, (a, b)) in our_vec.iter().zip(hf_out.iter()).enumerate() {
        let d = (a - b).abs();
        sum_diff += f64::from(d);
        if d > max_diff {
            max_diff = d;
            max_pos = i;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let mean_diff = sum_diff / our_vec.len() as f64;
    let plane = latent_size * latent_size;
    let (channel, pixel_idx) = (max_pos / plane, max_pos % plane);
    let (px_y, px_x) = (pixel_idx / latent_size, pixel_idx % latent_size);

    println!("\ndiff vs HF reference:");
    println!("  max abs diff:  {max_diff:.3e} at channel {channel} pixel ({px_y},{px_x})");
    println!("  mean abs diff: {mean_diff:.3e}");
    println!("  tolerance:     {TOL_ABS:.0e}");

    if max_diff > TOL_ABS {
        return Err(format!(
            "max diff {max_diff:.3e} exceeds tolerance {TOL_ABS:.0e} — UNet fails M6 gate"
        )
        .into());
    }
    println!("\n✓ M6 numerical-equivalence gate PASSED");
    Ok(())
}
