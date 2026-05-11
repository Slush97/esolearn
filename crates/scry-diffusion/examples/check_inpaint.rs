// SPDX-License-Identifier: MIT OR Apache-2.0
//! M11 numerical-equivalence gate for the inpainting code path.
//!
//! Three sub-checks against HF's reference dump, all at 1e-3 abs:
//!
//!   A. `masked_latent` — `vae.encode(image * (1 - mask)).mode * scaling_factor`.
//!      Validates the mask multiply, the VAE-encode-mode use (mean of the
//!      Gaussian rather than a sampled latent), and the scaling.
//!   B. `mask_latent`   — nearest downsample of the mask to latent resolution.
//!      Validates the integer-stride sampling against HF's `F.interpolate(..., mode='nearest')`.
//!   C. `unet_out_t0`   — one `UNet` forward at step 0 with the 9-channel
//!      input (noisy, mask, `masked_latent`). Validates `UnetConfig::sd_1_5_inpainting()`
//!      against the HF inpainting weights.
//!
//! The full denoise loop, CFG combine, and VAE decode are NOT exercised
//! here — they're already gated by `check_ddim`, `check_unet`, `check_vae_decoder`.
//! The new surface in M11 is exactly the three checks above.
//!
//! Usage:
//!
//! ```text
//! crates/scry-vision/.venv/bin/python \
//!     crates/scry-diffusion/python/dump_inpaint_ref.py
//! cargo run -p scry-diffusion --release \
//!     --features safetensors --example check_inpaint
//! ```

use std::path::PathBuf;
use std::time::Instant;

use scry_diffusion::conditioning::Conditioning;
use scry_diffusion::unet::{Unet, UnetConfig};
use scry_diffusion::vae::encoder::VaeEncoderConfig;
use scry_diffusion::vae::VaeEncoder;
use scry_diffusion::weights::SafetensorsCheckpoint;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::MathBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

const TOL_ABS: f32 = 1e-3;

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, f64, usize) {
    assert_eq!(a.len(), b.len());
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    let mut max_pos = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        sum_diff += f64::from(d);
        if d > max_diff {
            max_diff = d;
            max_pos = i;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let mean = sum_diff / a.len() as f64;
    (max_diff, mean, max_pos)
}

fn report(label: &str, max_diff: f32, mean: f64, _max_pos: usize) -> Result<(), String> {
    println!("  {label} max_abs={max_diff:.3e}  mean_abs={mean:.3e}  tol={TOL_ABS:.0e}");
    if max_diff > TOL_ABS {
        println!("    ✗ FAIL");
        return Err(format!("{label}: {max_diff:.3e} > {TOL_ABS:.0e}"));
    }
    println!("    ✓ pass");
    Ok(())
}

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

    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let base_snapshot = crate_root.join(".assets/sd-1-5");
    let inpaint_snapshot = crate_root.join(".assets/sd-1-5-inpainting");
    let ref_path = crate_root
        .join(".assets/refs")
        .join(format!("inpaint_seed{seed}.safetensors"));

    if !ref_path.exists() {
        return Err(format!(
            "reference dump not found at {}\n\
             generate it first:\n  \
             crates/scry-vision/.venv/bin/python \
             crates/scry-diffusion/python/dump_inpaint_ref.py --seed {seed}",
            ref_path.display()
        )
        .into());
    }

    // ---- Load reference dump --------------------------------------
    let ref_ckpt = SafetensorsCheckpoint::open(&ref_path)?;
    let image_flat = ref_ckpt.tensor_f32("image_norm")?;
    let mask_flat = ref_ckpt.tensor_f32("mask")?;
    let masked_image_flat = ref_ckpt.tensor_f32("masked_image")?;
    let hf_masked_latent = ref_ckpt.tensor_f32("masked_latent")?;
    let hf_mask_latent = ref_ckpt.tensor_f32("mask_latent")?;
    let hf_unet_input = ref_ckpt.tensor_f32("unet_input_t0")?;
    let hf_unet_out = ref_ckpt.tensor_f32("unet_out_t0")?;
    let cond_embed_flat = ref_ckpt.tensor_f32("cond_embed")?;
    let t0_vec = ref_ckpt.tensor_f32("t0")?;
    if t0_vec.len() != 1 {
        return Err(format!("t0 must be 1 element, got {}", t0_vec.len()).into());
    }
    let t0 = t0_vec[0];

    let image_size = ((image_flat.len() / 3) as f64).sqrt() as usize;
    let latent_size = image_size / 8;
    let img_numel = 3 * image_size * image_size;
    let mask_numel = image_size * image_size;
    let latent_numel = 4 * latent_size * latent_size;
    let mask_lat_numel = latent_size * latent_size;
    let input9_numel = 9 * latent_size * latent_size;

    if image_flat.len() != img_numel {
        return Err(format!(
            "image_norm: expected {img_numel} elems, got {}",
            image_flat.len()
        )
        .into());
    }
    if mask_flat.len() != mask_numel {
        return Err(format!("mask: expected {mask_numel} elems, got {}", mask_flat.len()).into());
    }
    if masked_image_flat.len() != img_numel
        || hf_masked_latent.len() != latent_numel
        || hf_mask_latent.len() != mask_lat_numel
        || hf_unet_input.len() != input9_numel
        || hf_unet_out.len() != latent_numel
    {
        return Err("reference tensor shape mismatch — re-run dump_inpaint_ref.py".into());
    }
    if cond_embed_flat.len() != 77 * 768 {
        return Err(format!(
            "cond_embed: expected 77*768={}, got {}",
            77 * 768,
            cond_embed_flat.len()
        )
        .into());
    }

    println!("seed: {seed}");
    println!("image shape:  [3, {image_size}, {image_size}]");
    println!("latent shape: [4, {latent_size}, {latent_size}]");
    println!("t0 timestep:  {t0}");

    // ---- Sub-check A: masked_latent ---------------------------------
    // masked_image = image * (1 - mask), broadcast mask across the 3 channels.
    // We expect HF's masked_image (also dumped) to match this exactly — sanity
    // check before VAE encode.
    println!("\n[A] masked_latent (mask multiply + VAE encode-mode + scaling)");
    let mut our_masked_image = vec![0.0f32; img_numel];
    let plane = image_size * image_size;
    for c in 0..3 {
        for p in 0..plane {
            our_masked_image[c * plane + p] = image_flat[c * plane + p] * (1.0 - mask_flat[p]);
        }
    }
    let (md, mn, mp) = max_abs_diff(&our_masked_image, &masked_image_flat);
    report("masked_image (sanity)", md, mn, mp)
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    println!("  loading VAE encoder...");
    let vae_path = base_snapshot.join("vae/diffusion_pytorch_model.safetensors");
    let vae_ckpt = SafetensorsCheckpoint::open(&vae_path)?;
    let vae_cfg = VaeEncoderConfig::sd_1_5();
    let scaling_factor = vae_cfg.scaling_factor;
    let t0_load = Instant::now();
    let encoder = VaeEncoder::<CpuBackend>::from_safetensors(vae_cfg, &vae_ckpt)?;
    println!("  loaded in {:.1}s", t0_load.elapsed().as_secs_f32());

    let masked_image_t =
        Tensor::<CpuBackend>::from_vec(our_masked_image, Shape::new(&[3, image_size, image_size]));
    println!("  encoding masked image...");
    let t_enc = Instant::now();
    let (mean, _logvar) = encoder.encode(&masked_image_t)?;
    println!("  encoded in {:.1}s", t_enc.elapsed().as_secs_f32());

    let scaled = CpuBackend::scale(&mean.data, scaling_factor);
    let our_masked_latent: Vec<f32> =
        Tensor::<CpuBackend>::new(scaled, mean.shape.clone()).to_vec();
    let (md, mn, mp) = max_abs_diff(&our_masked_latent, &hf_masked_latent);
    report("masked_latent", md, mn, mp).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    // ---- Sub-check B: mask_latent (nearest downsample) --------------
    // HF: F.interpolate(mask[1,1,H,W], size=(h,h), mode='nearest').
    // For integer downsample by 8 the source index is y_src = y_dst * 8.
    println!("\n[B] mask_latent (nearest downsample 8×)");
    let mut our_mask_latent = vec![0.0f32; mask_lat_numel];
    for y in 0..latent_size {
        for x in 0..latent_size {
            our_mask_latent[y * latent_size + x] = mask_flat[(y * 8) * image_size + (x * 8)];
        }
    }
    let (md, mn, mp) = max_abs_diff(&our_mask_latent, &hf_mask_latent);
    report("mask_latent", md, mn, mp).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    // ---- Sub-check C: UNet forward with 9-channel input -------------
    println!("\n[C] UNet forward (9-channel conv_in, inpainting weights)");
    println!("  loading inpainting UNet...");
    let unet_path = inpaint_snapshot.join("unet/diffusion_pytorch_model.fp16.safetensors");
    let unet_ckpt = SafetensorsCheckpoint::open(&unet_path)?;
    let t_unet_load = Instant::now();
    let mut unet =
        Unet::<CpuBackend>::from_safetensors(UnetConfig::sd_1_5_inpainting(), &unet_ckpt)?;
    println!("  loaded in {:.1}s", t_unet_load.elapsed().as_secs_f32());

    let cond_embed = Tensor::<CpuBackend>::from_vec(cond_embed_flat, Shape::new(&[77, 768]));
    let conditioning = Conditioning::<CpuBackend> {
        embeddings: cond_embed,
        extras: None,
    };
    let unet_input =
        Tensor::<CpuBackend>::from_vec(hf_unet_input, Shape::new(&[9, latent_size, latent_size]));
    println!("  running UNet forward at t={t0}...");
    let t_fwd = Instant::now();
    let unet_out = unet.forward(&unet_input, t0, &conditioning)?;
    println!("  forward took {:.1}s", t_fwd.elapsed().as_secs_f32());
    let our_unet_out = unet_out.to_vec();
    let (md, mn, mp) = max_abs_diff(&our_unet_out, &hf_unet_out);
    report("unet_out_t0", md, mn, mp).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    println!("\n✓ M11 inpaint parity gate PASSED (A + B + C all within {TOL_ABS:.0e})");
    Ok(())
}
