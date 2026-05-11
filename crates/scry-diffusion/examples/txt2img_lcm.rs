// SPDX-License-Identifier: MIT OR Apache-2.0
//! 4-step LCM txt2img driver — visual smoke for the M12 `LcmScheduler`.
//!
//! Loads the LCM-distilled `UNet` from `SimianLuo/LCM_Dreamshaper_v7`
//! (a full `SD 1.5`-compatible checkpoint) and pairs it with the base
//! `SD 1.5` text encoder / VAE / tokenizer — those three components are
//! byte-identical between the two snapshots, so we save ~825 MB of
//! redundant disk by only fetching the LCM `UNet`.
//!
//! Default config: 4 steps, `cfg=1.0` (LCM ignores classifier-free
//! guidance — the distillation is trained without it), seed=42.
//!
//! Usage:
//! ```bash
//! cargo run -p scry-diffusion --release \
//!     --features safetensors,decode,scry-gpu-cuda,scry-gpu-bf16 \
//!     --example txt2img_lcm -- \
//!     --prompt "a photo of a corgi in a chef's hat" \
//!     --out lcm.png
//! ```
//!
//! Asset layout (see `ASSETS.md` for download instructions):
//! ```text
//! crates/scry-diffusion/.assets/
//! ├── sd-1-5/                 (base SD 1.5: tokenizer + text_encoder + vae)
//! └── lcm-dreamshaper-v7/
//!     └── unet/diffusion_pytorch_model.safetensors
//! ```

use std::path::PathBuf;
use std::time::Instant;

use scry_diffusion::scheduler::lcm::{LcmConfig, LcmScheduler};
use scry_diffusion::text_encoder::clip_text::{ClipTextConfig, ClipTextEncoder};
use scry_diffusion::tokenizer::Tokenizer;
use scry_diffusion::unet::{Unet, UnetConfig};
use scry_diffusion::vae::decoder::{VaeDecoder, VaeDecoderConfig};
use scry_diffusion::weights::SafetensorsCheckpoint;
use scry_diffusion::{GenerationParams, SdPipeline};

#[cfg(not(feature = "scry-gpu-cuda"))]
use scry_llm::backend::cpu::CpuBackend as Backend;
#[cfg(feature = "scry-gpu-cuda")]
use scry_llm::backend::scry_gpu::ScryGpuBackend as Backend;

const BACKEND_NAME: &str = if cfg!(feature = "scry-gpu-cuda") {
    "scry-gpu (CUDA)"
} else {
    "CPU"
};

#[derive(Debug)]
struct Args {
    base_snapshot: PathBuf,
    lcm_snapshot: PathBuf,
    out: PathBuf,
    prompt: String,
    steps: u32,
    seed: u64,
    size: u32,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let assets = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".assets");
        let mut base_snapshot = assets.join("sd-1-5");
        let mut lcm_snapshot = assets.join("lcm-dreamshaper-v7");
        let mut out = PathBuf::from("lcm.png");
        let mut prompt =
            String::from("a photo of a corgi puppy wearing a chef's hat, soft studio lighting");
        let mut steps: u32 = 4;
        let mut seed: u64 = 42;
        let mut size: u32 = 512;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--base-snapshot" => {
                    base_snapshot = PathBuf::from(args.next().ok_or("--base-snapshot needs path")?);
                }
                "--lcm-snapshot" => {
                    lcm_snapshot = PathBuf::from(args.next().ok_or("--lcm-snapshot needs path")?);
                }
                "--out" => out = PathBuf::from(args.next().ok_or("--out needs path")?),
                "--prompt" => prompt = args.next().ok_or("--prompt needs string")?,
                "--steps" => {
                    steps = args
                        .next()
                        .ok_or("--steps needs u32")?
                        .parse()
                        .map_err(|e| format!("--steps: {e}"))?;
                }
                "--seed" => {
                    seed = args
                        .next()
                        .ok_or("--seed needs u64")?
                        .parse()
                        .map_err(|e| format!("--seed: {e}"))?;
                }
                "--size" => {
                    size = args
                        .next()
                        .ok_or("--size needs u32")?
                        .parse()
                        .map_err(|e| format!("--size: {e}"))?;
                }
                "-h" | "--help" => {
                    println!("{USAGE}");
                    std::process::exit(0);
                }
                other => return Err(format!("unknown flag: {other}\n\n{USAGE}")),
            }
        }
        Ok(Self {
            base_snapshot,
            lcm_snapshot,
            out,
            prompt,
            steps,
            seed,
            size,
        })
    }
}

const USAGE: &str = "\
Usage: txt2img_lcm [OPTIONS]

Options:
  --base-snapshot PATH  Base SD 1.5 snapshot root (default: .assets/sd-1-5)
  --lcm-snapshot PATH   LCM-Dreamshaper snapshot root (default: .assets/lcm-dreamshaper-v7)
  --out PATH            Output PNG path (default: lcm.png)
  --prompt STRING       Prompt
  --steps N             Denoising steps (default: 4 — LCM is trained for 4)
  --seed N              Latent noise seed (default: 42)
  --size N              Output side length, multiple of 8 (default: 512)";

#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    println!("scry-diffusion txt2img_lcm (4-step LCM)");
    println!("  backend:        {BACKEND_NAME}");
    enable_bf16_matmul_if_available();
    println!("  base snapshot:  {}", args.base_snapshot.display());
    println!("  lcm snapshot:   {}", args.lcm_snapshot.display());
    println!("  prompt:         {:?}", args.prompt);
    println!(
        "  size: {sz}×{sz}, steps: {n}, seed: {s} (cfg=1.0 — LCM skips CFG)",
        sz = args.size,
        n = args.steps,
        s = args.seed,
    );

    // ---- Load tokenizer + text encoder from base SD 1.5. ----
    let t0 = Instant::now();
    println!("\n[1/4] loading tokenizer + CLIP text encoder (base SD 1.5)…");
    let tokenizer = Tokenizer::from_dir(args.base_snapshot.join("tokenizer"))?;
    let text_ckpt =
        SafetensorsCheckpoint::open(args.base_snapshot.join("text_encoder/model.safetensors"))?;
    let text_encoder =
        ClipTextEncoder::<Backend>::from_safetensors(ClipTextConfig::clip_vit_l(), &text_ckpt)?;
    println!("  done in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Load LCM UNet + wire guidance-scale embedding. ----
    let t0 = Instant::now();
    println!("\n[2/4] loading LCM UNet (LCM_Dreamshaper_v7)…");
    let unet_ckpt = SafetensorsCheckpoint::open(
        args.lcm_snapshot
            .join("unet/diffusion_pytorch_model.safetensors"),
    )?;
    let mut unet = Unet::<Backend>::from_safetensors(UnetConfig::sd_1_5(), &unet_ckpt)?;
    // LCM-Dreamshaper was distilled with `guidance_scale = 8.0`; HF's LCM
    // pipeline passes `w = guidance_scale - 1 = 7.0` to `cond_proj`.
    unet.set_guidance_scale(7.0)?;
    println!("  done in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Load VAE decoder from base SD 1.5. ----
    let t0 = Instant::now();
    println!("\n[3/4] loading VAE decoder (base SD 1.5)…");
    let vae_ckpt = SafetensorsCheckpoint::open(
        args.base_snapshot
            .join("vae/diffusion_pytorch_model.safetensors"),
    )?;
    let vae = VaeDecoder::<Backend>::from_safetensors(VaeDecoderConfig::sd_1_5(), &vae_ckpt)?;
    println!("  done in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Build pipeline. ----
    let mut lcm_cfg = LcmConfig::sd_1_5();
    lcm_cfg.noise_seed = args.seed;
    let scheduler = LcmScheduler::new(lcm_cfg)?;
    let progress_start = Instant::now();
    let pipeline = SdPipeline {
        tokenizer,
        text_encoder,
        unet,
        vae,
        vae_encoder: None,
        scheduler,
        progress: None,
    };
    let mut pipeline = pipeline.with_progress(move |i, total, t| {
        let elapsed = progress_start.elapsed().as_secs_f32();
        println!(
            "  step {:>2}/{total} (t={t:>5.0})  [{elapsed:.1}s elapsed]",
            i + 1
        );
    });
    pipeline.to_device();

    let params = GenerationParams {
        prompt: args.prompt,
        // LCM is trained without classifier-free guidance, so the negative
        // prompt + uncond branch are skipped (cfg=1.0 → pipeline takes the
        // cond-only fast path).
        negative_prompt: String::new(),
        num_inference_steps: args.steps,
        guidance_scale: 1.0,
        seed: args.seed,
        size: (args.size, args.size),
    };

    println!("\n[4/4] generating…");
    let t0 = Instant::now();
    let img = pipeline.generate(&params)?;
    println!(
        "\n  total denoise+decode: {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    // ---- Save PNG. ----
    let dims = img.shape.dims().to_vec();
    let (channels, h, w) = match dims.as_slice() {
        [c, h, w] => (*c, *h, *w),
        other => return Err(format!("expected [C, H, W] image, got {other:?}").into()),
    };
    if channels != 3 {
        return Err(format!("expected 3 channels, got {channels}").into());
    }
    let pixels = img.to_vec();
    let mut hwc = vec![0_u8; h * w * 3];
    let plane = h * w;
    for y in 0..h {
        for x in 0..w {
            let src_offset = y * w + x;
            let dst = (y * w + x) * 3;
            for c in 0..3 {
                let v = pixels[c * plane + src_offset];
                let clamped = v.clamp(0.0, 1.0);
                hwc[dst + c] = (clamped * 255.0 + 0.5) as u8;
            }
        }
    }
    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let img_buf =
        image::RgbImage::from_raw(w as u32, h as u32, hwc).ok_or("failed to pack image buffer")?;
    img_buf.save(&args.out)?;
    println!("\nsaved {} ({}×{})", args.out.display(), w, h);
    Ok(())
}

#[cfg(all(feature = "scry-gpu-cuda", feature = "scry-gpu-bf16"))]
fn enable_bf16_matmul_if_available() {
    if let Err(e) = scry_llm::backend::scry_gpu::ScryGpuBackend::set_bf16_matmul(true) {
        eprintln!("warning: bf16 matmul opt-in failed ({e}); falling back to fp32");
    } else {
        println!("  matmul:         bf16 (cuBLAS GemmEx)");
    }
}

#[cfg(not(all(feature = "scry-gpu-cuda", feature = "scry-gpu-bf16")))]
fn enable_bf16_matmul_if_available() {}
