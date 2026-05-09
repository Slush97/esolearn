// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end txt2img driver — first runnable image generator.
//!
//! Loads SD 1.5 from a local HF snapshot, runs the full
//! tokenize → CLIP → DDIM-loop(UNet+CFG) → VAE pipeline, and writes a PNG.
//!
//! Usage:
//! ```bash
//! cargo run -p scry-diffusion --release \
//!     --features safetensors,decode --example txt2img -- \
//!     --prompt "a photo of an astronaut riding a horse on mars" \
//!     --out astronaut.png \
//!     --steps 30 --cfg 7.5 --seed 42 --size 512
//! ```

use std::path::PathBuf;
use std::time::Instant;

use scry_diffusion::scheduler::ddim::{DdimConfig, DdimScheduler};
use scry_diffusion::text_encoder::clip_text::{ClipTextConfig, ClipTextEncoder};
use scry_diffusion::tokenizer::Tokenizer;
use scry_diffusion::unet::{Unet, UnetConfig};
use scry_diffusion::vae::decoder::{VaeDecoder, VaeDecoderConfig};
use scry_diffusion::weights::SafetensorsCheckpoint;
use scry_diffusion::{GenerationParams, Txt2ImgPipeline};
use scry_llm::backend::cpu::CpuBackend;

#[derive(Debug)]
struct Args {
    snapshot: PathBuf,
    out: PathBuf,
    prompt: String,
    negative_prompt: String,
    steps: u32,
    cfg: f32,
    seed: u64,
    size: u32,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut snapshot = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".assets/sd-1-5");
        let mut out = PathBuf::from("output.png");
        let mut prompt = String::from("a photo of an astronaut riding a horse on mars");
        let mut negative_prompt = String::new();
        let mut steps: u32 = 30;
        let mut cfg: f32 = 7.5;
        let mut seed: u64 = 42;
        let mut size: u32 = 512;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--snapshot" => {
                    snapshot = PathBuf::from(args.next().ok_or("--snapshot needs path")?)
                }
                "--out" => out = PathBuf::from(args.next().ok_or("--out needs path")?),
                "--prompt" => prompt = args.next().ok_or("--prompt needs string")?,
                "--negative-prompt" | "--negative" => {
                    negative_prompt = args.next().ok_or("--negative-prompt needs string")?;
                }
                "--steps" => {
                    steps = args
                        .next()
                        .ok_or("--steps needs u32")?
                        .parse()
                        .map_err(|e| format!("--steps: {e}"))?;
                }
                "--cfg" | "--guidance" => {
                    cfg = args
                        .next()
                        .ok_or("--cfg needs f32")?
                        .parse()
                        .map_err(|e| format!("--cfg: {e}"))?;
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
                    println!("{}", USAGE);
                    std::process::exit(0);
                }
                other => return Err(format!("unknown flag: {other}\n\n{USAGE}")),
            }
        }
        Ok(Self {
            snapshot,
            out,
            prompt,
            negative_prompt,
            steps,
            cfg,
            seed,
            size,
        })
    }
}

const USAGE: &str = "\
Usage: txt2img [OPTIONS]

Options:
  --snapshot PATH       SD 1.5 snapshot root (default: crates/scry-diffusion/.assets/sd-1-5)
  --out PATH            Output PNG path (default: output.png)
  --prompt STRING       Prompt (default: \"a photo of an astronaut riding a horse on mars\")
  --negative-prompt STR Negative prompt (default: empty)
  --steps N             Denoising steps (default: 30)
  --cfg F               Classifier-free guidance scale (default: 7.5)
  --seed N              Latent noise seed (default: 42)
  --size N              Output side length, multiple of 8 (default: 512)";

#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    println!("scry-diffusion txt2img");
    println!("  snapshot: {}", args.snapshot.display());
    println!("  prompt:   {:?}", args.prompt);
    if !args.negative_prompt.is_empty() {
        println!("  negative: {:?}", args.negative_prompt);
    }
    println!(
        "  size: {sz}×{sz}, steps: {n}, cfg: {cfg}, seed: {s}",
        sz = args.size,
        n = args.steps,
        cfg = args.cfg,
        s = args.seed,
    );

    // ---- Load tokenizer + text encoder. ----
    let t0 = Instant::now();
    println!("\n[1/4] loading tokenizer + CLIP text encoder…");
    let tokenizer = Tokenizer::from_dir(args.snapshot.join("tokenizer"))?;
    let text_ckpt =
        SafetensorsCheckpoint::open(args.snapshot.join("text_encoder/model.safetensors"))?;
    let text_encoder =
        ClipTextEncoder::<CpuBackend>::from_safetensors(ClipTextConfig::clip_vit_l(), &text_ckpt)?;
    println!("  done in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Load UNet. ----
    let t0 = Instant::now();
    println!("\n[2/4] loading UNet…");
    let unet_ckpt = SafetensorsCheckpoint::open(
        args.snapshot
            .join("unet/diffusion_pytorch_model.safetensors"),
    )?;
    let unet = Unet::<CpuBackend>::from_safetensors(UnetConfig::sd_1_5(), &unet_ckpt)?;
    println!("  done in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Load VAE decoder. ----
    let t0 = Instant::now();
    println!("\n[3/4] loading VAE decoder…");
    let vae_ckpt = SafetensorsCheckpoint::open(
        args.snapshot
            .join("vae/diffusion_pytorch_model.safetensors"),
    )?;
    let vae = VaeDecoder::<CpuBackend>::from_safetensors(VaeDecoderConfig::sd_1_5(), &vae_ckpt)?;
    println!("  done in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Build pipeline. ----
    let scheduler = DdimScheduler::new(DdimConfig::sd_1_5())?;
    let progress_start = Instant::now();
    let pipeline = Txt2ImgPipeline {
        tokenizer,
        text_encoder,
        unet,
        vae,
        scheduler,
        progress: None,
    };
    let mut pipeline = pipeline.with_progress(move |i, total, t| {
        let elapsed = progress_start.elapsed().as_secs_f32();
        println!(
            "  step {:>3}/{total} (t={t:>5.0})  [{elapsed:.1}s elapsed]",
            i + 1
        );
    });

    let params = GenerationParams {
        prompt: args.prompt,
        negative_prompt: args.negative_prompt,
        num_inference_steps: args.steps,
        guidance_scale: args.cfg,
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
        other => return Err(format!("expected [C, H, W] image, got {:?}", other).into()),
    };
    if channels != 3 {
        return Err(format!("expected 3 channels, got {channels}").into());
    }
    let pixels = img.to_vec(); // [3, H, W] in [0, 1]
                               // Repack CHW -> HWC u8.
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
