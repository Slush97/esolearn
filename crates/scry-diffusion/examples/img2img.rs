// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end img2img driver — same pipeline as `txt2img`, but the
//! denoise loop starts from a VAE-encoded init image instead of pure
//! Gaussian noise.
//!
//! Usage:
//! ```bash
//! cargo run -p scry-diffusion --release \
//!     --features safetensors,decode --example img2img -- \
//!     --init crates/scry-diffusion/.assets/refs/sd15_512_30steps.png \
//!     --prompt "an oil painting of the same scene, golden hour" \
//!     --strength 0.6 --steps 30 --cfg 7.5 --seed 42 \
//!     --out img2img_out.png
//! ```

use std::path::PathBuf;
use std::time::Instant;

use scry_diffusion::scheduler::ddim::{DdimConfig, DdimScheduler};
use scry_diffusion::text_encoder::clip_text::{ClipTextConfig, ClipTextEncoder};
use scry_diffusion::tokenizer::Tokenizer;
use scry_diffusion::unet::{Unet, UnetConfig};
use scry_diffusion::vae::decoder::{VaeDecoder, VaeDecoderConfig};
use scry_diffusion::vae::encoder::{VaeEncoder, VaeEncoderConfig};
use scry_diffusion::weights::SafetensorsCheckpoint;
use scry_diffusion::{Img2ImgParams, SdPipeline};
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

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
    snapshot: PathBuf,
    init: PathBuf,
    out: PathBuf,
    prompt: String,
    negative_prompt: String,
    steps: u32,
    cfg: f32,
    seed: u64,
    strength: f32,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut snapshot = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".assets/sd-1-5");
        let mut init: Option<PathBuf> = None;
        let mut out = PathBuf::from("img2img_out.png");
        let mut prompt = String::from("an oil painting of the same scene, dramatic lighting");
        let mut negative_prompt = String::new();
        let mut steps: u32 = 30;
        let mut cfg: f32 = 7.5;
        let mut seed: u64 = 42;
        let mut strength: f32 = 0.6;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--snapshot" => {
                    snapshot = PathBuf::from(args.next().ok_or("--snapshot needs path")?);
                }
                "--init" => init = Some(PathBuf::from(args.next().ok_or("--init needs path")?)),
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
                "--strength" => {
                    strength = args
                        .next()
                        .ok_or("--strength needs f32")?
                        .parse()
                        .map_err(|e| format!("--strength: {e}"))?;
                }
                "-h" | "--help" => {
                    println!("{USAGE}");
                    std::process::exit(0);
                }
                other => return Err(format!("unknown flag: {other}\n\n{USAGE}")),
            }
        }
        let init = init.ok_or("--init <path> is required")?;
        Ok(Self {
            snapshot,
            init,
            out,
            prompt,
            negative_prompt,
            steps,
            cfg,
            seed,
            strength,
        })
    }
}

const USAGE: &str = "\
Usage: img2img --init <PATH> [OPTIONS]

Required:
  --init PATH           Init image PNG (RGB, side multiple of 8)

Options:
  --snapshot PATH       SD 1.5 snapshot root (default: crates/scry-diffusion/.assets/sd-1-5)
  --out PATH            Output PNG path (default: img2img_out.png)
  --prompt STRING       Prompt (default: \"an oil painting of the same scene, dramatic lighting\")
  --negative-prompt STR Negative prompt (default: empty)
  --steps N             Total denoising steps (default: 30)
  --cfg F               Classifier-free guidance scale (default: 7.5)
  --seed N              Latent noise seed (default: 42)
  --strength F          img2img strength in [0,1] (default: 0.6)";

#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    println!("scry-diffusion img2img");
    println!("  backend:  {BACKEND_NAME}");
    enable_bf16_matmul_if_available();
    println!("  snapshot: {}", args.snapshot.display());
    println!("  init:     {}", args.init.display());
    println!("  prompt:   {:?}", args.prompt);
    if !args.negative_prompt.is_empty() {
        println!("  negative: {:?}", args.negative_prompt);
    }
    println!(
        "  steps: {n}, strength: {st}, cfg: {cfg}, seed: {s}",
        n = args.steps,
        st = args.strength,
        cfg = args.cfg,
        s = args.seed,
    );

    // ---- Load init image PNG → [3, H, W] tensor in [-1, 1]. ----
    let img = image::open(&args.init)?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w % 8 != 0 || h % 8 != 0 {
        return Err(format!("init image {w}×{h} must have sides that are multiples of 8").into());
    }
    // HWC u8 → CHW f32 in [-1, 1]: pixel/127.5 - 1.0
    let plane = h * w;
    let mut chw = vec![0.0_f32; 3 * plane];
    let raw = img.as_raw();
    for y in 0..h {
        for x in 0..w {
            let src = (y * w + x) * 3;
            let dst = y * w + x;
            for c in 0..3 {
                chw[c * plane + dst] = f32::from(raw[src + c]) / 127.5 - 1.0;
            }
        }
    }
    let init_image: Tensor<Backend> = Tensor::from_vec(chw, Shape::new(&[3, h, w]));
    println!("  init shape: [3, {h}, {w}]");

    // ---- Load tokenizer + text encoder. ----
    let t0 = Instant::now();
    println!("\n[1/5] loading tokenizer + CLIP text encoder…");
    let tokenizer = Tokenizer::from_dir(args.snapshot.join("tokenizer"))?;
    let text_ckpt =
        SafetensorsCheckpoint::open(args.snapshot.join("text_encoder/model.safetensors"))?;
    let text_encoder =
        ClipTextEncoder::<Backend>::from_safetensors(ClipTextConfig::clip_vit_l(), &text_ckpt)?;
    println!("  done in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Load UNet. ----
    let t0 = Instant::now();
    println!("\n[2/5] loading UNet…");
    let unet_ckpt = SafetensorsCheckpoint::open(
        args.snapshot
            .join("unet/diffusion_pytorch_model.safetensors"),
    )?;
    let unet = Unet::<Backend>::from_safetensors(UnetConfig::sd_1_5(), &unet_ckpt)?;
    println!("  done in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Load VAE decoder + encoder. ----
    let t0 = Instant::now();
    println!("\n[3/5] loading VAE decoder + encoder…");
    let vae_ckpt = SafetensorsCheckpoint::open(
        args.snapshot
            .join("vae/diffusion_pytorch_model.safetensors"),
    )?;
    let vae = VaeDecoder::<Backend>::from_safetensors(VaeDecoderConfig::sd_1_5(), &vae_ckpt)?;
    let vae_encoder =
        VaeEncoder::<Backend>::from_safetensors(VaeEncoderConfig::sd_1_5(), &vae_ckpt)?;
    println!("  done in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Build pipeline. ----
    let scheduler = DdimScheduler::new(DdimConfig::sd_1_5())?;
    let progress_start = Instant::now();
    let pipeline = SdPipeline {
        tokenizer,
        text_encoder,
        unet,
        vae,
        vae_encoder: Some(vae_encoder),
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
    println!("\n[4/5] uploading weights to device…");
    let t0 = Instant::now();
    pipeline.to_device();
    println!("  done in {:.1}s", t0.elapsed().as_secs_f32());

    let params = Img2ImgParams {
        prompt: args.prompt,
        negative_prompt: args.negative_prompt,
        num_inference_steps: args.steps,
        guidance_scale: args.cfg,
        seed: args.seed,
        strength: args.strength,
    };

    println!("\n[5/5] running img2img…");
    let t0 = Instant::now();
    let out = pipeline.img2img(&params, &init_image)?;
    println!(
        "\n  total encode+denoise+decode: {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    // ---- Save PNG. ----
    let dims = out.shape.dims().to_vec();
    let (channels, oh, ow) = match dims.as_slice() {
        [c, h, w] => (*c, *h, *w),
        other => return Err(format!("expected [C, H, W] image, got {other:?}").into()),
    };
    if channels != 3 {
        return Err(format!("expected 3 channels, got {channels}").into());
    }
    let pixels = out.to_vec();
    let mut hwc = vec![0_u8; oh * ow * 3];
    let plane = oh * ow;
    for y in 0..oh {
        for x in 0..ow {
            let dst = (y * ow + x) * 3;
            for c in 0..3 {
                let v = pixels[c * plane + (y * ow + x)];
                hwc[dst + c] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    }
    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let img_buf = image::RgbImage::from_raw(ow as u32, oh as u32, hwc)
        .ok_or("failed to pack image buffer")?;
    img_buf.save(&args.out)?;
    println!("\nsaved {} ({}×{})", args.out.display(), ow, oh);
    Ok(())
}

#[cfg(all(feature = "scry-gpu-cuda", feature = "scry-gpu-bf16"))]
fn enable_bf16_matmul_if_available() {
    if let Err(e) = scry_llm::backend::scry_gpu::ScryGpuBackend::set_bf16_matmul(true) {
        eprintln!("warning: bf16 matmul opt-in failed ({e}); falling back to fp32");
    } else {
        println!("  matmul:   bf16 (cuBLAS GemmEx)");
    }
}

#[cfg(not(all(feature = "scry-gpu-cuda", feature = "scry-gpu-bf16")))]
fn enable_bf16_matmul_if_available() {}
