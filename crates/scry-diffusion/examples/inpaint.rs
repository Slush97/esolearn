// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end inpaint driver — re-synthesize masked regions of an image
//! conditioned on a text prompt. Pairs the SD 1.5 base text encoder + VAE
//! with the SD 1.5 inpainting UNet (9-channel conv_in).
//!
//! Usage:
//! ```bash
//! cargo run -p scry-diffusion --release \
//!     --features safetensors,decode --example inpaint -- \
//!     --init photo.png --mask photo_mask.png \
//!     --prompt "a red brick wall" \
//!     --steps 30 --cfg 7.5 --seed 42 \
//!     --out inpaint_out.png
//! ```
//!
//! Mask convention: any pixel with luminance > 0.5 (or value > 127 for 8-bit
//! grayscale / RGB) is treated as "inpaint here". The unmasked region is
//! anchored to the source image via the masked-latent channels at every step.

use std::path::PathBuf;
use std::time::Instant;

use scry_diffusion::scheduler::ddim::{DdimConfig, DdimScheduler};
use scry_diffusion::text_encoder::clip_text::{ClipTextConfig, ClipTextEncoder};
use scry_diffusion::tokenizer::Tokenizer;
use scry_diffusion::unet::{Unet, UnetConfig};
use scry_diffusion::vae::decoder::{VaeDecoder, VaeDecoderConfig};
use scry_diffusion::vae::encoder::{VaeEncoder, VaeEncoderConfig};
use scry_diffusion::weights::SafetensorsCheckpoint;
use scry_diffusion::{InpaintParams, SdPipeline};
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
    base_snapshot: PathBuf,
    inpaint_snapshot: PathBuf,
    init: PathBuf,
    mask: PathBuf,
    out: PathBuf,
    prompt: String,
    negative_prompt: String,
    steps: u32,
    cfg: f32,
    seed: u64,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut base_snapshot = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".assets/sd-1-5");
        let mut inpaint_snapshot =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".assets/sd-1-5-inpainting");
        let mut init: Option<PathBuf> = None;
        let mut mask: Option<PathBuf> = None;
        let mut out = PathBuf::from("inpaint_out.png");
        let mut prompt = String::from("a red brick wall");
        let mut negative_prompt = String::new();
        let mut steps: u32 = 30;
        let mut cfg: f32 = 7.5;
        let mut seed: u64 = 42;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--base-snapshot" => {
                    base_snapshot = PathBuf::from(args.next().ok_or("--base-snapshot needs path")?);
                }
                "--inpaint-snapshot" => {
                    inpaint_snapshot =
                        PathBuf::from(args.next().ok_or("--inpaint-snapshot needs path")?);
                }
                "--init" => init = Some(PathBuf::from(args.next().ok_or("--init needs path")?)),
                "--mask" => mask = Some(PathBuf::from(args.next().ok_or("--mask needs path")?)),
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
                "-h" | "--help" => {
                    println!("{USAGE}");
                    std::process::exit(0);
                }
                other => return Err(format!("unknown flag: {other}\n\n{USAGE}")),
            }
        }
        let init = init.ok_or("--init <path> is required")?;
        let mask = mask.ok_or("--mask <path> is required")?;
        Ok(Self {
            base_snapshot,
            inpaint_snapshot,
            init,
            mask,
            out,
            prompt,
            negative_prompt,
            steps,
            cfg,
            seed,
        })
    }
}

const USAGE: &str = "\
Usage: inpaint --init <PATH> --mask <PATH> [OPTIONS]

Required:
  --init PATH           Source image PNG (RGB, sides multiple of 8)
  --mask PATH           Mask PNG (sides match init; bright = inpaint here)

Options:
  --base-snapshot PATH      SD 1.5 base snapshot (tokenizer + text_encoder + vae).
                            Default: crates/scry-diffusion/.assets/sd-1-5
  --inpaint-snapshot PATH   SD 1.5 inpainting snapshot (unet).
                            Default: crates/scry-diffusion/.assets/sd-1-5-inpainting
  --out PATH                Output PNG path (default: inpaint_out.png)
  --prompt STRING           Prompt (default: \"a red brick wall\")
  --negative-prompt STR     Negative prompt (default: empty)
  --steps N                 Total denoising steps (default: 30)
  --cfg F                   Classifier-free guidance scale (default: 7.5)
  --seed N                  Latent noise seed (default: 42)";

#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    println!("scry-diffusion inpaint");
    println!("  backend:           {BACKEND_NAME}");
    enable_bf16_matmul_if_available();
    println!("  base snapshot:     {}", args.base_snapshot.display());
    println!("  inpaint snapshot:  {}", args.inpaint_snapshot.display());
    println!("  init:              {}", args.init.display());
    println!("  mask:              {}", args.mask.display());
    println!("  prompt:            {:?}", args.prompt);
    if !args.negative_prompt.is_empty() {
        println!("  negative:          {:?}", args.negative_prompt);
    }
    println!(
        "  steps: {n}, cfg: {cfg}, seed: {s}",
        n = args.steps,
        cfg = args.cfg,
        s = args.seed,
    );

    // ---- Load init image PNG → [3, H, W] in [-1, 1] -----------------
    let img = image::open(&args.init)?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w % 8 != 0 || h % 8 != 0 {
        return Err(format!("init image {w}×{h} must have sides that are multiples of 8").into());
    }
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
    println!("  init shape:        [3, {h}, {w}]");

    // ---- Load mask PNG → [1, H, W] in {0, 1} ------------------------
    // Accept either L8 grayscale or RGB. Threshold at 0.5 (i.e., 128/255)
    // so AA edges round predictably without overshoot.
    let mask_img = image::open(&args.mask)?.to_luma8();
    let (mw, mh) = (mask_img.width() as usize, mask_img.height() as usize);
    if mw != w || mh != h {
        return Err(format!(
            "mask {mw}×{mh} must match init {w}×{h}"
        )
        .into());
    }
    let mut mask_chw = vec![0.0_f32; plane];
    let mask_raw = mask_img.as_raw();
    for (i, px) in mask_raw.iter().enumerate() {
        mask_chw[i] = if *px > 127 { 1.0 } else { 0.0 };
    }
    let mask: Tensor<Backend> = Tensor::from_vec(mask_chw, Shape::new(&[1, h, w]));

    // ---- Load tokenizer + text encoder (base snapshot). -------------
    let t0 = Instant::now();
    println!("\n[1/5] loading tokenizer + CLIP text encoder…");
    let tokenizer = Tokenizer::from_dir(args.base_snapshot.join("tokenizer"))?;
    let text_ckpt =
        SafetensorsCheckpoint::open(args.base_snapshot.join("text_encoder/model.safetensors"))?;
    let text_encoder =
        ClipTextEncoder::<Backend>::from_safetensors(ClipTextConfig::clip_vit_l(), &text_ckpt)?;
    println!("  done in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Load inpainting UNet (inpaint snapshot, .fp16 file). -------
    let t0 = Instant::now();
    println!("\n[2/5] loading inpainting UNet…");
    let unet_ckpt = SafetensorsCheckpoint::open(
        args.inpaint_snapshot
            .join("unet/diffusion_pytorch_model.fp16.safetensors"),
    )?;
    let unet = Unet::<Backend>::from_safetensors(UnetConfig::sd_1_5_inpainting(), &unet_ckpt)?;
    println!("  done in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Load VAE decoder + encoder (base snapshot). ----------------
    let t0 = Instant::now();
    println!("\n[3/5] loading VAE decoder + encoder…");
    let vae_ckpt = SafetensorsCheckpoint::open(
        args.base_snapshot
            .join("vae/diffusion_pytorch_model.safetensors"),
    )?;
    let vae = VaeDecoder::<Backend>::from_safetensors(VaeDecoderConfig::sd_1_5(), &vae_ckpt)?;
    let vae_encoder =
        VaeEncoder::<Backend>::from_safetensors(VaeEncoderConfig::sd_1_5(), &vae_ckpt)?;
    println!("  done in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Build pipeline. --------------------------------------------
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

    let params = InpaintParams {
        prompt: args.prompt,
        negative_prompt: args.negative_prompt,
        num_inference_steps: args.steps,
        guidance_scale: args.cfg,
        seed: args.seed,
    };

    println!("\n[5/5] running inpaint…");
    let t0 = Instant::now();
    let out = pipeline.inpaint(&params, &init_image, &mask)?;
    println!(
        "\n  total encode+denoise+decode: {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    // ---- Save PNG. --------------------------------------------------
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
        println!("  matmul:            bf16 (cuBLAS GemmEx)");
    }
}

#[cfg(not(all(feature = "scry-gpu-cuda", feature = "scry-gpu-bf16")))]
fn enable_bf16_matmul_if_available() {}
