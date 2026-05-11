// SPDX-License-Identifier: MIT OR Apache-2.0
//! Stable Diffusion 1.5 text-to-image command handlers.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tauri::async_runtime::spawn_blocking;

use scry_diffusion::scheduler::ddim::{DdimConfig, DdimScheduler};
use scry_diffusion::text_encoder::clip_text::{ClipTextConfig, ClipTextEncoder};
use scry_diffusion::tokenizer::Tokenizer;
use scry_diffusion::unet::{Unet, UnetConfig};
use scry_diffusion::vae::decoder::{VaeDecoder, VaeDecoderConfig};
use scry_diffusion::weights::SafetensorsCheckpoint;
use scry_diffusion::{GenerationParams, SdPipeline};
use scry_llm::backend::DeviceBackend;

use crate::state::{AppState, Backend, LoadedDiffusion};

#[derive(Deserialize)]
pub struct DiffusionLoadArgs {
    /// Path to the SD-1-5 snapshot directory containing
    /// `tokenizer/`, `text_encoder/`, `unet/`, `vae/` subdirs.
    pub snapshot_path: String,
}

#[derive(Serialize)]
pub struct DiffusionLoadResult {
    pub backend: &'static str,
}

fn load_pipeline(snapshot: &Path) -> Result<crate::state::DiffusionPipeline, String> {
    let tokenizer = Tokenizer::from_dir(snapshot.join("tokenizer"))
        .map_err(|e| format!("tokenizer load failed: {e}"))?;

    let text_encoder = {
        let ckpt = SafetensorsCheckpoint::open(snapshot.join("text_encoder/model.safetensors"))
            .map_err(|e| format!("text_encoder open failed: {e}"))?;
        ClipTextEncoder::<Backend>::from_safetensors(ClipTextConfig::clip_vit_l(), &ckpt)
            .map_err(|e| format!("text_encoder load failed: {e}"))?
    };

    let unet = {
        let ckpt =
            SafetensorsCheckpoint::open(snapshot.join("unet/diffusion_pytorch_model.safetensors"))
                .map_err(|e| format!("unet open failed: {e}"))?;
        Unet::<Backend>::from_safetensors(UnetConfig::sd_1_5(), &ckpt)
            .map_err(|e| format!("unet load failed: {e}"))?
    };

    let vae = {
        let ckpt =
            SafetensorsCheckpoint::open(snapshot.join("vae/diffusion_pytorch_model.safetensors"))
                .map_err(|e| format!("vae open failed: {e}"))?;
        VaeDecoder::<Backend>::from_safetensors(VaeDecoderConfig::sd_1_5(), &ckpt)
            .map_err(|e| format!("vae load failed: {e}"))?
    };

    let scheduler = DdimScheduler::new(DdimConfig::sd_1_5())
        .map_err(|e| format!("scheduler init failed: {e}"))?;

    let mut pipeline = SdPipeline {
        tokenizer,
        text_encoder,
        unet,
        vae,
        vae_encoder: None,
        scheduler,
        progress: None,
    };
    pipeline.to_device();
    Ok(pipeline)
}

#[tauri::command]
pub async fn diffusion_load(
    state: tauri::State<'_, AppState>,
    args: DiffusionLoadArgs,
) -> Result<DiffusionLoadResult, String> {
    let slot = state.diffusion.clone();
    let snapshot = PathBuf::from(args.snapshot_path);
    spawn_blocking(move || -> Result<DiffusionLoadResult, String> {
        let pipeline = load_pipeline(&snapshot)?;
        *slot.lock() = Some(LoadedDiffusion { pipeline });
        Ok(DiffusionLoadResult {
            backend: crate::state::BACKEND_NAME,
        })
    })
    .await
    .map_err(|e| format!("join error: {e}"))?
}

#[derive(Deserialize)]
pub struct DiffusionGenerateArgs {
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    #[serde(default = "default_steps")]
    pub steps: u32,
    #[serde(default = "default_cfg")]
    pub cfg: f32,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_size")]
    pub size: u32,
}

fn default_steps() -> u32 {
    30
}
fn default_cfg() -> f32 {
    7.5
}
fn default_seed() -> u64 {
    42
}
fn default_size() -> u32 {
    512
}

#[derive(Serialize)]
pub struct DiffusionGenerateResult {
    /// PNG-encoded image as raw bytes (frontend wraps in Blob).
    pub png: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub elapsed_ms: u128,
}

#[tauri::command]
pub async fn diffusion_generate(
    state: tauri::State<'_, AppState>,
    args: DiffusionGenerateArgs,
) -> Result<DiffusionGenerateResult, String> {
    let slot = state.diffusion.clone();
    spawn_blocking(move || -> Result<DiffusionGenerateResult, String> {
        let mut guard = slot.lock();
        let loaded = guard.as_mut().ok_or("diffusion pipeline not loaded")?;
        let params = GenerationParams {
            prompt: args.prompt,
            negative_prompt: args.negative_prompt,
            num_inference_steps: args.steps,
            guidance_scale: args.cfg,
            seed: args.seed,
            size: (args.size, args.size),
        };
        let start = std::time::Instant::now();
        let tensor = loaded
            .pipeline
            .generate(&params)
            .map_err(|e| format!("generate failed: {e}"))?;
        let elapsed_ms = start.elapsed().as_millis();
        let (rgb, w, h) = tensor_to_rgb(&tensor)?;
        let png = encode_png(&rgb, w, h)?;
        Ok(DiffusionGenerateResult {
            png,
            width: w,
            height: h,
            elapsed_ms,
        })
    })
    .await
    .map_err(|e| format!("join error: {e}"))?
}

fn tensor_to_rgb<B: DeviceBackend>(
    tensor: &scry_llm::tensor::Tensor<B>,
) -> Result<(Vec<u8>, u32, u32), String> {
    let dims = tensor.shape.dims();
    if dims.len() != 3 || dims[0] != 3 {
        return Err(format!("expected [3,H,W] tensor, got {dims:?}"));
    }
    let h = dims[1];
    let w = dims[2];
    let flat = tensor.to_vec();
    let plane = h * w;
    let mut out = Vec::with_capacity(plane * 3);
    for i in 0..plane {
        for c in 0..3 {
            let v = flat[c * plane + i].clamp(0.0, 1.0);
            out.push((v * 255.0).round() as u8);
        }
    }
    Ok((out, w as u32, h as u32))
}

fn encode_png(rgb: &[u8], w: u32, h: u32) -> Result<Vec<u8>, String> {
    use image::{codecs::png::PngEncoder, ImageEncoder};
    let mut buf = Vec::with_capacity(rgb.len());
    PngEncoder::new(&mut buf)
        .write_image(rgb, w, h, image::ExtendedColorType::Rgb8)
        .map_err(|e| format!("png encode failed: {e}"))?;
    Ok(buf)
}
