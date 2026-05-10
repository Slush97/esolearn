// SPDX-License-Identifier: MIT OR Apache-2.0
//! CLIP text encoder microbenchmark.
//!
//! Loads `text_encoder/model.safetensors` from the SD 1.5 checkpoint and
//! times `ClipTextEncoder::forward(tokens)` over `--runs` iterations after
//! `--warmup` discarded warmups. Reports min / median / max wall-clock per
//! encode in milliseconds. Backend selection mirrors `bench_sd`:
//!
//! ```bash
//! # CPU baseline.
//! cargo run -p scry-diffusion --release --features safetensors \
//!     --example bench_clip
//!
//! # GPU (CUDA).
//! CUDARC_CUDA_VERSION=13010 cargo run -p scry-diffusion --release \
//!     --features safetensors,scry-gpu-cuda --example bench_clip
//! ```

use std::path::PathBuf;
use std::time::Instant;

use scry_diffusion::text_encoder::clip_text::{ClipTextConfig, ClipTextEncoder};
use scry_diffusion::tokenizer::Tokenizer;
use scry_diffusion::weights::SafetensorsCheckpoint;

#[cfg(not(feature = "scry-gpu-cuda"))]
use scry_llm::backend::cpu::CpuBackend as Backend;
#[cfg(feature = "scry-gpu-cuda")]
use scry_llm::backend::scry_gpu::ScryGpuBackend as Backend;

const BACKEND_NAME: &str = if cfg!(feature = "scry-gpu-cuda") {
    "scry-gpu (CUDA)"
} else {
    "CPU"
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut snapshot = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".assets/sd-1-5");
    let mut prompt = String::from("a photo of an astronaut riding a horse on mars");
    let mut runs: u32 = 50;
    let mut warmup: u32 = 5;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--snapshot" => {
                snapshot = PathBuf::from(args.next().ok_or("--snapshot needs path")?);
            }
            "--prompt" => prompt = args.next().ok_or("--prompt needs string")?,
            "--runs" => runs = args.next().ok_or("--runs needs u32")?.parse()?,
            "--warmup" => warmup = args.next().ok_or("--warmup needs u32")?.parse()?,
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }

    println!("backend     : {BACKEND_NAME}");
    println!("prompt      : {prompt:?}");
    println!("warmup/runs : {warmup}/{runs}");
    println!();

    let tok = Tokenizer::from_dir(snapshot.join("tokenizer"))?;
    let tokens = tok.encode(&prompt)?;
    println!("tokenized   : {} tokens", tokens.len());

    let model_path = snapshot.join("text_encoder/model.safetensors");
    let model_ckpt = SafetensorsCheckpoint::open(&model_path)?;
    let t0 = Instant::now();
    let mut encoder =
        ClipTextEncoder::<Backend>::from_safetensors(ClipTextConfig::clip_vit_l(), &model_ckpt)?;
    println!(
        "loaded      : {} layers, d_model={}, in {:.2}s",
        encoder.num_layers(),
        encoder.d_model(),
        t0.elapsed().as_secs_f32()
    );

    let t0 = Instant::now();
    encoder.to_device();
    backend_sync();
    println!(
        "to_device   : {:.1} ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );
    println!();

    for w in 0..warmup {
        let t0 = Instant::now();
        let _out = encoder.forward(&tokens)?;
        backend_sync();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("[warmup {w}] {ms:7.2} ms");
    }

    let mut times = Vec::with_capacity(runs as usize);
    for _ in 0..runs {
        let t0 = Instant::now();
        let _out = encoder.forward(&tokens)?;
        backend_sync();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mn = times[0];
    let mx = *times.last().unwrap();
    let med = times[times.len() / 2];
    let p10 = times[times.len() / 10];
    let p90 = times[times.len() * 9 / 10];

    println!();
    println!("=== summary over {runs} runs ===");
    println!("  min    : {mn:7.2} ms");
    println!("  p10    : {p10:7.2} ms");
    println!("  median : {med:7.2} ms");
    println!("  p90    : {p90:7.2} ms");
    println!("  max    : {mx:7.2} ms");
    Ok(())
}

#[cfg(feature = "scry-gpu-cuda")]
fn backend_sync() {
    let _ = scry_llm::backend::scry_gpu::ScryGpuBackend::synchronize();
}
#[cfg(not(feature = "scry-gpu-cuda"))]
fn backend_sync() {}
