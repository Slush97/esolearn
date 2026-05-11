// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end Stable Diffusion 1.5 txt2img bench.
//!
//! Times an entire generation with the same prompt + seed + steps + CFG +
//! resolution as `python/bench_pytorch.py`, so the two numbers can be
//! compared directly. Reports per-step median latency and total wall-clock,
//! averaged over `--runs` after `--warmup` discarded runs.
//!
//! ## Backend selection (compile-time)
//!
//! Default features compile against `CpuBackend`. To bench the GPU path
//! enable `scry-gpu-cuda` (and optionally `scry-gpu-bf16`,
//! `scry-gpu-cudnn`) — the bench will use `ScryGpuBackend` automatically.
//!
//! ```bash
//! # CPU baseline (slow — typically minutes per image at 512×512).
//! cargo run -p scry-diffusion --release --features safetensors,decode \
//!     --example bench_sd
//!
//! # GPU baseline (fp32 matmul, no cudnn).
//! CUDARC_CUDA_VERSION=13010 cargo run -p scry-diffusion --release \
//!     --features safetensors,decode,scry-gpu-cuda --example bench_sd
//!
//! # GPU + bf16 matmul + cuDNN convs (target perf row).
//! CUDARC_CUDA_VERSION=13010 cargo run -p scry-diffusion --release \
//!     --features safetensors,decode,scry-gpu-cuda,scry-gpu-bf16,scry-gpu-cudnn \
//!     --example bench_sd -- --bf16-matmul
//! ```
//!
//! ## Toggles
//!
//! `--bf16-matmul` (requires `scry-gpu-bf16`): route UNet+VAE matmuls
//! through cuBLAS GemmEx in bf16. ~2× speedup on Ampere+, with a small
//! numerical drift that's typically invisible in the final image.
//!
//! `--no-cudnn` (requires `scry-gpu-cudnn`): disable the cuDNN conv
//! fast-path and fall back to the legacy im2col + cuBLAS chain. Useful
//! to attribute speedup between bf16-matmul and cudnn.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use scry_diffusion::scheduler::ddim::{DdimConfig, DdimScheduler};
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
    snapshot: PathBuf,
    prompt: String,
    negative_prompt: String,
    steps: u32,
    cfg: f32,
    seed: u64,
    size: u32,
    runs: u32,
    warmup: u32,
    bf16_matmul: bool,
    no_cudnn: bool,
    profile: bool,
    json_out: Option<PathBuf>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut snapshot = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".assets/sd-1-5");
        let mut prompt = String::from("a photo of an astronaut riding a horse on mars");
        let mut negative_prompt = String::new();
        let mut steps: u32 = 30;
        let mut cfg: f32 = 7.5;
        let mut seed: u64 = 42;
        let mut size: u32 = 512;
        let mut runs: u32 = 3;
        let mut warmup: u32 = 1;
        let mut bf16_matmul = false;
        let mut no_cudnn = false;
        let mut profile = false;
        let mut json_out: Option<PathBuf> = None;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--snapshot" => {
                    snapshot = PathBuf::from(args.next().ok_or("--snapshot needs path")?)
                }
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
                "--runs" => {
                    runs = args
                        .next()
                        .ok_or("--runs needs u32")?
                        .parse()
                        .map_err(|e| format!("--runs: {e}"))?;
                }
                "--warmup" => {
                    warmup = args
                        .next()
                        .ok_or("--warmup needs u32")?
                        .parse()
                        .map_err(|e| format!("--warmup: {e}"))?;
                }
                "--bf16-matmul" => bf16_matmul = true,
                "--no-cudnn" => no_cudnn = true,
                "--profile" => profile = true,
                "--json-out" => {
                    json_out = Some(PathBuf::from(args.next().ok_or("--json-out needs path")?));
                }
                "-h" | "--help" => {
                    println!("{}", USAGE);
                    std::process::exit(0);
                }
                other => return Err(format!("unknown flag: {other}\n\n{USAGE}")),
            }
        }
        if runs == 0 {
            return Err("--runs must be > 0".into());
        }
        Ok(Self {
            snapshot,
            prompt,
            negative_prompt,
            steps,
            cfg,
            seed,
            size,
            runs,
            warmup,
            bf16_matmul,
            no_cudnn,
            profile,
            json_out,
        })
    }
}

const USAGE: &str = "\
Usage: bench_sd [OPTIONS]

Times the full SdPipeline at the same shape PyTorch's bench_pytorch.py
does, for a side-by-side perf comparison.

Options:
  --snapshot PATH       SD 1.5 snapshot root (default: crates/scry-diffusion/.assets/sd-1-5)
  --prompt STRING       Prompt (default: \"a photo of an astronaut riding a horse on mars\")
  --negative-prompt STR Negative prompt (default: empty)
  --steps N             Denoising steps (default: 30)
  --cfg F               CFG scale (default: 7.5)
  --seed N              Latent noise seed (default: 42)
  --size N              Output side length, multiple of 8 (default: 512)
  --runs N              Timed runs after warmup (default: 3)
  --warmup N            Discarded warmup runs (default: 1)
  --bf16-matmul         Enable bf16 matmul fast-path (requires scry-gpu-bf16)
  --no-cudnn            Disable cuDNN conv fast-path (requires scry-gpu-cudnn)
  --profile             Print per-section timing breakdown (requires `profile` feature)
  --json-out PATH       Append one JSON-line summary to PATH (for perf-regression gates).
                        Reads GIT_SHA from env to tag the entry; falls back to \"unknown\".";

fn median(mut xs: Vec<f64>) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;

    println!("=== scry-diffusion SD 1.5 bench ===");
    println!("Backend     : {BACKEND_NAME}");
    println!("Snapshot    : {}", args.snapshot.display());
    println!("Prompt      : {:?}", args.prompt);
    if !args.negative_prompt.is_empty() {
        println!("Negative    : {:?}", args.negative_prompt);
    }
    println!("Size        : {sz}x{sz}", sz = args.size);
    println!(
        "Steps       : {n}, CFG: {cfg}, seed: {s}",
        n = args.steps,
        cfg = args.cfg,
        s = args.seed,
    );
    println!(
        "Runs        : {} timed (after {} warmup)",
        args.runs, args.warmup
    );

    // ---- Apply runtime backend toggles before any forward pass. ----
    apply_backend_toggles(args.bf16_matmul, args.no_cudnn)?;

    // ---- Load weights once. The bench measures forward passes only. ----
    let t_load = Instant::now();
    println!("\n[load] tokenizer + CLIP + UNet + VAE…");
    let tokenizer = Tokenizer::from_dir(args.snapshot.join("tokenizer"))?;
    let text_ckpt =
        SafetensorsCheckpoint::open(args.snapshot.join("text_encoder/model.safetensors"))?;
    let text_encoder =
        ClipTextEncoder::<Backend>::from_safetensors(ClipTextConfig::clip_vit_l(), &text_ckpt)?;
    let unet_ckpt = SafetensorsCheckpoint::open(
        args.snapshot
            .join("unet/diffusion_pytorch_model.safetensors"),
    )?;
    let unet = Unet::<Backend>::from_safetensors(UnetConfig::sd_1_5(), &unet_ckpt)?;
    let vae_ckpt = SafetensorsCheckpoint::open(
        args.snapshot
            .join("vae/diffusion_pytorch_model.safetensors"),
    )?;
    let vae = VaeDecoder::<Backend>::from_safetensors(VaeDecoderConfig::sd_1_5(), &vae_ckpt)?;
    let scheduler = DdimScheduler::new(DdimConfig::sd_1_5())?;
    println!("  loaded in {:.1}s", t_load.elapsed().as_secs_f64());

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

    let params = GenerationParams {
        prompt: args.prompt.clone(),
        negative_prompt: args.negative_prompt.clone(),
        num_inference_steps: args.steps,
        guidance_scale: args.cfg,
        seed: args.seed,
        size: (args.size, args.size),
    };

    // ---- Warmup. ----
    for w in 0..args.warmup {
        let t0 = Instant::now();
        let _img = pipeline.generate(&params)?;
        backend_sync();
        let wall = t0.elapsed().as_secs_f64() * 1000.0;
        println!("\n[warmup {w}] {wall:7.1} ms total");
    }

    // ---- Timed runs. ----
    let mut total_times = Vec::with_capacity(args.runs as usize);
    let mut all_step_times: Vec<f64> = Vec::new();

    // Profile collector (no-op when `profile` feature is off). Reset
    // before the first timed run so warmup-phase sections don't pollute
    // the breakdown.
    if args.profile {
        scry_diffusion::profile::reset();
    }

    for r in 0..args.runs {
        // Shared mutable state for the progress callback. The callback fires
        // BEFORE each unet step, so successive deltas measure step durations
        // 0..N-2 (the last step's time runs after the final callback and is
        // not captured here). Total wall-clock covers everything including
        // the un-callbacked tail.
        let marks = Arc::new(Mutex::new(Vec::<Instant>::with_capacity(
            args.steps as usize + 1,
        )));
        let marks_cb = marks.clone();
        pipeline.progress = Some(Box::new(move |_i, _total, _t| {
            // Cheap host-side timestamp; on GPU we accept some launch-latency
            // bleed-through because synchronizing inside a hot loop would
            // distort the benchmark more than it would help.
            marks_cb.lock().unwrap().push(Instant::now());
        }));

        let t0 = Instant::now();
        let _img = pipeline.generate(&params)?;
        backend_sync();
        let wall = t0.elapsed().as_secs_f64() * 1000.0;
        total_times.push(wall);

        let marks_v = marks.lock().unwrap();
        let mut step_times = Vec::with_capacity(marks_v.len().saturating_sub(1));
        for w in marks_v.windows(2) {
            step_times.push(w[1].duration_since(w[0]).as_secs_f64() * 1000.0);
        }
        let med_step = median(step_times.clone());
        all_step_times.extend(step_times);

        println!("[run {r}]    {wall:7.1} ms total  median-step {med_step:6.2} ms");
    }

    // Drop the callback so subsequent runs (if any) don't accumulate.
    pipeline.progress = None;

    // ---- Summary. ----
    println!("\n=== summary ===");
    let med_total = median(total_times.clone());
    println!(
        "total wall-clock   : median {med_total:7.1} ms over {} runs",
        args.runs
    );
    if !all_step_times.is_empty() {
        let med_step = median(all_step_times.clone());
        let n = all_step_times.len();
        let mn = all_step_times.iter().copied().fold(f64::INFINITY, f64::min);
        let mx = all_step_times
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        println!("per-step latency   : median {med_step:6.2} ms  (n={n})");
        println!("  step min/max     : {mn:6.2} / {mx:6.2} ms");
    }

    if args.profile {
        scry_diffusion::profile::print_summary();
    }

    if let Some(path) = args.json_out.as_ref() {
        write_json_line(path, &args, med_total, &all_step_times)?;
        println!("\n[json-out] appended one entry to {}", path.display());
    }

    Ok(())
}

fn write_json_line(
    path: &std::path::Path,
    args: &Args,
    wall_ms_median: f64,
    all_step_times: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let epoch_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let sha = std::env::var("GIT_SHA").unwrap_or_else(|_| "unknown".to_string());

    let (step_med, step_min, step_max, n_step) = if all_step_times.is_empty() {
        (0.0, 0.0, 0.0, 0_usize)
    } else {
        (
            median(all_step_times.to_vec()),
            all_step_times.iter().copied().fold(f64::INFINITY, f64::min),
            all_step_times
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max),
            all_step_times.len(),
        )
    };

    let entry = serde_json::json!({
        "epoch_secs": epoch_secs,
        "sha": sha,
        "backend": BACKEND_NAME,
        "scheduler": "ddim",
        "size": args.size,
        "steps": args.steps,
        "runs": args.runs,
        "cfg": args.cfg,
        "seed": args.seed,
        "bf16_matmul": args.bf16_matmul,
        "no_cudnn": args.no_cudnn,
        "wall_ms_median": wall_ms_median,
        "step_ms_median": step_med,
        "step_ms_min": step_min,
        "step_ms_max": step_max,
        "n_steps_sampled": n_step,
    });

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{entry}")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Backend-specific helpers — keep main() tidy.
// ---------------------------------------------------------------------------

#[cfg(feature = "scry-gpu-cuda")]
fn backend_sync() {
    // Wall-clock numbers are meaningless on a fully-async GPU pipeline if the
    // timer stops while work is still in-flight — sync before reading the
    // clock. CpuBackend doesn't need this.
    let _ = scry_llm::backend::scry_gpu::ScryGpuBackend::synchronize();
}
#[cfg(not(feature = "scry-gpu-cuda"))]
fn backend_sync() {}

fn apply_backend_toggles(
    bf16_matmul: bool,
    no_cudnn: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if bf16_matmul {
        apply_bf16_matmul()?;
    }
    if no_cudnn {
        apply_disable_cudnn()?;
    }
    Ok(())
}

#[cfg(feature = "scry-gpu-bf16")]
fn apply_bf16_matmul() -> Result<(), Box<dyn std::error::Error>> {
    scry_llm::backend::scry_gpu::ScryGpuBackend::set_bf16_matmul(true)
        .map_err(|e| format!("set_bf16_matmul: {e}"))?;
    println!("[toggle] bf16 matmul: ON");
    Ok(())
}
#[cfg(not(feature = "scry-gpu-bf16"))]
fn apply_bf16_matmul() -> Result<(), Box<dyn std::error::Error>> {
    Err("--bf16-matmul requires the `scry-gpu-bf16` feature".into())
}

#[cfg(feature = "scry-gpu-cudnn")]
fn apply_disable_cudnn() -> Result<(), Box<dyn std::error::Error>> {
    scry_llm::backend::scry_gpu::ScryGpuBackend::set_cudnn_conv(false)
        .map_err(|e| format!("set_cudnn_conv: {e}"))?;
    println!("[toggle] cuDNN convs: OFF");
    Ok(())
}
#[cfg(not(feature = "scry-gpu-cudnn"))]
fn apply_disable_cudnn() -> Result<(), Box<dyn std::error::Error>> {
    Err("--no-cudnn requires the `scry-gpu-cudnn` feature".into())
}
