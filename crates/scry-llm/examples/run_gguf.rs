// SPDX-License-Identifier: MIT OR Apache-2.0

//! Load a Llama-architecture GGUF checkpoint and generate text.
//!
//! ```text
//! cargo run -p scry-llm --release \
//!   --features "gguf,tokenizer" \
//!   --example run_gguf -- \
//!   --model path/to/llama-3.2-1b-instruct-q4_k_m.gguf \
//!   --tokenizer path/to/tokenizer.json \
//!   --prompt "The capital of France is" \
//!   --max-tokens 32
//! ```
//!
//! With `--features cuda` the model runs on the live CUDA backend; otherwise
//! it falls back to the CPU backend (much slower; useful for smoke-testing
//! the loader without a GPU).
//!
//! The GGUF file embeds tokenizer state (`tokenizer.ggml.tokens` / `merges`),
//! but parsing those into an HF-compatible tokenizer is post-M14 work — for
//! now you supply a sibling `tokenizer.json` from the original HF repo.

#![cfg(all(feature = "gguf", feature = "tokenizer"))]

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use scry_llm::generate::{generate, SamplingConfig};
use scry_llm::nn::llama::LlamaModel;
use scry_llm::tokenizer::HfTokenizer;

// Pick the strongest backend available at build time.
#[cfg(feature = "cuda")]
type B = scry_llm::backend::cuda::CudaBackend;
#[cfg(all(not(feature = "cuda"), feature = "scry-gpu"))]
type B = scry_llm::backend::scry_gpu::ScryGpuBackend;
#[cfg(all(not(feature = "cuda"), not(feature = "scry-gpu")))]
type B = scry_llm::backend::cpu::CpuBackend;

struct Args {
    model: PathBuf,
    tokenizer: PathBuf,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    seed: u64,
}

fn parse_args() -> Result<Args, String> {
    let mut model: Option<PathBuf> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut prompt = String::from("The capital of France is");
    let mut max_tokens: usize = 32;
    let mut temperature: f32 = 0.0;
    let mut seed: u64 = 0;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut take = || it.next().ok_or_else(|| format!("missing value for {arg}"));
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(take()?)),
            "--tokenizer" => tokenizer = Some(PathBuf::from(take()?)),
            "--prompt" => prompt = take()?,
            "--max-tokens" => {
                max_tokens = take()?.parse().map_err(|e| format!("--max-tokens: {e}"))?;
            }
            "--temperature" => {
                temperature = take()?.parse().map_err(|e| format!("--temperature: {e}"))?;
            }
            "--seed" => {
                seed = take()?.parse().map_err(|e| format!("--seed: {e}"))?;
            }
            "-h" | "--help" => {
                return Err("see file header for usage".into());
            }
            other => return Err(format!("unknown arg {other}")),
        }
    }
    Ok(Args {
        model: model.ok_or("--model required")?,
        tokenizer: tokenizer.ok_or("--tokenizer required")?,
        prompt,
        max_tokens,
        temperature,
        seed,
    })
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    #[cfg(feature = "cuda")]
    scry_llm::backend::cuda::init_gpu(0);

    eprintln!("loading {}…", args.model.display());
    let t0 = Instant::now();
    let model = match LlamaModel::<B>::from_gguf(&args.model) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: load gguf: {e}");
            return ExitCode::from(1);
        }
    };
    eprintln!(
        "  loaded {} params in {:.2}s",
        model.n_params(),
        t0.elapsed().as_secs_f64()
    );

    let tokenizer = match HfTokenizer::from_file(&args.tokenizer) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: load tokenizer: {e}");
            return ExitCode::from(1);
        }
    };

    let mut token_ids = match tokenizer.bos_id() {
        Some(bos) => vec![bos],
        None => Vec::new(),
    };
    token_ids.extend(tokenizer.encode(&args.prompt));
    eprintln!("prompt: {:?} ({} tokens)", args.prompt, token_ids.len());

    let cfg = SamplingConfig {
        temperature: args.temperature,
        top_k: 0,
        top_p: 1.0,
        max_tokens: args.max_tokens,
    };
    let mut rng = fastrand::Rng::with_seed(args.seed);

    let t0 = Instant::now();
    let generated = generate(&model, &token_ids, &cfg, &mut rng);
    let elapsed = t0.elapsed();

    let text = tokenizer.decode(&generated);
    println!("{}{}", args.prompt, text);
    eprintln!(
        "{} tokens in {:.2}s ({:.1} tok/s)",
        generated.len(),
        elapsed.as_secs_f64(),
        generated.len() as f64 / elapsed.as_secs_f64()
    );
    ExitCode::SUCCESS
}
