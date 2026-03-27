//! Cold-start benchmark for scry-stt whisper-tiny.
//!
//! Measures model-load + first-transcription wall time, broken down by stage.
//!
//! Run: cargo run --release -p scry-stt --features safetensors --example bench_cold_start

#[cfg(not(feature = "safetensors"))]
compile_error!("This benchmark requires --features safetensors");

use std::path::PathBuf;
use std::time::Instant;

#[cfg(not(feature = "scry-gpu"))]
use scry_llm::backend::cpu::CpuBackend as Backend;
#[cfg(feature = "scry-gpu")]
use scry_llm::backend::scry_gpu::ScryGpuBackend as Backend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use scry_stt::checkpoint::load_whisper_checkpoint;
use scry_stt::decode::{greedy_decode, DecodeConfig};
use scry_stt::mel::{log_mel_spectrogram, pad_or_trim_audio, WHISPER_SAMPLE_RATE};
use scry_stt::model::config::WhisperConfig;
use scry_stt::tokenizer::WhisperTokenizer;

fn model_dir() -> PathBuf {
    // Same path convention as e2e.rs
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/whisper-tiny")
}

/// Read VmRSS from /proc/self/status (Linux only).
fn rss_mb() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let kb: f64 = line
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            return kb / 1024.0;
        }
    }
    0.0
}

fn main() {
    let model_path = model_dir().join("model.safetensors");
    let tokenizer_path = model_dir().join("tokenizer.json");

    if !model_path.exists() {
        eprintln!("ERROR: Model not found at {}", model_path.display());
        eprintln!("Download whisper-tiny from HuggingFace into {}", model_dir().display());
        std::process::exit(1);
    }

    let rss_baseline = rss_mb();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           scry-stt Cold-Start Benchmark (whisper-tiny)      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    #[cfg(feature = "scry-gpu")]
    println!("  Backend: scry-gpu (Vulkan compute)");
    #[cfg(all(feature = "mkl", not(feature = "scry-gpu")))]
    println!("  Backend: MKL (Intel Math Kernel Library)");
    #[cfg(all(feature = "blas", not(any(feature = "scry-gpu", feature = "mkl"))))]
    println!("  Backend: BLAS (OpenBLAS)");
    #[cfg(not(any(feature = "blas", feature = "mkl", feature = "scry-gpu")))]
    println!("  Backend: CPU (tiled + rayon)");
    println!();

    let config = WhisperConfig::tiny();

    // ── Stage 1: Model loading ──────────────────────────────────────────
    let t0 = Instant::now();
    let model = load_whisper_checkpoint::<Backend>(&model_path, &config)
        .expect("Failed to load checkpoint");
    let model_load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let rss_after_model = rss_mb();

    // ── Stage 2: Tokenizer loading ──────────────────────────────────────
    let t1 = Instant::now();
    let tokenizer = WhisperTokenizer::from_file(&tokenizer_path)
        .expect("Failed to load tokenizer");
    let tokenizer_load_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // ── Stage 3: Audio generation (2s 440Hz sine wave) ──────────────────
    let t2 = Instant::now();
    let duration_samples = 2 * WHISPER_SAMPLE_RATE as usize;
    let samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            let t = i as f32 / WHISPER_SAMPLE_RATE as f32;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
        })
        .collect();
    let audio_gen_ms = t2.elapsed().as_secs_f64() * 1000.0;

    // ── Stage 4: Mel spectrogram ────────────────────────────────────────
    let t3 = Instant::now();
    let audio_chunk = pad_or_trim_audio(&samples);
    let mel = log_mel_spectrogram(&audio_chunk);
    let mel_tensor = Tensor::<Backend>::from_vec(
        mel.data.clone(),
        Shape::new(&[mel.n_mels, mel.n_frames]),
    );
    let mel_ms = t3.elapsed().as_secs_f64() * 1000.0;

    // ── Stage 5: Encoder forward pass ───────────────────────────────────
    let t4 = Instant::now();
    let encoder_output = model.encode(&mel_tensor);
    let encoder_ms = t4.elapsed().as_secs_f64() * 1000.0;

    // ── Stage 6: Greedy decode (10 tokens) ──────────────────────────────
    let t5 = Instant::now();
    let decode_config = DecodeConfig {
        max_tokens: 10,
        ..DecodeConfig::default()
    };
    let tokens = greedy_decode(&model, &encoder_output, &decode_config);
    let decode_ms = t5.elapsed().as_secs_f64() * 1000.0;

    // ── Stage 7: Detokenization ─────────────────────────────────────────
    let t6 = Instant::now();
    let text = tokenizer.decode(&tokens);
    let detok_ms = t6.elapsed().as_secs_f64() * 1000.0;

    let rss_after_inference = rss_mb();

    // ── Totals ──────────────────────────────────────────────────────────
    let total_load_ms = model_load_ms + tokenizer_load_ms;
    let total_inference_ms = mel_ms + encoder_ms + decode_ms + detok_ms;
    let total_cold_start_ms = total_load_ms + audio_gen_ms + total_inference_ms;

    // ── Results table ───────────────────────────────────────────────────
    println!("┌─────────────────────────┬──────────────┐");
    println!("│ Stage                   │ Time (ms)    │");
    println!("├─────────────────────────┼──────────────┤");
    println!("│ Model load              │ {:>12.2} │", model_load_ms);
    println!("│ Tokenizer load          │ {:>12.2} │", tokenizer_load_ms);
    println!("│ Audio generation (2s)   │ {:>12.2} │", audio_gen_ms);
    println!("│ Mel spectrogram         │ {:>12.2} │", mel_ms);
    println!("│ Encoder forward         │ {:>12.2} │", encoder_ms);
    println!("│ Greedy decode (10 tok)  │ {:>12.2} │", decode_ms);
    println!("│ Detokenization          │ {:>12.2} │", detok_ms);
    println!("├─────────────────────────┼──────────────┤");
    println!("│ Total load              │ {:>12.2} │", total_load_ms);
    println!("│ Total inference         │ {:>12.2} │", total_inference_ms);
    println!("│ Total cold-start        │ {:>12.2} │", total_cold_start_ms);
    println!("└─────────────────────────┴──────────────┘");
    println!();

    println!("┌─────────────────────────┬──────────────┐");
    println!("│ Memory                  │ RSS (MB)     │");
    println!("├─────────────────────────┼──────────────┤");
    println!("│ Baseline                │ {:>12.1} │", rss_baseline);
    println!("│ After model load        │ {:>12.1} │", rss_after_model);
    println!("│ After inference         │ {:>12.1} │", rss_after_inference);
    println!("│ Delta (model)           │ {:>12.1} │", rss_after_model - rss_baseline);
    println!("│ Delta (total)           │ {:>12.1} │", rss_after_inference - rss_baseline);
    println!("└─────────────────────────┴──────────────┘");
    println!();

    println!("Tokens: {:?}", tokens);
    println!("Text:   '{}'", text);
    println!();
    println!("Vocab size: {}", tokenizer.vocab_size());

    // ── Warmup + averaged warm inference ────────────────────────────────
    println!();
    println!("── Warm inference (3-run average) ───────────────────────────");
    let mut warm_times = Vec::new();
    for run in 0..3 {
        let t_warm = Instant::now();
        let mel_w = log_mel_spectrogram(&audio_chunk);
        let mel_w_tensor = Tensor::<Backend>::from_vec(
            mel_w.data,
            Shape::new(&[mel_w.n_mels, mel_w.n_frames]),
        );
        let enc_w = model.encode(&mel_w_tensor);
        let tok_w = greedy_decode(&model, &enc_w, &decode_config);
        let _ = tokenizer.decode(&tok_w);
        let warm_ms = t_warm.elapsed().as_secs_f64() * 1000.0;
        println!("  Run {}: {:.2} ms", run + 1, warm_ms);
        warm_times.push(warm_ms);
    }
    let avg_warm: f64 = warm_times.iter().sum::<f64>() / warm_times.len() as f64;
    println!("  Average: {:.2} ms", avg_warm);
}
