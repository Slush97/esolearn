//! Side-by-side benchmark: scry-stt vs Python openai-whisper using live mic input.
//!
//! Records audio from your microphone, transcribes with scry-stt (Rust), then
//! calls the companion Python script to transcribe the same WAV with
//! openai-whisper, and displays timing + output side by side.
//!
//! Run:
//!   cargo run --release -p scry-stt --features "safetensors,live" --example bench_vs_python
//!
//! Prerequisites:
//!   pip install openai-whisper

#[cfg(not(feature = "safetensors"))]
compile_error!("This benchmark requires --features safetensors");
#[cfg(not(feature = "live"))]
compile_error!("This benchmark requires --features live");

use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use scry_llm::backend::cpu::CpuBackend as Backend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use scry_stt::checkpoint::load_whisper_checkpoint;
use scry_stt::decode::{greedy_decode, DecodeConfig};
use scry_stt::mel::{
    log_mel_spectrogram, pad_or_trim_audio, WHISPER_CHUNK_SAMPLES, WHISPER_SAMPLE_RATE,
};
use scry_stt::model::config::WhisperConfig;
use scry_stt::tokenizer::WhisperTokenizer;

fn model_dir(model_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("models/whisper-{model_name}"))
}

fn scripts_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts")
}

fn wait_for_enter() {
    let mut buf = String::new();
    io::stdin().read_line(&mut buf).unwrap();
}

/// Linearly resample audio from `src_rate` to `dst_rate`.
fn resample(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate {
        return samples.to_vec();
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = (samples.len() as f64 * ratio) as usize;
    (0..out_len)
        .map(|i| {
            let src_pos = i as f64 / ratio;
            let idx = src_pos as usize;
            let frac = (src_pos - idx as f64) as f32;
            let s0 = samples[idx.min(samples.len() - 1)];
            let s1 = samples[(idx + 1).min(samples.len() - 1)];
            s0 + frac * (s1 - s0)
        })
        .collect()
}

/// Write mono f32 PCM samples as a 16-bit WAV file.
fn write_wav(path: &std::path::Path, samples: &[f32], sample_rate: u32) {
    let num_samples = samples.len() as u32;
    let data_size = num_samples * 2;
    let file_size = 36 + data_size;

    let mut f = std::fs::File::create(path).expect("Failed to create WAV file");
    f.write_all(b"RIFF").unwrap();
    f.write_all(&file_size.to_le_bytes()).unwrap();
    f.write_all(b"WAVE").unwrap();
    f.write_all(b"fmt ").unwrap();
    f.write_all(&16u32.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap(); // PCM
    f.write_all(&1u16.to_le_bytes()).unwrap(); // mono
    f.write_all(&sample_rate.to_le_bytes()).unwrap();
    f.write_all(&(sample_rate * 2).to_le_bytes()).unwrap();
    f.write_all(&2u16.to_le_bytes()).unwrap();
    f.write_all(&16u16.to_le_bytes()).unwrap();
    f.write_all(b"data").unwrap();
    f.write_all(&data_size.to_le_bytes()).unwrap();
    for &s in samples {
        let i16_val = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        f.write_all(&i16_val.to_le_bytes()).unwrap();
    }
}

fn venv_python() -> PathBuf {
    let venv = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.venv/bin/python3");
    if venv.exists() {
        venv
    } else {
        PathBuf::from("python3")
    }
}

/// Run the Python whisper benchmark and parse JSON output.
fn run_python_benchmark(wav_path: &std::path::Path, model_name: &str) -> Option<serde_json::Value> {
    let script = scripts_dir().join("bench_whisper.py");
    if !script.exists() {
        eprintln!(
            "  \x1b[33mPython script not found: {}\x1b[0m",
            script.display()
        );
        return None;
    }

    print!("  Running Python openai-whisper... ");
    io::stdout().flush().unwrap();

    let output = Command::new(&venv_python())
        .arg(&script)
        .arg(wav_path)
        .arg(model_name)
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            println!("done");
            serde_json::from_str(stdout.trim()).ok()
        }
        Ok(out) => {
            println!("failed");
            let stderr = String::from_utf8_lossy(&out.stderr);
            eprintln!("  \x1b[33mPython error:\x1b[0m {stderr}");
            None
        }
        Err(e) => {
            println!("failed");
            eprintln!("  \x1b[33mCouldn't run python3: {e}\x1b[0m");
            eprintln!("  Make sure openai-whisper is installed: pip install openai-whisper");
            None
        }
    }
}

/// Run whisper.cpp benchmark and parse JSON output.
fn run_whispercpp_benchmark(wav_path: &std::path::Path, model_name: &str) -> Option<serde_json::Value> {
    let script = scripts_dir().join("bench_whisper_cpp.py");
    if !script.exists() {
        return None;
    }

    print!("  Running whisper.cpp... ");
    io::stdout().flush().unwrap();

    let output = Command::new(&venv_python())
        .arg(&script)
        .arg(wav_path)
        .arg(model_name)
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            // Check for error in JSON
            if stdout.contains("\"error\"") {
                println!("not available");
                return None;
            }
            println!("done");
            serde_json::from_str(stdout.trim()).ok()
        }
        Ok(out) => {
            println!("failed");
            let stderr = String::from_utf8_lossy(&out.stderr);
            eprintln!("  \x1b[33mwhisper.cpp error:\x1b[0m {stderr}");
            None
        }
        Err(_) => {
            println!("not available");
            None
        }
    }
}

struct RustResult {
    text: String,
    mel_ms: f64,
    encoder_ms: f64,
    decode_ms: f64,
    detok_ms: f64,
    total_inference_ms: f64,
    warm_runs: Vec<f64>,
    warm_avg: f64,
}

fn record_audio(device: &cpal::Device, input_config: &cpal::SupportedStreamConfig) -> Vec<f32> {
    let sample_rate = input_config.sample_rate().0;
    let channels = input_config.channels() as usize;
    let max_samples = WHISPER_CHUNK_SAMPLES * channels;

    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let buf_clone = Arc::clone(&buffer);

    let stream_config: cpal::StreamConfig = input_config.clone().into();
    let stream = device
        .build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut buf = buf_clone.lock().unwrap();
                let remaining = max_samples.saturating_sub(buf.len());
                if remaining > 0 {
                    let take = data.len().min(remaining);
                    buf.extend_from_slice(&data[..take]);
                }
            },
            |err| eprintln!("  Audio error: {err}"),
            None,
        )
        .expect("Failed to build input stream");

    stream.play().expect("Failed to start recording");
    print!("  \x1b[31m●\x1b[0m Recording... press \x1b[1mENTER\x1b[0m to stop  ");
    io::stdout().flush().unwrap();

    wait_for_enter();
    drop(stream);

    let raw_samples = buffer.lock().unwrap().clone();

    // Downmix to mono
    let mono: Vec<f32> = if channels > 1 {
        raw_samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        raw_samples
    };

    // Resample to 16kHz
    resample(&mono, sample_rate, WHISPER_SAMPLE_RATE)
}

fn transcribe_rust(
    audio_16k: &[f32],
    model: &scry_stt::model::WhisperModel<Backend>,
    tokenizer: &WhisperTokenizer,
    decode_config: &DecodeConfig,
) -> RustResult {
    // ── Cold inference ───────────────────────────────────────────────────
    let t_mel = Instant::now();
    let audio_chunk = pad_or_trim_audio(audio_16k);
    let mel = log_mel_spectrogram(&audio_chunk);
    let mel_tensor =
        Tensor::<Backend>::from_vec(mel.data, Shape::new(&[mel.n_mels, mel.n_frames]));
    let mel_ms = t_mel.elapsed().as_secs_f64() * 1000.0;

    let t_enc = Instant::now();
    let encoder_output = model.encode(&mel_tensor);
    let encoder_ms = t_enc.elapsed().as_secs_f64() * 1000.0;

    let t_dec = Instant::now();
    let tokens = greedy_decode(model, &encoder_output, decode_config);
    let decode_ms = t_dec.elapsed().as_secs_f64() * 1000.0;

    let t_detok = Instant::now();
    let text = tokenizer.decode(&tokens);
    let detok_ms = t_detok.elapsed().as_secs_f64() * 1000.0;

    let total_inference_ms = mel_ms + encoder_ms + decode_ms + detok_ms;

    // ── Warm runs ────────────────────────────────────────────────────────
    let mut warm_runs = Vec::new();
    let profile = std::env::var("SCRY_DECODE_PROFILE").is_ok();
    for run_i in 0..3 {
        let tw = Instant::now();
        let ac = pad_or_trim_audio(audio_16k);
        let m = log_mel_spectrogram(&ac);
        let mt = Tensor::<Backend>::from_vec(m.data, Shape::new(&[m.n_mels, m.n_frames]));
        let te = Instant::now();
        let enc = model.encode(&mt);
        let enc_ms = te.elapsed().as_secs_f64() * 1000.0;
        let td = Instant::now();
        let tok = greedy_decode(model, &enc, decode_config);
        let dec_ms = td.elapsed().as_secs_f64() * 1000.0;
        let _ = tokenizer.decode(&tok);
        let total = tw.elapsed().as_secs_f64() * 1000.0;
        if profile {
            eprintln!(
                "  [warm {run_i}] total={total:.1}ms  enc={enc_ms:.1}ms  dec={dec_ms:.1}ms"
            );
        }
        warm_runs.push(total);
    }
    let warm_avg = warm_runs.iter().sum::<f64>() / warm_runs.len() as f64;

    RustResult {
        text,
        mel_ms,
        encoder_ms,
        decode_ms,
        detok_ms,
        total_inference_ms,
        warm_runs,
        warm_avg,
    }
}

fn print_comparison(
    rust: &RustResult,
    python: Option<&serde_json::Value>,
    wcpp: Option<&serde_json::Value>,
    rust_model_load_ms: f64,
) {
    println!();
    println!("╔════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                   scry-stt vs whisper.cpp vs Python openai-whisper                       ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // ── Transcription comparison ─────────────────────────────────────────
    println!("  \x1b[1;36mscry-stt:\x1b[0m     \"{}\"", rust.text.trim());
    if let Some(w) = wcpp {
        println!(
            "  \x1b[1;35mwhisper.cpp:\x1b[0m  \"{}\"",
            w["text"].as_str().unwrap_or("(no output)").trim()
        );
    } else {
        println!("  \x1b[1;35mwhisper.cpp:\x1b[0m  \x1b[90m(skipped)\x1b[0m");
    }
    if let Some(py) = python {
        println!(
            "  \x1b[1;33mPython:\x1b[0m       \"{}\"",
            py["text"].as_str().unwrap_or("(no output)")
        );
    } else {
        println!("  \x1b[1;33mPython:\x1b[0m       \x1b[90m(skipped)\x1b[0m");
    }
    println!();

    // ── Extract whisper.cpp timings ──────────────────────────────────────
    let w_mel = wcpp.and_then(|w| w["mel_ms"].as_f64());
    let w_enc = wcpp.and_then(|w| w["encode_ms"].as_f64());
    let w_dec = wcpp.and_then(|w| w["decode_ms"].as_f64());
    let w_total = wcpp.and_then(|w| w["total_inference_ms"].as_f64());
    let w_warm = wcpp.and_then(|w| w["warm_avg_ms"].as_f64());
    let w_model = wcpp.and_then(|w| w["model_load_ms"].as_f64());

    // ── Extract Python timings ───────────────────────────────────────────
    let py_mel = python.and_then(|p| p["mel_ms"].as_f64());
    let py_decode = python.and_then(|p| p["decode_ms"].as_f64());
    let py_total = python.and_then(|p| p["total_inference_ms"].as_f64());
    let py_warm = python.and_then(|p| p["warm_avg_ms"].as_f64());
    let py_model = python.and_then(|p| p["model_load_ms"].as_f64());

    println!("┌─────────────────────────┬──────────────┬──────────────┬──────────────┬────────────┬────────────┐");
    println!("│ Stage                   │ scry-stt(ms) │ w.cpp (ms)   │ Python (ms)  │ vs w.cpp   │ vs Python  │");
    println!("├─────────────────────────┼──────────────┼──────────────┼──────────────┼────────────┼────────────┤");

    let fmt_speedup = |rust_ms: f64, other_ms: Option<f64>| -> String {
        match other_ms {
            Some(o) => {
                let ratio = o / rust_ms;
                let color = if ratio >= 1.0 { "32" } else { "31" };
                format!("\x1b[{color}m{ratio:>9.2}x\x1b[0m")
            }
            None => "         -".to_string(),
        }
    };

    let print_row = |label: &str, rust_ms: f64, wcpp_ms: Option<f64>, py_ms: Option<f64>| {
        let w_str = wcpp_ms.map_or("           -".to_string(), |v| format!("{v:>12.2}"));
        let p_str = py_ms.map_or("           -".to_string(), |v| format!("{v:>12.2}"));
        let vs_w = fmt_speedup(rust_ms, wcpp_ms);
        let vs_p = fmt_speedup(rust_ms, py_ms);
        println!("│ {label:<23} │ {rust_ms:>12.2} │ {w_str} │ {p_str} │ {vs_w} │ {vs_p} │");
    };

    print_row("Mel spectrogram", rust.mel_ms, w_mel, py_mel);
    print_row("Encoder", rust.encoder_ms, w_enc, None);
    print_row("Decode + detok", rust.decode_ms + rust.detok_ms, w_dec, py_decode);
    print_row("Total inference", rust.total_inference_ms, w_total, py_total);
    println!("├─────────────────────────┼──────────────┼──────────────┼──────────────┼────────────┼────────────┤");
    print_row("Warm avg (3 runs)", rust.warm_avg, w_warm, py_warm);
    println!("└─────────────────────────┴──────────────┴──────────────┴──────────────┴────────────┴────────────┘");
    println!();

    // ── Individual warm runs ─────────────────────────────────────────────
    println!("  Warm runs (ms):");
    let py_runs: Vec<f64> = python
        .and_then(|p| p["warm_runs_ms"].as_array())
        .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
        .unwrap_or_default();
    let w_runs: Vec<f64> = wcpp
        .and_then(|w| w["warm_runs_ms"].as_array())
        .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
        .unwrap_or_default();

    for i in 0..3 {
        let r = rust.warm_runs[i];
        let w = w_runs.get(i).map_or("       -".to_string(), |v| format!("{v:>8.2}"));
        let p = py_runs.get(i).map_or("       -".to_string(), |v| format!("{v:>8.2}"));
        println!(
            "    Run {}: scry-stt {r:>8.2}  |  whisper.cpp {w}  |  Python {p}",
            i + 1
        );
    }

    // ── Model load times (excluded from all inference totals above) ─────
    println!();
    println!("  Model load (excluded from inference):");
    println!("    scry-stt:    {rust_model_load_ms:.0}ms");
    if let Some(wm) = w_model {
        println!("    whisper.cpp: {wm:.0}ms");
    }
    if let Some(pm) = py_model {
        println!("    Python:      {pm:.0}ms");
    }
    println!();
}

fn main() {
    // Parse --model flag (default: tiny)
    let args: Vec<String> = std::env::args().collect();
    let model_name = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("tiny");

    let config = match model_name {
        "tiny" => WhisperConfig::tiny(),
        "base" => WhisperConfig::base(),
        "small" => WhisperConfig::small(),
        "medium" => WhisperConfig::medium(),
        other => {
            eprintln!("Unknown model: {other}. Choose from: tiny, base, small, medium");
            std::process::exit(1);
        }
    };

    let model_path = model_dir(model_name).join("model.safetensors");
    let tokenizer_path = model_dir(model_name).join("tokenizer.json");

    if !model_path.exists() {
        eprintln!("ERROR: Model not found at {}", model_path.display());
        eprintln!(
            "Convert weights: python3 scripts/convert_openai_to_hf.py {model_name}"
        );
        eprintln!(
            "Then copy tokenizer.json into {}",
            model_dir(model_name).display()
        );
        std::process::exit(1);
    }

    // ── Load Rust model ──────────────────────────────────────────────────
    println!();
    println!(
        "  \x1b[1;36mscry-stt\x1b[0m vs \x1b[1;33mPython whisper\x1b[0m — live benchmark (\x1b[1m{model_name}\x1b[0m)"
    );
    println!();
    print!("  Loading scry-stt model... ");
    io::stdout().flush().unwrap();

    let t0 = Instant::now();
    let model =
        load_whisper_checkpoint::<Backend>(&model_path, &config).expect("Failed to load model");
    let tokenizer =
        WhisperTokenizer::from_file(&tokenizer_path).expect("Failed to load tokenizer");
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("done ({load_ms:.0}ms)");

    // ── Set up audio device ──────────────────────────────────────────────
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("No input device available");
    let dev_name = device.name().unwrap_or_else(|_| "unknown".into());
    let input_config = device
        .default_input_config()
        .expect("No default input config");
    let sample_rate = input_config.sample_rate().0;
    let channels = input_config.channels();

    println!(
        "  Mic: \x1b[33m{dev_name}\x1b[0m ({sample_rate}Hz, {channels}ch)"
    );

    let decode_config = DecodeConfig {
        max_tokens: 224,
        ..DecodeConfig::default()
    };

    let wav_dir = std::env::temp_dir();

    // ── Main loop ────────────────────────────────────────────────────────
    let mut round = 0u32;
    loop {
        round += 1;
        println!();
        println!(
            "  ─── Round {round} ─── Press \x1b[1mENTER\x1b[0m to start recording (Ctrl+C to quit)"
        );
        wait_for_enter();

        let recording_start = Instant::now();
        let audio_16k = record_audio(&device, &input_config);
        let duration_s = recording_start.elapsed().as_secs_f64();

        println!(
            "  Captured {:.1}s ({} samples @ 16kHz)",
            duration_s,
            audio_16k.len()
        );

        if audio_16k.is_empty() {
            println!("  \x1b[33mNo audio captured.\x1b[0m");
            continue;
        }

        let peak = audio_16k
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        let rms =
            (audio_16k.iter().map(|s| s * s).sum::<f32>() / audio_16k.len() as f32).sqrt();
        println!("  peak={peak:.4}, rms={rms:.4}");

        if peak < 0.001 {
            println!("  \x1b[33mAudio is silent (peak < 0.001), skipping.\x1b[0m");
            continue;
        }

        // ── Save WAV for Python ──────────────────────────────────────────
        let wav_path = wav_dir.join(format!("scry_bench_r{round}.wav"));
        write_wav(&wav_path, &audio_16k, WHISPER_SAMPLE_RATE);
        println!("  Saved: {}", wav_path.display());
        println!();

        // ── Rust transcription ───────────────────────────────────────────
        print!("  Running scry-stt... ");
        io::stdout().flush().unwrap();
        let rust_result = transcribe_rust(&audio_16k, &model, &tokenizer, &decode_config);
        println!("done ({:.0}ms)", rust_result.total_inference_ms);

        // ── whisper.cpp transcription ────────────────────────────────────
        let wcpp_result = run_whispercpp_benchmark(&wav_path, model_name);

        // ── Python transcription ─────────────────────────────────────────
        let python_result = run_python_benchmark(&wav_path, model_name);

        // ── Side-by-side comparison ──────────────────────────────────────
        print_comparison(&rust_result, python_result.as_ref(), wcpp_result.as_ref(), load_ms);

        // Clean up temp WAV
        let _ = std::fs::remove_file(&wav_path);
    }
}
