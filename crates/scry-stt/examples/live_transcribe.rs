//! Live microphone transcription using scry-stt.
//!
//! Records audio from the default input device, then transcribes on stop.
//!
//! Run:
//!   cargo run --release -p scry-stt --features "safetensors,live" --example live_transcribe

#[cfg(not(feature = "safetensors"))]
compile_error!("This example requires --features safetensors");
#[cfg(not(feature = "live"))]
compile_error!("This example requires --features live");

use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use scry_llm::backend::cpu::CpuBackend as Backend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use scry_stt::checkpoint::load_whisper_checkpoint;
use scry_stt::decode::{greedy_decode, DecodeConfig};
use scry_stt::mel::{log_mel_spectrogram, pad_or_trim_audio, WHISPER_CHUNK_SAMPLES, WHISPER_SAMPLE_RATE};
use scry_stt::model::config::WhisperConfig;
use scry_stt::tokenizer::WhisperTokenizer;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/whisper-tiny")
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
fn write_wav_f32(path: &str, samples: &[f32], sample_rate: u32) {
    use std::io::Write;
    let num_samples = samples.len() as u32;
    let data_size = num_samples * 2; // 16-bit = 2 bytes per sample
    let file_size = 36 + data_size;

    let mut f = std::fs::File::create(path).expect("Failed to create WAV file");
    // RIFF header
    f.write_all(b"RIFF").unwrap();
    f.write_all(&file_size.to_le_bytes()).unwrap();
    f.write_all(b"WAVE").unwrap();
    // fmt chunk
    f.write_all(b"fmt ").unwrap();
    f.write_all(&16u32.to_le_bytes()).unwrap(); // chunk size
    f.write_all(&1u16.to_le_bytes()).unwrap(); // PCM
    f.write_all(&1u16.to_le_bytes()).unwrap(); // mono
    f.write_all(&sample_rate.to_le_bytes()).unwrap();
    f.write_all(&(sample_rate * 2).to_le_bytes()).unwrap(); // byte rate
    f.write_all(&2u16.to_le_bytes()).unwrap(); // block align
    f.write_all(&16u16.to_le_bytes()).unwrap(); // bits per sample
    // data chunk
    f.write_all(b"data").unwrap();
    f.write_all(&data_size.to_le_bytes()).unwrap();
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let i16_val = (clamped * 32767.0) as i16;
        f.write_all(&i16_val.to_le_bytes()).unwrap();
    }
}

fn main() {
    let model_path = model_dir().join("model.safetensors");
    let tokenizer_path = model_dir().join("tokenizer.json");

    if !model_path.exists() {
        eprintln!("ERROR: Model not found at {}", model_path.display());
        eprintln!(
            "Download whisper-tiny from HuggingFace into {}",
            model_dir().display()
        );
        std::process::exit(1);
    }

    // ── Load model ──────────────────────────────────────────────────────
    println!();
    println!("  \x1b[1;36mscry-stt\x1b[0m live transcription (whisper-tiny)");
    print!("  Loading model... ");
    io::stdout().flush().unwrap();

    let t0 = Instant::now();
    let config = WhisperConfig::tiny();
    let model =
        load_whisper_checkpoint::<Backend>(&model_path, &config).expect("Failed to load model");
    let tokenizer =
        WhisperTokenizer::from_file(&tokenizer_path).expect("Failed to load tokenizer");
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!("done ({load_ms:.0}ms)");
    println!();

    // ── Set up audio device ─────────────────────────────────────────────
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("No input device available");
    let dev_name = device.name().unwrap_or_else(|_| "unknown".into());
    println!("  Using: \x1b[33m{dev_name}\x1b[0m");

    let input_config = device
        .default_input_config()
        .expect("No default input config");
    let sample_rate = input_config.sample_rate().0;
    let channels = input_config.channels() as usize;
    println!(
        "  Sample rate: {sample_rate} Hz, channels: {channels}"
    );
    println!();

    let decode_config = DecodeConfig {
        max_tokens: 224,
        ..DecodeConfig::default()
    };

    // ── Main loop ───────────────────────────────────────────────────────
    loop {
        println!(
            "  Press \x1b[1mENTER\x1b[0m to start recording (Ctrl+C to quit)..."
        );
        wait_for_enter();

        // Start recording
        let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let buf_clone = Arc::clone(&buffer);
        let recording_start = Instant::now();

        let max_samples = WHISPER_CHUNK_SAMPLES * channels; // 30s cap, pre-downmix

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
        print!(
            "  \x1b[31m●\x1b[0m Recording... press \x1b[1mENTER\x1b[0m to stop  "
        );
        io::stdout().flush().unwrap();

        wait_for_enter();
        drop(stream); // stop recording

        let duration_s = recording_start.elapsed().as_secs_f64();
        let raw_samples = buffer.lock().unwrap().clone();

        println!(
            "  \x1b[90m■\x1b[0m Stopped. ({:.1}s captured, {} samples)",
            duration_s,
            raw_samples.len()
        );

        if raw_samples.is_empty() {
            println!("  \x1b[33mNo audio captured.\x1b[0m");
            println!();
            continue;
        }

        // ── Downmix to mono ─────────────────────────────────────────────
        let mono: Vec<f32> = if channels > 1 {
            raw_samples
                .chunks_exact(channels)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                .collect()
        } else {
            raw_samples
        };

        // ── Resample to 16kHz ───────────────────────────────────────────
        let audio_16k = resample(&mono, sample_rate, WHISPER_SAMPLE_RATE);

        // ── Audio diagnostics ───────────────────────────────────────────
        let peak = audio_16k.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let rms = (audio_16k.iter().map(|s| s * s).sum::<f32>() / audio_16k.len() as f32).sqrt();
        println!(
            "  Audio: {} samples @ 16kHz ({:.1}s), peak={:.4}, rms={:.4}",
            audio_16k.len(),
            audio_16k.len() as f64 / 16000.0,
            peak,
            rms,
        );

        if peak < 0.001 {
            println!("  \x1b[33m⚠ Audio is essentially silent (peak < 0.001)\x1b[0m");
            println!();
            continue;
        }

        // Save debug WAV so you can listen back (set SCRY_WAV=path to enable)
        if let Ok(wav_path) = std::env::var("SCRY_WAV") {
            write_wav_f32(&wav_path, &audio_16k, WHISPER_SAMPLE_RATE);
            println!("  Saved debug WAV to: {wav_path}");
        }

        // ── Transcribe ──────────────────────────────────────────────────
        print!("  Transcribing... ");
        io::stdout().flush().unwrap();

        let t_infer = Instant::now();
        // Pad audio to 30 seconds before computing mel spectrogram.
        // This matches the Python Whisper pipeline and ensures silence frames
        // receive correct normalized values (not 0.0 which represents moderate energy).
        let audio_chunk = pad_or_trim_audio(&audio_16k);
        let mel = log_mel_spectrogram(&audio_chunk);
        let mel_tensor = Tensor::<Backend>::from_vec(
            mel.data,
            Shape::new(&[mel.n_mels, mel.n_frames]),
        );
        let encoder_output = model.encode(&mel_tensor);
        let tokens = greedy_decode(&model, &encoder_output, &decode_config);
        let text = tokenizer.decode(&tokens);
        let infer_ms = t_infer.elapsed().as_secs_f64() * 1000.0;

        println!("done ({infer_ms:.0}ms)");
        println!("  Tokens: {:?}", &tokens[..tokens.len().min(20)]);
        println!();
        if text.trim().is_empty() {
            println!(
                "  \x1b[90m(no speech detected)\x1b[0m"
            );
        } else {
            println!(
                "  \x1b[1;32m>\x1b[0m \x1b[1m\"{}\"\x1b[0m",
                text.trim()
            );
        }
        println!();
    }
}
