//! Transcribe a WAV file using scry-stt whisper-tiny.
//!
//! Run: cargo run --release -p scry-stt --features safetensors --example transcribe_wav -- path/to/file.wav

#[cfg(not(feature = "safetensors"))]
compile_error!("This example requires --features safetensors");

use std::path::PathBuf;

#[cfg(not(feature = "wgpu"))]
use scry_llm::backend::cpu::CpuBackend as Backend;
#[cfg(feature = "wgpu")]
use scry_llm::backend::wgpu::WgpuBackend as Backend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use scry_stt::checkpoint::load_whisper_checkpoint;
use scry_stt::decode::{greedy_decode, DecodeConfig};
use scry_stt::mel::{log_mel_spectrogram, pad_or_trim_audio, WHISPER_SAMPLE_RATE};
use scry_stt::model::config::WhisperConfig;
use scry_stt::tokenizer::WhisperTokenizer;

fn load_wav_pcm_16k(path: &std::path::Path) -> Vec<f32> {
    let data = std::fs::read(path).expect("Failed to read WAV file");
    // Parse WAV header - expect 16-bit PCM mono 16kHz
    assert!(&data[0..4] == b"RIFF", "Not a RIFF file");
    assert!(&data[8..12] == b"WAVE", "Not a WAVE file");

    // Find "data" chunk
    let mut pos = 12;
    while pos + 8 < data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if chunk_id == b"data" {
            let pcm_data = &data[pos + 8..pos + 8 + chunk_size];
            // Convert 16-bit PCM to f32
            return pcm_data
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                .collect();
        }
        pos += 8 + chunk_size;
    }
    panic!("No data chunk found in WAV file");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let wav_path = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/jfk.wav")
    };

    eprintln!("Loading WAV: {}", wav_path.display());
    let samples = load_wav_pcm_16k(&wav_path);
    eprintln!("Audio: {} samples ({:.2}s at {}Hz)", samples.len(), samples.len() as f64 / WHISPER_SAMPLE_RATE as f64, WHISPER_SAMPLE_RATE);

    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/whisper-tiny");
    let config = WhisperConfig::tiny();
    let model = load_whisper_checkpoint::<Backend>(&model_dir.join("model.safetensors"), &config)
        .expect("Failed to load model");
    let tokenizer = WhisperTokenizer::from_file(&model_dir.join("tokenizer.json"))
        .expect("Failed to load tokenizer");

    let audio_chunk = pad_or_trim_audio(&samples);
    eprintln!("Audio chunk: {} samples, first 10: {:?}", audio_chunk.len(), &audio_chunk[..10]);
    eprintln!("Audio non-zero samples: {}", audio_chunk.iter().filter(|x| x.abs() > 1e-8).count());
    eprintln!("Audio max abs: {}", audio_chunk.iter().map(|x| x.abs()).fold(0.0f32, f32::max));

    let mel = log_mel_spectrogram(&audio_chunk);
    eprintln!("Mel: {}x{}", mel.n_mels, mel.n_frames);
    // Print first few mel values
    eprintln!("Mel[0][0..5]: {:?}", &mel.data[0..5]);
    eprintln!("Mel[0][100..105]: {:?}", &mel.data[100..105]);
    let mel_min = mel.data.iter().cloned().fold(f32::INFINITY, f32::min);
    let mel_max = mel.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mel_mean = mel.data.iter().sum::<f32>() / mel.data.len() as f32;
    eprintln!("Mel stats: min={mel_min}, max={mel_max}, mean={mel_mean}");

    let mel_tensor = Tensor::<Backend>::from_vec(
        mel.data,
        Shape::new(&[mel.n_mels, mel.n_frames]),
    );

    // === Debug: compare first decode step against Python reference ===
    let sot_token = 50258usize;
    let tok_emb = &model.decoder.token_embedding;
    let pos_emb = &model.decoder.positional_embedding;
    let d = model.config.d_model;

    // Get token embedding for SOT
    let tok_data = tok_emb.to_vec();
    let tok_vec = &tok_data[sot_token * d..(sot_token + 1) * d];
    eprintln!("Rust token emb SOT first 5: {:?}", &tok_vec[..5]);

    let pos_data = pos_emb.to_vec();
    let pos_vec = &pos_data[0..d];
    eprintln!("Rust pos emb [0] first 5: {:?}", &pos_vec[..5]);

    let x: Vec<f32> = tok_vec.iter().zip(pos_vec.iter()).map(|(a, b)| a + b).collect();
    eprintln!("Rust x = tok + pos first 5: {:?}", &x[..5]);

    // Layer norm
    let ln = &model.decoder.blocks[0].attn_ln;
    let x_tensor = scry_llm::tensor::Tensor::<Backend>::from_vec(x.clone(), Shape::new(&[1, d]));
    let x_normed = ln.forward(&x_tensor);
    let xn = x_normed.to_vec();
    eprintln!("Rust after LN first 5: {:?}", &xn[..5]);

    // Q projection via fused QKV
    let qkv_w = &model.decoder.blocks[0].self_attn.qkv_weight;
    let qkv_b = &model.decoder.blocks[0].self_attn.qkv_bias;
    let qkv = scry_llm::ops::matmul_bias(&x_normed, qkv_w, qkv_b, 1, d, 3 * d, false, false);
    let qkv_data = qkv.to_vec();
    eprintln!("Rust Q (from fused QKV) first 5: {:?}", &qkv_data[..5]);
    let q_slice = &qkv_data[..d];
    let q_min = q_slice.iter().cloned().fold(f32::INFINITY, f32::min);
    let q_max = q_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let q_mean = q_slice.iter().sum::<f32>() / d as f32;
    eprintln!("Rust Q stats: min={q_min}, max={q_max}, mean={q_mean}");

    // === End debug ===

    let encoder_output = model.encode(&mel_tensor);
    let tokens = greedy_decode(&model, &encoder_output, &DecodeConfig::default());
    let text = tokenizer.decode(&tokens);

    println!("{text}");
}
