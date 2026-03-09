//! # scry-stt — Native Rust Speech-to-Text Engine
//!
//! A production-grade Whisper implementation built on the `scry-llm` tensor
//! and backend infrastructure. Supports CPU and CUDA inference with zero-copy
//! model loading via memory-mapped safetensors.
//!
//! ## Architecture
//!
//! ```text
//! WAV/PCM → Mel Spectrogram → Whisper Encoder → Whisper Decoder → Text
//!           (STFT + mel)      (Conv1D + Transformer)  (cross-attn + causal)
//! ```
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use scry_stt::mel::log_mel_spectrogram;
//! use scry_stt::model::WhisperModel;
//! use scry_stt::model::config::WhisperConfig;
//! use scry_stt::decode::{greedy_decode, DecodeConfig};
//! use scry_llm::backend::cpu::CpuBackend;
//! use scry_llm::tensor::shape::Shape;
//! use scry_llm::tensor::Tensor;
//!
//! // 1. Compute mel spectrogram from audio
//! let audio_samples: Vec<f32> = load_wav("audio.wav");
//! let mel = log_mel_spectrogram(&audio_samples);
//! let mel_padded = mel.pad_or_truncate(3000);
//!
//! // 2. Load model
//! let config = WhisperConfig::tiny();
//! let model = WhisperModel::<CpuBackend>::new(config);
//!
//! // 3. Encode audio
//! let mel_tensor = Tensor::from_vec(mel_padded.data, Shape::new(&[80, 3000]));
//! let encoder_output = model.encode(&mel_tensor);
//!
//! // 4. Decode text
//! let decode_config = DecodeConfig::default();
//! let tokens = greedy_decode(&model, &encoder_output, &decode_config);
//! ```

pub mod decode;
pub mod error;
pub mod mel;
pub mod model;
pub mod tokenizer;

#[cfg(feature = "safetensors")]
pub mod checkpoint;
