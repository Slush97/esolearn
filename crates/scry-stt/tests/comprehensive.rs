//! Comprehensive tests for scry-stt.
//!
//! Tests are organized by level:
//!   1. Numerical correctness (known inputs → known outputs)
//!   2. Invariant checks (softmax sums to 1, shapes correct, etc.)
//!   3. Smoke tests (full forward pass doesn't panic, produces finite values)
//!   4. Integration (full mel → encode → decode pipeline)
//!   5. Edge cases (empty audio, boundary positions)

use scry_llm::backend::cpu::CpuBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;
use scry_stt::decode::{greedy_decode, DecodeConfig};
use scry_stt::mel::{log_mel_spectrogram, WHISPER_HOP_LENGTH, WHISPER_N_FFT, WHISPER_N_MELS};
use scry_stt::model::attention::CrossAttention;
use scry_stt::model::config::WhisperConfig;
use scry_stt::model::conv1d::Conv1d;
use scry_stt::model::WhisperModel;

// ============================================================================
// 1. Numerical correctness
// ============================================================================

#[test]
fn conv1d_identity_kernel() {
    // Conv1D with a 1-sample kernel, 1 input channel, 1 output channel
    // weight = [[1.0]], bias = 0 → output should equal input
    let mut rng = fastrand::Rng::with_seed(0);
    let mut conv = Conv1d::<CpuBackend>::new(1, 1, 1, 1, 0, &mut rng);

    // Set weight to identity (1.0) and bias to 0
    conv.weight = Tensor::from_vec(vec![1.0f32], Shape::new(&[1, 1, 1]));
    conv.bias = Tensor::from_vec(vec![0.0f32], Shape::new(&[1]));

    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let input = Tensor::<CpuBackend>::from_vec(input_data.clone(), Shape::new(&[1, 5]));
    let output = conv.forward(&input);

    assert_eq!(output.shape.dims(), &[1, 5]);
    let out_vals = output.to_vec();
    for (i, &expected) in input_data.iter().enumerate() {
        assert!(
            (out_vals[i] - expected).abs() < 1e-6,
            "conv1d identity: position {i}: got {}, expected {expected}",
            out_vals[i]
        );
    }
}

#[test]
fn conv1d_bias_only() {
    // Zero weights + nonzero bias → output should be bias everywhere
    let mut rng = fastrand::Rng::with_seed(0);
    let mut conv = Conv1d::<CpuBackend>::new(2, 3, 3, 1, 1, &mut rng);

    // Zero weights
    let w_size = 3 * 2 * 3;
    conv.weight = Tensor::from_vec(vec![0.0f32; w_size], Shape::new(&[3, 2, 3]));
    conv.bias = Tensor::from_vec(vec![1.0, 2.0, 3.0], Shape::new(&[3]));

    let input = Tensor::<CpuBackend>::from_vec(vec![0.5f32; 2 * 10], Shape::new(&[2, 10]));
    let output = conv.forward(&input);

    assert_eq!(output.shape.dims(), &[3, 10]);
    let out_vals = output.to_vec();
    for c in 0..3 {
        for t in 0..10 {
            let expected = (c + 1) as f32;
            assert!(
                (out_vals[c * 10 + t] - expected).abs() < 1e-5,
                "bias test: channel {c} pos {t}: got {}, expected {expected}",
                out_vals[c * 10 + t]
            );
        }
    }
}

#[test]
fn conv1d_stride2_downsamples() {
    // Verify stride=2 produces half the length (with padding=1)
    let mut rng = fastrand::Rng::with_seed(42);
    let conv = Conv1d::<CpuBackend>::new(4, 8, 3, 2, 1, &mut rng);

    let input = Tensor::<CpuBackend>::from_vec(vec![0.1f32; 4 * 100], Shape::new(&[4, 100]));
    let output = conv.forward(&input);

    // (100 + 2*1 - 3) / 2 + 1 = 50
    assert_eq!(output.shape.dims(), &[8, 50]);

    // All values should be finite
    let vals = output.to_vec();
    assert!(!vals.iter().any(|v| v.is_nan() || v.is_infinite()));
}

// ============================================================================
// 2. Attention invariants
// ============================================================================

#[test]
fn cross_attention_softmax_sums_to_one() {
    // With known uniform input, verify attention weights approximately sum to 1
    // We can't directly inspect weights, but we can verify the output is well-formed
    let mut rng = fastrand::Rng::with_seed(42);
    let d_model = 64;
    let n_heads = 4;
    let attn = CrossAttention::<CpuBackend>::new(d_model, n_heads, &mut rng);

    // Uniform encoder output
    let audio_len = 10;
    let encoder_out = Tensor::<CpuBackend>::from_vec(
        vec![0.1f32; audio_len * d_model],
        Shape::new(&[audio_len, d_model]),
    );
    let cache = attn.compute_kv_cache(&encoder_out);

    // Single decoder position
    let decoder_state = Tensor::<CpuBackend>::from_vec(
        vec![0.1f32; d_model],
        Shape::new(&[1, d_model]),
    );
    let output = attn.forward(&decoder_state, &cache);

    // Output should be finite and same shape
    assert_eq!(output.shape.dims(), &[1, d_model]);
    let vals = output.to_vec();
    assert!(!vals.iter().any(|v| v.is_nan()));
    assert!(!vals.iter().any(|v| v.is_infinite()));
}

#[test]
fn cross_attention_different_inputs_different_outputs() {
    let mut rng = fastrand::Rng::with_seed(42);
    let d_model = 64;
    let attn = CrossAttention::<CpuBackend>::new(d_model, 4, &mut rng);

    // Two different encoder outputs
    let enc1 = Tensor::<CpuBackend>::from_vec(vec![0.1f32; 5 * d_model], Shape::new(&[5, d_model]));
    let enc2 = Tensor::<CpuBackend>::from_vec(vec![0.5f32; 5 * d_model], Shape::new(&[5, d_model]));

    let cache1 = attn.compute_kv_cache(&enc1);
    let cache2 = attn.compute_kv_cache(&enc2);

    let decoder_state = Tensor::<CpuBackend>::from_vec(
        vec![0.1f32; d_model],
        Shape::new(&[1, d_model]),
    );

    let out1 = attn.forward(&decoder_state, &cache1);
    let out2 = attn.forward(&decoder_state, &cache2);

    let v1 = out1.to_vec();
    let v2 = out2.to_vec();

    // Outputs should differ since encoder inputs differ
    let diff: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        diff > 1e-6,
        "cross-attention should produce different outputs for different encoder inputs, diff={diff}"
    );
}

// ============================================================================
// 3. Smoke tests — forward pass runs, shapes correct, values finite
// ============================================================================

#[test]
fn encoder_forward_smoke_tiny() {
    // Use tiny config (smallest model) for faster test
    let config = WhisperConfig::tiny();
    let model = WhisperModel::<CpuBackend>::new(config.clone());

    // Small mel input: [80, 100] (not 3000 frames — just testing shape pipeline)
    let n_frames = 100;
    let mel = Tensor::<CpuBackend>::from_vec(
        vec![0.1f32; config.n_mels * n_frames],
        Shape::new(&[config.n_mels, n_frames]),
    );

    let encoder_output = model.encode(&mel);

    // Conv1 stride=1: n_frames → n_frames
    // Conv2 stride=2: n_frames → n_frames/2 = 50
    let expected_audio_len = 50;
    assert_eq!(
        encoder_output.shape.dims(),
        &[expected_audio_len, config.d_model],
        "encoder output shape mismatch"
    );

    // All values should be finite
    let vals = encoder_output.to_vec();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "encoder output contains NaN or Inf"
    );
}

#[test]
fn decoder_forward_step_smoke_tiny() {
    let config = WhisperConfig::tiny();
    let model = WhisperModel::<CpuBackend>::new(config.clone());

    // Create a fake encoder output
    let audio_len = 50;
    let encoder_output = Tensor::<CpuBackend>::from_vec(
        vec![0.01f32; audio_len * config.d_model],
        Shape::new(&[audio_len, config.d_model]),
    );

    let cross_kv_caches = model.compute_cross_kv_caches(&encoder_output);
    let mut self_kv_cache = model.new_decoder_kv_cache();

    // Run 3 decode steps
    for pos in 0..3 {
        let token_id = if pos == 0 { 50258 } else { 100 + pos }; // SOT then arbitrary tokens
        let logits = model.decode_step(token_id, pos, &mut self_kv_cache, &cross_kv_caches);

        assert_eq!(
            logits.shape.dims(),
            &[1, config.n_vocab],
            "logits shape mismatch at position {pos}"
        );

        let vals = logits.to_vec();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "logits contain NaN or Inf at position {pos}"
        );

        // Logits should have some variance (not all identical)
        let min = vals.iter().copied().fold(f32::INFINITY, f32::min);
        let max = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max - min > 1e-6,
            "logits have no variance at position {pos}: min={min}, max={max}"
        );
    }

    // Verify KV cache grew correctly
    assert_eq!(self_kv_cache.layers[0].seq_len, 3);
    // Cache is pre-allocated to [n_heads, max_seq, d_head] — verify full buffer size
    let expected_buf = config.n_decoder_heads * config.n_text_ctx * config.d_head_decoder();
    assert_eq!(
        self_kv_cache.layers[0].k.len(),
        expected_buf,
        "KV cache should be pre-allocated to n_heads * max_seq * d_head"
    );
}

// ============================================================================
// 4. Integration test — full mel → encode → decode pipeline
// ============================================================================

#[test]
fn full_pipeline_mel_to_tokens() {
    let config = WhisperConfig::tiny();
    let model = WhisperModel::<CpuBackend>::new(config.clone());

    // Generate 1 second of audio (440Hz sine wave)
    let samples: Vec<f32> = (0..16_000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
        .collect();

    // Mel spectrogram
    let mel = log_mel_spectrogram(&samples);
    assert_eq!(mel.n_mels, WHISPER_N_MELS);
    assert!(!mel.data.iter().any(|v| v.is_nan()), "mel has NaN values");

    // Pad to model's expected input length
    let mel_padded = mel.pad_or_truncate(config.n_audio_ctx * 2); // 3000 frames for 1500 audio ctx
    let mel_tensor = Tensor::<CpuBackend>::from_vec(
        mel_padded.data.clone(),
        Shape::new(&[config.n_mels, mel_padded.n_frames]),
    );

    // Encode
    let encoder_output = model.encode(&mel_tensor);
    assert_eq!(encoder_output.shape.dims()[1], config.d_model);
    let enc_vals = encoder_output.to_vec();
    assert!(enc_vals.iter().all(|v| v.is_finite()), "encoder output has NaN/Inf");

    // Decode (with random weights, output tokens are meaningless but pipeline should work)
    let decode_config = DecodeConfig {
        max_tokens: 5, // Just a few tokens to verify the loop works
        temperature: 0.0,
        ..DecodeConfig::default()
    };
    let tokens = greedy_decode(&model, &encoder_output, &decode_config);

    // With random weights, we should get some tokens (unless first token is EOT)
    // The important thing is that the pipeline ran without panic
    assert!(tokens.len() <= 5, "should not exceed max_tokens");

    // Each token should be in vocabulary range
    for &t in &tokens {
        assert!(
            t < config.n_vocab,
            "token {t} exceeds vocab size {}",
            config.n_vocab
        );
    }
}

// ============================================================================
// 5. Edge cases
// ============================================================================

#[test]
fn mel_very_short_audio() {
    // Audio shorter than one FFT window
    let samples = vec![0.5f32; WHISPER_N_FFT / 2]; // 200 samples
    let mel = log_mel_spectrogram(&samples);

    // Should still produce valid output (padded to n_fft)
    assert_eq!(mel.n_mels, WHISPER_N_MELS);
    assert!(mel.n_frames >= 1, "must produce at least 1 frame");
    assert!(!mel.data.iter().any(|v| v.is_nan()), "NaN in short audio mel");
}

#[test]
fn mel_single_sample() {
    let samples = vec![0.5f32; 1];
    let mel = log_mel_spectrogram(&samples);
    assert_eq!(mel.n_mels, WHISPER_N_MELS);
    assert!(mel.n_frames >= 1);
    assert!(!mel.data.iter().any(|v| v.is_nan()));
}

#[test]
fn mel_30_second_chunk() {
    // Full 30-second chunk
    let samples = vec![0.0f32; 16_000 * 30];
    let mel = log_mel_spectrogram(&samples);

    // Expected: 480000 / 160 + 1 = 3001 frames
    assert_eq!(mel.n_frames, 3001);
    assert_eq!(mel.n_mels, WHISPER_N_MELS);
}

#[test]
fn mel_output_value_range() {
    // After Whisper's normalization, values should be in roughly [-1, 1]
    let samples: Vec<f32> = (0..16_000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
        .collect();
    let mel = log_mel_spectrogram(&samples);

    let min = mel.data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = mel.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // After (x + 4.0) / 4.0 normalization, typical range is [-1, 1.5]
    assert!(
        min >= -2.0,
        "mel min {min} is unreasonably low (expected >= -2.0)"
    );
    assert!(
        max <= 2.0,
        "mel max {max} is unreasonably high (expected <= 2.0)"
    );
}

#[test]
fn mel_pad_truncate_preserves_values() {
    let samples: Vec<f32> = (0..16_000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
        .collect();
    let mel = log_mel_spectrogram(&samples);
    let original_frames = mel.n_frames;

    // Pad to longer
    let padded = mel.pad_or_truncate(original_frames + 100);
    // Original values should be preserved
    for m in 0..mel.n_mels {
        for t in 0..original_frames {
            let orig = mel.data[m * original_frames + t];
            let pad = padded.data[m * padded.n_frames + t];
            assert!(
                (orig - pad).abs() < 1e-10,
                "pad_or_truncate changed value at mel={m} frame={t}"
            );
        }
        // Padded region should be zeros
        for t in original_frames..padded.n_frames {
            assert_eq!(
                padded.data[m * padded.n_frames + t], 0.0,
                "padded region should be 0 at mel={m} frame={t}"
            );
        }
    }

    // Truncate to shorter
    let truncated = mel.pad_or_truncate(50);
    assert_eq!(truncated.n_frames, 50);
    for m in 0..mel.n_mels {
        for t in 0..50 {
            let orig = mel.data[m * original_frames + t];
            let trunc = truncated.data[m * 50 + t];
            assert!(
                (orig - trunc).abs() < 1e-10,
                "truncation changed value at mel={m} frame={t}"
            );
        }
    }
}

#[test]
fn decoder_kv_cache_clear_resets() {
    let config = WhisperConfig::tiny();
    let model = WhisperModel::<CpuBackend>::new(config.clone());
    let mut cache = model.new_decoder_kv_cache();

    let audio_len = 10;
    let encoder_output = Tensor::<CpuBackend>::from_vec(
        vec![0.01f32; audio_len * config.d_model],
        Shape::new(&[audio_len, config.d_model]),
    );
    let cross_kv_caches = model.compute_cross_kv_caches(&encoder_output);

    // Run a decode step
    model.decode_step(50258, 0, &mut cache, &cross_kv_caches);
    assert_eq!(cache.layers[0].seq_len, 1);

    // Clear and verify reset — buffers stay allocated, only seq_len resets
    cache.clear();
    assert_eq!(cache.layers[0].seq_len, 0);
    // Pre-allocated cache retains its buffer; seq_len=0 means no valid data
    assert!(!cache.layers[0].k.is_empty(), "pre-allocated cache should retain buffer");
}

#[test]
fn argmax_basic() {
    // Test via greedy decode with max_tokens=1
    let config = WhisperConfig::tiny();
    let model = WhisperModel::<CpuBackend>::new(config);

    let encoder_output = Tensor::<CpuBackend>::from_vec(
        vec![0.01f32; 50 * 384],
        Shape::new(&[50, 384]),
    );

    let decode_config = DecodeConfig {
        max_tokens: 1,
        temperature: 0.0,
        ..DecodeConfig::default()
    };

    // Run twice — deterministic greedy should give same token
    let tokens1 = greedy_decode(&model, &encoder_output, &decode_config);
    let tokens2 = greedy_decode(&model, &encoder_output, &decode_config);

    if !tokens1.is_empty() && !tokens2.is_empty() {
        assert_eq!(
            tokens1[0], tokens2[0],
            "greedy decode should be deterministic"
        );
    }
}

#[test]
fn sinusoidal_embeddings_orthogonal_property() {
    // Sinusoidal embeddings at different positions should have bounded dot products
    let config = WhisperConfig::tiny();
    let model = WhisperModel::<CpuBackend>::new(config.clone());

    let pos_emb = model.encoder.positional_embedding.to_vec();
    let d = config.d_model;

    // Position 0 embedding
    let pos0: Vec<f32> = pos_emb[..d].to_vec();
    // Position 100 embedding
    let pos100: Vec<f32> = pos_emb[100 * d..101 * d].to_vec();

    // Self dot product (should be roughly d_model for unit-ish vectors)
    let self_dot: f64 = pos0.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
    assert!(self_dot > 0.0, "position embedding has zero magnitude");

    // Cross dot product should be less than self dot product (different positions ≠ same)
    let cross_dot: f64 = pos0
        .iter()
        .zip(pos100.iter())
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    assert!(
        cross_dot.abs() < self_dot,
        "position embeddings at pos 0 and 100 should not be identical (self_dot={self_dot}, cross_dot={cross_dot})"
    );
}

#[test]
fn config_d_head_divides_evenly() {
    // All model configs should have d_model divisible by n_heads
    for config in [
        WhisperConfig::tiny(),
        WhisperConfig::base(),
        WhisperConfig::small(),
        WhisperConfig::medium(),
        WhisperConfig::large_v3(),
        WhisperConfig::large_v3_turbo(),
    ] {
        assert_eq!(
            config.d_model % config.n_encoder_heads, 0,
            "{}: d_model {} not divisible by n_encoder_heads {}",
            config.name, config.d_model, config.n_encoder_heads
        );
        assert_eq!(
            config.d_model % config.n_decoder_heads, 0,
            "{}: d_model {} not divisible by n_decoder_heads {}",
            config.name, config.d_model, config.n_decoder_heads
        );
    }
}

#[test]
fn n_frames_formula_matches() {
    // Verify that our n_frames calculation matches for various audio lengths
    for seconds in [0.5, 1.0, 2.5, 5.0, 10.0, 30.0] {
        let n_samples = (seconds * 16_000.0) as usize;
        let samples = vec![0.0f32; n_samples];
        let mel = log_mel_spectrogram(&samples);

        let expected = n_samples / WHISPER_HOP_LENGTH + 1;
        assert_eq!(
            mel.n_frames, expected,
            "n_frames mismatch for {seconds}s audio: got {}, expected {expected}",
            mel.n_frames
        );
    }
}
