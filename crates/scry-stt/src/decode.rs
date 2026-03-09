use scry_llm::backend::MathBackend;
use scry_llm::tensor::Tensor;

use crate::model::WhisperModel;
use crate::model::attention::CrossKvCache;
use crate::model::decoder::DecoderKvCache;
use crate::tokenizer::{EOT_TOKEN, SOT_TOKEN, NO_TIMESTAMPS_TOKEN};

/// Language token for English.
pub const LANG_EN_TOKEN: usize = 50259;
/// Task token for transcription (as opposed to translation).
pub const TRANSCRIBE_TOKEN: usize = 50359;

/// Configuration for decoding behavior.
#[derive(Clone, Debug)]
pub struct DecodeConfig {
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Temperature for sampling (0 = greedy).
    pub temperature: f32,
    /// End-of-text token ID.
    pub eot_token: usize,
    /// Start-of-transcript token ID.
    pub sot_token: usize,
    /// Conditioning prompt tokens force-fed before generation.
    ///
    /// For Whisper, this should be `[SOT, <|en|>, <|transcribe|>, <|notimestamps|>]`.
    /// The last token's logits produce the first generated token.
    pub prompt_tokens: Vec<usize>,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            max_tokens: 224,
            temperature: 0.0, // greedy by default
            eot_token: EOT_TOKEN,
            sot_token: SOT_TOKEN,
            prompt_tokens: vec![
                SOT_TOKEN,
                LANG_EN_TOKEN,
                TRANSCRIBE_TOKEN,
                NO_TIMESTAMPS_TOKEN,
            ],
        }
    }
}

/// A transcribed segment with text and timing information.
#[derive(Clone, Debug)]
pub struct Segment {
    /// Transcribed text for this segment.
    pub text: String,
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
    /// Token IDs that produced this segment.
    pub tokens: Vec<usize>,
}

/// GPT-2's `<|endoftext|>` token — distinct from Whisper's EOT (50257).
const GPT2_ENDOFTEXT: usize = 50256;

/// Suppress non-text tokens in logits (in-place).
///
/// Matches Whisper's Python `SuppressTokens` + `ApplyTimestampRules` logic:
/// - GPT-2's `<|endoftext|>` (50256) is suppressed (not our EOT).
/// - All special tokens from SOT (50258) onwards are set to -inf.
/// - EOT (50257) is left alone so the model can terminate the sequence.
fn suppress_special_tokens(logits: &mut [f32]) {
    // Suppress GPT-2 endoftext — it's not Whisper's EOT but leaks through otherwise
    logits[GPT2_ENDOFTEXT] = f32::NEG_INFINITY;
    // Suppress SOT and every Whisper special token after it
    for i in SOT_TOKEN..logits.len() {
        logits[i] = f32::NEG_INFINITY;
    }
}

/// Fused argmax with token suppression — zero allocations.
///
/// Scans logits once, skipping suppressed token indices (GPT-2 endoftext 50256
/// and all tokens >= SOT 50258), returning the index of the maximum allowed logit.
/// EOT (50257) is allowed so the model can terminate naturally.
fn argmax_suppressed(logits: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        // Skip suppressed tokens: GPT-2 endoftext and SOT+
        if i == GPT2_ENDOFTEXT || i >= SOT_TOKEN {
            // Allow EOT through
            if i != EOT_TOKEN {
                continue;
            }
        }
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

/// Greedy decode: generate tokens until EOT or max length.
///
/// Force-feeds the `prompt_tokens` (conditioning prefix) before switching to
/// autoregressive generation. Returns only the generated token IDs.
///
/// Applies Whisper-style token suppression: all special tokens (language tags,
/// timestamps, task tokens, etc.) are suppressed during generation so only
/// text tokens and EOT can be selected.
pub fn greedy_decode<B: MathBackend>(
    model: &WhisperModel<B>,
    encoder_output: &Tensor<B>,
    config: &DecodeConfig,
) -> Vec<usize> {
    let cross_kv_caches: Vec<CrossKvCache<B>> = model.compute_cross_kv_caches(encoder_output);
    let mut self_kv_cache: DecoderKvCache<B> = model.new_decoder_kv_cache();

    let mut tokens = Vec::with_capacity(config.max_tokens);
    let prompt = &config.prompt_tokens;

    let profile = std::env::var("SCRY_DECODE_PROFILE").is_ok();

    let greedy = config.temperature < 1e-8;

    // Phase 1: Force-feed conditioning prompt tokens.
    // For all but the last prompt token, skip the logit projection entirely
    // (saves ~1-2ms × (prompt_len-1) by avoiding the vocab GEMV).
    for (pos, &tok) in prompt.iter().enumerate() {
        if pos < prompt.len() - 1 {
            model.decode_step_no_logits(tok, pos, &mut self_kv_cache, &cross_kv_caches);
        } else {
            let logits = model.decode_step_profiled(tok, pos, &mut self_kv_cache, &cross_kv_caches, profile);
            let next_token = if greedy {
                // Zero-alloc: borrow logits directly with fused suppression
                argmax_suppressed(&logits.as_slice())
            } else {
                let mut logits_vec = logits.to_vec();
                suppress_special_tokens(&mut logits_vec);
                sample_with_temperature(&logits_vec, config.temperature)
            };
            if next_token == config.eot_token {
                return tokens;
            }
            tokens.push(next_token);
        }
    }

    // Phase 2: Autoregressive generation from the last generated token.
    let prompt_len = prompt.len();
    for step in 0..(config.max_tokens - 1) {
        let pos = prompt_len + step;
        let current_token = *tokens.last().expect("tokens non-empty after prompt phase");

        let t_step = std::time::Instant::now();
        let logits = model.decode_step_profiled(
            current_token,
            pos,
            &mut self_kv_cache,
            &cross_kv_caches,
            profile,
        );
        let step_ms = t_step.elapsed().as_secs_f64() * 1000.0;

        let next_token = if greedy {
            argmax_suppressed(&logits.as_slice())
        } else {
            let mut logits_vec = logits.to_vec();
            suppress_special_tokens(&mut logits_vec);
            sample_with_temperature(&logits_vec, config.temperature)
        };

        if profile {
            eprintln!("  [decode] step {step} tok={current_token} → {next_token} ({step_ms:.2}ms)");
        }

        if next_token == config.eot_token {
            break;
        }

        tokens.push(next_token);
    }

    tokens
}

/// Sample with temperature scaling.
fn sample_with_temperature(logits: &[f32], temperature: f32) -> usize {
    let scaled: Vec<f64> = logits
        .iter()
        .map(|&x| f64::from(x) / f64::from(temperature))
        .collect();

    // Softmax
    let max_val = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probs: Vec<f64> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    // Weighted sample
    let mut rng = fastrand::Rng::new();
    let r = rng.f64();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    probs.len() - 1
}
