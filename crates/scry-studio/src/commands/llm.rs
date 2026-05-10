// SPDX-License-Identifier: MIT OR Apache-2.0
//! GPT-2 generation, tokenization, and logit introspection.

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tauri::async_runtime::spawn_blocking;

use scry_llm::backend::DeviceBackend;
use scry_llm::generate::{generate, SamplingConfig};
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
use scry_llm::tokenizer::HfTokenizer;

use crate::state::{AppState, Backend, LoadedLlm};

#[derive(Deserialize)]
pub struct LlmLoadArgs {
    /// Path to the safetensors checkpoint.
    pub model_path: String,
    /// Path to a HuggingFace `tokenizer.json`.
    pub tokenizer_path: String,
    /// Architecture preset name. Currently only `"gpt2"` (small, 124M) is supported.
    pub preset: String,
}

#[derive(Serialize)]
pub struct LlmLoadResult {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub n_layers: usize,
    pub d_model: usize,
}

fn config_for(preset: &str) -> Result<Gpt2Config, String> {
    match preset {
        "gpt2" | "gpt2-small" => Ok(Gpt2Config::gpt2_small()),
        other => Err(format!(
            "unknown gpt2 preset: {other} (only 'gpt2' is supported today)"
        )),
    }
}

#[tauri::command]
pub async fn llm_load(
    state: tauri::State<'_, AppState>,
    args: LlmLoadArgs,
) -> Result<LlmLoadResult, String> {
    let slot: Arc<Mutex<Option<LoadedLlm>>> = state.llm.clone();
    let model_path = PathBuf::from(args.model_path);
    let tokenizer_path = PathBuf::from(args.tokenizer_path);
    let preset = args.preset;

    spawn_blocking(move || -> Result<LlmLoadResult, String> {
        let config = config_for(&preset)?;
        let model = Gpt2Model::<Backend>::load_safetensors(config.clone(), &model_path)
            .map_err(|e| format!("model load failed: {e}"))?;
        let tokenizer =
            HfTokenizer::from_file(&tokenizer_path).map_err(|e| format!("tokenizer load failed: {e}"))?;

        let result = LlmLoadResult {
            vocab_size: config.vocab_size,
            max_seq_len: config.max_seq_len,
            n_layers: config.n_layers,
            d_model: config.d_model,
        };

        *slot.lock() = Some(LoadedLlm {
            model,
            tokenizer,
            config,
        });
        Ok(result)
    })
    .await
    .map_err(|e| format!("join error: {e}"))?
}

#[derive(Deserialize)]
pub struct LlmGenerateArgs {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_max_tokens() -> usize {
    128
}
fn default_temperature() -> f32 {
    1.0
}
fn default_top_p() -> f32 {
    1.0
}
fn default_seed() -> u64 {
    42
}

#[derive(Serialize)]
pub struct LlmGenerateResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub elapsed_ms: u128,
}

#[tauri::command]
pub async fn llm_generate(
    state: tauri::State<'_, AppState>,
    args: LlmGenerateArgs,
) -> Result<LlmGenerateResult, String> {
    let slot = state.llm.clone();

    spawn_blocking(move || -> Result<LlmGenerateResult, String> {
        let guard = slot.lock();
        let llm = guard.as_ref().ok_or("LLM model not loaded")?;

        let prompt_tokens = llm.tokenizer.encode(&args.prompt);
        let cfg = SamplingConfig {
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            max_tokens: args.max_tokens,
        };
        let mut rng = fastrand::Rng::with_seed(args.seed);
        let start = std::time::Instant::now();
        let out_tokens = generate(&llm.model, &prompt_tokens, &cfg, &mut rng);
        let elapsed_ms = start.elapsed().as_millis();
        let text = llm.tokenizer.decode(&out_tokens);
        Ok(LlmGenerateResult {
            text,
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: out_tokens.len(),
            elapsed_ms,
        })
    })
    .await
    .map_err(|e| format!("join error: {e}"))?
}

#[derive(Deserialize)]
pub struct TokenizeArgs {
    pub text: String,
}

#[derive(Serialize)]
pub struct TokenizeResult {
    pub ids: Vec<usize>,
    pub pieces: Vec<String>,
}

#[tauri::command]
pub async fn llm_tokenize(
    state: tauri::State<'_, AppState>,
    args: TokenizeArgs,
) -> Result<TokenizeResult, String> {
    let slot = state.llm.clone();
    spawn_blocking(move || -> Result<TokenizeResult, String> {
        let guard = slot.lock();
        let llm = guard.as_ref().ok_or("LLM model not loaded")?;
        let ids = llm.tokenizer.encode(&args.text);
        let pieces = ids
            .iter()
            .map(|id| llm.tokenizer.decode(&[*id]))
            .collect();
        Ok(TokenizeResult { ids, pieces })
    })
    .await
    .map_err(|e| format!("join error: {e}"))?
}

#[derive(Deserialize)]
pub struct DecodeArgs {
    pub ids: Vec<usize>,
}

#[derive(Serialize)]
pub struct DecodeResult {
    pub text: String,
}

#[tauri::command]
pub async fn llm_decode(
    state: tauri::State<'_, AppState>,
    args: DecodeArgs,
) -> Result<DecodeResult, String> {
    let slot = state.llm.clone();
    spawn_blocking(move || -> Result<DecodeResult, String> {
        let guard = slot.lock();
        let llm = guard.as_ref().ok_or("LLM model not loaded")?;
        Ok(DecodeResult {
            text: llm.tokenizer.decode(&args.ids),
        })
    })
    .await
    .map_err(|e| format!("join error: {e}"))?
}

#[derive(Deserialize)]
pub struct LogitsArgs {
    pub prompt: String,
    /// Number of top tokens to return for the *next-token* distribution.
    #[serde(default = "default_top_k_logits")]
    pub top_k: usize,
}

fn default_top_k_logits() -> usize {
    20
}

#[derive(Serialize)]
pub struct TopToken {
    pub id: usize,
    pub piece: String,
    pub logit: f32,
    pub prob: f32,
}

#[derive(Serialize)]
pub struct LogitsResult {
    pub prompt_tokens: Vec<usize>,
    pub top: Vec<TopToken>,
}

#[tauri::command]
pub async fn llm_logits(
    state: tauri::State<'_, AppState>,
    args: LogitsArgs,
) -> Result<LogitsResult, String> {
    let slot = state.llm.clone();
    spawn_blocking(move || -> Result<LogitsResult, String> {
        let guard = slot.lock();
        let llm = guard.as_ref().ok_or("LLM model not loaded")?;

        let prompt_tokens = llm.tokenizer.encode(&args.prompt);
        if prompt_tokens.is_empty() {
            return Err("prompt is empty after tokenization".into());
        }

        let logits = llm.model.forward(&prompt_tokens);
        // [seq, vocab] — take the last row.
        let row = last_row_to_vec(&logits)
            .map_err(|e| format!("logits readback failed: {e}"))?;

        let mut top = top_k_softmax(&row, args.top_k);
        for entry in &mut top {
            entry.piece = llm.tokenizer.decode(&[entry.id]);
        }
        Ok(LogitsResult {
            prompt_tokens,
            top,
        })
    })
    .await
    .map_err(|e| format!("join error: {e}"))?
}

/// Read the last row of a `[seq, vocab]` logits tensor back to host.
fn last_row_to_vec<B: DeviceBackend>(
    logits: &scry_llm::tensor::Tensor<B>,
) -> Result<Vec<f32>, String> {
    let dims = logits.shape.dims();
    if dims.len() != 2 {
        return Err(format!("expected 2-D logits, got {dims:?}"));
    }
    let seq = dims[0];
    let vocab = dims[1];
    let flat = logits.to_vec();
    if flat.len() != seq * vocab {
        return Err(format!(
            "tensor data length {} != seq*vocab {}",
            flat.len(),
            seq * vocab
        ));
    }
    let start = (seq - 1) * vocab;
    Ok(flat[start..].to_vec())
}

fn top_k_softmax(logits: &[f32], k: usize) -> Vec<TopToken> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|x| x / sum).collect();

    let mut idx: Vec<usize> = (0..logits.len()).collect();
    let kk = k.min(logits.len());
    idx.select_nth_unstable_by(kk.saturating_sub(1).max(0), |&a, &b| {
        probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut top: Vec<TopToken> = idx
        .into_iter()
        .take(kk)
        .map(|id| TopToken {
            id,
            piece: String::new(),
            logit: logits[id],
            prob: probs[id],
        })
        .collect();
    top.sort_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap_or(std::cmp::Ordering::Equal));
    top
}
