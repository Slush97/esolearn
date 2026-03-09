//! Safetensors checkpoint loader for Whisper models.
//!
//! Loads HuggingFace `model.safetensors` files (e.g. from `openai/whisper-tiny`)
//! into our `WhisperModel` by mapping HF tensor names to struct fields.

use std::path::Path;

use half::f16;
use scry_llm::backend::MathBackend;
use scry_llm::nn::layernorm::LayerNormModule;
use scry_llm::nn::linear::Linear;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use crate::error::{ModelError, SttError};
use crate::model::attention::CrossAttention;
use crate::model::config::WhisperConfig;
use crate::model::conv1d::Conv1d;
use crate::model::decoder::{DecoderBlock, DecoderSelfAttention, WhisperDecoder};
use crate::model::encoder::{EncoderBlock, EncoderSelfAttention, WhisperEncoder};
use crate::model::WhisperModel;

/// Load a Whisper model from a HuggingFace safetensors checkpoint.
///
/// The `safetensors_path` should point to a `model.safetensors` file
/// downloaded from a HuggingFace Whisper model (e.g. `openai/whisper-tiny`).
///
/// # Errors
///
/// Returns an error if the file cannot be read, tensors are missing,
/// or shapes don't match the provided config.
pub fn load_whisper_checkpoint<B: MathBackend>(
    safetensors_path: &Path,
    config: &WhisperConfig,
) -> crate::error::Result<WhisperModel<B>> {
    // Memory-map the file for zero-copy access
    let file = std::fs::File::open(safetensors_path)
        .map_err(SttError::Io)?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(SttError::Io)?;

    let tensors = safetensors::SafeTensors::deserialize(&mmap)
        .map_err(|e| ModelError::Checkpoint(format!("deserialize: {e}")))?;

    // Helper: load a tensor by HF name, auto-detecting dtype (F16 or F32)
    let load = |name: &str| -> Result<Vec<f32>, SttError> {
        let t = tensors
            .tensor(name)
            .map_err(|_| ModelError::MissingWeight(name.to_string()))?;
        match t.dtype() {
            safetensors::Dtype::F16 => Ok(f16_bytes_to_f32(t.data())),
            safetensors::Dtype::F32 => Ok(f32_bytes_to_f32(t.data())),
            safetensors::Dtype::BF16 => Ok(bf16_bytes_to_f32(t.data())),
            other => Err(ModelError::Checkpoint(
                format!("unsupported dtype {other:?} for tensor '{name}'"),
            ).into()),
        }
    };

    // Helper: load and build a Tensor with given shape
    let load_tensor = |name: &str, shape: &[usize]| -> Result<Tensor<B>, SttError> {
        let data = load(name)?;
        let expected = shape.iter().product::<usize>();
        if data.len() != expected {
            return Err(ModelError::ShapeMismatch {
                name: name.to_string(),
                expected: shape.to_vec(),
                got: vec![data.len()],
            }
            .into());
        }
        Ok(Tensor::from_vec(data, Shape::new(shape)))
    };

    // Helper: load a linear weight (transpose from HF [out, in] → our [in, out])
    let load_linear_weight = |name: &str, in_f: usize, out_f: usize| -> Result<Tensor<B>, SttError> {
        let data = load(name)?;
        let transposed = transpose_2d(&data, out_f, in_f);
        Ok(Tensor::from_vec(transposed, Shape::new(&[in_f, out_f])))
    };

    let d = config.d_model;
    let d4 = d * 4;

    // ========================================================================
    // Encoder
    // ========================================================================

    // Conv layers (same layout, no transpose needed)
    let enc_conv1 = load_conv1d(&load, "model.encoder.conv1", config.n_mels, d, 3, 1, 1)?;
    let enc_conv2 = load_conv1d(&load, "model.encoder.conv2", d, d, 3, 2, 1)?;

    // Positional embedding (learned in HF, replaces our sinusoidal init)
    let enc_pos_emb = load_tensor(
        "model.encoder.embed_positions.weight",
        &[config.n_audio_ctx, d],
    )?;

    // Encoder blocks
    let mut enc_blocks = Vec::with_capacity(config.n_encoder_layers);
    for i in 0..config.n_encoder_layers {
        let prefix = format!("model.encoder.layers.{i}");
        let block = load_encoder_block(&load, &load_linear_weight, &load_tensor, &prefix, d, d4, config.n_encoder_heads)?;
        enc_blocks.push(block);
    }

    // Final encoder layer norm
    let enc_ln_post = load_layer_norm(&load_tensor, "model.encoder.layer_norm", d)?;

    let pos_data = enc_pos_emb.to_vec();
    let encoder = WhisperEncoder {
        conv1: enc_conv1,
        conv2: enc_conv2,
        positional_embedding: enc_pos_emb,
        pos_data,
        blocks: enc_blocks,
        ln_post: enc_ln_post,
        d_model: d,
    };

    // ========================================================================
    // Decoder
    // ========================================================================

    // Infer actual vocab size from the checkpoint's token embedding shape
    // (HF models may differ slightly from the config, e.g. 51864 vs 51865)
    let tok_emb_tensor = tensors
        .tensor("model.decoder.embed_tokens.weight")
        .map_err(|_| ModelError::MissingWeight("model.decoder.embed_tokens.weight".into()))?;
    let actual_vocab = tok_emb_tensor.shape()[0];

    let dec_tok_emb = load_tensor("model.decoder.embed_tokens.weight", &[actual_vocab, d])?;
    let dec_pos_emb = load_tensor(
        "model.decoder.embed_positions.weight",
        &[config.n_text_ctx, d],
    )?;

    let mut dec_blocks = Vec::with_capacity(config.n_decoder_layers);
    for i in 0..config.n_decoder_layers {
        let prefix = format!("model.decoder.layers.{i}");
        let block = load_decoder_block(&load, &load_linear_weight, &load_tensor, &prefix, d, d4, config.n_decoder_heads)?;
        dec_blocks.push(block);
    }

    let dec_ln = load_layer_norm(&load_tensor, "model.decoder.layer_norm", d)?;

    let logit_proj_weight = WhisperDecoder::transpose_2d(&dec_tok_emb, actual_vocab, d);
    let decoder = WhisperDecoder {
        token_embedding: dec_tok_emb,
        logit_proj_weight,
        positional_embedding: dec_pos_emb,
        blocks: dec_blocks,
        ln: dec_ln,
        d_model: d,
        vocab_size: actual_vocab,
        n_text_ctx: config.n_text_ctx,
    };

    let mut loaded_config = config.clone();
    loaded_config.n_vocab = actual_vocab;

    Ok(WhisperModel {
        encoder,
        decoder,
        config: loaded_config,
    })
}

/// Load a Whisper model and quantize all Linear layer weights to INT8.
///
/// Same as [`load_whisper_checkpoint`] but applies symmetric per-tensor INT8
/// quantization (W8A32) to all MLP linear layers, reducing weight memory ~4x.
#[cfg(feature = "quantize")]
pub fn load_whisper_checkpoint_quantized<B: MathBackend>(
    safetensors_path: &Path,
    config: &WhisperConfig,
) -> crate::error::Result<WhisperModel<B>> {
    let mut model = load_whisper_checkpoint(safetensors_path, config)?;
    quantize_model_weights(&mut model);
    Ok(model)
}

/// Quantize all MLP Linear layer weights in a Whisper model to INT8.
#[cfg(feature = "quantize")]
pub fn quantize_model_weights<B: MathBackend>(model: &mut WhisperModel<B>) {
    for block in &mut model.encoder.blocks {
        block.mlp_fc1.quantize_weights();
        block.mlp_fc2.quantize_weights();
    }
    for block in &mut model.decoder.blocks {
        block.mlp_fc1.quantize_weights();
        block.mlp_fc2.quantize_weights();
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Convert f16 bytes (little-endian) to f32 values.
fn f16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f16::from_bits(bits).to_f32()
        })
        .collect()
}

/// Convert f32 bytes (little-endian) to f32 values.
fn f32_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Convert bf16 bytes (little-endian) to f32 values.
fn bf16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            half::bf16::from_bits(bits).to_f32()
        })
        .collect()
}

/// Transpose a row-major `[rows, cols]` matrix to `[cols, rows]`.
fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Load a Conv1d layer (weights share the same [out, in, kernel] layout).
fn load_conv1d<B: MathBackend>(
    load: &dyn Fn(&str) -> Result<Vec<f32>, SttError>,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<Conv1d<B>, SttError> {
    let w_name = format!("{prefix}.weight");
    let b_name = format!("{prefix}.bias");
    let w_data = load(&w_name)?;
    let b_data = load(&b_name)?;
    // Pre-allocate workspace for the expected input length (3000 frames for 30s audio).
    let expected_len = 3000;
    let expected_out = (expected_len + 2 * padding - kernel) / stride + 1;
    let col_rows = kernel * in_ch;
    let workspace = std::cell::RefCell::new(vec![0.0f32; col_rows * expected_out]);

    Ok(Conv1d {
        weight: Tensor::from_vec(w_data, Shape::new(&[out_ch, in_ch, kernel])),
        bias: Tensor::from_vec(b_data, Shape::new(&[out_ch])),
        in_channels: in_ch,
        out_channels: out_ch,
        kernel_size: kernel,
        stride,
        padding,
        workspace,
    })
}

/// Load a LayerNormModule from HF weight/bias names.
fn load_layer_norm<B: MathBackend>(
    load_tensor: &dyn Fn(&str, &[usize]) -> Result<Tensor<B>, SttError>,
    prefix: &str,
    d: usize,
) -> Result<LayerNormModule<B>, SttError> {
    let gamma = load_tensor(&format!("{prefix}.weight"), &[d])?;
    let beta = load_tensor(&format!("{prefix}.bias"), &[d])?;
    Ok(LayerNormModule {
        gamma,
        beta,
        eps: 1e-5,
    })
}

/// Load a Linear layer (transpose weight, copy bias).
fn load_linear<B: MathBackend>(
    load_linear_weight: &dyn Fn(&str, usize, usize) -> Result<Tensor<B>, SttError>,
    load_tensor: &dyn Fn(&str, &[usize]) -> Result<Tensor<B>, SttError>,
    prefix: &str,
    in_f: usize,
    out_f: usize,
) -> Result<Linear<B>, SttError> {
    let weight = load_linear_weight(&format!("{prefix}.weight"), in_f, out_f)?;
    let bias = load_tensor(&format!("{prefix}.bias"), &[out_f])?;
    Ok(Linear {
        weight,
        bias,
        in_features: in_f,
        out_features: out_f,
        #[cfg(feature = "quantize")]
        quant: None,
    })
}

/// Load fused QKV weights for self-attention.
///
/// HF stores separate `q_proj`, `k_proj`, `v_proj` each `[d, d]`.
/// We fuse into `qkv_weight [d, 3d]` and `qkv_bias [3d]`.
fn load_fused_qkv<B: MathBackend>(
    load: &dyn Fn(&str) -> Result<Vec<f32>, SttError>,
    prefix: &str,
    d: usize,
) -> Result<(Tensor<B>, Tensor<B>), SttError> {
    // Load and transpose each projection weight: HF [d, d] → our [d, d]
    let q_w = transpose_2d(&load(&format!("{prefix}.q_proj.weight"))?, d, d);
    let k_w = transpose_2d(&load(&format!("{prefix}.k_proj.weight"))?, d, d);
    let v_w = transpose_2d(&load(&format!("{prefix}.v_proj.weight"))?, d, d);

    // Horizontally concatenate: for each row i, [q_row_i | k_row_i | v_row_i]
    let mut qkv_weight = vec![0.0f32; d * 3 * d];
    for row in 0..d {
        for col in 0..d {
            qkv_weight[row * 3 * d + col] = q_w[row * d + col];
            qkv_weight[row * 3 * d + d + col] = k_w[row * d + col];
            qkv_weight[row * 3 * d + 2 * d + col] = v_w[row * d + col];
        }
    }

    // Concatenate biases: [q_bias | k_bias | v_bias]
    // Note: Whisper's K projection has no bias — use zeros as fallback
    let q_b = load(&format!("{prefix}.q_proj.bias"))?;
    let k_b = load(&format!("{prefix}.k_proj.bias")).unwrap_or_else(|_| vec![0.0f32; d]);
    let v_b = load(&format!("{prefix}.v_proj.bias"))?;
    let mut qkv_bias = Vec::with_capacity(3 * d);
    qkv_bias.extend_from_slice(&q_b);
    qkv_bias.extend_from_slice(&k_b);
    qkv_bias.extend_from_slice(&v_b);

    Ok((
        Tensor::from_vec(qkv_weight, Shape::new(&[d, 3 * d])),
        Tensor::from_vec(qkv_bias, Shape::new(&[3 * d])),
    ))
}

/// Load an encoder block.
fn load_encoder_block<B: MathBackend>(
    load: &dyn Fn(&str) -> Result<Vec<f32>, SttError>,
    load_linear_weight: &dyn Fn(&str, usize, usize) -> Result<Tensor<B>, SttError>,
    load_tensor: &dyn Fn(&str, &[usize]) -> Result<Tensor<B>, SttError>,
    prefix: &str,
    d: usize,
    d4: usize,
    n_heads: usize,
) -> Result<EncoderBlock<B>, SttError> {
    let attn_ln = load_layer_norm(load_tensor, &format!("{prefix}.self_attn_layer_norm"), d)?;

    let (qkv_weight, qkv_bias) = load_fused_qkv(load, &format!("{prefix}.self_attn"), d)?;
    let out_weight = load_linear_weight(&format!("{prefix}.self_attn.out_proj.weight"), d, d)?;
    let out_bias = load_tensor(&format!("{prefix}.self_attn.out_proj.bias"), &[d])?;

    let attn = EncoderSelfAttention {
        qkv_weight,
        qkv_bias,
        out_weight,
        out_bias,
        n_heads,
        d_model: d,
        d_head: d / n_heads,
    };

    let mlp_ln = load_layer_norm(load_tensor, &format!("{prefix}.final_layer_norm"), d)?;
    let mlp_fc1 = load_linear(load_linear_weight, load_tensor, &format!("{prefix}.fc1"), d, d4)?;
    let mlp_fc2 = load_linear(load_linear_weight, load_tensor, &format!("{prefix}.fc2"), d4, d)?;

    Ok(EncoderBlock {
        attn_ln,
        attn,
        mlp_ln,
        mlp_fc1,
        mlp_fc2,
    })
}

/// Load cross-attention layer (separate Q, K, V — K has no bias).
fn load_cross_attention<B: MathBackend>(
    load_linear_weight: &dyn Fn(&str, usize, usize) -> Result<Tensor<B>, SttError>,
    load_tensor: &dyn Fn(&str, &[usize]) -> Result<Tensor<B>, SttError>,
    prefix: &str,
    d: usize,
    n_heads: usize,
) -> Result<CrossAttention<B>, SttError> {
    let q_weight = load_linear_weight(&format!("{prefix}.q_proj.weight"), d, d)?;
    let q_bias = load_tensor(&format!("{prefix}.q_proj.bias"), &[d])?;
    let k_weight = load_linear_weight(&format!("{prefix}.k_proj.weight"), d, d)?;
    // K has no bias in Whisper cross-attention
    let v_weight = load_linear_weight(&format!("{prefix}.v_proj.weight"), d, d)?;
    let v_bias = load_tensor(&format!("{prefix}.v_proj.bias"), &[d])?;
    let out_weight = load_linear_weight(&format!("{prefix}.out_proj.weight"), d, d)?;
    let out_bias = load_tensor(&format!("{prefix}.out_proj.bias"), &[d])?;

    Ok(CrossAttention {
        q_weight,
        q_bias,
        k_weight,
        v_weight,
        v_bias,
        out_weight,
        out_bias,
        n_heads,
        d_model: d,
        d_head: d / n_heads,
    })
}

/// Load a decoder block.
fn load_decoder_block<B: MathBackend>(
    load: &dyn Fn(&str) -> Result<Vec<f32>, SttError>,
    load_linear_weight: &dyn Fn(&str, usize, usize) -> Result<Tensor<B>, SttError>,
    load_tensor: &dyn Fn(&str, &[usize]) -> Result<Tensor<B>, SttError>,
    prefix: &str,
    d: usize,
    d4: usize,
    n_heads: usize,
) -> Result<DecoderBlock<B>, SttError> {
    // Self-attention
    let attn_ln = load_layer_norm(load_tensor, &format!("{prefix}.self_attn_layer_norm"), d)?;
    let (qkv_weight, qkv_bias) = load_fused_qkv(load, &format!("{prefix}.self_attn"), d)?;
    let out_weight = load_linear_weight(&format!("{prefix}.self_attn.out_proj.weight"), d, d)?;
    let out_bias = load_tensor(&format!("{prefix}.self_attn.out_proj.bias"), &[d])?;

    let self_attn = DecoderSelfAttention {
        qkv_weight,
        qkv_bias,
        out_weight,
        out_bias,
        n_heads,
        d_model: d,
        d_head: d / n_heads,
    };

    // Cross-attention
    let cross_attn_ln = load_layer_norm(load_tensor, &format!("{prefix}.encoder_attn_layer_norm"), d)?;
    let cross_attn = load_cross_attention(
        load_linear_weight, load_tensor,
        &format!("{prefix}.encoder_attn"),
        d, n_heads,
    )?;

    // MLP
    let mlp_ln = load_layer_norm(load_tensor, &format!("{prefix}.final_layer_norm"), d)?;
    let mlp_fc1 = load_linear(load_linear_weight, load_tensor, &format!("{prefix}.fc1"), d, d4)?;
    let mlp_fc2 = load_linear(load_linear_weight, load_tensor, &format!("{prefix}.fc2"), d4, d)?;

    Ok(DecoderBlock {
        attn_ln,
        self_attn,
        cross_attn_ln,
        cross_attn,
        mlp_ln,
        mlp_fc1,
        mlp_fc2,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_2d_basic() {
        // [2, 3] matrix:
        // 1 2 3
        // 4 5 6
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = transpose_2d(&data, 2, 3);
        // Expected [3, 2]:
        // 1 4
        // 2 5
        // 3 6
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn f16_roundtrip() {
        // Create f16 bytes for known values
        let values = [0.0f32, 1.0, -1.0, 0.5, 42.0];
        let mut bytes = Vec::new();
        for &v in &values {
            let h = f16::from_f32(v);
            bytes.extend_from_slice(&h.to_bits().to_le_bytes());
        }
        let result = f16_bytes_to_f32(&bytes);
        for (got, &expected) in result.iter().zip(&values) {
            assert!(
                (got - expected).abs() < 0.01,
                "expected {expected}, got {got}"
            );
        }
    }
}
