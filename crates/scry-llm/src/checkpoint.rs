use std::path::Path;

use crate::backend::MathBackend;
use crate::error::ScryLlmError;
use crate::nn::gpt2::{Gpt2Config, Gpt2Model};
use crate::nn::Module;

/// Load model weights from a safetensors checkpoint file.
///
/// Returns `(model, step, rng_seed)`.
///
/// # Errors
///
/// Returns an error if the checkpoint cannot be read or parsed.
pub fn load_checkpoint<B: MathBackend>(
    path: &Path,
    config: &Gpt2Config,
) -> crate::error::Result<(Gpt2Model<B>, usize, u64)> {
    let data = std::fs::read(path)
        .map_err(|e| ScryLlmError::CheckpointError(format!("read error: {e}")))?;

    let (_, meta_obj) = safetensors::SafeTensors::read_metadata(&data)
        .map_err(|e| ScryLlmError::CheckpointError(format!("metadata error: {e}")))?;

    let loaded = safetensors::SafeTensors::deserialize(&data)
        .map_err(|e| ScryLlmError::CheckpointError(format!("deserialize error: {e}")))?;

    let meta = meta_obj
        .metadata()
        .as_ref()
        .ok_or_else(|| ScryLlmError::CheckpointError("no metadata in checkpoint".into()))?;

    let step: usize = meta
        .get("step")
        .ok_or_else(|| ScryLlmError::CheckpointError("missing 'step' in metadata".into()))?
        .parse()
        .map_err(|e| ScryLlmError::CheckpointError(format!("parse step: {e}")))?;

    let seed: u64 = meta
        .get("seed")
        .ok_or_else(|| ScryLlmError::CheckpointError("missing 'seed' in metadata".into()))?
        .parse()
        .map_err(|e| ScryLlmError::CheckpointError(format!("parse seed: {e}")))?;

    let mut rng = fastrand::Rng::with_seed(0);
    let mut model = Gpt2Model::<B>::new(config.clone(), &mut rng);

    let n_params = model.parameters().len();
    for i in 0..n_params {
        let tensor_name = format!("param.{i}");
        let t = loaded
            .tensor(&tensor_name)
            .map_err(|e| ScryLlmError::CheckpointError(format!("missing {tensor_name}: {e}")))?;
        let floats = bytes_to_f32(t.data());
        let shape = {
            let params = model.parameters();
            params[i].shape.clone()
        };
        // Need parameters_mut — re-add it for checkpoint loading
        let storage = B::from_vec(floats, &shape);
        // Access field directly since we own the model
        set_param_data(&mut model, i, storage);
    }

    Ok((model, step, seed))
}

fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Set parameter data by index. This traverses the model in the same order as `parameters()`.
fn set_param_data<B: MathBackend>(model: &mut Gpt2Model<B>, idx: usize, data: B::Storage) {
    // Embedding: token_embedding(0), position_embedding(1)
    // Per block (12 params): ln1.gamma, ln1.beta, qkv_weight, qkv_bias, proj_weight, proj_bias,
    //                         ln2.gamma, ln2.beta, fc1.weight, fc1.bias, fc2.weight, fc2.bias
    // Final: ln_f.gamma, ln_f.beta
    if idx == 0 {
        model.embedding.token_embedding.data = data;
        return;
    }
    if idx == 1 {
        model.embedding.position_embedding.data = data;
        return;
    }

    let block_params = 12;
    let block_start = 2;
    let block_end = block_start + model.blocks.len() * block_params;

    if idx >= block_start && idx < block_end {
        let rel = idx - block_start;
        let block_idx = rel / block_params;
        let param_in_block = rel % block_params;
        let block = &mut model.blocks[block_idx];
        match param_in_block {
            0 => block.ln1.gamma.data = data,
            1 => block.ln1.beta.data = data,
            2 => block.attn.qkv_weight.data = data,
            3 => block.attn.qkv_bias.data = data,
            4 => block.attn.proj_weight.data = data,
            5 => block.attn.proj_bias.data = data,
            6 => block.ln2.gamma.data = data,
            7 => block.ln2.beta.data = data,
            8 => block.mlp.fc1.weight.data = data,
            9 => block.mlp.fc1.bias.data = data,
            10 => block.mlp.fc2.weight.data = data,
            11 => block.mlp.fc2.bias.data = data,
            _ => unreachable!(),
        }
        return;
    }

    // Final layernorm
    if idx == block_end {
        model.ln_f.gamma.data = data;
    } else if idx == block_end + 1 {
        model.ln_f.beta.data = data;
    }
}
