use std::collections::HashMap;

use crate::backend::{DeviceBackend, MathBackend};
use crate::tensor::shape::Shape;
use crate::tensor::TensorId;

#[allow(unused_imports)]
use super::{GradTape, Operation, SavedData};

/// Gradient storage: maps [`TensorId`] to its accumulated gradient.
pub type Gradients<B> = HashMap<TensorId, <B as DeviceBackend>::Storage>;

/// Run backward pass from a scalar loss.
/// `loss_id` must be the output of the last recorded op.
/// Returns gradients for all tensors that require them.
pub fn backward<B: MathBackend>(tape: &GradTape<B>, loss_id: TensorId) -> Gradients<B> {
    let mut grads: Gradients<B> = HashMap::new();

    // Seed: gradient of loss w.r.t. itself is 1.0
    let ones = B::from_vec(vec![1.0], &Shape::new(&[1]));
    grads.insert(loss_id, ones);

    // Reverse iteration through tape
    for node in tape.nodes.iter().rev() {
        let d_out = match grads.get(&node.output_id) {
            Some(g) => B::clone_storage(g),
            None => continue, // no gradient flows to this node
        };
        backward_node(node, &d_out, &mut grads);
    }

    grads
}

/// Process a single tape node's backward pass, accumulating gradients.
///
/// This is the per-node dispatch used by both the standard [`backward`] and
/// the checkpointed backward in [`Gpt2Model::backward_checkpointed`].
#[allow(clippy::too_many_lines)]
pub fn backward_node<B: MathBackend>(
    node: &super::TapeNode<B>,
    d_out: &B::Storage,
    grads: &mut Gradients<B>,
) {
    match (&node.op, &node.saved) {
        (
            Operation::Matmul,
            SavedData::Matmul {
                a,
                b,
                m,
                k,
                n,
                trans_a,
                trans_b,
            },
        ) => {
            let (d_a, d_b) = B::matmul_backward(d_out, a, b, *m, *k, *n, *trans_a, *trans_b);
            accumulate_grad::<B>(grads, node.input_ids[0], d_a);
            accumulate_grad::<B>(grads, node.input_ids[1], d_b);
        }
        (
            Operation::Add,
            SavedData::Add {
                a_shape,
                b_shape,
                out_shape,
            },
        ) => {
            let (d_a, d_b) = B::add_backward(d_out, a_shape, b_shape, out_shape);
            accumulate_grad::<B>(grads, node.input_ids[0], d_a);
            accumulate_grad::<B>(grads, node.input_ids[1], d_b);
        }
        (Operation::Softmax, SavedData::Softmax { output, shape }) => {
            let d_input = B::softmax_backward(d_out, output, shape);
            accumulate_grad::<B>(grads, node.input_ids[0], d_input);
        }
        (
            Operation::LayerNorm,
            SavedData::LayerNorm {
                input,
                gamma,
                mean,
                rstd,
                shape,
                gamma_id,
                beta_id,
            },
        ) => {
            let (d_input, d_gamma, d_beta) =
                B::layernorm_backward(d_out, input, gamma, mean, rstd, shape);
            accumulate_grad::<B>(grads, node.input_ids[0], d_input);
            accumulate_grad::<B>(grads, *gamma_id, d_gamma);
            accumulate_grad::<B>(grads, *beta_id, d_beta);
        }
        (Operation::Gelu, SavedData::Gelu { input }) => {
            let d_input = B::gelu_backward(d_out, input);
            accumulate_grad::<B>(grads, node.input_ids[0], d_input);
        }
        (
            Operation::CrossEntropy,
            SavedData::CrossEntropy {
                logits,
                targets,
                batch,
                vocab,
            },
        ) => {
            let d_logits = B::cross_entropy_backward(logits, targets, *batch, *vocab);
            let d_out_scalar = B::to_vec(d_out)[0];
            let d_logits_scaled = B::scale(&d_logits, d_out_scalar);
            accumulate_grad::<B>(grads, node.input_ids[0], d_logits_scaled);
        }
        (
            Operation::Embedding,
            SavedData::Embedding {
                indices,
                vocab,
                dim,
                weight_id,
            },
        ) => {
            let d_weight = B::embedding_backward(d_out, indices, *vocab, *dim);
            accumulate_grad::<B>(grads, *weight_id, d_weight);
        }
        (Operation::Sum, SavedData::Sum { input_shape }) => {
            let d_out_scalar = B::to_vec(d_out)[0];
            let d_input = B::from_vec(vec![d_out_scalar; input_shape.numel()], input_shape);
            accumulate_grad::<B>(grads, node.input_ids[0], d_input);
        }
        (
            Operation::Attention,
            SavedData::Attention {
                input,
                qkv_weight,
                proj_weight,
                attn_weights,
                q_per_head,
                k_per_head,
                v_per_head,
                attn_dropout_masks,
                head_concat,
                n_heads,
                d_model,
                d_head,
                seq_len,
                qkv_weight_id,
                qkv_bias_id,
                proj_weight_id,
                proj_bias_id,
            },
        ) => {
            let n_heads = *n_heads;
            let d_model = *d_model;
            let d_head = *d_head;
            let seq_len = *seq_len;

            attention_backward::<B>(
                d_out,
                input,
                qkv_weight,
                proj_weight,
                head_concat,
                attn_weights,
                q_per_head,
                k_per_head,
                v_per_head,
                attn_dropout_masks,
                n_heads,
                d_model,
                d_head,
                seq_len,
                grads,
                node.input_ids[0],
                *qkv_weight_id,
                *qkv_bias_id,
                *proj_weight_id,
                *proj_bias_id,
            );
        }
        (Operation::Dropout, SavedData::Dropout { mask }) => {
            let d_input = B::mul_elementwise(d_out, mask);
            accumulate_grad::<B>(grads, node.input_ids[0], d_input);
        }
        (
            Operation::AttentionBatched,
            SavedData::AttentionBatched {
                input,
                qkv_weight,
                proj_weight,
                attn_weights,
                q_heads,
                k_heads,
                v_heads,
                attn_dropout_mask,
                head_concat,
                n_heads,
                d_model,
                d_head,
                batch_size,
                seq_len,
                qkv_weight_id,
                qkv_bias_id,
                proj_weight_id,
                proj_bias_id,
            },
        ) => {
            attention_batched_backward::<B>(
                d_out,
                input,
                qkv_weight,
                proj_weight,
                attn_weights,
                q_heads,
                k_heads,
                v_heads,
                attn_dropout_mask,
                head_concat,
                *n_heads,
                *d_model,
                *d_head,
                *batch_size,
                *seq_len,
                grads,
                node.input_ids[0],
                *qkv_weight_id,
                *qkv_bias_id,
                *proj_weight_id,
                *proj_bias_id,
            );
        }
        (Operation::Checkpoint, _) => {
            // Checkpoint nodes are handled by backward_checkpointed in Gpt2Model.
            // If encountered in standard backward, treat as a pass-through.
            accumulate_grad::<B>(grads, node.input_ids[0], B::clone_storage(d_out));
        }
        _ => unreachable!("mismatched Operation/SavedData variants: backward dispatch bug"),
    }
}

fn accumulate_grad<B: MathBackend>(grads: &mut Gradients<B>, id: TensorId, grad: B::Storage) {
    if let Some(existing) = grads.get_mut(&id) {
        B::add_inplace(existing, &grad);
    } else {
        grads.insert(id, grad);
    }
}

/// Backward pass for the fused multi-head causal self-attention op.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names
)]
fn attention_backward<B: MathBackend>(
    d_out: &B::Storage,
    input: &B::Storage,
    qkv_weight: &B::Storage,
    proj_weight: &B::Storage,
    head_concat: &B::Storage,
    attn_weights: &[B::Storage],
    q_per_head: &[B::Storage],
    k_per_head: &[B::Storage],
    v_per_head: &[B::Storage],
    attn_dropout_masks: &[B::Storage],
    n_heads: usize,
    d_model: usize,
    d_head: usize,
    seq_len: usize,
    grads: &mut Gradients<B>,
    input_id: TensorId,
    qkv_weight_id: TensorId,
    qkv_bias_id: TensorId,
    proj_weight_id: TensorId,
    proj_bias_id: TensorId,
) {
    // d_out is [seq, d_model]
    // output = head_concat @ W_proj + b_proj

    // d_proj_bias = sum over seq of d_out => [d_model]
    // Use GPU reduce_rows: treat d_out as [seq_len, d_model] and sum over rows.
    let (_, d_proj_bias) = B::add_backward(
        d_out,
        &Shape::new(&[seq_len, d_model]),
        &Shape::new(&[1, d_model]),
        &Shape::new(&[seq_len, d_model]),
    );
    accumulate_grad::<B>(grads, proj_bias_id, d_proj_bias);

    // d_head_concat = d_out @ W_proj^T => [seq, d_model]
    let d_head_concat = B::matmul(d_out, proj_weight, seq_len, d_model, d_model, false, true);

    // d_W_proj = head_concat^T @ d_out => [d_model, d_model]
    let d_proj_weight = B::matmul(head_concat, d_out, d_model, seq_len, d_model, true, false);
    accumulate_grad::<B>(grads, proj_weight_id, d_proj_weight);

    let scale = 1.0 / (d_head as f64).sqrt();

    // Accumulate d_QKV using gather/scatter (avoids host round-trips on GPU)
    let mut d_qkv_storage = B::zeros(&Shape::new(&[seq_len, 3 * d_model]));

    for h in 0..n_heads {
        // Extract d_out_h from d_head_concat for this head
        let d_out_h = B::gather_columns(&d_head_concat, seq_len, d_model, h * d_head, d_head);

        let attn = &attn_weights[h];
        let dropout_mask = &attn_dropout_masks[h];
        let q_h = &q_per_head[h];
        let k_h = &k_per_head[h];
        let v_h = &v_per_head[h];

        // attn_dropped = attn * dropout_mask
        let attn_dropped = B::mul_elementwise(attn, dropout_mask);

        // d_attn_dropped = d_out_h @ V_h^T => [seq, seq]
        let d_attn_dropped = B::matmul(&d_out_h, v_h, seq_len, d_head, seq_len, false, true);

        // d_V_h = attn_dropped^T @ d_out_h => [seq, d_head]
        let d_v_h = B::matmul(&attn_dropped, &d_out_h, seq_len, seq_len, d_head, true, false);

        // Dropout backward: d_attn = d_attn_dropped * dropout_mask
        let d_attn = B::mul_elementwise(&d_attn_dropped, dropout_mask);

        // d_scores = softmax_backward(d_attn, attn) => [seq, seq]
        let d_scores = B::softmax_backward(&d_attn, attn, &Shape::new(&[seq_len, seq_len]));

        // Apply scale and causal mask to d_scores (zeros upper triangle, scales lower)
        // In backward, the causal mask zeros gradients for masked positions,
        // and scale is applied to the unmasked ones — same operation as forward.
        let mut d_scores_scaled = d_scores;
        B::apply_causal_mask_and_scale(&mut d_scores_scaled, seq_len, scale as f32, 0.0);

        // d_Q_h = d_scores @ K_h => [seq, d_head]
        let d_q_h = B::matmul(&d_scores_scaled, k_h, seq_len, seq_len, d_head, false, false);

        // d_K_h = d_scores^T @ Q_h => [seq, d_head]
        let d_k_h = B::matmul(&d_scores_scaled, q_h, seq_len, seq_len, d_head, true, false);

        // Scatter d_Q, d_K, d_V into d_QKV
        B::scatter_columns(&mut d_qkv_storage, &d_q_h, seq_len, 3 * d_model, h * d_head, d_head);
        B::scatter_columns(&mut d_qkv_storage, &d_k_h, seq_len, 3 * d_model, d_model + h * d_head, d_head);
        B::scatter_columns(&mut d_qkv_storage, &d_v_h, seq_len, 3 * d_model, 2 * d_model + h * d_head, d_head);
    }

    // d_input = d_QKV @ W_qkv^T => [seq, d_model]
    let d_input = B::matmul(
        &d_qkv_storage,
        qkv_weight,
        seq_len,
        3 * d_model,
        d_model,
        false,
        true,
    );
    accumulate_grad::<B>(grads, input_id, d_input);

    // d_W_qkv = input^T @ d_QKV => [d_model, 3*d_model]
    let d_qkv_weight = B::matmul(
        input,
        &d_qkv_storage,
        d_model,
        seq_len,
        3 * d_model,
        true,
        false,
    );
    accumulate_grad::<B>(grads, qkv_weight_id, d_qkv_weight);

    // d_qkv_bias = sum over seq of d_QKV => [3*d_model]
    // Use GPU reduce_rows: treat d_qkv as [seq_len, 3*d_model] and sum over rows.
    let (_, d_qkv_bias) = B::add_backward(
        &d_qkv_storage,
        &Shape::new(&[seq_len, 3 * d_model]),
        &Shape::new(&[1, 3 * d_model]),
        &Shape::new(&[seq_len, 3 * d_model]),
    );
    accumulate_grad::<B>(grads, qkv_bias_id, d_qkv_bias);
}

/// Backward pass for batched attention using strided batched GEMMs.
/// No per-batch or per-head loops — all ops are batched.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn attention_batched_backward<B: MathBackend>(
    d_out: &B::Storage,
    input: &B::Storage,
    qkv_weight: &B::Storage,
    proj_weight: &B::Storage,
    attn_weights: &B::Storage,
    q_heads: &B::Storage,
    k_heads: &B::Storage,
    v_heads: &B::Storage,
    attn_dropout_mask: &B::Storage,
    head_concat: &B::Storage,
    n_heads: usize,
    d_model: usize,
    d_head: usize,
    batch_size: usize,
    seq_len: usize,
    grads: &mut Gradients<B>,
    input_id: TensorId,
    qkv_weight_id: TensorId,
    qkv_bias_id: TensorId,
    proj_weight_id: TensorId,
    proj_bias_id: TensorId,
) {
    let total_tokens = batch_size * seq_len;
    let bh = batch_size * n_heads;

    // d_out: [B*S, D]

    // d_proj_bias = reduce_rows(d_out) -> [D]
    let (_, d_proj_bias) = B::add_backward(
        d_out,
        &Shape::new(&[total_tokens, d_model]),
        &Shape::new(&[1, d_model]),
        &Shape::new(&[total_tokens, d_model]),
    );
    accumulate_grad::<B>(grads, proj_bias_id, d_proj_bias);

    // d_head_concat = d_out @ W_proj^T -> [B*S, D]
    let d_head_concat = B::matmul(d_out, proj_weight, total_tokens, d_model, d_model, false, true);

    // d_W_proj = head_concat^T @ d_out -> [D, D]
    let d_proj_weight = B::matmul(head_concat, d_out, d_model, total_tokens, d_model, true, false);
    accumulate_grad::<B>(grads, proj_weight_id, d_proj_weight);

    // Reshape d_head_concat [B*S, D] -> [B*H, S, d_head]
    let d_out_heads = B::reshape_for_heads(&d_head_concat, batch_size, seq_len, n_heads, d_head);

    let scale = 1.0 / (d_head as f64).sqrt();

    // attn_dropped = attn_weights * dropout_mask  [B*H, S, S] elementwise
    let attn_dropped = B::mul_elementwise(attn_weights, attn_dropout_mask);

    // d_attn_dropped = d_out_heads @ V^T: [B*H, S, d_head] @ [B*H, d_head, S] -> [B*H, S, S]
    let d_attn_dropped = B::matmul_strided_batched(
        &d_out_heads, v_heads, bh, seq_len, d_head, seq_len, false, true,
    );

    // d_V = attn_dropped^T @ d_out_heads: [B*H, S, S]^T @ [B*H, S, d_head] -> [B*H, S, d_head]
    let d_v_heads = B::matmul_strided_batched(
        &attn_dropped, &d_out_heads, bh, seq_len, seq_len, d_head, true, false,
    );

    // d_attn = d_attn_dropped * dropout_mask
    let d_attn = B::mul_elementwise(&d_attn_dropped, attn_dropout_mask);

    // d_scores = softmax_backward(d_attn, attn_weights)
    let d_scores = B::softmax_backward(
        &d_attn, attn_weights, &Shape::new(&[bh * seq_len, seq_len]),
    );

    // Apply causal mask to d_scores (zeros upper triangle, scales lower)
    let mut d_scores_masked = d_scores;
    B::apply_batched_causal_mask_and_scale(&mut d_scores_masked, bh, seq_len, scale as f32, 0.0);

    // d_Q = d_scores @ K: [B*H, S, S] @ [B*H, S, d_head] -> [B*H, S, d_head]
    let d_q_heads = B::matmul_strided_batched(
        &d_scores_masked, k_heads, bh, seq_len, seq_len, d_head, false, false,
    );

    // d_K = d_scores^T @ Q: [B*H, S, S]^T @ [B*H, S, d_head] -> [B*H, S, d_head]
    let d_k_heads = B::matmul_strided_batched(
        &d_scores_masked, q_heads, bh, seq_len, seq_len, d_head, true, false,
    );

    // Reshape d_Q, d_K, d_V from [B*H, S, d_head] -> [B*S, D]
    let d_q_flat = B::reshape_from_heads(&d_q_heads, batch_size, seq_len, n_heads, d_head);
    let d_k_flat = B::reshape_from_heads(&d_k_heads, batch_size, seq_len, n_heads, d_head);
    let d_v_flat = B::reshape_from_heads(&d_v_heads, batch_size, seq_len, n_heads, d_head);

    // Assemble d_QKV [B*S, 3D] from d_Q, d_K, d_V
    let mut d_qkv = B::zeros(&Shape::new(&[total_tokens, 3 * d_model]));
    B::scatter_columns(&mut d_qkv, &d_q_flat, total_tokens, 3 * d_model, 0, d_model);
    B::scatter_columns(&mut d_qkv, &d_k_flat, total_tokens, 3 * d_model, d_model, d_model);
    B::scatter_columns(&mut d_qkv, &d_v_flat, total_tokens, 3 * d_model, 2 * d_model, d_model);

    // d_input = d_QKV @ W_qkv^T -> [B*S, D]
    let d_input = B::matmul(
        &d_qkv,
        qkv_weight,
        total_tokens,
        3 * d_model,
        d_model,
        false,
        true,
    );
    accumulate_grad::<B>(grads, input_id, d_input);

    // d_W_qkv = input^T @ d_QKV -> [D, 3D]
    let d_qkv_weight = B::matmul(
        input,
        &d_qkv,
        d_model,
        total_tokens,
        3 * d_model,
        true,
        false,
    );
    accumulate_grad::<B>(grads, qkv_weight_id, d_qkv_weight);

    // d_qkv_bias = reduce_rows(d_QKV) -> [3D]
    let (_, d_qkv_bias) = B::add_backward(
        &d_qkv,
        &Shape::new(&[total_tokens, 3 * d_model]),
        &Shape::new(&[1, 3 * d_model]),
        &Shape::new(&[total_tokens, 3 * d_model]),
    );
    accumulate_grad::<B>(grads, qkv_bias_id, d_qkv_bias);
}
