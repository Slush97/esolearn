// SPDX-License-Identifier: MIT OR Apache-2.0
//! SAM mask decoder — two-way transformer decoder with upscaling heads.
//!
//! Takes image embeddings + prompt embeddings and produces multi-mask outputs
//! with `IoU` confidence scores.

use scry_llm::backend::MathBackend;
use scry_llm::nn::Module;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use crate::nn::{ConvTranspose2d, LayerNorm2d};

use super::{SamConfig, SamOutput};

// ── Cross-attention ─────────────────────────────────────────────────────────

/// Multi-head attention with separate Q, K, V projections.
///
/// Used for both self-attention and cross-attention in the two-way transformer.
struct Attention<B: MathBackend> {
    q_weight: Tensor<B>,
    q_bias: Tensor<B>,
    k_weight: Tensor<B>,
    k_bias: Tensor<B>,
    v_weight: Tensor<B>,
    v_bias: Tensor<B>,
    out_weight: Tensor<B>,
    out_bias: Tensor<B>,
    n_heads: usize,
    d_head: usize,
    d_model: usize,
}

impl<B: MathBackend> Attention<B> {
    fn new(d_model: usize, n_heads: usize) -> Self {
        let d_head = d_model / n_heads;
        let make_weight =
            |rows, cols| Tensor::from_vec(vec![0.0; rows * cols], Shape::new(&[rows, cols]));
        let make_bias = |dim| Tensor::from_vec(vec![0.0; dim], Shape::new(&[dim]));

        Self {
            q_weight: make_weight(d_model, d_model),
            q_bias: make_bias(d_model),
            k_weight: make_weight(d_model, d_model),
            k_bias: make_bias(d_model),
            v_weight: make_weight(d_model, d_model),
            v_bias: make_bias(d_model),
            out_weight: make_weight(d_model, d_model),
            out_bias: make_bias(d_model),
            n_heads,
            d_head,
            d_model,
        }
    }

    /// Attention: `q_input` attends to `kv_input`.
    ///
    /// `q_input`: `[q_len, d_model]`, `kv_input`: `[kv_len, d_model]`.
    /// Returns: `[q_len, d_model]`.
    fn forward(&self, q_input: &Tensor<B>, kv_input: &Tensor<B>) -> Tensor<B> {
        let q_len = q_input.shape.dims()[0];
        let kv_len = kv_input.shape.dims()[0];
        let dm = self.d_model;
        let nh = self.n_heads;
        let dh = self.d_head;

        // Project Q, K, V
        let q = self.linear_project(q_input, &self.q_weight, &self.q_bias, q_len);
        let k = self.linear_project(kv_input, &self.k_weight, &self.k_bias, kv_len);
        let v = self.linear_project(kv_input, &self.v_weight, &self.v_bias, kv_len);

        // Reshape to [n_heads, seq, d_head]
        let q_heads = self.to_heads(&q, q_len);
        let k_heads = self.to_heads(&k, kv_len);
        let v_heads = self.to_heads(&v, kv_len);

        // Scaled dot-product attention per head
        let scale = 1.0 / (dh as f32).sqrt();
        let mut context = vec![0.0f32; nh * q_len * dh];

        for h in 0..nh {
            // Compute scores: [q_len, kv_len]
            let mut scores = vec![0.0f32; q_len * kv_len];
            for i in 0..q_len {
                for j in 0..kv_len {
                    let mut dot = 0.0f32;
                    for d in 0..dh {
                        dot += q_heads[h * q_len * dh + i * dh + d]
                            * k_heads[h * kv_len * dh + j * dh + d];
                    }
                    scores[i * kv_len + j] = dot * scale;
                }
            }

            // Softmax per row
            for i in 0..q_len {
                let off = i * kv_len;
                let row = &mut scores[off..off + kv_len];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                let inv = 1.0 / sum;
                for v in row.iter_mut() {
                    *v *= inv;
                }
            }

            // Context: scores @ V -> [q_len, d_head]
            for i in 0..q_len {
                for d in 0..dh {
                    let mut val = 0.0f32;
                    for j in 0..kv_len {
                        val += scores[i * kv_len + j] * v_heads[h * kv_len * dh + j * dh + d];
                    }
                    context[h * q_len * dh + i * dh + d] = val;
                }
            }
        }

        // Reshape [n_heads, q_len, d_head] -> [q_len, d_model]
        let mut combined = vec![0.0f32; q_len * dm];
        for s in 0..q_len {
            for h in 0..nh {
                for d in 0..dh {
                    combined[s * dm + h * dh + d] = context[h * q_len * dh + s * dh + d];
                }
            }
        }

        // Output projection
        let combined_s = B::from_vec(combined, &Shape::new(&[q_len, dm]));
        let out = B::matmul(
            &combined_s,
            &self.out_weight.data,
            q_len,
            dm,
            dm,
            false,
            false,
        );
        let out_shape = Shape::new(&[q_len, dm]);
        let bias_shape = Shape::new(&[1, dm]);
        let result = B::add(
            &out,
            &self.out_bias.data,
            &out_shape,
            &bias_shape,
            &out_shape,
        );

        Tensor::new(result, out_shape)
    }

    fn linear_project(
        &self,
        input: &Tensor<B>,
        weight: &Tensor<B>,
        bias: &Tensor<B>,
        seq: usize,
    ) -> Vec<f32> {
        let dm = self.d_model;
        let out = B::matmul(&input.data, &weight.data, seq, dm, dm, false, false);
        let out_shape = Shape::new(&[seq, dm]);
        let bias_shape = Shape::new(&[1, dm]);
        let result = B::add(&out, &bias.data, &out_shape, &bias_shape, &out_shape);
        B::into_vec(result)
    }

    fn to_heads(&self, data: &[f32], seq: usize) -> Vec<f32> {
        let nh = self.n_heads;
        let dh = self.d_head;
        let dm = self.d_model;
        let mut heads = vec![0.0f32; nh * seq * dh];
        for s in 0..seq {
            for h in 0..nh {
                for d in 0..dh {
                    heads[h * seq * dh + s * dh + d] = data[s * dm + h * dh + d];
                }
            }
        }
        heads
    }
}

impl<B: MathBackend> Module<B> for Attention<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![
            &self.q_weight,
            &self.q_bias,
            &self.k_weight,
            &self.k_bias,
            &self.v_weight,
            &self.v_bias,
            &self.out_weight,
            &self.out_bias,
        ]
    }
}

// ── Two-way attention block ─────────────────────────────────────────────────

/// Two-way transformer block.
///
/// 1. Self-attention on tokens
/// 2. Cross-attention: tokens attend to image
/// 3. MLP on tokens
/// 4. Cross-attention: image attends to tokens
pub struct TwoWayAttentionBlock<B: MathBackend> {
    self_attn: Attention<B>,
    ln_self: (Tensor<B>, Tensor<B>),
    cross_attn_token_to_image: Attention<B>,
    ln_cross_ti: (Tensor<B>, Tensor<B>),
    mlp_lin1_weight: Tensor<B>,
    mlp_lin1_bias: Tensor<B>,
    mlp_lin2_weight: Tensor<B>,
    mlp_lin2_bias: Tensor<B>,
    ln_mlp: (Tensor<B>, Tensor<B>),
    cross_attn_image_to_token: Attention<B>,
    ln_cross_it: (Tensor<B>, Tensor<B>),
    d_model: usize,
    d_ff: usize,
}

impl<B: MathBackend> TwoWayAttentionBlock<B> {
    fn new(d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        let make_ln = || {
            (
                Tensor::from_vec(vec![1.0; d_model], Shape::new(&[d_model])),
                Tensor::from_vec(vec![0.0; d_model], Shape::new(&[d_model])),
            )
        };
        let make_weight =
            |rows, cols| Tensor::from_vec(vec![0.0; rows * cols], Shape::new(&[rows, cols]));
        let make_bias = |dim| Tensor::from_vec(vec![0.0; dim], Shape::new(&[dim]));

        Self {
            self_attn: Attention::new(d_model, n_heads),
            ln_self: make_ln(),
            cross_attn_token_to_image: Attention::new(d_model, n_heads),
            ln_cross_ti: make_ln(),
            mlp_lin1_weight: make_weight(d_model, d_ff),
            mlp_lin1_bias: make_bias(d_ff),
            mlp_lin2_weight: make_weight(d_ff, d_model),
            mlp_lin2_bias: make_bias(d_model),
            ln_mlp: make_ln(),
            cross_attn_image_to_token: Attention::new(d_model, n_heads),
            ln_cross_it: make_ln(),
            d_model,
            d_ff,
        }
    }

    /// Forward pass.
    ///
    /// `tokens`: `[num_tokens, d_model]`, `image`: `[hw, d_model]`.
    /// Returns `(updated_tokens, updated_image)`.
    fn forward(&self, tokens: &Tensor<B>, image: &Tensor<B>) -> (Tensor<B>, Tensor<B>) {
        let t_shape = tokens.shape.clone();
        let i_shape = image.shape.clone();

        // 1. Self-attention on tokens
        let t_normed = B::layernorm_inference(
            &tokens.data,
            &self.ln_self.0.data,
            &self.ln_self.1.data,
            &t_shape,
            1e-5,
        );
        let t_normed_tensor = Tensor::new(t_normed, t_shape.clone());
        let attn_out = self.self_attn.forward(&t_normed_tensor, &t_normed_tensor);
        let tokens_data = B::add(&tokens.data, &attn_out.data, &t_shape, &t_shape, &t_shape);

        // 2. Cross-attention: tokens -> image
        let t_normed = B::layernorm_inference(
            &tokens_data,
            &self.ln_cross_ti.0.data,
            &self.ln_cross_ti.1.data,
            &t_shape,
            1e-5,
        );
        let t_normed_tensor = Tensor::new(t_normed, t_shape.clone());
        let cross_out = self
            .cross_attn_token_to_image
            .forward(&t_normed_tensor, image);
        let tokens_data = B::add(&tokens_data, &cross_out.data, &t_shape, &t_shape, &t_shape);

        // 3. MLP on tokens
        let t_normed = B::layernorm_inference(
            &tokens_data,
            &self.ln_mlp.0.data,
            &self.ln_mlp.1.data,
            &t_shape,
            1e-5,
        );
        let mlp_out = self.mlp_forward(&t_normed, t_shape.dims()[0]);
        let tokens_data = B::add(&tokens_data, &mlp_out, &t_shape, &t_shape, &t_shape);

        // 4. Cross-attention: image -> tokens
        let i_normed = B::layernorm_inference(
            &image.data,
            &self.ln_cross_it.0.data,
            &self.ln_cross_it.1.data,
            &i_shape,
            1e-5,
        );
        let i_normed_tensor = Tensor::new(i_normed, i_shape.clone());
        let tokens_tensor = Tensor::new(tokens_data.clone(), t_shape.clone());
        let img_cross = self
            .cross_attn_image_to_token
            .forward(&i_normed_tensor, &tokens_tensor);
        let image_data = B::add(&image.data, &img_cross.data, &i_shape, &i_shape, &i_shape);

        (
            Tensor::new(tokens_data, t_shape),
            Tensor::new(image_data, i_shape),
        )
    }

    fn mlp_forward(&self, input: &B::Storage, seq: usize) -> B::Storage {
        let dm = self.d_model;
        let df = self.d_ff;

        // fc1 + relu
        let h = B::matmul(input, &self.mlp_lin1_weight.data, seq, dm, df, false, false);
        let h_shape = Shape::new(&[seq, df]);
        let b1_shape = Shape::new(&[1, df]);
        let h = B::add(&h, &self.mlp_lin1_bias.data, &h_shape, &b1_shape, &h_shape);
        // ReLU
        let h_vec: Vec<f32> = B::into_vec(h).into_iter().map(|x| x.max(0.0)).collect();
        let h = B::from_vec(h_vec, &h_shape);

        // fc2
        let out = B::matmul(&h, &self.mlp_lin2_weight.data, seq, df, dm, false, false);
        let out_shape = Shape::new(&[seq, dm]);
        let b2_shape = Shape::new(&[1, dm]);
        B::add(
            &out,
            &self.mlp_lin2_bias.data,
            &out_shape,
            &b2_shape,
            &out_shape,
        )
    }
}

impl<B: MathBackend> Module<B> for TwoWayAttentionBlock<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = self.self_attn.parameters();
        params.push(&self.ln_self.0);
        params.push(&self.ln_self.1);
        params.extend(self.cross_attn_token_to_image.parameters());
        params.push(&self.ln_cross_ti.0);
        params.push(&self.ln_cross_ti.1);
        params.push(&self.mlp_lin1_weight);
        params.push(&self.mlp_lin1_bias);
        params.push(&self.mlp_lin2_weight);
        params.push(&self.mlp_lin2_bias);
        params.push(&self.ln_mlp.0);
        params.push(&self.ln_mlp.1);
        params.extend(self.cross_attn_image_to_token.parameters());
        params.push(&self.ln_cross_it.0);
        params.push(&self.ln_cross_it.1);
        params
    }
}

// ── MLP head ────────────────────────────────────────────────────────────────

/// Multi-layer perceptron head with `ReLU` activations.
///
/// Used for mask prediction (hypernetwork MLPs) and `IoU` score prediction.
pub struct MlpHead<B: MathBackend> {
    /// `(weight, bias)` pairs for each layer.
    pub layers: Vec<(Tensor<B>, Tensor<B>)>,
}

impl<B: MathBackend> MlpHead<B> {
    /// Create an MLP with `num_layers` layers.
    ///
    /// Hidden layers have dimension `hidden_dim`, final output has dimension `output_dim`.
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, num_layers: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let (in_d, out_d) = if i == 0 {
                (
                    input_dim,
                    if num_layers == 1 {
                        output_dim
                    } else {
                        hidden_dim
                    },
                )
            } else if i == num_layers - 1 {
                (hidden_dim, output_dim)
            } else {
                (hidden_dim, hidden_dim)
            };
            layers.push((
                Tensor::from_vec(vec![0.0; in_d * out_d], Shape::new(&[in_d, out_d])),
                Tensor::from_vec(vec![0.0; out_d], Shape::new(&[out_d])),
            ));
        }
        Self { layers }
    }

    /// Forward pass: `[d_in]` -> `[d_out]`.
    ///
    /// `ReLU` between all layers except the last.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut current = input.to_vec();
        for (i, (weight, bias)) in self.layers.iter().enumerate() {
            let in_dim = weight.shape.dims()[0];
            let out_dim = weight.shape.dims()[1];
            let w = weight.to_vec();
            let b = bias.to_vec();

            let mut out = vec![0.0f32; out_dim];
            for o in 0..out_dim {
                let mut val = b[o];
                for k in 0..in_dim {
                    val += current[k] * w[k * out_dim + o];
                }
                // ReLU on all but last layer
                if i < self.layers.len() - 1 {
                    val = val.max(0.0);
                }
                out[o] = val;
            }
            current = out;
        }
        current
    }
}

impl<B: MathBackend> Module<B> for MlpHead<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        self.layers.iter().flat_map(|(w, b)| [w, b]).collect()
    }
}

// ── Mask decoder ────────────────────────────────────────────────────────────

/// SAM mask decoder.
///
/// Takes image embeddings and prompt embeddings, produces multi-mask output
/// with `IoU` confidence scores.
pub struct MaskDecoder<B: MathBackend> {
    /// Two-way transformer blocks.
    pub transformer_blocks: Vec<TwoWayAttentionBlock<B>>,
    /// Final layer norm for tokens after transformer.
    pub ln_final_gamma: Tensor<B>,
    pub ln_final_beta: Tensor<B>,
    /// Learned `IoU` prediction token: `[1, embed_dim]`.
    pub iou_token: Tensor<B>,
    /// Learned mask tokens: `[num_mask_tokens, embed_dim]`.
    pub mask_tokens: Tensor<B>,
    /// Upscaling: `ConvTranspose2d` 256->64 k=2 s=2.
    pub upscale_conv1: ConvTranspose2d<B>,
    pub upscale_ln: LayerNorm2d<B>,
    /// Upscaling: `ConvTranspose2d` 64->32 k=2 s=2.
    pub upscale_conv2: ConvTranspose2d<B>,
    /// Per-mask hypernetwork MLPs.
    pub mask_mlps: Vec<MlpHead<B>>,
    /// `IoU` prediction head.
    pub iou_head: MlpHead<B>,
    pub embed_dim: usize,
    pub num_mask_tokens: usize,
}

impl<B: MathBackend> MaskDecoder<B> {
    /// Create a zero-initialized mask decoder from config.
    pub fn new(config: &SamConfig) -> Self {
        let ed = config.out_channels; // 256
        let num_mask_tokens = config.num_multimask_outputs + 1; // 4 (1 single + 3 multi)
        let upscale_ch1 = ed / 4; // 64
        let upscale_ch2 = ed / 8; // 32

        let mut mask_mlps = Vec::with_capacity(num_mask_tokens);
        for _ in 0..num_mask_tokens {
            mask_mlps.push(MlpHead::new(ed, ed, upscale_ch2, 3));
        }

        Self {
            transformer_blocks: (0..config.decoder_depth)
                .map(|_| TwoWayAttentionBlock::new(ed, config.decoder_num_heads, ed * 2))
                .collect(),
            ln_final_gamma: Tensor::from_vec(vec![1.0; ed], Shape::new(&[ed])),
            ln_final_beta: Tensor::from_vec(vec![0.0; ed], Shape::new(&[ed])),
            iou_token: Tensor::from_vec(vec![0.0; ed], Shape::new(&[1, ed])),
            mask_tokens: Tensor::from_vec(
                vec![0.0; num_mask_tokens * ed],
                Shape::new(&[num_mask_tokens, ed]),
            ),
            upscale_conv1: ConvTranspose2d::square(ed, upscale_ch1, 2, 2, 0),
            upscale_ln: LayerNorm2d::new(upscale_ch1, 1e-6),
            upscale_conv2: ConvTranspose2d::square(upscale_ch1, upscale_ch2, 2, 2, 0),
            mask_mlps,
            iou_head: MlpHead::new(
                ed,
                config.iou_head_hidden_dim,
                num_mask_tokens,
                config.iou_head_depth,
            ),
            embed_dim: ed,
            num_mask_tokens,
        }
    }

    /// Predict masks and `IoU` scores.
    ///
    /// - `image_embeddings`: `[embed_dim, grid_h, grid_w]` from the image encoder
    /// - `image_pe`: `[embed_dim, grid_h, grid_w]` positional encoding
    /// - `sparse_prompt`: `[num_points, embed_dim]` from prompt encoder
    /// - `dense_prompt`: `[embed_dim, grid_h, grid_w]` from prompt encoder
    pub fn forward(
        &self,
        image_embeddings: &Tensor<B>,
        image_pe: &Tensor<B>,
        sparse_prompt: &Tensor<B>,
        dense_prompt: &Tensor<B>,
    ) -> SamOutput {
        let img_dims = image_embeddings.shape.dims();
        let ed = img_dims[0];
        let grid_h = img_dims[1];
        let grid_w = img_dims[2];
        let hw = grid_h * grid_w;
        let num_prompt = sparse_prompt.shape.dims()[0];
        let num_output_tokens = 1 + self.num_mask_tokens; // iou_token + mask_tokens
        let total_tokens = num_output_tokens + num_prompt;

        // Build token sequence: [iou_token, mask_tokens, sparse_prompt_tokens]
        let iou_vec = self.iou_token.to_vec();
        let mask_vec = self.mask_tokens.to_vec();
        let prompt_vec = sparse_prompt.to_vec();
        let mut tokens = Vec::with_capacity(total_tokens * ed);
        tokens.extend_from_slice(&iou_vec);
        tokens.extend_from_slice(&mask_vec);
        tokens.extend_from_slice(&prompt_vec);
        let tokens_tensor = Tensor::<B>::from_vec(tokens, Shape::new(&[total_tokens, ed]));

        // Image features + dense prompt: add dense_prompt to image_embeddings
        let img_vec = image_embeddings.to_vec();
        let dense_vec = dense_prompt.to_vec();
        let src: Vec<f32> = img_vec
            .iter()
            .zip(dense_vec.iter())
            .map(|(a, b)| a + b)
            .collect();

        // Flatten image to [hw, ed] for transformer (transpose from [ed, h, w])
        let mut src_flat = vec![0.0f32; hw * ed];
        for d in 0..ed {
            for i in 0..hw {
                src_flat[i * ed + d] = src[d * hw + i];
            }
        }
        let src_tensor = Tensor::<B>::from_vec(src_flat, Shape::new(&[hw, ed]));

        // Also flatten image_pe to [hw, ed]
        let pe_vec = image_pe.to_vec();
        let mut pe_flat = vec![0.0f32; hw * ed];
        for d in 0..ed {
            for i in 0..hw {
                pe_flat[i * ed + d] = pe_vec[d * hw + i];
            }
        }

        // Add positional encoding to image (keys)
        let src_with_pe: Vec<f32> = src_tensor
            .to_vec()
            .iter()
            .zip(pe_flat.iter())
            .map(|(a, b)| a + b)
            .collect();
        let src_pe_tensor = Tensor::<B>::from_vec(src_with_pe, Shape::new(&[hw, ed]));

        // Run two-way transformer blocks
        let mut tokens_t = tokens_tensor;
        let mut image_t = src_pe_tensor;
        for block in &self.transformer_blocks {
            let (t, i) = block.forward(&tokens_t, &image_t);
            tokens_t = t;
            image_t = i;
        }

        // Final layer norm on tokens
        let t_shape = tokens_t.shape.clone();
        let tokens_normed = B::layernorm_inference(
            &tokens_t.data,
            &self.ln_final_gamma.data,
            &self.ln_final_beta.data,
            &t_shape,
            1e-5,
        );
        let tokens_vec = B::into_vec(tokens_normed);

        // Extract IoU token (first token) and mask tokens (next num_mask_tokens)
        let iou_token_vec = tokens_vec[..ed].to_vec();
        let mask_token_vecs: Vec<&[f32]> = (0..self.num_mask_tokens)
            .map(|i| &tokens_vec[(1 + i) * ed..(2 + i) * ed])
            .collect();

        // Upscale image features: [ed, h, w] -> [ch2, 4h, 4w]
        // First reshape image_t back to [ed, grid_h, grid_w]
        let image_vec = image_t.to_vec();
        let mut image_chw = vec![0.0f32; ed * hw];
        for i in 0..hw {
            for d in 0..ed {
                image_chw[d * hw + i] = image_vec[i * ed + d];
            }
        }
        let upscale_input = Tensor::<B>::from_vec(image_chw, Shape::new(&[ed, grid_h, grid_w]));

        let upscaled = self.upscale_conv1.forward(&upscale_input);
        let upscaled = self.upscale_ln.forward(&upscaled);
        // GELU activation
        let up_vec: Vec<f32> = upscaled.to_vec();
        let up_gelu: Vec<f32> = up_vec
            .iter()
            .map(|&x| {
                // Tanh approximation of GELU
                let inner = std::f32::consts::FRAC_2_SQRT_PI
                    * std::f32::consts::FRAC_1_SQRT_2
                    * (x + 0.044_715 * x * x * x);
                x * 0.5 * (1.0 + inner.tanh())
            })
            .collect();
        let upscaled = Tensor::<B>::from_vec(up_gelu, upscaled.shape.clone());
        let upscaled = self.upscale_conv2.forward(&upscaled);

        let up_dims = upscaled.shape.dims();
        let mask_h = up_dims[1]; // grid_h * 4
        let mask_w = up_dims[2]; // grid_w * 4
        let upscaled_vec = upscaled.to_vec();

        // Generate masks: dot(MLP(mask_token), upscaled) for each mask
        let mut masks = Vec::with_capacity(self.num_mask_tokens);
        for (mlp, &token) in self.mask_mlps.iter().zip(mask_token_vecs.iter()) {
            let hyper = mlp.forward(token);
            // hyper: [up_ch], upscaled: [up_ch, mask_h, mask_w]
            // mask = sum over channels: hyper[c] * upscaled[c, :, :]
            let mut mask = vec![0.0f32; mask_h * mask_w];
            for (c, &h) in hyper.iter().enumerate() {
                let ch_off = c * mask_h * mask_w;
                for p in 0..mask_h * mask_w {
                    mask[p] += h * upscaled_vec[ch_off + p];
                }
            }
            masks.push(mask);
        }

        // IoU prediction
        let iou_scores = self.iou_head.forward(&iou_token_vec);

        SamOutput {
            masks,
            mask_width: mask_w as u32,
            mask_height: mask_h as u32,
            iou_scores,
        }
    }
}

impl<B: MathBackend> Module<B> for MaskDecoder<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = Vec::new();
        for block in &self.transformer_blocks {
            params.extend(block.parameters());
        }
        params.push(&self.ln_final_gamma);
        params.push(&self.ln_final_beta);
        params.push(&self.iou_token);
        params.push(&self.mask_tokens);
        params.extend(self.upscale_conv1.parameters());
        params.extend(self.upscale_ln.parameters());
        params.extend(self.upscale_conv2.parameters());
        for mlp in &self.mask_mlps {
            params.extend(mlp.parameters());
        }
        params.extend(self.iou_head.parameters());
        params
    }
}

// ── Safetensors loading ─────────────────────────────────────────────────────

#[cfg(feature = "safetensors")]
impl<B: MathBackend> MaskDecoder<B> {
    /// Load a SAM mask decoder from safetensors.
    ///
    /// `prefix` is typically `"mask_decoder."`.
    pub fn from_safetensors(
        config: &SamConfig,
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
    ) -> crate::error::Result<Self> {
        use crate::checkpoint::{load_conv_transpose2d, load_layernorm2d, load_tensor};

        let ed = config.out_channels;
        let nh = config.decoder_num_heads;
        let num_mask_tokens = config.num_multimask_outputs + 1;
        let upscale_ch1 = ed / 4;
        let upscale_ch2 = ed / 8;
        let d_ff = ed * 2;

        let mut transformer_blocks = Vec::with_capacity(config.decoder_depth);
        for i in 0..config.decoder_depth {
            let bp = format!("{prefix}transformer.layers.{i}");
            transformer_blocks.push(load_two_way_block(tensors, &bp, ed, nh, d_ff)?);
        }

        let ln_final_gamma = load_tensor(
            tensors,
            &format!("{prefix}transformer.final_attn_token_to_image.norm.weight"),
            &[ed],
        )
        .or_else(|_| {
            load_tensor(
                tensors,
                &format!("{prefix}transformer.norm_final_attn.weight"),
                &[ed],
            )
        })?;
        let ln_final_beta = load_tensor(
            tensors,
            &format!("{prefix}transformer.final_attn_token_to_image.norm.bias"),
            &[ed],
        )
        .or_else(|_| {
            load_tensor(
                tensors,
                &format!("{prefix}transformer.norm_final_attn.bias"),
                &[ed],
            )
        })?;

        let iou_token = load_tensor(tensors, &format!("{prefix}iou_token.weight"), &[1, ed])?;
        let mask_tokens = load_tensor(
            tensors,
            &format!("{prefix}mask_tokens.weight"),
            &[num_mask_tokens, ed],
        )?;

        let upscale_conv1 = load_conv_transpose2d(
            tensors,
            &format!("{prefix}output_upscaling.0"),
            ed,
            upscale_ch1,
            2,
            2,
            2,
            0,
            0,
        )?;
        let upscale_ln = load_layernorm2d(
            tensors,
            &format!("{prefix}output_upscaling.1"),
            upscale_ch1,
            1e-6,
        )?;
        let upscale_conv2 = load_conv_transpose2d(
            tensors,
            &format!("{prefix}output_upscaling.3"),
            upscale_ch1,
            upscale_ch2,
            2,
            2,
            2,
            0,
            0,
        )?;

        let mut mask_mlps = Vec::with_capacity(num_mask_tokens);
        for i in 0..num_mask_tokens {
            mask_mlps.push(load_mlp_head(
                tensors,
                &format!("{prefix}output_hypernetworks_mlps.{i}"),
                ed,
                ed,
                upscale_ch2,
                3,
            )?);
        }

        let iou_head = load_mlp_head(
            tensors,
            &format!("{prefix}iou_prediction_head"),
            ed,
            config.iou_head_hidden_dim,
            num_mask_tokens,
            config.iou_head_depth,
        )?;

        Ok(Self {
            transformer_blocks,
            ln_final_gamma,
            ln_final_beta,
            iou_token,
            mask_tokens,
            upscale_conv1,
            upscale_ln,
            upscale_conv2,
            mask_mlps,
            iou_head,
            embed_dim: ed,
            num_mask_tokens,
        })
    }
}

#[cfg(feature = "safetensors")]
fn load_two_way_block<B: MathBackend>(
    tensors: &safetensors::SafeTensors<'_>,
    prefix: &str,
    d_model: usize,
    n_heads: usize,
    d_ff: usize,
) -> crate::error::Result<TwoWayAttentionBlock<B>> {
    use crate::checkpoint::{load_tensor, load_tensor_transposed};

    let load_attn = |ap: &str| -> crate::error::Result<Attention<B>> {
        let dm = d_model;
        let q_weight = load_tensor_transposed(tensors, &format!("{ap}.q_proj.weight"), dm, dm)?;
        let q_bias = load_tensor(tensors, &format!("{ap}.q_proj.bias"), &[dm])?;
        let k_weight = load_tensor_transposed(tensors, &format!("{ap}.k_proj.weight"), dm, dm)?;
        let k_bias = load_tensor(tensors, &format!("{ap}.k_proj.bias"), &[dm])?;
        let v_weight = load_tensor_transposed(tensors, &format!("{ap}.v_proj.weight"), dm, dm)?;
        let v_bias = load_tensor(tensors, &format!("{ap}.v_proj.bias"), &[dm])?;
        let out_weight = load_tensor_transposed(tensors, &format!("{ap}.out_proj.weight"), dm, dm)?;
        let out_bias = load_tensor(tensors, &format!("{ap}.out_proj.bias"), &[dm])?;
        Ok(Attention {
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            out_weight,
            out_bias,
            n_heads,
            d_head: dm / n_heads,
            d_model: dm,
        })
    };

    let load_ln = |lp: &str| -> crate::error::Result<(Tensor<B>, Tensor<B>)> {
        Ok((
            load_tensor(tensors, &format!("{lp}.weight"), &[d_model])?,
            load_tensor(tensors, &format!("{lp}.bias"), &[d_model])?,
        ))
    };

    Ok(TwoWayAttentionBlock {
        self_attn: load_attn(&format!("{prefix}.self_attn"))?,
        ln_self: load_ln(&format!("{prefix}.norm1"))?,
        cross_attn_token_to_image: load_attn(&format!("{prefix}.cross_attn_token_to_image"))?,
        ln_cross_ti: load_ln(&format!("{prefix}.norm2"))?,
        mlp_lin1_weight: load_tensor_transposed(
            tensors,
            &format!("{prefix}.mlp.lin1.weight"),
            d_ff,
            d_model,
        )?,
        mlp_lin1_bias: load_tensor(tensors, &format!("{prefix}.mlp.lin1.bias"), &[d_ff])?,
        mlp_lin2_weight: load_tensor_transposed(
            tensors,
            &format!("{prefix}.mlp.lin2.weight"),
            d_model,
            d_ff,
        )?,
        mlp_lin2_bias: load_tensor(tensors, &format!("{prefix}.mlp.lin2.bias"), &[d_model])?,
        ln_mlp: load_ln(&format!("{prefix}.norm3"))?,
        cross_attn_image_to_token: load_attn(&format!("{prefix}.cross_attn_image_to_token"))?,
        ln_cross_it: load_ln(&format!("{prefix}.norm4"))?,
        d_model,
        d_ff,
    })
}

#[cfg(feature = "safetensors")]
fn load_mlp_head<B: MathBackend>(
    tensors: &safetensors::SafeTensors<'_>,
    prefix: &str,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    num_layers: usize,
) -> crate::error::Result<MlpHead<B>> {
    use crate::checkpoint::load_tensor_transposed;
    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let (in_d, out_d) = if i == 0 {
            (
                input_dim,
                if num_layers == 1 {
                    output_dim
                } else {
                    hidden_dim
                },
            )
        } else if i == num_layers - 1 {
            (hidden_dim, output_dim)
        } else {
            (hidden_dim, hidden_dim)
        };
        let weight =
            load_tensor_transposed(tensors, &format!("{prefix}.layers.{i}.weight"), out_d, in_d)?;
        let bias = crate::checkpoint::load_tensor(
            tensors,
            &format!("{prefix}.layers.{i}.bias"),
            &[out_d],
        )?;
        layers.push((weight, bias));
    }
    Ok(MlpHead { layers })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scry_llm::backend::cpu::CpuBackend;

    fn tiny_config() -> SamConfig {
        SamConfig {
            image_size: 64,
            patch_size: 16,
            embed_dim: 32,
            depth: 1,
            num_heads: 2,
            mlp_ratio: 2.0,
            global_attn_indices: vec![0],
            window_size: 2,
            out_channels: 16,
            num_multimask_outputs: 3,
            iou_head_depth: 2,
            iou_head_hidden_dim: 16,
            decoder_depth: 1,
            decoder_num_heads: 2,
        }
    }

    #[test]
    fn two_way_block_output_shapes() {
        let block = TwoWayAttentionBlock::<CpuBackend>::new(16, 2, 32);
        let tokens = Tensor::from_vec(vec![0.0; 5 * 16], Shape::new(&[5, 16]));
        let image = Tensor::from_vec(vec![0.0; 16 * 16], Shape::new(&[16, 16]));
        let (t_out, i_out) = block.forward(&tokens, &image);
        assert_eq!(t_out.shape.dims(), &[5, 16]);
        assert_eq!(i_out.shape.dims(), &[16, 16]);
    }

    #[test]
    fn mlp_head_output_shape() {
        let mlp = MlpHead::<CpuBackend>::new(16, 32, 8, 3);
        let input = vec![0.0f32; 16];
        let output = mlp.forward(&input);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn mask_decoder_output_shapes() {
        let config = tiny_config();
        let decoder = MaskDecoder::<CpuBackend>::new(&config);

        let ed = config.out_channels;
        let grid = config.num_patches_per_side(); // 4

        let image_emb =
            Tensor::from_vec(vec![0.0; ed * grid * grid], Shape::new(&[ed, grid, grid]));
        let image_pe = Tensor::from_vec(vec![0.0; ed * grid * grid], Shape::new(&[ed, grid, grid]));
        let sparse = Tensor::from_vec(vec![0.0; 1 * ed], Shape::new(&[1, ed]));
        let dense = Tensor::from_vec(vec![0.0; ed * grid * grid], Shape::new(&[ed, grid, grid]));

        let output = decoder.forward(&image_emb, &image_pe, &sparse, &dense);

        // 4 masks (1 + 3 multimask)
        assert_eq!(output.masks.len(), 4);
        // Mask resolution: grid * 4 = 16
        assert_eq!(output.mask_width, 16);
        assert_eq!(output.mask_height, 16);
        assert_eq!(output.masks[0].len(), 16 * 16);
        // 4 IoU scores
        assert_eq!(output.iou_scores.len(), 4);
    }

    #[test]
    fn upscaling_shape() {
        let config = tiny_config();
        let ed = config.out_channels; // 16

        // Test upscaling path: ed -> ed/4 -> ed/8
        let up1 = ConvTranspose2d::<CpuBackend>::square(ed, ed / 4, 2, 2, 0);
        let up2 = ConvTranspose2d::<CpuBackend>::square(ed / 4, ed / 8, 2, 2, 0);

        let input = Tensor::from_vec(vec![0.0; ed * 4 * 4], Shape::new(&[ed, 4, 4]));
        let x = up1.forward(&input);
        assert_eq!(x.shape.dims(), &[ed / 4, 8, 8]);
        let x = up2.forward(&x);
        assert_eq!(x.shape.dims(), &[ed / 8, 16, 16]);
    }
}
