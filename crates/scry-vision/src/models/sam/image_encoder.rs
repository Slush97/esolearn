// SPDX-License-Identifier: MIT OR Apache-2.0
//! SAM image encoder — windowed Vision Transformer with relative position bias.
//!
//! Key differences from the standard [`crate::models::vit::Vit`]:
//! - No CLS token — outputs a 2D `[out_channels, H/patch, W/patch]` feature map
//! - Window attention (default 14×14) with global attention at selected layers
//! - Decomposed relative position bias instead of absolute position embeddings
//! - Neck convolutions reduce `embed_dim` → `out_channels` (256)

use scry_llm::backend::MathBackend;
use scry_llm::nn::Module;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use crate::models::vit::VitMlp;
use crate::nn::conv2d::Conv2d;
use crate::nn::layernorm2d::LayerNorm2d;
use crate::nn::PatchEmbedding;

use super::SamConfig;

// ── Relative position bias ──────────────────────────────────────────────────

/// Decomposed relative position bias for attention.
///
/// Instead of a full 2D `[M*M, M*M]` table, stores separate horizontal and
/// vertical position embeddings. The bias for position `(i, j)` attending to
/// `(i', j')` is `rel_pos_h[i - i' + M - 1] · q_h + rel_pos_w[j - j' + M - 1] · q_w`.
pub struct RelativePositionBias<B: MathBackend> {
    /// Row relative position embeddings: `[2 * size - 1, d_head]`.
    pub rel_pos_h: Tensor<B>,
    /// Column relative position embeddings: `[2 * size - 1, d_head]`.
    pub rel_pos_w: Tensor<B>,
    pub size: usize,
    pub d_head: usize,
}

impl<B: MathBackend> RelativePositionBias<B> {
    /// Create zero-initialized bias for a window of `size × size` tokens.
    pub fn new(size: usize, d_head: usize) -> Self {
        let table_len = 2 * size - 1;
        Self {
            rel_pos_h: Tensor::from_vec(
                vec![0.0; table_len * d_head],
                Shape::new(&[table_len, d_head]),
            ),
            rel_pos_w: Tensor::from_vec(
                vec![0.0; table_len * d_head],
                Shape::new(&[table_len, d_head]),
            ),
            size,
            d_head,
        }
    }

    /// Compute the additive attention bias for a `grid_h × grid_w` feature map.
    ///
    /// Returns a flat `[grid_h * grid_w, grid_h * grid_w]` bias matrix where
    /// `bias[i, j]` is the scalar bias added to the attention logit of token `i`
    /// attending to token `j`.
    ///
    /// `q` has shape `[n_heads, seq, d_head]` (flattened).
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    pub fn compute(&self, q: &[f32], n_heads: usize, grid_h: usize, grid_w: usize) -> Vec<f32> {
        let seq = grid_h * grid_w;
        let rel_h = self.rel_pos_h.to_vec();
        let rel_w = self.rel_pos_w.to_vec();
        let dh = self.d_head;

        // Interpolate relative position tables if grid size != stored size
        let (rel_h_interp, interp_h) = interpolate_rel_pos(&rel_h, self.size, grid_h, dh);
        let (rel_w_interp, interp_w) = interpolate_rel_pos(&rel_w, self.size, grid_w, dh);
        let rel_h = if interp_h { &rel_h_interp } else { &rel_h };
        let rel_w = if interp_w { &rel_w_interp } else { &rel_w };

        let mut bias = vec![0.0f32; n_heads * seq * seq];

        for head in 0..n_heads {
            let q_base = head * seq * dh;

            // Compute rel_h bias: q[head, i, :] @ rel_pos_h[rel_idx, :]^T
            // For each source row i_r and target row j_r, rel_idx = i_r - j_r + grid_h - 1
            for i in 0..seq {
                let i_r = i / grid_w;
                let i_c = i % grid_w;
                let q_off = q_base + i * dh;

                for j in 0..seq {
                    let j_r = j / grid_w;
                    let j_c = j % grid_w;

                    // Height component: dot(q[i], rel_pos_h[i_r - j_r + grid_h - 1])
                    let rh_idx = (i_r as isize - j_r as isize + grid_h as isize - 1) as usize;
                    let rh_off = rh_idx * dh;
                    let mut dot_h = 0.0f32;
                    for d in 0..dh {
                        dot_h += q[q_off + d] * rel_h[rh_off + d];
                    }

                    // Width component: dot(q[i], rel_pos_w[i_c - j_c + grid_w - 1])
                    let rw_idx = (i_c as isize - j_c as isize + grid_w as isize - 1) as usize;
                    let rw_off = rw_idx * dh;
                    let mut dot_w = 0.0f32;
                    for d in 0..dh {
                        dot_w += q[q_off + d] * rel_w[rw_off + d];
                    }

                    bias[head * seq * seq + i * seq + j] = dot_h + dot_w;
                }
            }
        }

        bias
    }
}

/// Linearly interpolate a relative position table from `orig_size` to `target_size`.
///
/// Returns `(interpolated_data, did_interpolate)`.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
fn interpolate_rel_pos(
    data: &[f32],
    orig_size: usize,
    target_size: usize,
    d_head: usize,
) -> (Vec<f32>, bool) {
    let orig_len = 2 * orig_size - 1;
    let target_len = 2 * target_size - 1;
    if orig_len == target_len {
        return (Vec::new(), false);
    }
    // Linear interpolation per d_head column
    let mut out = vec![0.0f32; target_len * d_head];
    let scale = orig_len as f32 / target_len as f32;
    for i in 0..target_len {
        let src_f = (i as f32 + 0.5) * scale - 0.5;
        let src_lo = (src_f.floor() as isize).clamp(0, orig_len as isize - 1) as usize;
        let src_hi = (src_lo + 1).min(orig_len - 1);
        let frac = src_f - src_lo as f32;
        for d in 0..d_head {
            out[i * d_head + d] =
                data[src_lo * d_head + d] * (1.0 - frac) + data[src_hi * d_head + d] * frac;
        }
    }
    (out, true)
}

impl<B: MathBackend> Module<B> for RelativePositionBias<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.rel_pos_h, &self.rel_pos_w]
    }
}

// ── Window attention ────────────────────────────────────────────────────────

/// SAM-style attention with optional window partitioning and relative position bias.
pub struct SamAttention<B: MathBackend> {
    /// Fused QKV projection: `[embed_dim, 3 * embed_dim]`.
    pub qkv_weight: Tensor<B>,
    pub qkv_bias: Tensor<B>,
    /// Output projection: `[embed_dim, embed_dim]`.
    pub proj_weight: Tensor<B>,
    pub proj_bias: Tensor<B>,
    pub rel_pos: RelativePositionBias<B>,
    pub n_heads: usize,
    pub d_head: usize,
    pub d_model: usize,
    /// If true, use windowed attention; if false, use global attention.
    pub use_window: bool,
    pub window_size: usize,
}

impl<B: MathBackend> SamAttention<B> {
    /// Create a zero-initialized attention block.
    pub fn new(d_model: usize, n_heads: usize, window_size: usize, use_window: bool) -> Self {
        let d_head = d_model / n_heads;
        let qkv_dim = 3 * d_model;
        let rel_size = window_size;
        Self {
            qkv_weight: Tensor::from_vec(
                vec![0.0; d_model * qkv_dim],
                Shape::new(&[d_model, qkv_dim]),
            ),
            qkv_bias: Tensor::from_vec(vec![0.0; qkv_dim], Shape::new(&[qkv_dim])),
            proj_weight: Tensor::from_vec(
                vec![0.0; d_model * d_model],
                Shape::new(&[d_model, d_model]),
            ),
            proj_bias: Tensor::from_vec(vec![0.0; d_model], Shape::new(&[d_model])),
            rel_pos: RelativePositionBias::new(rel_size, d_head),
            n_heads,
            d_head,
            d_model,
            use_window,
            window_size,
        }
    }

    /// Forward pass: `[seq, d_model]` -> `[seq, d_model]`.
    ///
    /// `grid_h` and `grid_w` define the 2D layout of the tokens (seq = `grid_h * grid_w`).
    /// When `use_window` is true, tokens are partitioned into non-overlapping
    /// `window_size × window_size` windows before attention.
    pub fn forward(&self, input: &Tensor<B>, grid_h: usize, grid_w: usize) -> Tensor<B> {
        if self.use_window {
            self.forward_windowed(input, grid_h, grid_w)
        } else {
            self.forward_global(input, grid_h, grid_w)
        }
    }

    fn forward_global(&self, input: &Tensor<B>, grid_h: usize, grid_w: usize) -> Tensor<B> {
        let seq = input.shape.dims()[0];
        let dm = self.d_model;

        // QKV: [seq, dm] @ [dm, 3*dm] + bias -> [seq, 3*dm]
        let qkv = B::matmul(
            &input.data,
            &self.qkv_weight.data,
            seq,
            dm,
            3 * dm,
            false,
            false,
        );
        let qkv_shape = Shape::new(&[seq, 3 * dm]);
        let bias_shape = Shape::new(&[1, 3 * dm]);
        let qkv = B::add(
            &qkv,
            &self.qkv_bias.data,
            &qkv_shape,
            &bias_shape,
            &qkv_shape,
        );
        let qkv_vec = B::into_vec(qkv);

        // Split into Q, K, V and reshape to [n_heads, seq, d_head]
        let nh = self.n_heads;
        let dh = self.d_head;
        let mut q = vec![0.0f32; nh * seq * dh];
        let mut k = vec![0.0f32; nh * seq * dh];
        let mut v = vec![0.0f32; nh * seq * dh];

        for s in 0..seq {
            for h in 0..nh {
                for d in 0..dh {
                    let src = s * 3 * dm + h * dh + d;
                    let dst = h * seq * dh + s * dh + d;
                    q[dst] = qkv_vec[src];
                    k[dst] = qkv_vec[src + dm];
                    v[dst] = qkv_vec[src + 2 * dm];
                }
            }
        }

        // Attention scores: Q @ K^T / sqrt(d_head)
        // [n_heads, seq, d_head] @ [n_heads, d_head, seq] -> [n_heads, seq, seq]
        let scale = 1.0 / (dh as f32).sqrt();
        let k_storage = B::from_vec(k, &Shape::new(&[nh * seq, dh]));
        let mut scores = vec![0.0f32; nh * seq * seq];

        for h in 0..nh {
            let q_off = h * seq * dh;
            let k_off = h * seq * dh;
            for i in 0..seq {
                for j in 0..seq {
                    let mut dot = 0.0f32;
                    for d in 0..dh {
                        dot += q[q_off + i * dh + d]
                            * B::into_vec(k_storage.clone())[k_off + j * dh + d];
                    }
                    scores[h * seq * seq + i * seq + j] = dot * scale;
                }
            }
        }

        // Add relative position bias
        let rel_bias = self.rel_pos.compute(&q, nh, grid_h, grid_w);
        for i in 0..scores.len() {
            scores[i] += rel_bias[i];
        }

        // Softmax per head per row
        for h in 0..nh {
            for i in 0..seq {
                let off = h * seq * seq + i * seq;
                let row = &mut scores[off..off + seq];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                let inv_sum = 1.0 / sum;
                for v in row.iter_mut() {
                    *v *= inv_sum;
                }
            }
        }

        // Context: scores @ V -> [n_heads, seq, d_head]
        let v_vec = B::into_vec(B::from_vec(v, &Shape::new(&[nh * seq, dh])));
        let mut context = vec![0.0f32; nh * seq * dh];
        for h in 0..nh {
            for i in 0..seq {
                for d in 0..dh {
                    let mut val = 0.0f32;
                    for j in 0..seq {
                        val +=
                            scores[h * seq * seq + i * seq + j] * v_vec[h * seq * dh + j * dh + d];
                    }
                    context[h * seq * dh + i * dh + d] = val;
                }
            }
        }

        // Reshape [n_heads, seq, d_head] -> [seq, dm]
        let mut combined = vec![0.0f32; seq * dm];
        for s in 0..seq {
            for h in 0..nh {
                for d in 0..dh {
                    combined[s * dm + h * dh + d] = context[h * seq * dh + s * dh + d];
                }
            }
        }

        // Output projection: [seq, dm] @ [dm, dm] + bias
        let combined_s = B::from_vec(combined, &Shape::new(&[seq, dm]));
        let out = B::matmul(
            &combined_s,
            &self.proj_weight.data,
            seq,
            dm,
            dm,
            false,
            false,
        );
        let out_shape = Shape::new(&[seq, dm]);
        let bias_shape = Shape::new(&[1, dm]);
        let result = B::add(
            &out,
            &self.proj_bias.data,
            &out_shape,
            &bias_shape,
            &out_shape,
        );

        Tensor::new(result, out_shape)
    }

    fn forward_windowed(&self, input: &Tensor<B>, grid_h: usize, grid_w: usize) -> Tensor<B> {
        let ws = self.window_size;
        let dm = self.d_model;
        let total_seq = grid_h * grid_w;

        // Pad grid to be divisible by window_size
        let pad_h = (ws - grid_h % ws) % ws;
        let pad_w = (ws - grid_w % ws) % ws;
        let gh = grid_h + pad_h;
        let gw = grid_w + pad_w;

        let input_vec = input.to_vec();

        // Pad input if needed: [grid_h*grid_w, dm] -> [gh*gw, dm]
        let padded = if pad_h > 0 || pad_w > 0 {
            let mut p = vec![0.0f32; gh * gw * dm];
            for r in 0..grid_h {
                let src_off = r * grid_w * dm;
                let dst_off = r * gw * dm;
                p[dst_off..dst_off + grid_w * dm]
                    .copy_from_slice(&input_vec[src_off..src_off + grid_w * dm]);
            }
            p
        } else {
            input_vec.clone()
        };

        // Partition into windows: [num_windows, ws*ws, dm]
        let nwh = gh / ws;
        let nww = gw / ws;
        let num_windows = nwh * nww;
        let win_seq = ws * ws;
        let mut windowed = vec![0.0f32; num_windows * win_seq * dm];

        for wh in 0..nwh {
            for ww in 0..nww {
                let win_idx = wh * nww + ww;
                for r in 0..ws {
                    for c in 0..ws {
                        let src_row = wh * ws + r;
                        let src_col = ww * ws + c;
                        let src_off = (src_row * gw + src_col) * dm;
                        let dst_off = (win_idx * win_seq + r * ws + c) * dm;
                        windowed[dst_off..dst_off + dm]
                            .copy_from_slice(&padded[src_off..src_off + dm]);
                    }
                }
            }
        }

        // Run attention on each window independently
        let mut out_windowed = vec![0.0f32; num_windows * win_seq * dm];

        for w in 0..num_windows {
            let win_off = w * win_seq * dm;
            let win_data = &windowed[win_off..win_off + win_seq * dm];
            let win_tensor = Tensor::<B>::from_vec(win_data.to_vec(), Shape::new(&[win_seq, dm]));
            let win_out = self.forward_global(&win_tensor, ws, ws);
            let win_out_vec = win_out.to_vec();
            out_windowed[win_off..win_off + win_seq * dm].copy_from_slice(&win_out_vec);
        }

        // Unpartition: [num_windows, ws*ws, dm] -> [gh*gw, dm]
        let mut unpadded = vec![0.0f32; gh * gw * dm];
        for wh in 0..nwh {
            for ww in 0..nww {
                let win_idx = wh * nww + ww;
                for r in 0..ws {
                    for c in 0..ws {
                        let dst_row = wh * ws + r;
                        let dst_col = ww * ws + c;
                        let src_off = (win_idx * win_seq + r * ws + c) * dm;
                        let dst_off = (dst_row * gw + dst_col) * dm;
                        unpadded[dst_off..dst_off + dm]
                            .copy_from_slice(&out_windowed[src_off..src_off + dm]);
                    }
                }
            }
        }

        // Remove padding
        let mut result = vec![0.0f32; total_seq * dm];
        for r in 0..grid_h {
            let src_off = r * gw * dm;
            let dst_off = r * grid_w * dm;
            result[dst_off..dst_off + grid_w * dm]
                .copy_from_slice(&unpadded[src_off..src_off + grid_w * dm]);
        }

        Tensor::from_vec(result, Shape::new(&[total_seq, dm]))
    }
}

impl<B: MathBackend> Module<B> for SamAttention<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = vec![
            &self.qkv_weight,
            &self.qkv_bias,
            &self.proj_weight,
            &self.proj_bias,
        ];
        params.extend(self.rel_pos.parameters());
        params
    }
}

// ── Transformer block ───────────────────────────────────────────────────────

/// A single SAM `ViT` transformer block (pre-norm).
///
/// `x -> LN -> Attn(windowed or global) -> + -> LN -> MLP -> +`
pub struct SamVitBlock<B: MathBackend> {
    pub ln1_gamma: Tensor<B>,
    pub ln1_beta: Tensor<B>,
    pub attn: SamAttention<B>,
    pub ln2_gamma: Tensor<B>,
    pub ln2_beta: Tensor<B>,
    pub mlp: VitMlp<B>,
    pub d_model: usize,
}

impl<B: MathBackend> SamVitBlock<B> {
    /// Create a zero-initialized block.
    pub fn new(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        window_size: usize,
        use_window: bool,
    ) -> Self {
        Self {
            ln1_gamma: Tensor::from_vec(vec![1.0; d_model], Shape::new(&[d_model])),
            ln1_beta: Tensor::from_vec(vec![0.0; d_model], Shape::new(&[d_model])),
            attn: SamAttention::new(d_model, n_heads, window_size, use_window),
            ln2_gamma: Tensor::from_vec(vec![1.0; d_model], Shape::new(&[d_model])),
            ln2_beta: Tensor::from_vec(vec![0.0; d_model], Shape::new(&[d_model])),
            mlp: VitMlp::new(d_model, d_ff),
            d_model,
        }
    }

    /// Forward pass: `[seq, d_model]` -> `[seq, d_model]`.
    pub fn forward(&self, input: &Tensor<B>, grid_h: usize, grid_w: usize) -> Tensor<B> {
        let shape = input.shape.clone();

        // LN -> Attention -> residual
        let normed = B::layernorm_inference(
            &input.data,
            &self.ln1_gamma.data,
            &self.ln1_beta.data,
            &shape,
            1e-6,
        );
        let attn_out = self
            .attn
            .forward(&Tensor::new(normed, shape.clone()), grid_h, grid_w);
        let x = B::add(&input.data, &attn_out.data, &shape, &shape, &shape);

        // LN -> MLP -> residual
        let normed2 =
            B::layernorm_inference(&x, &self.ln2_gamma.data, &self.ln2_beta.data, &shape, 1e-6);
        let mlp_out = self.mlp.forward(&Tensor::new(normed2, shape.clone()));
        let result = B::add(&x, &mlp_out.data, &shape, &shape, &shape);

        Tensor::new(result, shape)
    }
}

impl<B: MathBackend> Module<B> for SamVitBlock<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = vec![
            &self.ln1_gamma,
            &self.ln1_beta,
            &self.ln2_gamma,
            &self.ln2_beta,
        ];
        params.extend(self.attn.parameters());
        params.push(&self.mlp.fc1_weight);
        params.push(&self.mlp.fc1_bias);
        params.push(&self.mlp.fc2_weight);
        params.push(&self.mlp.fc2_bias);
        params
    }
}

// ── Neck ────────────────────────────────────────────────────────────────────

/// Neck that reduces encoder output from `embed_dim` to `out_channels`.
///
/// `1×1 Conv -> LayerNorm2d -> 3×3 Conv -> LayerNorm2d`
pub struct SamNeck<B: MathBackend> {
    pub conv1: Conv2d<B>,
    pub ln1: LayerNorm2d<B>,
    pub conv2: Conv2d<B>,
    pub ln2: LayerNorm2d<B>,
}

impl<B: MathBackend> SamNeck<B> {
    /// Create a zero-initialized neck.
    pub fn new(embed_dim: usize, out_channels: usize) -> Self {
        Self {
            conv1: Conv2d::new(embed_dim, out_channels, 1, 1, 1, 0),
            ln1: LayerNorm2d::new(out_channels, 1e-6),
            conv2: Conv2d::new(out_channels, out_channels, 3, 3, 1, 1),
            ln2: LayerNorm2d::new(out_channels, 1e-6),
        }
    }

    /// Forward pass: `[embed_dim, H, W]` -> `[out_channels, H, W]`.
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let x = self.conv1.forward(input);
        let x = self.ln1.forward(&x);
        let x = self.conv2.forward(&x);
        self.ln2.forward(&x)
    }
}

impl<B: MathBackend> Module<B> for SamNeck<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = self.conv1.parameters();
        params.extend(self.ln1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.ln2.parameters());
        params
    }
}

// ── Full image encoder ──────────────────────────────────────────────────────

/// SAM image encoder: windowed `ViT` + neck.
///
/// Input: `[3, image_size, image_size]` -> Output: `[out_channels, H_grid, W_grid]`.
pub struct SamVit<B: MathBackend> {
    pub patch_embed: PatchEmbedding<B>,
    /// Absolute positional embeddings: `[H_grid * W_grid, embed_dim]`.
    pub pos_embed: Tensor<B>,
    pub blocks: Vec<SamVitBlock<B>>,
    pub neck: SamNeck<B>,
    pub config: SamConfig,
}

impl<B: MathBackend> SamVit<B> {
    /// Create a zero-initialized encoder from config.
    pub fn new(config: &SamConfig) -> Self {
        let grid = config.num_patches_per_side();
        let num_patches = grid * grid;
        let d_ff = config.d_ff();

        let blocks = (0..config.depth)
            .map(|i| {
                let use_window = !config.global_attn_indices.contains(&i);
                SamVitBlock::new(
                    config.embed_dim,
                    config.num_heads,
                    d_ff,
                    config.window_size,
                    use_window,
                )
            })
            .collect();

        Self {
            patch_embed: PatchEmbedding::new(
                config.in_channels(),
                config.embed_dim,
                config.patch_size,
            ),
            pos_embed: Tensor::from_vec(
                vec![0.0; num_patches * config.embed_dim],
                Shape::new(&[num_patches, config.embed_dim]),
            ),
            blocks,
            neck: SamNeck::new(config.embed_dim, config.out_channels),
            config: config.clone(),
        }
    }

    /// Forward pass: `[3, image_size, image_size]` -> `[out_channels, grid_h, grid_w]`.
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let grid_h = self.config.num_patches_per_side();
        let grid_w = grid_h;
        let dm = self.config.embed_dim;

        // Patch embed: [3, H, W] -> [num_patches, embed_dim]
        let x = self.patch_embed.forward(input);

        // Add positional embeddings
        let shape = x.shape.clone();
        let x = B::add(&x.data, &self.pos_embed.data, &shape, &shape, &shape);

        // Transformer blocks
        let mut x = Tensor::new(x, shape);
        for block in &self.blocks {
            x = block.forward(&x, grid_h, grid_w);
        }

        // Reshape [grid_h * grid_w, embed_dim] -> [embed_dim, grid_h, grid_w]
        let x_vec = x.to_vec();
        let mut reshaped = vec![0.0f32; dm * grid_h * grid_w];
        for r in 0..grid_h {
            for c in 0..grid_w {
                let token_idx = r * grid_w + c;
                for d in 0..dm {
                    reshaped[d * grid_h * grid_w + r * grid_w + c] = x_vec[token_idx * dm + d];
                }
            }
        }

        let feature_map = Tensor::from_vec(reshaped, Shape::new(&[dm, grid_h, grid_w]));

        // Neck: [embed_dim, grid_h, grid_w] -> [out_channels, grid_h, grid_w]
        self.neck.forward(&feature_map)
    }
}

impl<B: MathBackend> Module<B> for SamVit<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = self.patch_embed.parameters();
        params.push(&self.pos_embed);
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.neck.parameters());
        params
    }
}

// ── Safetensors loading ─────────────────────────────────────────────────────

#[cfg(feature = "safetensors")]
impl<B: MathBackend> SamVit<B> {
    /// Load a SAM image encoder from safetensors.
    ///
    /// `prefix` is typically `"image_encoder."`.
    ///
    /// # Key naming (HuggingFace SAM format)
    ///
    /// ```text
    /// {prefix}patch_embed.proj.weight           [embed_dim, 3, P, P]
    /// {prefix}patch_embed.proj.bias             [embed_dim]
    /// {prefix}pos_embed                         [1, grid_h, grid_w, embed_dim]
    /// {prefix}blocks.{i}.norm1.weight           [embed_dim]
    /// {prefix}blocks.{i}.attn.qkv.weight        [3*embed_dim, embed_dim]
    /// {prefix}blocks.{i}.attn.qkv.bias          [3*embed_dim]
    /// {prefix}blocks.{i}.attn.proj.weight        [embed_dim, embed_dim]
    /// {prefix}blocks.{i}.attn.proj.bias          [embed_dim]
    /// {prefix}blocks.{i}.attn.rel_pos_h         [2*win-1, d_head]
    /// {prefix}blocks.{i}.attn.rel_pos_w         [2*win-1, d_head]
    /// {prefix}blocks.{i}.norm2.weight
    /// {prefix}blocks.{i}.mlp.lin1.weight         [d_ff, embed_dim]
    /// {prefix}blocks.{i}.mlp.lin1.bias           [d_ff]
    /// {prefix}blocks.{i}.mlp.lin2.weight         [embed_dim, d_ff]
    /// {prefix}blocks.{i}.mlp.lin2.bias           [embed_dim]
    /// {prefix}neck.0.weight                      [out_ch, embed_dim, 1, 1]
    /// {prefix}neck.0.bias                        [out_ch]
    /// {prefix}neck.1.weight                      [out_ch]  (LayerNorm2d)
    /// {prefix}neck.1.bias                        [out_ch]
    /// {prefix}neck.2.weight                      [out_ch, out_ch, 3, 3]
    /// {prefix}neck.2.bias                        [out_ch]
    /// {prefix}neck.3.weight                      [out_ch]
    /// {prefix}neck.3.bias                        [out_ch]
    /// ```
    pub fn from_safetensors(
        config: &SamConfig,
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
    ) -> crate::error::Result<Self> {
        use crate::checkpoint::{
            load_conv2d_with_bias, load_f32, load_layernorm2d, load_tensor, load_tensor_transposed,
        };

        let d = config.embed_dim;
        let d_ff = config.d_ff();
        let dh = config.d_head();
        let nh = config.num_heads;
        let oc = config.out_channels;
        let p = config.patch_size;
        let in_ch = config.in_channels();
        let patch_len = in_ch * p * p;
        let grid = config.num_patches_per_side();
        let num_patches = grid * grid;

        // ── Patch embedding ──
        // SAM: patch_embed.proj.weight [embed_dim, 3, P, P] → proj [embed_dim, 3*P*P]
        let proj = load_tensor(
            tensors,
            &format!("{prefix}patch_embed.proj.weight"),
            &[d, patch_len],
        )?;
        let patch_bias = load_tensor(tensors, &format!("{prefix}patch_embed.proj.bias"), &[d])?;
        let patch_embed = PatchEmbedding {
            proj,
            bias: patch_bias,
            patch_size: p,
            in_channels: in_ch,
            embed_dim: d,
        };

        // ── Positional embedding ──
        // SAM stores as [1, grid_h, grid_w, embed_dim], we need [num_patches, embed_dim]
        let pos_data = load_f32(tensors, &format!("{prefix}pos_embed"))?;
        let pos_embed = if pos_data.len() == num_patches * d {
            Tensor::from_vec(pos_data, Shape::new(&[num_patches, d]))
        } else if pos_data.len() == 1 * grid * grid * d {
            // Reshape from [1, H, W, D] to [H*W, D]
            let mut reshaped = vec![0.0f32; num_patches * d];
            for i in 0..num_patches {
                for j in 0..d {
                    reshaped[i * d + j] = pos_data[i * d + j];
                }
            }
            Tensor::from_vec(reshaped, Shape::new(&[num_patches, d]))
        } else {
            return Err(crate::error::VisionError::ShapeMismatch {
                name: format!("{prefix}pos_embed"),
                expected: vec![num_patches, d],
                got: vec![pos_data.len()],
            });
        };

        // ── Transformer blocks ──
        let mut blocks = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            let bp = format!("{prefix}blocks.{i}");
            let use_window = !config.global_attn_indices.contains(&i);

            // For global attention blocks, rel_pos table size = grid_h
            // For window attention blocks, rel_pos table size = window_size
            let rel_size = if use_window { config.window_size } else { grid };
            let rel_table_len = 2 * rel_size - 1;

            let ln1_gamma = load_tensor(tensors, &format!("{bp}.norm1.weight"), &[d])?;
            let ln1_beta = load_tensor(tensors, &format!("{bp}.norm1.bias"), &[d])?;

            // QKV: [3*d, d] → [d, 3*d] (transpose)
            let qkv_weight =
                load_tensor_transposed(tensors, &format!("{bp}.attn.qkv.weight"), 3 * d, d)?;
            let qkv_bias = load_tensor(tensors, &format!("{bp}.attn.qkv.bias"), &[3 * d])?;

            let proj_weight =
                load_tensor_transposed(tensors, &format!("{bp}.attn.proj.weight"), d, d)?;
            let proj_bias = load_tensor(tensors, &format!("{bp}.attn.proj.bias"), &[d])?;

            let rel_pos_h = load_tensor(
                tensors,
                &format!("{bp}.attn.rel_pos_h"),
                &[rel_table_len, dh],
            )?;
            let rel_pos_w = load_tensor(
                tensors,
                &format!("{bp}.attn.rel_pos_w"),
                &[rel_table_len, dh],
            )?;

            let attn = SamAttention {
                qkv_weight,
                qkv_bias,
                proj_weight,
                proj_bias,
                rel_pos: RelativePositionBias {
                    rel_pos_h,
                    rel_pos_w,
                    size: rel_size,
                    d_head: dh,
                },
                n_heads: nh,
                d_head: dh,
                d_model: d,
                use_window,
                window_size: config.window_size,
            };

            let ln2_gamma = load_tensor(tensors, &format!("{bp}.norm2.weight"), &[d])?;
            let ln2_beta = load_tensor(tensors, &format!("{bp}.norm2.bias"), &[d])?;

            // MLP: lin1.weight [d_ff, d] → [d, d_ff], lin2.weight [d, d_ff] → [d_ff, d]
            let fc1_weight =
                load_tensor_transposed(tensors, &format!("{bp}.mlp.lin1.weight"), d_ff, d)?;
            let fc1_bias = load_tensor(tensors, &format!("{bp}.mlp.lin1.bias"), &[d_ff])?;
            let fc2_weight =
                load_tensor_transposed(tensors, &format!("{bp}.mlp.lin2.weight"), d, d_ff)?;
            let fc2_bias = load_tensor(tensors, &format!("{bp}.mlp.lin2.bias"), &[d])?;

            let mlp = crate::models::vit::VitMlp {
                fc1_weight,
                fc1_bias,
                fc2_weight,
                fc2_bias,
                d_model: d,
                d_ff,
            };

            blocks.push(SamVitBlock {
                ln1_gamma,
                ln1_beta,
                attn,
                ln2_gamma,
                ln2_beta,
                mlp,
                d_model: d,
            });
        }

        // ── Neck ──
        // neck.0 = Conv2d 1x1, neck.1 = LayerNorm2d, neck.2 = Conv2d 3x3, neck.3 = LayerNorm2d
        let neck_conv1 =
            load_conv2d_with_bias(tensors, &format!("{prefix}neck.0"), d, oc, 1, 1, 1, 0)?;
        let neck_ln1 = load_layernorm2d(tensors, &format!("{prefix}neck.1"), oc, 1e-6)?;
        let neck_conv2 =
            load_conv2d_with_bias(tensors, &format!("{prefix}neck.2"), oc, oc, 3, 3, 1, 1)?;
        let neck_ln2 = load_layernorm2d(tensors, &format!("{prefix}neck.3"), oc, 1e-6)?;

        let neck = SamNeck {
            conv1: neck_conv1,
            ln1: neck_ln1,
            conv2: neck_conv2,
            ln2: neck_ln2,
        };

        Ok(Self {
            patch_embed,
            pos_embed,
            blocks,
            neck,
            config: config.clone(),
        })
    }
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
            depth: 2,
            num_heads: 2,
            mlp_ratio: 2.0,
            global_attn_indices: vec![1],
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
    fn sam_attention_global_output_shape() {
        let attn = SamAttention::<CpuBackend>::new(32, 2, 4, false);
        let input = Tensor::from_vec(vec![0.0; 16 * 32], Shape::new(&[16, 32]));
        let output = attn.forward(&input, 4, 4);
        assert_eq!(output.shape.dims(), &[16, 32]);
    }

    #[test]
    fn sam_attention_windowed_output_shape() {
        let attn = SamAttention::<CpuBackend>::new(32, 2, 2, true);
        let input = Tensor::from_vec(vec![0.0; 16 * 32], Shape::new(&[16, 32]));
        let output = attn.forward(&input, 4, 4);
        assert_eq!(output.shape.dims(), &[16, 32]);
    }

    #[test]
    fn sam_vit_block_output_shape() {
        let block = SamVitBlock::<CpuBackend>::new(32, 2, 64, 2, true);
        let input = Tensor::from_vec(vec![0.0; 16 * 32], Shape::new(&[16, 32]));
        let output = block.forward(&input, 4, 4);
        assert_eq!(output.shape.dims(), &[16, 32]);
    }

    #[test]
    fn sam_neck_output_shape() {
        let neck = SamNeck::<CpuBackend>::new(32, 16);
        let input = Tensor::from_vec(vec![0.0; 32 * 4 * 4], Shape::new(&[32, 4, 4]));
        let output = neck.forward(&input);
        assert_eq!(output.shape.dims(), &[16, 4, 4]);
    }

    #[test]
    fn sam_vit_full_output_shape() {
        let config = tiny_config();
        let encoder = SamVit::<CpuBackend>::new(&config);
        let input = Tensor::from_vec(vec![0.0; 3 * 64 * 64], Shape::new(&[3, 64, 64]));
        let output = encoder.forward(&input);
        // 64/16 = 4 grid size
        assert_eq!(output.shape.dims(), &[16, 4, 4]);
    }

    #[test]
    fn window_partition_unpartition_roundtrip() {
        // Verify windowed attention preserves shape with zero weights
        let attn = SamAttention::<CpuBackend>::new(8, 1, 2, true);
        let input_data: Vec<f32> = (0..16 * 8).map(|i| i as f32 * 0.1).collect();
        let input = Tensor::from_vec(input_data, Shape::new(&[16, 8]));
        let output = attn.forward(&input, 4, 4);
        assert_eq!(output.shape.dims(), &[16, 8]);
    }

    #[test]
    fn relative_position_bias_shape() {
        let rpb = RelativePositionBias::<CpuBackend>::new(4, 8);
        // 2*4-1 = 7 entries in each table
        assert_eq!(rpb.rel_pos_h.shape.dims(), &[7, 8]);
        assert_eq!(rpb.rel_pos_w.shape.dims(), &[7, 8]);
    }
}
