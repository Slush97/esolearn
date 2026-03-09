use scry_llm::backend::MathBackend;
use scry_llm::nn::layernorm::LayerNormModule;
use scry_llm::nn::linear::Linear;
use scry_llm::nn::Module;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use crate::model::attention::{CrossAttention, CrossKvCache};

/// In-place residual add: `dst.data[i] += src.data[i]`.
/// Eliminates a Vec allocation per residual connection.
fn add_inplace<B: MathBackend>(dst: &mut Tensor<B>, src: &Tensor<B>) {
    B::add_inplace(&mut dst.data, &src.data);
}

/// Whisper text decoder.
///
/// Architecture:
///   1. Token embedding + learned positional embedding
///   2. N Transformer decoder blocks (causal self-attention + cross-attention + MLP)
///   3. Final layer norm
///   4. Linear projection to vocabulary logits (tied with token embedding)
///
/// Input: token IDs
/// Output: logits `[seq_len, vocab_size]`
pub struct WhisperDecoder<B: MathBackend> {
    /// Token embedding: `[vocab_size, d_model]`.
    pub token_embedding: Tensor<B>,
    /// Pre-transposed token embedding for logit projection: `[d_model, vocab_size]`.
    /// Avoids `gemv_trans_b` (row-wise dot products that don't auto-vectorize)
    /// in favour of the contiguous-memory `gemv` path (~5x faster).
    pub logit_proj_weight: Tensor<B>,
    /// Learned positional embedding: `[n_text_ctx, d_model]`.
    pub positional_embedding: Tensor<B>,
    /// Decoder blocks.
    pub blocks: Vec<DecoderBlock<B>>,
    /// Final layer norm.
    pub ln: LayerNormModule<B>,
    /// Model dimension.
    pub d_model: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum text context length.
    pub n_text_ctx: usize,
}

/// Single Transformer decoder block.
pub struct DecoderBlock<B: MathBackend> {
    /// Pre-self-attention layer norm.
    pub attn_ln: LayerNormModule<B>,
    /// Causal self-attention.
    pub self_attn: DecoderSelfAttention<B>,
    /// Pre-cross-attention layer norm.
    pub cross_attn_ln: LayerNormModule<B>,
    /// Cross-attention to encoder output.
    pub cross_attn: CrossAttention<B>,
    /// Pre-MLP layer norm.
    pub mlp_ln: LayerNormModule<B>,
    /// MLP first projection.
    pub mlp_fc1: Linear<B>,
    /// MLP second projection.
    pub mlp_fc2: Linear<B>,
}

/// Decoder causal self-attention with KV cache.
pub struct DecoderSelfAttention<B: MathBackend> {
    /// Combined QKV projection: `[d_model, 3 * d_model]`.
    pub qkv_weight: Tensor<B>,
    /// Combined QKV bias: `[3 * d_model]`.
    pub qkv_bias: Tensor<B>,
    /// Output projection: `[d_model, d_model]`.
    pub out_weight: Tensor<B>,
    /// Output bias: `[d_model]`.
    pub out_bias: Tensor<B>,
    /// Number of heads.
    pub n_heads: usize,
    /// Model dimension.
    pub d_model: usize,
    /// Per-head dimension.
    pub d_head: usize,
}

/// KV cache for decoder self-attention (grows with each decode step).
pub struct DecoderKvCache<B: MathBackend> {
    /// Per-layer KV cache. Each entry stores past K,V for one block.
    pub layers: Vec<DecoderLayerKv<B>>,
}

/// Per-layer KV storage for decoder self-attention.
///
/// Keys and values are stored pre-shaped in head-major layout
/// `[n_heads, max_seq, d_head]` so that new entries can be written
/// directly without a per-token reshape transpose.
pub struct DecoderLayerKv<B: MathBackend> {
    /// Cached keys: `[n_heads, max_seq, d_head]` (head-major, pre-shaped).
    pub k: Vec<f32>,
    /// Cached values: `[n_heads, max_seq, d_head]` (head-major, pre-shaped).
    pub v: Vec<f32>,
    /// Current sequence length in cache.
    pub seq_len: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Per-head dimension.
    pub d_head: usize,
    /// Maximum sequence length (pre-allocated capacity).
    pub max_seq: usize,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: MathBackend> DecoderKvCache<B> {
    /// Create a pre-allocated KV cache for `n_layers` decoder blocks.
    ///
    /// Each layer's K/V buffer is pre-allocated to `[n_heads, max_seq, d_head]`
    /// in head-major layout, eliminating per-token reshape transposes.
    pub fn new(n_layers: usize, n_heads: usize, d_head: usize, max_seq: usize) -> Self {
        let buf_size = n_heads * max_seq * d_head;
        let layers = (0..n_layers)
            .map(|_| DecoderLayerKv {
                k: vec![0.0; buf_size],
                v: vec![0.0; buf_size],
                seq_len: 0,
                n_heads,
                d_head,
                max_seq,
                _phantom: std::marker::PhantomData,
            })
            .collect();
        Self { layers }
    }

    /// Reset the cache (for a new audio chunk).
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.seq_len = 0;
        }
    }
}

impl<B: MathBackend> WhisperDecoder<B> {
    /// Transpose a `[rows, cols]` tensor to `[cols, rows]` (row-major).
    pub(crate) fn transpose_2d(src: &Tensor<B>, rows: usize, cols: usize) -> Tensor<B> {
        let data = src.to_vec();
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = data[r * cols + c];
            }
        }
        Tensor::from_vec(out, Shape::new(&[cols, rows]))
    }

    /// Create a new decoder with random initialization.
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        n_layers: usize,
        n_heads: usize,
        n_text_ctx: usize,
        rng: &mut fastrand::Rng,
    ) -> Self {
        let std_dev = 0.02;
        let mut rand_vec = |size: usize| -> Vec<f32> {
            (0..size)
                .map(|_| ((rng.f64() * 2.0 - 1.0) * std_dev) as f32)
                .collect()
        };

        let token_embedding = Tensor::from_vec(
            rand_vec(vocab_size * d_model),
            Shape::new(&[vocab_size, d_model]),
        );
        let logit_proj_weight = Self::transpose_2d(&token_embedding, vocab_size, d_model);
        let positional_embedding = Tensor::from_vec(
            rand_vec(n_text_ctx * d_model),
            Shape::new(&[n_text_ctx, d_model]),
        );

        let blocks = (0..n_layers)
            .map(|_| DecoderBlock::new(d_model, n_heads, rng))
            .collect();

        let ln = LayerNormModule::new(d_model);

        Self {
            token_embedding,
            logit_proj_weight,
            positional_embedding,
            blocks,
            ln,
            d_model,
            vocab_size,
            n_text_ctx,
        }
    }

    /// Run decoder blocks only (no logit projection).
    ///
    /// Updates KV caches but skips the expensive `[1, d_model] @ [vocab_size, d_model]^T`
    /// logit projection. Use for prompt tokens whose logits are discarded.
    pub fn forward_step_blocks_only(
        &self,
        token_id: usize,
        position: usize,
        self_kv_cache: &mut DecoderKvCache<B>,
        cross_kv_caches: &[CrossKvCache<B>],
    ) {
        let x = self.run_blocks(token_id, position, self_kv_cache, cross_kv_caches);
        // x is dropped — we only needed the KV cache side-effects.
        drop(x);
    }

    /// Forward pass for a single decode step with KV cache.
    ///
    /// `token_id`: the current token being decoded.
    /// `position`: the position in the output sequence.
    /// `self_kv_cache`: decoder self-attention KV cache (grows each step).
    /// `cross_kv_caches`: pre-computed encoder KV for each decoder layer.
    ///
    /// Returns logits `[1, vocab_size]`.
    pub fn forward_step(
        &self,
        token_id: usize,
        position: usize,
        self_kv_cache: &mut DecoderKvCache<B>,
        cross_kv_caches: &[CrossKvCache<B>],
    ) -> Tensor<B> {
        self.forward_step_profiled(token_id, position, self_kv_cache, cross_kv_caches, false)
    }

    /// Forward pass for a single decode step with optional sub-block profiling.
    ///
    /// When `profile` is true, prints per-block self-attn/cross-attn/MLP timings.
    pub fn forward_step_profiled(
        &self,
        token_id: usize,
        position: usize,
        self_kv_cache: &mut DecoderKvCache<B>,
        cross_kv_caches: &[CrossKvCache<B>],
        profile: bool,
    ) -> Tensor<B> {
        let (x, blocks_ms) = self.run_blocks_timed(token_id, position, self_kv_cache, cross_kv_caches, profile);

        // Project to vocab logits: x @ logit_proj_weight → [1, vocab_size]
        // Uses pre-transposed weight [d_model, vocab_size] so we hit the fast
        // contiguous-memory gemv_f32 path instead of row-wise gemv_trans_b.
        let t_logit = std::time::Instant::now();
        let logits = scry_llm::ops::matmul(
            &x,
            &self.logit_proj_weight,
            1,
            self.d_model,
            self.vocab_size,
            false,
            false,
        );
        let logit_ms = t_logit.elapsed().as_secs_f64() * 1000.0;

        if profile {
            eprintln!("    blocks={blocks_ms:.2}ms logit_proj={logit_ms:.2}ms");
        }

        logits
    }

    /// Shared blocks + layer norm path.
    fn run_blocks(
        &self,
        token_id: usize,
        position: usize,
        self_kv_cache: &mut DecoderKvCache<B>,
        cross_kv_caches: &[CrossKvCache<B>],
    ) -> Tensor<B> {
        let tok_emb = scry_llm::ops::embedding(
            &self.token_embedding,
            &[token_id],
            self.vocab_size,
            self.d_model,
        );
        let pos = scry_llm::ops::embedding(
            &self.positional_embedding,
            &[position],
            self.n_text_ctx,
            self.d_model,
        );
        let mut x = scry_llm::ops::add(&tok_emb, &pos);

        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward_step(&x, &mut self_kv_cache.layers[i], &cross_kv_caches[i], false);
        }

        self.ln.forward(&x)
    }

    /// Blocks + layer norm with optional timing.
    fn run_blocks_timed(
        &self,
        token_id: usize,
        position: usize,
        self_kv_cache: &mut DecoderKvCache<B>,
        cross_kv_caches: &[CrossKvCache<B>],
        profile: bool,
    ) -> (Tensor<B>, f64) {
        let tok_emb = scry_llm::ops::embedding(
            &self.token_embedding,
            &[token_id],
            self.vocab_size,
            self.d_model,
        );
        let pos = scry_llm::ops::embedding(
            &self.positional_embedding,
            &[position],
            self.n_text_ctx,
            self.d_model,
        );
        let mut x = scry_llm::ops::add(&tok_emb, &pos);

        let t_blocks = std::time::Instant::now();
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward_step(&x, &mut self_kv_cache.layers[i], &cross_kv_caches[i], profile);
        }
        let blocks_ms = if profile {
            t_blocks.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };

        (self.ln.forward(&x), blocks_ms)
    }
}

impl<B: MathBackend> DecoderBlock<B> {
    fn new(d_model: usize, n_heads: usize, rng: &mut fastrand::Rng) -> Self {
        Self {
            attn_ln: LayerNormModule::new(d_model),
            self_attn: DecoderSelfAttention::new(d_model, n_heads, rng),
            cross_attn_ln: LayerNormModule::new(d_model),
            cross_attn: CrossAttention::new(d_model, n_heads, rng),
            mlp_ln: LayerNormModule::new(d_model),
            mlp_fc1: Linear::new(d_model, d_model * 4, rng),
            mlp_fc2: Linear::new(d_model * 4, d_model, rng),
        }
    }

    fn forward_step(
        &self,
        x: &Tensor<B>,
        self_kv: &mut DecoderLayerKv<B>,
        cross_kv: &CrossKvCache<B>,
        profile: bool,
    ) -> Tensor<B> {
        // Causal self-attention with KV cache + residual
        let t0 = std::time::Instant::now();
        let attn_in = self.attn_ln.forward(x);
        let attn_out = self.self_attn.forward_with_cache(&attn_in, self_kv);
        let mut x = scry_llm::ops::add(x, &attn_out);
        let self_attn_us = t0.elapsed().as_micros();

        // Cross-attention with encoder KV cache + residual
        let t1 = std::time::Instant::now();
        let cross_in = self.cross_attn_ln.forward(&x);
        let cross_out = self.cross_attn.forward(&cross_in, cross_kv);
        add_inplace::<B>(&mut x, &cross_out);
        let cross_attn_us = t1.elapsed().as_micros();

        // MLP with residual
        let t2 = std::time::Instant::now();
        let mlp_in = self.mlp_ln.forward(&x);
        let h = self.mlp_fc1.forward(&mlp_in);
        let h = scry_llm::ops::gelu(&h);
        let mlp_out = self.mlp_fc2.forward(&h);
        add_inplace::<B>(&mut x, &mlp_out);
        let mlp_us = t2.elapsed().as_micros();

        if profile {
            eprintln!("      block: self_attn={self_attn_us}us cross_attn={cross_attn_us}us mlp={mlp_us}us");
        }

        x
    }
}

impl<B: MathBackend> DecoderSelfAttention<B> {
    fn new(d_model: usize, n_heads: usize, rng: &mut fastrand::Rng) -> Self {
        let d_head = d_model / n_heads;
        let std_dev = 0.02;
        let mut rand_vec = |size: usize| -> Vec<f32> {
            (0..size)
                .map(|_| ((rng.f64() * 2.0 - 1.0) * std_dev) as f32)
                .collect()
        };

        Self {
            qkv_weight: Tensor::from_vec(
                rand_vec(d_model * 3 * d_model),
                Shape::new(&[d_model, 3 * d_model]),
            ),
            qkv_bias: Tensor::from_vec(vec![0.0; 3 * d_model], Shape::new(&[3 * d_model])),
            out_weight: Tensor::from_vec(
                rand_vec(d_model * d_model),
                Shape::new(&[d_model, d_model]),
            ),
            out_bias: Tensor::from_vec(vec![0.0; d_model], Shape::new(&[d_model])),
            n_heads,
            d_model,
            d_head,
        }
    }

    /// Single-token causal self-attention with KV cache.
    ///
    /// KV cache is stored pre-shaped in `[n_heads, max_seq, d_head]` layout.
    /// New K/V entries are written directly into head-major position — no
    /// per-token reshape transpose needed.
    fn forward_with_cache(
        &self,
        input: &Tensor<B>,
        cache: &mut DecoderLayerKv<B>,
    ) -> Tensor<B> {
        let d_model = self.d_model;
        let n_heads = self.n_heads;
        let d_head = self.d_head;
        let seq_pos = cache.seq_len;
        let max_seq = cache.max_seq;

        // QKV = input @ W_qkv + b_qkv → [1, 3 * d_model] (fused)
        let qkv = scry_llm::ops::matmul_bias(
            input,
            &self.qkv_weight,
            &self.qkv_bias,
            1,
            d_model,
            3 * d_model,
            false,
            false,
        );
        // Move the underlying storage directly — avoids cloning 1152 floats.
        let qkv_vec = qkv.into_vec();

        // Write new K/V directly into pre-shaped cache [n_heads, max_seq, d_head].
        // QKV layout is [d_model | d_model | d_model] where each d_model = n_heads * d_head
        // interleaved as [h0_d0..h0_dH, h1_d0..h1_dH, ...].
        let k_slice = &qkv_vec[d_model..2 * d_model];
        let v_slice = &qkv_vec[2 * d_model..3 * d_model];
        for h in 0..n_heads {
            let cache_offset = h * max_seq * d_head + seq_pos * d_head;
            let head_offset = h * d_head;
            cache.k[cache_offset..cache_offset + d_head]
                .copy_from_slice(&k_slice[head_offset..head_offset + d_head]);
            cache.v[cache_offset..cache_offset + d_head]
                .copy_from_slice(&v_slice[head_offset..head_offset + d_head]);
        }
        cache.seq_len += 1;
        let cached_len = cache.seq_len;

        // Q [1, d_model] → [n_heads, 1, d_head] (identity permutation for seq=1)
        // Reuse the QKV allocation by truncating to Q portion only.
        // Skip reshape_for_heads — when batch=1, seq=1 it's a no-op permutation.
        let mut q_vec = qkv_vec;
        q_vec.truncate(d_model);
        let q_heads = B::from_vec(q_vec, &Shape::new(&[n_heads, d_head]));

        // Build K/V views: extract [n_heads, cached_len, d_head] from the
        // pre-shaped [n_heads, max_seq, d_head] buffer (strided copy of used rows only).
        let k_heads = extract_cached_heads(&cache.k, n_heads, cached_len, d_head, max_seq);
        let v_heads = extract_cached_heads(&cache.v, n_heads, cached_len, d_head, max_seq);

        let k_stor = B::from_vec(k_heads, &Shape::new(&[n_heads * cached_len, d_head]));
        let v_stor = B::from_vec(v_heads, &Shape::new(&[n_heads * cached_len, d_head]));

        // Batched scores = Q_heads @ K_heads^T → [n_heads * 1, cached_len]
        let scores = B::matmul_strided_batched(
            &q_heads, &k_stor, n_heads, 1, d_head, cached_len, false, true,
        );

        // Fused scale + softmax — [n_heads, cached_len]
        let scale = 1.0 / (d_head as f32).sqrt();
        let attn = B::scaled_softmax(
            &scores,
            scale,
            &Shape::new(&[n_heads, cached_len]),
        );

        // Batched out = attn @ V_heads → [n_heads * 1, d_head]
        let out_heads = B::matmul_strided_batched(
            &attn, &v_stor, n_heads, 1, cached_len, d_head, false, false,
        );

        // [n_heads, d_head] = [d_model] — identity permutation for seq=1, skip reshape.
        // Output projection (fused matmul + bias)
        let hc = Tensor::<B>::new(out_heads, Shape::new(&[1, d_model]));
        scry_llm::ops::matmul_bias(
            &hc,
            &self.out_weight,
            &self.out_bias,
            1,
            d_model,
            d_model,
            false,
            false,
        )
    }
}

/// Extract `[n_heads, seq_len, d_head]` from a pre-shaped `[n_heads, max_seq, d_head]` buffer.
///
/// When `seq_len == max_seq` this is a simple clone; otherwise it copies only the
/// used rows from each head's contiguous block, avoiding the O(n²) transpose that
/// `reshape_for_heads_from_host` would perform on the flat `[seq, d_model]` layout.
fn extract_cached_heads(
    buf: &[f32],
    n_heads: usize,
    seq_len: usize,
    d_head: usize,
    max_seq: usize,
) -> Vec<f32> {
    if seq_len == max_seq {
        return buf.to_vec();
    }
    let mut out = Vec::with_capacity(n_heads * seq_len * d_head);
    for h in 0..n_heads {
        let head_start = h * max_seq * d_head;
        let used = &buf[head_start..head_start + seq_len * d_head];
        out.extend_from_slice(used);
    }
    out
}

impl<B: MathBackend> Module<B> for DecoderSelfAttention<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.qkv_weight, &self.qkv_bias, &self.out_weight, &self.out_bias]
    }
}

impl<B: MathBackend> Module<B> for DecoderBlock<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = self.attn_ln.parameters();
        params.extend(self.self_attn.parameters());
        params.extend(self.cross_attn_ln.parameters());
        params.extend(self.cross_attn.parameters());
        params.extend(self.mlp_ln.parameters());
        params.extend(self.mlp_fc1.parameters());
        params.extend(self.mlp_fc2.parameters());
        params
    }
}

impl<B: MathBackend> Module<B> for WhisperDecoder<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = vec![&self.token_embedding, &self.logit_proj_weight, &self.positional_embedding];
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.ln.parameters());
        params
    }
}
