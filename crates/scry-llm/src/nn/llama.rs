use crate::backend::MathBackend;
use crate::nn::kv_cache::{KvCache, LayerKvCache, LlamaKvCache, LlamaLayerKvCache};
use crate::nn::rmsnorm::RMSNorm;
use crate::nn::Module;
use crate::ops;
use crate::tensor::shape::Shape;
use crate::tensor::Tensor;

/// Llama model configuration.
/// Llama 3 RoPE scaling configuration.
#[derive(Clone, Debug)]
pub struct RopeScaling {
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub original_max_position_embeddings: usize,
}

#[derive(Clone, Debug)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// Whether lm_head weights are tied to embed_tokens.
    pub tie_word_embeddings: bool,
    /// Llama 3 RoPE frequency scaling. `None` for standard RoPE.
    pub rope_scaling: Option<RopeScaling>,
}

impl LlamaConfig {
    fn llama3_rope_scaling() -> Option<RopeScaling> {
        Some(RopeScaling {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        })
    }

    /// Llama 3.2 1B.
    pub fn llama_1b() -> Self {
        Self {
            vocab_size: 128_256,
            hidden_size: 2048,
            intermediate_size: 8192,
            n_layers: 16,
            n_heads: 32,
            n_kv_heads: 8,
            max_seq_len: 131_072,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            tie_word_embeddings: true,
            rope_scaling: Self::llama3_rope_scaling(),
        }
    }

    /// Llama 3.2 3B.
    pub fn llama_3b() -> Self {
        Self {
            vocab_size: 128_256,
            hidden_size: 3072,
            intermediate_size: 8192,
            n_layers: 28,
            n_heads: 24,
            n_kv_heads: 8,
            max_seq_len: 131_072,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            tie_word_embeddings: true,
            rope_scaling: Self::llama3_rope_scaling(),
        }
    }

    /// Llama 3.1 8B.
    pub fn llama_8b() -> Self {
        Self {
            vocab_size: 128_256,
            hidden_size: 4096,
            intermediate_size: 14_336,
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: 8,
            max_seq_len: 131_072,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            tie_word_embeddings: false,
            rope_scaling: Self::llama3_rope_scaling(),
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.n_heads
    }

    /// Parse from HuggingFace `config.json`.
    #[cfg(feature = "tokenizer")]
    pub fn from_hf_config(json: &serde_json::Value) -> Option<Self> {
        Some(Self {
            vocab_size: json["vocab_size"].as_u64()? as usize,
            hidden_size: json["hidden_size"].as_u64()? as usize,
            intermediate_size: json["intermediate_size"].as_u64()? as usize,
            n_layers: json["num_hidden_layers"].as_u64()? as usize,
            n_heads: json["num_attention_heads"].as_u64()? as usize,
            n_kv_heads: json["num_key_value_heads"].as_u64()? as usize,
            max_seq_len: json
                .get("max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(131_072) as usize,
            rms_norm_eps: json
                .get("rms_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-5) as f32,
            rope_theta: json
                .get("rope_theta")
                .and_then(|v| v.as_f64())
                .unwrap_or(500_000.0) as f32,
            tie_word_embeddings: json
                .get("tie_word_embeddings")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            rope_scaling: json.get("rope_scaling").and_then(|rs| {
                Some(RopeScaling {
                    factor: rs.get("factor")?.as_f64()?,
                    low_freq_factor: rs.get("low_freq_factor")?.as_f64()?,
                    high_freq_factor: rs.get("high_freq_factor")?.as_f64()?,
                    original_max_position_embeddings: rs
                        .get("original_max_position_embeddings")?
                        .as_u64()? as usize,
                })
            }),
        })
    }
}

/// Llama attention with GQA and RoPE.
pub struct LlamaAttention<B: MathBackend> {
    pub q_proj: Tensor<B>,   // [hidden, n_heads * head_dim]
    pub k_proj: Tensor<B>,   // [hidden, n_kv_heads * head_dim]
    pub v_proj: Tensor<B>,   // [hidden, n_kv_heads * head_dim]
    pub o_proj: Tensor<B>,   // [n_heads * head_dim, hidden]
    pub qkv_proj: Tensor<B>, // [hidden, q_dim + k_dim + v_dim] — fused for single-GEMM projection
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Pre-computed RoPE frequencies per dimension pair: `freqs[i]` for pair `(2i, 2i+1)`.
    pub rope_freqs: Vec<f64>,
    /// GPU-resident RoPE frequencies (uploaded once at model load, reused every token).
    pub rope_freqs_device: B::Storage,
}

impl<B: MathBackend> LlamaAttention<B> {
    /// Full-sequence forward (prefill) — batched multi-head attention.
    ///
    /// Uses `matmul_strided_batched` to compute all heads in a single cuBLAS call,
    /// reducing kernel launch overhead from `O(n_heads × n_layers)` to `O(n_layers)`.
    pub fn forward(&self, input: &Tensor<B>, start_pos: usize) -> Tensor<B> {
        let seq_len = input.shape.dims()[0];
        let hidden = input.shape.dims()[1];
        let n_heads = self.n_heads;
        let n_kv = self.n_kv_heads;
        let hd = self.head_dim;

        // Fused QKV projection — 1 matmul instead of 3
        let q_dim = n_heads * hd;
        let k_dim = n_kv * hd;
        let qkv_dim = q_dim + k_dim + k_dim;
        let qkv = ops::matmul(input, &self.qkv_proj, seq_len, hidden, qkv_dim, false, false);

        // Split Q, K, V via column slicing
        let q_data = B::gather_columns(&qkv.data, seq_len, qkv_dim, 0, q_dim);
        let k_data = B::gather_columns(&qkv.data, seq_len, qkv_dim, q_dim, k_dim);
        let v_data = B::gather_columns(&qkv.data, seq_len, qkv_dim, q_dim + k_dim, k_dim);

        // Apply RoPE to all Q and K heads at once (freqs already on GPU — zero upload)
        let q_rope = B::rope_with_freqs_preloaded(&q_data, seq_len, n_heads, hd, start_pos, &self.rope_freqs_device);
        let k_rope = B::rope_with_freqs_preloaded(&k_data, seq_len, n_kv, hd, start_pos, &self.rope_freqs_device);

        // Reshape: [seq, H*d] → [H, seq, d]
        let q_heads = B::reshape_for_heads(&q_rope, 1, seq_len, n_heads, hd);
        let k_heads = B::reshape_for_heads(&k_rope, 1, seq_len, n_kv, hd);
        let v_heads = B::reshape_for_heads(&v_data, 1, seq_len, n_kv, hd);

        // Expand KV for GQA: [n_kv, seq, d] → [n_heads, seq, d]
        let k_expanded = B::repeat_kv(&k_heads, n_kv, n_heads, seq_len, hd);
        let v_expanded = B::repeat_kv(&v_heads, n_kv, n_heads, seq_len, hd);

        // Batched attention: scores = Q @ K^T → [n_heads, seq, seq]
        let mut scores = B::matmul_strided_batched(
            &q_heads, &k_expanded, n_heads, seq_len, hd, seq_len, false, true,
        );

        // Causal mask + scale
        let scale = 1.0 / (hd as f64).sqrt();
        B::apply_batched_causal_mask_and_scale(
            &mut scores, n_heads, seq_len, scale as f32, f32::NEG_INFINITY,
        );

        // Softmax per row: [n_heads * seq, seq]
        let attn = B::softmax(&scores, &Shape::new(&[n_heads * seq_len, seq_len]));

        // Output: attn @ V → [n_heads, seq, d]
        let out = B::matmul_strided_batched(
            &attn, &v_expanded, n_heads, seq_len, seq_len, hd, false, false,
        );

        // Reshape back: [n_heads, seq, d] → [seq, n_heads*d]
        let head_out = B::reshape_from_heads(&out, 1, seq_len, n_heads, hd);

        // Output projection
        let head_tensor = Tensor::new(head_out, Shape::new(&[seq_len, n_heads * hd]));
        ops::matmul(&head_tensor, &self.o_proj, seq_len, n_heads * hd, hidden, false, false)
    }

    /// Single-token forward with KV cache (legacy per-head cache).
    pub fn forward_with_cache(
        &self,
        input: &Tensor<B>,
        pos: usize,
        cache: &mut LayerKvCache<B>,
    ) -> Tensor<B> {
        let hidden = input.shape.dims()[1];
        let n_heads = self.n_heads;
        let n_kv = self.n_kv_heads;
        let hd = self.head_dim;

        // Fused QKV projection — 1 matmul instead of 3
        let q_dim = n_heads * hd;
        let k_dim = n_kv * hd;
        let qkv_dim = q_dim + k_dim + k_dim;
        let qkv = ops::matmul(input, &self.qkv_proj, 1, hidden, qkv_dim, false, false);

        let q_data = B::gather_columns(&qkv.data, 1, qkv_dim, 0, q_dim);
        let k_data = B::gather_columns(&qkv.data, 1, qkv_dim, q_dim, k_dim);
        let v_all_data = B::gather_columns(&qkv.data, 1, qkv_dim, q_dim + k_dim, k_dim);

        // Pre-scale Q by 1/√d fused into RoPE — eliminates a separate scale kernel
        let scale = (1.0 / (hd as f64).sqrt()) as f32;
        let q_scaled = B::rope_with_freqs_scaled_preloaded(&q_data, 1, n_heads, hd, pos, &self.rope_freqs_device, scale);
        let k_rope = B::rope_with_freqs_preloaded(&k_data, 1, n_kv, hd, pos, &self.rope_freqs_device);

        let mut new_k: Vec<B::Storage> = Vec::with_capacity(n_kv);
        let mut new_v: Vec<B::Storage> = Vec::with_capacity(n_kv);
        for kv_h in 0..n_kv {
            let k_h = B::gather_columns(&k_rope, 1, n_kv * hd, kv_h * hd, hd);
            let v_h = B::gather_columns(&v_all_data, 1, n_kv * hd, kv_h * hd, hd);
            new_k.push(k_h);
            new_v.push(v_h);
        }
        cache.append(new_k, new_v);
        let cached_len = cache.seq_len;

        let mut k_stacked = B::zeros(&Shape::new(&[n_kv * cached_len, hd]));
        let mut v_stacked = B::zeros(&Shape::new(&[n_kv * cached_len, hd]));
        for kv_h in 0..n_kv {
            B::scatter_rows(
                &mut k_stacked, &cache.k_per_head[kv_h],
                n_kv * cached_len, hd, kv_h * cached_len, cached_len,
            );
            B::scatter_rows(
                &mut v_stacked, &cache.v_per_head[kv_h],
                n_kv * cached_len, hd, kv_h * cached_len, cached_len,
            );
        }

        let k_expanded = B::repeat_kv(&k_stacked, n_kv, n_heads, cached_len, hd);
        let v_expanded = B::repeat_kv(&v_stacked, n_kv, n_heads, cached_len, hd);
        let q_heads = B::reshape_for_heads(&q_scaled, 1, 1, n_heads, hd);

        // Scores come out pre-scaled since Q was pre-scaled
        let scores = B::matmul_strided_batched(
            &q_heads, &k_expanded, n_heads, 1, hd, cached_len, false, true,
        );
        let attn = B::softmax(&scores, &Shape::new(&[n_heads, cached_len]));
        let out = B::matmul_strided_batched(
            &attn, &v_expanded, n_heads, 1, cached_len, hd, false, false,
        );
        let head_out = B::reshape_from_heads(&out, 1, 1, n_heads, hd);
        let proj = B::matmul(&head_out, &self.o_proj.data, 1, n_heads * hd, hidden, false, false);
        Tensor::new(proj, Shape::new(&[1, hidden]))
    }

    /// Single-token forward with pre-allocated contiguous KV cache.
    ///
    /// Instead of per-head splits + concat, appends `[1, n_kv*hd]` directly
    /// and reads the full cached K/V in 2 operations total (vs 2*n_kv before).
    pub fn forward_with_llama_cache(
        &self,
        input: &Tensor<B>,
        pos: usize,
        cache: &mut LlamaLayerKvCache<B>,
    ) -> Tensor<B> {
        let hidden = input.shape.dims()[1];
        let n_heads = self.n_heads;
        let n_kv = self.n_kv_heads;
        let hd = self.head_dim;

        // Fused QKV projection — 1 matmul instead of 3
        let q_dim = n_heads * hd;
        let k_dim = n_kv * hd;
        let qkv_dim = q_dim + k_dim + k_dim;
        let qkv = ops::matmul(input, &self.qkv_proj, 1, hidden, qkv_dim, false, false);

        // Split Q, K, V via column slicing
        let q_data = B::gather_columns(&qkv.data, 1, qkv_dim, 0, q_dim);
        let k_data = B::gather_columns(&qkv.data, 1, qkv_dim, q_dim, k_dim);
        let v_data = B::gather_columns(&qkv.data, 1, qkv_dim, q_dim + k_dim, k_dim);

        // Apply RoPE — fuse Q pre-scaling (1/√d) into RoPE to eliminate a separate scale kernel
        let scale = (1.0 / (hd as f64).sqrt()) as f32;
        let q_scaled = B::rope_with_freqs_scaled_preloaded(&q_data, 1, n_heads, hd, pos, &self.rope_freqs_device, scale);
        let k_rope = B::rope_with_freqs_preloaded(&k_data, 1, n_kv, hd, pos, &self.rope_freqs_device);

        // Append [1, n_kv*hd] directly — 2 scatter_rows total (vs 2*n_kv before)
        cache.append(&k_rope, &v_data);
        let cached_len = cache.seq_len;

        // Fused gather + reshape + GQA repeat: read directly from pre-allocated cache
        // and produce [n_heads, cached_len, hd] in a single kernel per tensor.
        // Eliminates 4 kernel launches + 4 GPU allocs per layer vs the 3-step path.
        let k_expanded = B::gather_reshape_repeat_kv(
            &cache.k_cache, cache.max_seq_len, cached_len, n_kv, n_heads, hd,
        );
        let v_expanded = B::gather_reshape_repeat_kv(
            &cache.v_cache, cache.max_seq_len, cached_len, n_kv, n_heads, hd,
        );

        // Reshape Q: [1, n_heads*hd] → [n_heads, 1, hd]
        // For seq=1, [1, H*d] and [H, 1, d] have identical memory layout — this is a no-op
        let q_heads = B::reshape_for_heads(&q_scaled, 1, 1, n_heads, hd);

        // Batched attention: scores come out pre-scaled since Q was pre-scaled
        let scores = B::matmul_strided_batched(
            &q_heads, &k_expanded, n_heads, 1, hd, cached_len, false, true,
        );

        // Softmax: [n_heads, cached_len] — no separate scale needed
        let attn = B::softmax(&scores, &Shape::new(&[n_heads, cached_len]));

        // Output: attn @ V → [n_heads, 1, hd]
        let out = B::matmul_strided_batched(
            &attn, &v_expanded, n_heads, 1, cached_len, hd, false, false,
        );

        // Reshape: [n_heads, 1, hd] → [1, n_heads*hd]
        let head_out = B::reshape_from_heads(&out, 1, 1, n_heads, hd);

        // Output projection
        let proj = B::matmul(&head_out, &self.o_proj.data, 1, n_heads * hd, hidden, false, false);
        Tensor::new(proj, Shape::new(&[1, hidden]))
    }
}

impl<B: MathBackend> Module<B> for LlamaAttention<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj]
    }
}

/// Llama MLP with `SwiGLU` activation.
pub struct LlamaMLP<B: MathBackend> {
    pub gate_proj: Tensor<B>, // [hidden, intermediate]
    pub up_proj: Tensor<B>,   // [hidden, intermediate]
    pub down_proj: Tensor<B>, // [intermediate, hidden]
}

impl<B: MathBackend> LlamaMLP<B> {
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let seq = input.shape.dims()[0];
        let hidden = input.shape.dims()[1];
        let inter = self.gate_proj.shape.dims()[1];

        let gate = ops::matmul(input, &self.gate_proj, seq, hidden, inter, false, false);
        let up = ops::matmul(input, &self.up_proj, seq, hidden, inter, false, false);

        // Use fused SwiGLU+bf16 when available to avoid a separate f32→bf16 cast kernel
        #[cfg(feature = "bf16")]
        let activated = {
            let data = B::swiglu_with_bf16(&gate.data, &up.data);
            Tensor::new(data, gate.shape.clone())
        };
        #[cfg(not(feature = "bf16"))]
        let activated = ops::swiglu(&gate, &up);

        ops::matmul(&activated, &self.down_proj, seq, inter, hidden, false, false)
    }
}

impl<B: MathBackend> Module<B> for LlamaMLP<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.gate_proj, &self.up_proj, &self.down_proj]
    }
}

/// Single Llama decoder layer.
pub struct LlamaDecoderLayer<B: MathBackend> {
    pub input_layernorm: RMSNorm<B>,
    pub self_attn: LlamaAttention<B>,
    pub post_attention_layernorm: RMSNorm<B>,
    pub mlp: LlamaMLP<B>,
}

impl<B: MathBackend> LlamaDecoderLayer<B> {
    pub fn forward(&self, input: &Tensor<B>, start_pos: usize) -> Tensor<B> {
        // Pre-norm attention with residual
        let normed = self.input_layernorm.forward(input);
        let attn_out = self.self_attn.forward(&normed, start_pos);
        let residual = ops::add(input, &attn_out);

        // Pre-norm MLP with residual
        let normed2 = self.post_attention_layernorm.forward(&residual);
        let mlp_out = self.mlp.forward(&normed2);
        ops::add(&residual, &mlp_out)
    }

    pub fn forward_with_cache(
        &self,
        input: &Tensor<B>,
        pos: usize,
        cache: &mut LayerKvCache<B>,
    ) -> Tensor<B> {
        let normed = self.input_layernorm.forward(input);
        let attn_out = self.self_attn.forward_with_cache(&normed, pos, cache);
        let residual = ops::add(input, &attn_out);

        let normed2 = self.post_attention_layernorm.forward(&residual);
        let mlp_out = self.mlp.forward(&normed2);
        ops::add(&residual, &mlp_out)
    }

    pub fn forward_with_llama_cache(
        &self,
        input: &Tensor<B>,
        pos: usize,
        cache: &mut LlamaLayerKvCache<B>,
    ) -> Tensor<B> {
        let normed = self.input_layernorm.forward(input);
        let attn_out = self.self_attn.forward_with_llama_cache(&normed, pos, cache);
        let residual = ops::add(input, &attn_out);

        let normed2 = self.post_attention_layernorm.forward(&residual);
        let mlp_out = self.mlp.forward(&normed2);
        ops::add(&residual, &mlp_out)
    }
}

impl<B: MathBackend> Module<B> for LlamaDecoderLayer<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = self.input_layernorm.parameters();
        params.extend(self.self_attn.parameters());
        params.extend(self.post_attention_layernorm.parameters());
        params.extend(self.mlp.parameters());
        params
    }
}

/// Full Llama language model.
pub struct LlamaModel<B: MathBackend> {
    pub config: LlamaConfig,
    pub embed_tokens: Tensor<B>,    // [vocab, hidden]
    pub layers: Vec<LlamaDecoderLayer<B>>,
    pub norm: RMSNorm<B>,
    pub lm_head: Option<Tensor<B>>, // None if tied to embed_tokens
}

impl<B: MathBackend> LlamaModel<B> {
    /// Create a random-initialized model (for testing).
    pub fn new(config: LlamaConfig, rng: &mut fastrand::Rng) -> Self {
        use crate::nn::init;

        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let hd = config.head_dim();
        let std = 0.02;

        let embed_tokens = Tensor::from_vec(
            init::normal_vec(rng, config.vocab_size * h, 0.0, std),
            Shape::new(&[config.vocab_size, h]),
        );

        let mut layers = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            let layer = LlamaDecoderLayer {
                input_layernorm: RMSNorm::new(h),
                self_attn: {
                    let q_dim = config.n_heads * hd;
                    let k_dim = config.n_kv_heads * hd;
                    let v_dim = k_dim;
                    let q_w = init::normal_vec(rng, h * q_dim, 0.0, std);
                    let k_w = init::normal_vec(rng, h * k_dim, 0.0, std);
                    let v_w = init::normal_vec(rng, h * v_dim, 0.0, std);
                    let qkv_w = fuse_qkv_weights(&q_w, &k_w, &v_w, h, q_dim, k_dim, v_dim);
                    {
                        let freqs = compute_rope_freqs(hd, config.rope_theta, config.rope_scaling.as_ref());
                        let freqs_f32: Vec<f32> = freqs.iter().map(|&f| f as f32).collect();
                        let freqs_device = B::from_vec(freqs_f32, &Shape::new(&[hd / 2]));
                        LlamaAttention {
                            q_proj: Tensor::from_vec(q_w, Shape::new(&[h, q_dim])),
                            k_proj: Tensor::from_vec(k_w, Shape::new(&[h, k_dim])),
                            v_proj: Tensor::from_vec(v_w, Shape::new(&[h, v_dim])),
                            o_proj: Tensor::from_vec(
                                init::normal_vec(rng, q_dim * h, 0.0, std),
                                Shape::new(&[q_dim, h]),
                            ),
                            qkv_proj: Tensor::from_vec(qkv_w, Shape::new(&[h, q_dim + k_dim + v_dim])),
                            n_heads: config.n_heads,
                            n_kv_heads: config.n_kv_heads,
                            head_dim: hd,
                            rope_freqs: freqs,
                            rope_freqs_device: freqs_device,
                        }
                    }
                },
                post_attention_layernorm: RMSNorm::new(h),
                mlp: LlamaMLP {
                    gate_proj: Tensor::from_vec(
                        init::normal_vec(rng, h * inter, 0.0, std),
                        Shape::new(&[h, inter]),
                    ),
                    up_proj: Tensor::from_vec(
                        init::normal_vec(rng, h * inter, 0.0, std),
                        Shape::new(&[h, inter]),
                    ),
                    down_proj: Tensor::from_vec(
                        init::normal_vec(rng, inter * h, 0.0, std),
                        Shape::new(&[inter, h]),
                    ),
                },
            };
            layers.push(layer);
        }

        let norm = RMSNorm::new(h);

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(Tensor::from_vec(
                init::normal_vec(rng, config.vocab_size * h, 0.0, std),
                Shape::new(&[config.vocab_size, h]),
            ))
        };

        Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
        }
    }

    /// Full-sequence forward (prefill). Returns logits `[seq, vocab]`.
    pub fn forward(&self, token_ids: &[usize]) -> Tensor<B> {
        let seq = token_ids.len();
        let hidden = self.config.hidden_size;
        let vocab = self.config.vocab_size;

        // Token embedding lookup
        let mut x = ops::embedding(&self.embed_tokens, token_ids, vocab, hidden);

        for layer in &self.layers {
            x = layer.forward(&x, 0);
        }

        x = self.norm.forward(&x);

        // LM head
        let head_weight = self.lm_head.as_ref().unwrap_or(&self.embed_tokens);
        ops::matmul(&x, head_weight, seq, hidden, vocab, false, true)
    }

    /// Create a new KV cache sized for this model (legacy per-head style).
    pub fn new_kv_cache(&self) -> KvCache<B> {
        KvCache::new(self.config.n_layers, self.config.n_kv_heads, self.config.head_dim())
    }

    /// Create a pre-allocated contiguous KV cache for Llama.
    pub fn new_llama_kv_cache(&self, max_seq: usize) -> LlamaKvCache<B> {
        LlamaKvCache::new(
            self.config.n_layers,
            max_seq,
            self.config.n_kv_heads,
            self.config.head_dim(),
        )
    }

    /// Single-token forward with KV cache (legacy). Returns logits `[1, vocab]`.
    pub fn forward_with_cache(
        &self,
        token_id: usize,
        pos: usize,
        cache: &mut KvCache<B>,
    ) -> Tensor<B> {
        let hidden = self.config.hidden_size;
        let vocab = self.config.vocab_size;

        let mut x = ops::embedding(&self.embed_tokens, &[token_id], vocab, hidden);

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward_with_cache(&x, pos, &mut cache.layers[i]);
        }

        x = self.norm.forward(&x);

        let head_weight = self.lm_head.as_ref().unwrap_or(&self.embed_tokens);
        ops::matmul(&x, head_weight, 1, hidden, vocab, false, true)
    }

    /// Single-token forward with pre-allocated Llama KV cache. Returns logits `[1, vocab]`.
    pub fn forward_with_llama_cache(
        &self,
        token_id: usize,
        pos: usize,
        cache: &mut LlamaKvCache<B>,
    ) -> Tensor<B> {
        let hidden = self.config.hidden_size;
        let vocab = self.config.vocab_size;

        let mut x = ops::embedding(&self.embed_tokens, &[token_id], vocab, hidden);

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward_with_llama_cache(&x, pos, &mut cache.layers[i]);
        }

        x = self.norm.forward(&x);

        let head_weight = self.lm_head.as_ref().unwrap_or(&self.embed_tokens);
        ops::matmul(&x, head_weight, 1, hidden, vocab, false, true)
    }

    /// Load from HuggingFace safetensors files.
    #[cfg(feature = "safetensors")]
    #[allow(clippy::too_many_lines)]
    pub fn from_safetensors(
        config: LlamaConfig,
        paths: &[&std::path::Path],
    ) -> crate::error::Result<Self> {
        use crate::error::ScryLlmError;

        // Read all shard files
        let mut all_data: Vec<Vec<u8>> = Vec::new();
        for path in paths {
            let data = std::fs::read(path).map_err(|e| {
                ScryLlmError::WeightLoadError(format!("failed to read {}: {e}", path.display()))
            })?;
            all_data.push(data);
        }

        let mut all_tensors: Vec<safetensors::SafeTensors<'_>> = Vec::new();
        // SAFETY: all_data lives as long as all_tensors since both are in scope
        // We need to use unsafe because SafeTensors borrows the data
        for data in &all_data {
            let tensors = safetensors::SafeTensors::deserialize(data).map_err(|e| {
                ScryLlmError::WeightLoadError(format!("failed to parse safetensors: {e}"))
            })?;
            all_tensors.push(tensors);
        }

        // Returns (f32_data, Option<raw_bf16_bytes>) — the bf16 bytes are passed through
        // when the source dtype is bf16, enabling direct GPU upload without f32→bf16 conversion.
        let load = |name: &str| -> crate::error::Result<(Vec<f32>, Option<Vec<u8>>)> {
            for tensors in &all_tensors {
                if let Ok(t) = tensors.tensor(name) {
                    return match t.dtype() {
                        safetensors::Dtype::F32 => Ok((bytes_to_f32(t.data()), None)),
                        safetensors::Dtype::BF16 => {
                            Ok((bf16_bytes_to_f32(t.data()), Some(t.data().to_vec())))
                        }
                        safetensors::Dtype::F16 => Ok((f16_bytes_to_f32(t.data()), None)),
                        other => Err(ScryLlmError::WeightLoadError(format!(
                            "unsupported dtype {other:?} for tensor '{name}'"
                        ))),
                    };
                }
            }
            Err(ScryLlmError::WeightLoadError(format!(
                "tensor '{name}' not found in any shard"
            )))
        };

        let load_and_transpose =
            |name: &str, rows: usize, cols: usize| -> crate::error::Result<(Vec<f32>, Option<Vec<u8>>)> {
                let (data, bf16_raw) = load(name)?;
                // HF stores as [out, in], we want [in, out]
                let transposed = transpose(&data, rows, cols);
                // bf16 bytes must also be transposed to match the f32 layout
                let bf16_transposed = bf16_raw.map(|raw| transpose_bf16_bytes(&raw, rows, cols));
                Ok((transposed, bf16_transposed))
            };

        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let hd = config.head_dim();

        // Helper: construct a Tensor, using direct bf16 upload when bf16 bytes are available.
        #[cfg(feature = "bf16")]
        let make_tensor = |data: Vec<f32>, bf16_raw: Option<Vec<u8>>, shape: Shape| -> Tensor<B> {
            if let Some(raw) = bf16_raw {
                Tensor::new(B::from_vec_with_bf16(data, &raw, &shape), shape)
            } else {
                Tensor::from_vec(data, shape)
            }
        };
        #[cfg(not(feature = "bf16"))]
        let make_tensor = |data: Vec<f32>, _bf16_raw: Option<Vec<u8>>, shape: Shape| -> Tensor<B> {
            Tensor::from_vec(data, shape)
        };

        let (embed, embed_bf16) = load("model.embed_tokens.weight")?;
        let embed_tokens = make_tensor(embed, embed_bf16, Shape::new(&[config.vocab_size, h]));

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let p = format!("model.layers.{i}");

            let (q_w, q_bf16) = load_and_transpose(
                &format!("{p}.self_attn.q_proj.weight"),
                config.n_heads * hd,
                h,
            )?;
            let (k_w, k_bf16) = load_and_transpose(
                &format!("{p}.self_attn.k_proj.weight"),
                config.n_kv_heads * hd,
                h,
            )?;
            let (v_w, v_bf16) = load_and_transpose(
                &format!("{p}.self_attn.v_proj.weight"),
                config.n_kv_heads * hd,
                h,
            )?;
            let (o_w, o_bf16) = load_and_transpose(
                &format!("{p}.self_attn.o_proj.weight"),
                h,
                config.n_heads * hd,
            )?;

            let (gate_w, gate_bf16) =
                load_and_transpose(&format!("{p}.mlp.gate_proj.weight"), inter, h)?;
            let (up_w, up_bf16) =
                load_and_transpose(&format!("{p}.mlp.up_proj.weight"), inter, h)?;
            let (down_w, down_bf16) =
                load_and_transpose(&format!("{p}.mlp.down_proj.weight"), h, inter)?;

            let (input_ln_w, _) = load(&format!("{p}.input_layernorm.weight"))?;
            let (post_ln_w, _) = load(&format!("{p}.post_attention_layernorm.weight"))?;

            let q_dim = config.n_heads * hd;
            let k_dim = config.n_kv_heads * hd;
            let v_dim = k_dim;
            let qkv_w = fuse_qkv_weights(&q_w, &k_w, &v_w, h, q_dim, k_dim, v_dim);
            let qkv_bf16 = match (&q_bf16, &k_bf16, &v_bf16) {
                (Some(q), Some(k), Some(v)) => Some(fuse_qkv_bf16_bytes(q, k, v, h, q_dim, k_dim, v_dim)),
                _ => None,
            };

            let layer = LlamaDecoderLayer {
                input_layernorm: RMSNorm {
                    weight: Tensor::from_vec(input_ln_w, Shape::new(&[h])),
                    eps: config.rms_norm_eps,
                },
                self_attn: {
                    let freqs = compute_rope_freqs(hd, config.rope_theta, config.rope_scaling.as_ref());
                    let freqs_f32: Vec<f32> = freqs.iter().map(|&f| f as f32).collect();
                    let freqs_device = B::from_vec(freqs_f32, &Shape::new(&[hd / 2]));
                    LlamaAttention {
                        q_proj: make_tensor(q_w, q_bf16, Shape::new(&[h, q_dim])),
                        k_proj: make_tensor(k_w, k_bf16, Shape::new(&[h, k_dim])),
                        v_proj: make_tensor(v_w, v_bf16, Shape::new(&[h, v_dim])),
                        o_proj: make_tensor(o_w, o_bf16, Shape::new(&[q_dim, h])),
                        qkv_proj: make_tensor(qkv_w, qkv_bf16, Shape::new(&[h, q_dim + k_dim + v_dim])),
                        n_heads: config.n_heads,
                        n_kv_heads: config.n_kv_heads,
                        head_dim: hd,
                        rope_freqs: freqs,
                        rope_freqs_device: freqs_device,
                    }
                },
                post_attention_layernorm: RMSNorm {
                    weight: Tensor::from_vec(post_ln_w, Shape::new(&[h])),
                    eps: config.rms_norm_eps,
                },
                mlp: LlamaMLP {
                    gate_proj: make_tensor(gate_w, gate_bf16, Shape::new(&[h, inter])),
                    up_proj: make_tensor(up_w, up_bf16, Shape::new(&[h, inter])),
                    down_proj: make_tensor(down_w, down_bf16, Shape::new(&[inter, h])),
                },
            };
            layers.push(layer);
        }

        let (norm_w, _) = load("model.norm.weight")?;
        let norm = RMSNorm {
            weight: Tensor::from_vec(norm_w, Shape::new(&[h])),
            eps: config.rms_norm_eps,
        };

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            let (lm_w, lm_bf16) = load("lm_head.weight")?;
            // lm_head is [vocab, hidden] — used as logits = hidden @ lm_head^T
            Some(make_tensor(lm_w, lm_bf16, Shape::new(&[config.vocab_size, h])))
        };

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    /// Total parameter count.
    pub fn n_params(&self) -> usize {
        self.parameters().iter().map(|t| t.numel()).sum()
    }
}

impl<B: MathBackend> Module<B> for LlamaModel<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = vec![&self.embed_tokens];
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params.extend(self.norm.parameters());
        if let Some(ref lm_head) = self.lm_head {
            params.push(lm_head);
        }
        params
    }
}

/// Implement `CausalLM` for `LlamaModel` using pre-allocated contiguous KV cache.
impl<B: MathBackend> crate::generate::CausalLM<B> for LlamaModel<B> {
    type Cache = LlamaKvCache<B>;

    fn forward(&self, token_ids: &[usize]) -> Tensor<B> {
        self.forward(token_ids)
    }

    fn forward_with_cache(
        &self,
        token_id: usize,
        pos: usize,
        cache: &mut LlamaKvCache<B>,
    ) -> Tensor<B> {
        self.forward_with_llama_cache(token_id, pos, cache)
    }

    fn new_kv_cache(&self, max_seq: usize) -> LlamaKvCache<B> {
        self.new_llama_kv_cache(max_seq)
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}

// ---- Helpers ----

/// Concatenate Q, K, V weight matrices column-wise into a single `[h, q_dim+k_dim+v_dim]` matrix.
fn fuse_qkv_weights(
    q_w: &[f32], k_w: &[f32], v_w: &[f32],
    h: usize, q_dim: usize, k_dim: usize, v_dim: usize,
) -> Vec<f32> {
    let qkv_dim = q_dim + k_dim + v_dim;
    let mut qkv = vec![0.0f32; h * qkv_dim];
    for row in 0..h {
        qkv[row * qkv_dim..row * qkv_dim + q_dim]
            .copy_from_slice(&q_w[row * q_dim..row * q_dim + q_dim]);
        qkv[row * qkv_dim + q_dim..row * qkv_dim + q_dim + k_dim]
            .copy_from_slice(&k_w[row * k_dim..row * k_dim + k_dim]);
        qkv[row * qkv_dim + q_dim + k_dim..row * qkv_dim + qkv_dim]
            .copy_from_slice(&v_w[row * v_dim..row * v_dim + v_dim]);
    }
    qkv
}

/// Compute pre-scaled RoPE frequencies for each dimension pair.
///
/// With Llama 3 scaling, low-frequency components are scaled by `factor`
/// and high-frequency components are left unchanged, with smooth interpolation.
pub fn compute_rope_freqs(head_dim: usize, theta: f32, scaling: Option<&RopeScaling>) -> Vec<f64> {
    let theta_f64 = f64::from(theta);
    let n_pairs = head_dim / 2;
    let mut freqs = Vec::with_capacity(n_pairs);

    for i in 0..n_pairs {
        let base_freq = 1.0 / theta_f64.powf(2.0 * i as f64 / head_dim as f64);
        freqs.push(base_freq);
    }

    if let Some(sc) = scaling {
        let old_ctx = sc.original_max_position_embeddings as f64;
        let low_freq_wavelen = old_ctx / sc.low_freq_factor;
        let high_freq_wavelen = old_ctx / sc.high_freq_factor;

        for freq in &mut freqs {
            let wavelen = 2.0 * std::f64::consts::PI / *freq;
            let scale_factor = if wavelen < high_freq_wavelen {
                1.0 // High frequency: no scaling
            } else if wavelen > low_freq_wavelen {
                sc.factor // Low frequency: full scaling
            } else {
                // Smooth interpolation
                let smooth = (old_ctx / wavelen - sc.low_freq_factor)
                    / (sc.high_freq_factor - sc.low_freq_factor);
                1.0 / ((1.0 - smooth) / sc.factor + smooth)
            };
            *freq /= scale_factor;
        }
    }

    freqs
}

#[cfg(feature = "safetensors")]
fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Convert bf16 bytes (little-endian) to f32.
/// bf16 is the upper 16 bits of an f32, so we just shift left by 16.
#[cfg(feature = "safetensors")]
fn bf16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f32::from_bits(u32::from(bits) << 16)
        })
        .collect()
}

/// Convert f16 bytes (little-endian) to f32 via IEEE 754 half-precision decoding.
#[cfg(feature = "safetensors")]
fn f16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f16_to_f32(bits)
        })
        .collect()
}

#[cfg(feature = "safetensors")]
fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1F);
    let mant = u32::from(h & 0x3FF);

    if exp == 0 {
        if mant == 0 {
            // Zero
            f32::from_bits(sign)
        } else {
            // Subnormal: convert to normalized f32
            let mut m = mant;
            let mut e: i32 = -14;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let f32_exp = ((e + 127) as u32) << 23;
            f32::from_bits(sign | f32_exp | (m << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits(sign | 0x7F80_0000 | (mant << 13))
    } else {
        // Normalized
        let f32_exp = (exp + 112) << 23; // 112 = 127 - 15
        f32::from_bits(sign | f32_exp | (mant << 13))
    }
}

/// Transpose `[rows, cols]` → `[cols, rows]` (row-major).
#[cfg(feature = "safetensors")]
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Fuse Q, K, V bf16 byte buffers column-wise into `[h, q_dim+k_dim+v_dim]` (2 bytes per element).
#[cfg(feature = "safetensors")]
fn fuse_qkv_bf16_bytes(
    q: &[u8], k: &[u8], v: &[u8],
    h: usize, q_dim: usize, k_dim: usize, v_dim: usize,
) -> Vec<u8> {
    let qkv_dim = q_dim + k_dim + v_dim;
    let mut out = vec![0u8; h * qkv_dim * 2];
    for row in 0..h {
        let dst_off = row * qkv_dim * 2;
        let q_off = row * q_dim * 2;
        let k_off = row * k_dim * 2;
        let v_off = row * v_dim * 2;
        out[dst_off..dst_off + q_dim * 2].copy_from_slice(&q[q_off..q_off + q_dim * 2]);
        out[dst_off + q_dim * 2..dst_off + (q_dim + k_dim) * 2]
            .copy_from_slice(&k[k_off..k_off + k_dim * 2]);
        out[dst_off + (q_dim + k_dim) * 2..dst_off + qkv_dim * 2]
            .copy_from_slice(&v[v_off..v_off + v_dim * 2]);
    }
    out
}

/// Transpose bf16 bytes `[rows, cols]` → `[cols, rows]` (2 bytes per element).
#[cfg(feature = "safetensors")]
fn transpose_bf16_bytes(data: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let mut out = vec![0u8; rows * cols * 2];
    for r in 0..rows {
        for c in 0..cols {
            let src = (r * cols + c) * 2;
            let dst = (c * rows + r) * 2;
            out[dst] = data[src];
            out[dst + 1] = data[src + 1];
        }
    }
    out
}
