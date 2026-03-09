use crate::backend::MathBackend;
use crate::tensor::Tensor;

/// Trait for causal language models, decoupling generation from specific architectures.
pub trait CausalLM<B: MathBackend> {
    /// Model-specific KV cache type.
    type Cache;

    /// Full-sequence forward pass (prefill). Returns logits `[seq, vocab]`.
    fn forward(&self, token_ids: &[usize]) -> Tensor<B>;

    /// Single-token forward with KV cache. Returns logits `[1, vocab]`.
    fn forward_with_cache(
        &self,
        token_id: usize,
        pos: usize,
        cache: &mut Self::Cache,
    ) -> Tensor<B>;

    /// Create a new KV cache sized for this model.
    fn new_kv_cache(&self, max_seq: usize) -> Self::Cache;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;
}

/// Configuration for text generation sampling.
#[derive(Clone, Debug)]
pub struct SamplingConfig {
    /// Temperature for logit scaling (0 = greedy, 1 = standard, >1 = more random).
    pub temperature: f32,
    /// Top-k filtering: keep only the k most probable tokens (0 = disabled).
    pub top_k: usize,
    /// Top-p (nucleus) filtering: keep smallest set with cumulative prob >= p (1.0 = disabled).
    pub top_p: f32,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            max_tokens: 128,
        }
    }
}

/// Generate tokens autoregressively from a prompt.
///
/// Uses KV-cache for efficient single-token forward passes.
/// Returns the generated token IDs (not including the prompt).
pub fn generate<B: MathBackend, M: CausalLM<B>>(
    model: &M,
    prompt_tokens: &[usize],
    config: &SamplingConfig,
    rng: &mut fastrand::Rng,
) -> Vec<usize> {
    assert!(
        !prompt_tokens.is_empty(),
        "generate: prompt_tokens must not be empty"
    );

    let max_seq = prompt_tokens.len() + config.max_tokens;
    let mut cache = model.new_kv_cache(max_seq);
    let vocab_size = model.vocab_size();
    let mut generated = Vec::with_capacity(config.max_tokens);

    // Prefill: process prompt tokens through the model
    for (pos, &token_id) in prompt_tokens.iter().enumerate() {
        let logits = model.forward_with_cache(token_id, pos, &mut cache);
        if pos == prompt_tokens.len() - 1 {
            let logits_vec = logits.to_vec();
            let token = sample_token(&logits_vec[..vocab_size], config, rng);
            generated.push(token);
        }
    }

    // Autoregressive generation
    for i in 1..config.max_tokens {
        let last_token = generated[generated.len() - 1];
        let position = prompt_tokens.len() + i - 1;

        let logits = model.forward_with_cache(last_token, position, &mut cache);
        let logits_vec = logits.to_vec();
        let token = sample_token(&logits_vec[..vocab_size], config, rng);
        generated.push(token);
    }

    generated
}

/// Sample a single token from logits using temperature, top-k, and top-p filtering.
pub fn sample_token(logits: &[f32], config: &SamplingConfig, rng: &mut fastrand::Rng) -> usize {
    let n = logits.len();

    // Temperature = 0 or very small: greedy
    if config.temperature < 1e-8 {
        return argmax(logits);
    }

    // Apply temperature scaling
    let mut scaled: Vec<f64> = logits
        .iter()
        .map(|&x| f64::from(x) / f64::from(config.temperature))
        .collect();

    // Top-k filtering
    if config.top_k > 0 && config.top_k < n {
        let mut indexed: Vec<(usize, f64)> = scaled.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = indexed[config.top_k - 1].1;
        for v in &mut scaled {
            if *v < threshold {
                *v = f64::NEG_INFINITY;
            }
        }
    }

    // Softmax
    let max_val = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probs: Vec<f64> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    } else {
        let uniform = 1.0 / n as f64;
        probs.fill(uniform);
    }

    // Top-p (nucleus) filtering
    if config.top_p < 1.0 && config.top_p > 0.0 {
        let mut indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0;
        let mut cutoff_idx = indexed.len();
        for (i, &(_, p)) in indexed.iter().enumerate() {
            cumsum += p;
            if cumsum >= f64::from(config.top_p) {
                cutoff_idx = i + 1;
                break;
            }
        }

        let mut keep = vec![false; n];
        for &(idx, _) in &indexed[..cutoff_idx] {
            keep[idx] = true;
        }
        for (i, p) in probs.iter_mut().enumerate() {
            if !keep[i] {
                *p = 0.0;
            }
        }

        let sum2: f64 = probs.iter().sum();
        if sum2 > 0.0 {
            for p in &mut probs {
                *p /= sum2;
            }
        }
    }

    weighted_sample(&probs, rng)
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i)
}

fn weighted_sample(probs: &[f64], rng: &mut fastrand::Rng) -> usize {
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

// Implement CausalLM for Gpt2Model
impl<B: MathBackend> CausalLM<B> for crate::nn::gpt2::Gpt2Model<B> {
    type Cache = crate::nn::kv_cache::KvCache<B>;

    fn forward(&self, token_ids: &[usize]) -> Tensor<B> {
        self.forward(token_ids)
    }

    fn forward_with_cache(
        &self,
        token_id: usize,
        pos: usize,
        cache: &mut crate::nn::kv_cache::KvCache<B>,
    ) -> Tensor<B> {
        self.forward_with_cache(token_id, pos, cache)
    }

    fn new_kv_cache(&self, _max_seq: usize) -> crate::nn::kv_cache::KvCache<B> {
        self.new_kv_cache()
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}
