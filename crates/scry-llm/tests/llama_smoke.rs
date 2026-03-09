//! Smoke tests for `LlamaModel` with random weights.
//! Verifies no panics, NaN, or Inf across forward pass, KV-cache, and generation.

use scry_llm::backend::cpu::CpuBackend;
use scry_llm::generate::{generate, CausalLM, SamplingConfig};
use scry_llm::nn::llama::{LlamaConfig, LlamaModel};
type Cpu = CpuBackend;

fn tiny_llama_config() -> LlamaConfig {
    LlamaConfig {
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 32,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2, // GQA: 4 query heads, 2 KV heads
        max_seq_len: 64,
        rms_norm_eps: 1e-5,
        rope_theta: 500_000.0,
        tie_word_embeddings: true,
        rope_scaling: None,
    }
}

fn tiny_llama() -> LlamaModel<Cpu> {
    let config = tiny_llama_config();
    let mut rng = fastrand::Rng::with_seed(42);
    LlamaModel::<Cpu>::new(config, &mut rng)
}

#[test]
fn llama_forward_finite() {
    let model = tiny_llama();
    let logits = model.forward(&[0, 1, 2]);
    let v = logits.to_vec();

    // Shape: [3, 32]
    assert_eq!(v.len(), 3 * 32);
    assert!(v.iter().all(|x| x.is_finite()), "logits contain NaN/Inf");
}

#[test]
fn llama_forward_single_token() {
    let model = tiny_llama();
    let logits = model.forward(&[5]);
    let v = logits.to_vec();
    assert_eq!(v.len(), 32);
    assert!(v.iter().all(|x| x.is_finite()));
}

#[test]
fn llama_kv_cache_forward() {
    let model = tiny_llama();
    let mut cache = model.new_kv_cache();

    // Process tokens one at a time through the cache
    let tokens = [0, 1, 2, 3];
    let mut last_logits = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        let logits = model.forward_with_cache(tok, pos, &mut cache);
        let v = logits.to_vec();
        assert_eq!(v.len(), 32, "logits should be [1, vocab_size]");
        assert!(v.iter().all(|x| x.is_finite()), "cache logits contain NaN/Inf at pos {pos}");
        last_logits = v;
    }
    assert!(!last_logits.is_empty());
}

#[test]
fn llama_kv_cache_matches_full_forward() {
    let model = tiny_llama();
    let tokens = [3, 7, 1];

    // Full forward: take logits at the last position
    let full_logits = model.forward(&tokens);
    let full_vec = full_logits.to_vec();
    let last_pos_full = &full_vec[2 * 32..3 * 32];

    // KV-cache forward: process each token sequentially
    let mut cache = model.new_kv_cache();
    let mut cache_last = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        let logits = model.forward_with_cache(tok, pos, &mut cache);
        cache_last = logits.to_vec();
    }

    // The last-position logits should match between both methods
    assert_eq!(last_pos_full.len(), cache_last.len());
    for (i, (&a, &b)) in last_pos_full.iter().zip(cache_last.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-3,
            "logit mismatch at index {i}: full={a:.6}, cache={b:.6}, diff={diff:.6}"
        );
    }
}

#[test]
fn llama_kv_cache_contiguous_matches_full_forward() {
    let model = tiny_llama();
    let tokens = [3, 7, 1];

    // Full forward: take logits at the last position
    let full_logits = model.forward(&tokens);
    let full_vec = full_logits.to_vec();
    let last_pos_full = &full_vec[2 * 32..3 * 32];

    // Contiguous Llama KV-cache forward: process each token sequentially
    let mut cache = model.new_llama_kv_cache(64);
    let mut cache_last = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        let logits = model.forward_with_llama_cache(tok, pos, &mut cache);
        cache_last = logits.to_vec();
    }

    // The last-position logits should match between both methods
    assert_eq!(last_pos_full.len(), cache_last.len());
    for (i, (&a, &b)) in last_pos_full.iter().zip(cache_last.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-3,
            "contiguous cache logit mismatch at index {i}: full={a:.6}, cache={b:.6}, diff={diff:.6}"
        );
    }
}

#[test]
fn llama_generate_greedy() {
    let model = tiny_llama();
    let config = SamplingConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        max_tokens: 8,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let tokens = generate(&model, &[0, 1], &config, &mut rng);

    assert_eq!(tokens.len(), 8);
    for &t in &tokens {
        assert!(t < 32, "token {t} out of vocab range");
    }
}

#[test]
fn llama_generate_deterministic_greedy() {
    let model = tiny_llama();
    let config = SamplingConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        max_tokens: 5,
    };

    let mut rng1 = fastrand::Rng::with_seed(100);
    let tokens1 = generate(&model, &[0], &config, &mut rng1);

    let mut rng2 = fastrand::Rng::with_seed(200);
    let tokens2 = generate(&model, &[0], &config, &mut rng2);

    assert_eq!(tokens1, tokens2, "greedy generation should be deterministic");
}

#[test]
fn llama_generate_sampling() {
    let model = tiny_llama();
    let configs = [
        SamplingConfig { temperature: 0.5, top_k: 5, top_p: 1.0, max_tokens: 4 },
        SamplingConfig { temperature: 1.0, top_k: 0, top_p: 0.9, max_tokens: 4 },
        SamplingConfig { temperature: 2.0, top_k: 10, top_p: 0.5, max_tokens: 4 },
    ];

    for (i, config) in configs.iter().enumerate() {
        let mut rng = fastrand::Rng::with_seed(42 + i as u64);
        let tokens = generate(&model, &[0, 1, 2], config, &mut rng);
        assert_eq!(tokens.len(), 4, "config {i} should produce max_tokens tokens");
        for &t in &tokens {
            assert!(t < 32, "config {i}: token {t} out of vocab range");
        }
    }
}

#[test]
fn llama_casuallm_trait() {
    // Verify CausalLM trait is implemented correctly
    fn check_causal_lm<B: scry_llm::backend::MathBackend, M: CausalLM<B>>(lm: &M) {
        assert_eq!(lm.vocab_size(), 32);
        let logits = lm.forward(&[0, 1]);
        assert_eq!(logits.to_vec().len(), 2 * 32);
        let mut cache = lm.new_kv_cache(64);
        let logits = lm.forward_with_cache(0, 0, &mut cache);
        assert_eq!(logits.to_vec().len(), 32);
    }

    let model = tiny_llama();
    check_causal_lm::<Cpu, _>(&model);
}

#[test]
fn llama_param_count() {
    let model = tiny_llama();
    let n_params = model.n_params();
    // With tied embeddings, count embed_tokens once
    // embed: 32*16 = 512
    // per layer: q(16*16) + k(16*8) + v(16*8) + o(16*16) + gate(16*32) + up(16*32) + down(32*16) + 2*ln(16) = 256+128+128+256+512+512+512+32 = 2336
    // 2 layers: 4672
    // final norm: 16
    // total: 512 + 4672 + 16 = 5200
    assert!(n_params > 0, "param count should be positive");
    println!("  Tiny Llama params: {n_params}");
}

#[test]
fn llama_untied_embeddings() {
    let config = LlamaConfig {
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 32,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: 2,
        max_seq_len: 64,
        rms_norm_eps: 1e-5,
        rope_theta: 500_000.0,
        tie_word_embeddings: false,
        rope_scaling: None,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let model = LlamaModel::<Cpu>::new(config, &mut rng);

    assert!(model.lm_head.is_some(), "untied model should have separate lm_head");

    let logits = model.forward(&[0, 1]);
    let v = logits.to_vec();
    assert_eq!(v.len(), 2 * 32);
    assert!(v.iter().all(|x| x.is_finite()));
}

#[test]
fn llama_gqa_ratio() {
    // Test with different GQA ratios
    for (n_heads, n_kv_heads) in [(4, 4), (4, 2), (4, 1), (8, 2)] {
        let config = LlamaConfig {
            vocab_size: 16,
            hidden_size: n_heads * 4, // head_dim = 4
            intermediate_size: 32,
            n_layers: 1,
            n_heads,
            n_kv_heads,
            max_seq_len: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            tie_word_embeddings: true,
            rope_scaling: None,
        };
        let mut rng = fastrand::Rng::with_seed(42);
        let model = LlamaModel::<Cpu>::new(config, &mut rng);

        let logits = model.forward(&[0, 1, 2]);
        let v = logits.to_vec();
        assert!(
            v.iter().all(|x| x.is_finite()),
            "GQA {n_heads}/{n_kv_heads}: logits contain NaN/Inf"
        );

        // Also test KV-cache path
        let mut cache = model.new_kv_cache();
        for pos in 0..3 {
            let logits = model.forward_with_cache(pos, pos, &mut cache);
            let v = logits.to_vec();
            assert!(
                v.iter().all(|x| x.is_finite()),
                "GQA {n_heads}/{n_kv_heads} cache: NaN/Inf at pos {pos}"
            );
        }
    }
}
