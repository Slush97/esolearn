//! Tests for text generation.

use scry_llm::backend::cpu::CpuBackend;
use scry_llm::generate::{generate, SamplingConfig};
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};

type Cpu = CpuBackend;

fn tiny_model() -> Gpt2Model<Cpu> {
    let config = Gpt2Config {
        vocab_size: 10,
        max_seq_len: 32,
        d_model: 8,
        n_heads: 2,
        n_layers: 2,
        d_ff: 16,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    Gpt2Model::<Cpu>::new(config, &mut rng)
}

#[test]
fn generate_valid_token_ids() {
    let model = tiny_model();
    let config = SamplingConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        max_tokens: 10,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let tokens = generate(&model, &[0, 1, 2], &config, &mut rng);
    assert!(!tokens.is_empty());
    for &t in &tokens {
        assert!(t < model.config.vocab_size, "token {t} >= vocab_size");
    }
}

#[test]
fn generate_respects_max_tokens() {
    let model = tiny_model();
    let config = SamplingConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        max_tokens: 5,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let tokens = generate(&model, &[0], &config, &mut rng);
    assert_eq!(tokens.len(), 5, "should generate exactly max_tokens tokens");
}

#[test]
fn greedy_with_low_temperature() {
    let model = tiny_model();
    let config = SamplingConfig {
        temperature: 0.0, // greedy
        top_k: 0,
        top_p: 1.0,
        max_tokens: 5,
    };

    // Greedy should be deterministic
    let mut rng1 = fastrand::Rng::with_seed(100);
    let tokens1 = generate(&model, &[0, 1], &config, &mut rng1);

    let mut rng2 = fastrand::Rng::with_seed(200);
    let tokens2 = generate(&model, &[0, 1], &config, &mut rng2);

    assert_eq!(tokens1, tokens2, "greedy generation should be deterministic");
}

#[test]
fn no_panics_tiny_model() {
    let model = tiny_model();

    // Test various sampling configs
    let configs = [
        SamplingConfig { temperature: 0.0, top_k: 0, top_p: 1.0, max_tokens: 3 },
        SamplingConfig { temperature: 0.5, top_k: 3, top_p: 1.0, max_tokens: 3 },
        SamplingConfig { temperature: 1.0, top_k: 0, top_p: 0.9, max_tokens: 3 },
        SamplingConfig { temperature: 2.0, top_k: 5, top_p: 0.5, max_tokens: 3 },
    ];

    for (i, config) in configs.iter().enumerate() {
        let mut rng = fastrand::Rng::with_seed(42 + i as u64);
        let tokens = generate(&model, &[0], config, &mut rng);
        assert!(!tokens.is_empty(), "config {i} should produce tokens");
        for &t in &tokens {
            assert!(t < model.config.vocab_size, "config {i}: token {t} >= vocab_size");
        }
    }
}
