//! Tests for text generation.

use scry_llm::backend::cpu::CpuBackend;
use scry_llm::generate::{generate, generate_stream, CancelToken, SamplingConfig};
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

    assert_eq!(
        tokens1, tokens2,
        "greedy generation should be deterministic"
    );
}

#[test]
fn no_panics_tiny_model() {
    let model = tiny_model();

    // Test various sampling configs
    let configs = [
        SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            max_tokens: 3,
        },
        SamplingConfig {
            temperature: 0.5,
            top_k: 3,
            top_p: 1.0,
            max_tokens: 3,
        },
        SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.9,
            max_tokens: 3,
        },
        SamplingConfig {
            temperature: 2.0,
            top_k: 5,
            top_p: 0.5,
            max_tokens: 3,
        },
    ];

    for (i, config) in configs.iter().enumerate() {
        let mut rng = fastrand::Rng::with_seed(42 + i as u64);
        let tokens = generate(&model, &[0], config, &mut rng);
        assert!(!tokens.is_empty(), "config {i} should produce tokens");
        for &t in &tokens {
            assert!(
                t < model.config.vocab_size,
                "config {i}: token {t} >= vocab_size"
            );
        }
    }
}

#[test]
fn stream_matches_blocking_for_same_seed() {
    let model = tiny_model();
    let config = SamplingConfig {
        temperature: 0.7,
        top_k: 0,
        top_p: 1.0,
        max_tokens: 8,
    };
    let prompt = [0, 1, 2];

    let mut rng_blocking = fastrand::Rng::with_seed(7);
    let blocking = generate(&model, &prompt, &config, &mut rng_blocking);

    let mut rng_streaming = fastrand::Rng::with_seed(7);
    let streamed: Vec<usize> = generate_stream(
        &model,
        &prompt,
        config.clone(),
        &mut rng_streaming,
        CancelToken::new(),
    )
    .collect();

    assert_eq!(
        blocking, streamed,
        "stream and blocking must produce identical tokens for the same seed"
    );
}

#[test]
fn cancel_before_first_token_returns_none() {
    let model = tiny_model();
    let config = SamplingConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        max_tokens: 16,
    };
    let mut rng = fastrand::Rng::with_seed(1);
    let cancel = CancelToken::new();
    cancel.cancel();
    let collected: Vec<usize> =
        generate_stream(&model, &[0, 1], config, &mut rng, cancel).collect();
    assert!(
        collected.is_empty(),
        "pre-cancelled stream should yield nothing"
    );
}

#[test]
fn cancel_mid_stream_short_circuits() {
    let model = tiny_model();
    let config = SamplingConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        max_tokens: 32,
    };
    let mut rng = fastrand::Rng::with_seed(2);
    let cancel = CancelToken::new();
    let mut stream = generate_stream(&model, &[0, 1], config, &mut rng, cancel.clone());

    let mut collected = Vec::new();
    for _ in 0..3 {
        let t = stream.next().expect("first 3 tokens must arrive");
        collected.push(t);
    }
    cancel.cancel();
    assert!(stream.next().is_none(), "cancel must terminate the stream");
    assert!(
        stream.next().is_none(),
        "stream stays exhausted post-cancel"
    );
    assert_eq!(
        collected.len(),
        3,
        "exactly the pre-cancel tokens are observed"
    );
}

#[test]
fn iterator_take_yields_n_tokens() {
    let model = tiny_model();
    let config = SamplingConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        max_tokens: 100,
    };
    let mut rng = fastrand::Rng::with_seed(3);
    let stream = generate_stream(&model, &[0], config, &mut rng, CancelToken::new());
    let taken: Vec<usize> = stream.take(4).collect();
    assert_eq!(taken.len(), 4, ".take(4) must yield exactly 4 tokens");
}

#[test]
fn cancel_token_clones_share_state() {
    let token = CancelToken::new();
    let mirror = token.clone();
    assert!(!token.is_cancelled());
    assert!(!mirror.is_cancelled());
    mirror.cancel();
    assert!(token.is_cancelled(), "cancel on clone must propagate");
}
