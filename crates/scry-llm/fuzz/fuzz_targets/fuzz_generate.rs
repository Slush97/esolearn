//! Fuzz text generation: tiny model, arbitrary prompt tokens + sampling config.
//! Must never panic and all output tokens must be < vocab_size.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::generate::{generate, SamplingConfig};
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};

type Cpu = CpuBackend;

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }

    let vocab_size = 5;

    // Parse sampling config from fuzz bytes
    let temperature = (data[0] as f32) / 50.0; // 0..5.1
    let top_k = (data[1] % 6) as usize; // 0..=5
    let top_p = (data[2] as f32) / 255.0; // 0..=1

    // Prompt tokens
    let prompt_len = (data[3] % 3) as usize + 1; // 1..=3
    let prompt: Vec<usize> = data[4..4 + prompt_len.min(data.len() - 4)]
        .iter()
        .map(|&b| (b as usize) % vocab_size)
        .collect();
    if prompt.is_empty() {
        return;
    }

    let model_config = Gpt2Config {
        vocab_size,
        max_seq_len: 16,
        d_model: 4,
        n_heads: 2,
        n_layers: 1,
        d_ff: 8,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(model_config, &mut rng);

    let config = SamplingConfig {
        temperature,
        top_k,
        top_p,
        max_tokens: 5,
    };

    let mut gen_rng = fastrand::Rng::with_seed(data[5] as u64);
    let tokens = generate(&model, &prompt, &config, &mut gen_rng);

    for &t in &tokens {
        assert!(t < vocab_size, "generated token {t} >= vocab_size {vocab_size}");
    }
});
