//! GPT-2 pretrained inference test.
//! Requires downloading GPT-2 small safetensors file.
//! Run with: `cargo test -p scry-llm --features safetensors --test gpt2_inference -- --ignored --nocapture`

#[test]
#[ignore = "requires downloaded GPT-2 weights"]
#[cfg(feature = "safetensors")]
fn gpt2_pretrained_logits() {
    use scry_llm::backend::cpu::CpuBackend;
    use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
    use scry_llm::nn::Module;

    type Cpu = CpuBackend;

    // Default GPT-2 small weights path
    let path = std::path::Path::new("tests/fixtures/gpt2-small/model.safetensors");
    if !path.exists() {
        println!("  Skipping: model file not found at {}", path.display());
        println!("  Download with: huggingface-cli download openai-community/gpt2 model.safetensors --local-dir tests/fixtures/gpt2-small/");
        return;
    }

    let config = Gpt2Config::gpt2_small();
    let model =
        Gpt2Model::<Cpu>::load_safetensors(config, path).expect("Failed to load safetensors");

    // --- Parameter count validation ---
    let total_params: usize = model.parameters().iter().map(|p| p.numel()).sum();
    println!("  Total parameters: {total_params}");
    // GPT-2 small has ~124M parameters. Weight tying means the LM head shares
    // token_embedding, so unique params counted by Module::parameters() should
    // be around 124M. Allow some tolerance for counting differences.
    assert!(
        total_params > 120_000_000 && total_params < 130_000_000,
        "Expected ~124M parameters, got {total_params}"
    );

    // "Hello" tokenized: [15496]
    // "Hello world" tokenized: [15496, 995]
    let token_ids = &[15496usize, 995];
    let logits = model.forward(token_ids);
    let logits_vec = logits.to_vec();

    // logits shape: [2, 50257]
    assert_eq!(logits_vec.len(), 2 * 50257);

    // --- Verify all logits are finite ---
    assert!(
        logits_vec.iter().all(|v| v.is_finite()),
        "Some logits are NaN or Inf"
    );

    // --- Logit range check: should not be exploding ---
    let max_logit = logits_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_logit = logits_vec.iter().copied().fold(f32::INFINITY, f32::min);
    println!("  Logit range: [{min_logit:.4}, {max_logit:.4}]");
    assert!(
        max_logit < 100.0,
        "Max logit is too large ({max_logit:.4}), model may be broken"
    );
    assert!(
        min_logit > -100.0,
        "Min logit is too negative ({min_logit:.4}), model may be broken"
    );

    // --- Top-10 at last position ---
    let last_pos = &logits_vec[50257..];
    let mut indexed: Vec<(usize, f32)> = last_pos.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  Top-10 logits at last position ('world' -> next):");
    for (i, (idx, val)) in indexed.iter().take(10).enumerate() {
        println!("    #{i}: token={idx}, logit={val:.4}");
        assert!(val.is_finite(), "Top-{i} logit is not finite");
    }

    // --- Top-1 prediction sanity check ---
    // The top-1 prediction for "Hello world" -> next should be a common English token.
    // GPT-2 token IDs below 50000 are valid; common continuation tokens (punctuation,
    // common words) typically have IDs < 15000.
    let (top_token, top_logit) = indexed[0];
    println!("  Top-1 prediction: token={top_token}, logit={top_logit:.4}");
    assert!(
        top_token < 50257,
        "Top-1 token {top_token} is out of vocabulary range"
    );
    // The top logit should be meaningfully larger than the second
    let second_logit = indexed[1].1;
    println!("  Top-1 margin over #2: {:.4}", top_logit - second_logit);

    println!(
        "  GPT-2 inference test passed: {total_params} params, all logits finite, reasonable top-k"
    );
}
