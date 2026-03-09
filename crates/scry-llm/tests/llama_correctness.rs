//! Llama 3 correctness test against `HuggingFace` reference output.
//!
//! Requires downloading Llama 3.2 1B weights and tokenizer.
//! Run with: `cargo test -p scry-llm --features "safetensors,tokenizer" --test llama_correctness -- --ignored --nocapture`

#[test]
#[ignore = "requires downloaded Llama 3.2 1B weights"]
#[cfg(all(feature = "safetensors", feature = "tokenizer"))]
fn llama_pretrained_logits() {
    use scry_llm::backend::cpu::CpuBackend;
    use scry_llm::nn::llama::{LlamaConfig, LlamaModel};
    use scry_llm::tokenizer::HfTokenizer;

    type Cpu = CpuBackend;

    let fixture_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let model_dir = fixture_dir.join("llama-3.2-1b");

    // Check for model files
    let config_path = model_dir.join("config.json");
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !config_path.exists() || !tokenizer_path.exists() {
        println!("  Skipping: model files not found at {}", model_dir.display());
        println!("  Download with:");
        println!("    huggingface-cli download meta-llama/Llama-3.2-1B --local-dir tests/fixtures/llama-3.2-1b/");
        return;
    }

    // Find safetensors shard files
    let mut shard_paths: Vec<std::path::PathBuf> = std::fs::read_dir(&model_dir)
        .expect("failed to read model dir")
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    shard_paths.sort();

    if shard_paths.is_empty() {
        println!("  Skipping: no .safetensors files in {}", model_dir.display());
        return;
    }

    println!("  Found {} safetensors shard(s)", shard_paths.len());

    // Load config
    let config_str = std::fs::read_to_string(&config_path).expect("failed to read config.json");
    let config_json: serde_json::Value =
        serde_json::from_str(&config_str).expect("failed to parse config.json");
    let config = LlamaConfig::from_hf_config(&config_json).expect("failed to parse LlamaConfig");
    println!("  Config: {}h {}l {}kv vocab={}", config.hidden_size, config.n_layers, config.n_kv_heads, config.vocab_size);

    // Load model
    let shard_refs: Vec<&std::path::Path> = shard_paths.iter().map(std::path::PathBuf::as_path).collect();
    let model =
        LlamaModel::<Cpu>::from_safetensors(config.clone(), &shard_refs).expect("failed to load model");

    // Parameter count
    let n_params = model.n_params();
    println!("  Parameters: {n_params}");
    assert!(
        n_params > 1_000_000_000 && n_params < 1_500_000_000,
        "Expected ~1.2B parameters, got {n_params}"
    );

    // Load tokenizer
    let tokenizer =
        HfTokenizer::from_file(&tokenizer_path).expect("failed to load tokenizer.json");
    println!("  Vocab size: {}", tokenizer.vocab_size());

    // Encode a test prompt (Llama 3 requires BOS token)
    let prompt = "The capital of France is";
    let bos = tokenizer.bos_id().expect("tokenizer should have BOS token");
    let mut token_ids = vec![bos];
    token_ids.extend(tokenizer.encode(prompt));
    println!("  Prompt: {prompt:?} -> {token_ids:?}");

    // Full forward pass
    let logits = model.forward(&token_ids);
    let logits_vec = logits.to_vec();
    let seq_len = token_ids.len();
    let vocab_size = config.vocab_size;
    assert_eq!(logits_vec.len(), seq_len * vocab_size);

    // All logits finite
    assert!(
        logits_vec.iter().all(|v| v.is_finite()),
        "Some logits are NaN or Inf"
    );

    // Logit range check
    let max_logit = logits_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_logit = logits_vec.iter().copied().fold(f32::INFINITY, f32::min);
    println!("  Logit range: [{min_logit:.4}, {max_logit:.4}]");
    assert!(max_logit < 200.0, "Max logit too large: {max_logit}");
    assert!(min_logit > -200.0, "Min logit too negative: {min_logit}");

    // Top-10 at last position
    let last_pos = &logits_vec[(seq_len - 1) * vocab_size..seq_len * vocab_size];
    let mut indexed: Vec<(usize, f32)> = last_pos.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  Top-10 predictions for '{prompt}' -> next:");
    for (rank, &(idx, val)) in indexed.iter().take(10).enumerate() {
        let token_str = tokenizer.decode(&[idx]);
        println!("    #{rank}: token={idx} ({token_str:?}), logit={val:.4}");
    }

    // "Paris" should appear in the top-5 predictions for the 1B model
    let top5_strs: Vec<String> = indexed
        .iter()
        .take(5)
        .map(|&(idx, _)| tokenizer.decode(&[idx]))
        .collect();
    println!("  Top-5 decoded: {top5_strs:?}");
    let paris_in_top5 = top5_strs
        .iter()
        .any(|s| s.trim().to_lowercase().contains("paris"));
    assert!(
        paris_in_top5,
        "Expected 'Paris' in top-5, got {top5_strs:?}"
    );
}

#[test]
#[ignore = "requires downloaded Llama 3.2 1B weights"]
#[cfg(all(feature = "safetensors", feature = "tokenizer"))]
fn llama_pretrained_generate() {
    use scry_llm::backend::cpu::CpuBackend;
    use scry_llm::generate::{generate, SamplingConfig};
    use scry_llm::nn::llama::{LlamaConfig, LlamaModel};
    use scry_llm::tokenizer::HfTokenizer;

    type Cpu = CpuBackend;

    let fixture_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let model_dir = fixture_dir.join("llama-3.2-1b");

    let config_path = model_dir.join("config.json");
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !config_path.exists() || !tokenizer_path.exists() {
        println!("  Skipping: model files not found");
        return;
    }

    let mut shard_paths: Vec<std::path::PathBuf> = std::fs::read_dir(&model_dir)
        .expect("failed to read model dir")
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    shard_paths.sort();

    if shard_paths.is_empty() {
        println!("  Skipping: no safetensors files");
        return;
    }

    let config_str = std::fs::read_to_string(&config_path).expect("failed to read config.json");
    let config_json: serde_json::Value =
        serde_json::from_str(&config_str).expect("failed to parse config.json");
    let config = LlamaConfig::from_hf_config(&config_json).expect("failed to parse LlamaConfig");

    let shard_refs: Vec<&std::path::Path> = shard_paths.iter().map(std::path::PathBuf::as_path).collect();
    let model =
        LlamaModel::<Cpu>::from_safetensors(config, &shard_refs).expect("failed to load model");

    let tokenizer =
        HfTokenizer::from_file(&tokenizer_path).expect("failed to load tokenizer.json");

    let prompt = "The capital of France is";
    let bos = tokenizer.bos_id().expect("tokenizer should have BOS token");
    let mut prompt_tokens = vec![bos];
    prompt_tokens.extend(tokenizer.encode(prompt));
    println!("  Prompt: {prompt:?} -> {prompt_tokens:?}");

    // Greedy generation
    let config = SamplingConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        max_tokens: 20,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let generated = generate(&model, &prompt_tokens, &config, &mut rng);

    let all_tokens: Vec<usize> = prompt_tokens.iter().copied().chain(generated.iter().copied()).collect();
    let output = tokenizer.decode(&all_tokens);
    println!("  Generated: {output:?}");

    // Verify coherent output
    assert!(
        output.to_lowercase().contains("paris"),
        "Expected output to contain 'paris', got: {output:?}"
    );
}
