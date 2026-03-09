//! Quick throughput measurement for Llama inference.
//! Run: cargo test -p scry-llm --features "safetensors,tokenizer" --test llama_bench --release -- --ignored --nocapture

#[test]
#[ignore = "requires downloaded Llama 3.2 1B weights"]
#[cfg(all(feature = "safetensors", feature = "tokenizer"))]
fn llama_throughput() {
    use scry_llm::backend::cpu::CpuBackend;
    use scry_llm::generate::{generate, SamplingConfig};
    use scry_llm::nn::llama::{LlamaConfig, LlamaModel};
    use scry_llm::tokenizer::HfTokenizer;
    use std::time::Instant;

    type Cpu = CpuBackend;

    let fixture_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let model_dir = fixture_dir.join("llama-3.2-1b");

    let config_path = model_dir.join("config.json");
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !config_path.exists() {
        println!("Skipping: model not found");
        return;
    }

    // Load model
    let t0 = Instant::now();
    let config_str = std::fs::read_to_string(&config_path).unwrap();
    let config_json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = LlamaConfig::from_hf_config(&config_json).unwrap();

    let mut shard_paths: Vec<std::path::PathBuf> = std::fs::read_dir(&model_dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    shard_paths.sort();
    let shard_refs: Vec<&std::path::Path> = shard_paths.iter().map(std::path::PathBuf::as_path).collect();
    let model = LlamaModel::<Cpu>::from_safetensors(config.clone(), &shard_refs).unwrap();
    let load_time = t0.elapsed();
    println!("Model load: {:.2}s", load_time.as_secs_f64());

    let tokenizer = HfTokenizer::from_file(&tokenizer_path).unwrap();

    // Prepare prompt with BOS
    let prompt = "The capital of France is";
    let bos = tokenizer.bos_id().unwrap();
    let mut prompt_tokens = vec![bos];
    prompt_tokens.extend(tokenizer.encode(prompt));
    let prompt_len = prompt_tokens.len();
    println!("Prompt: {prompt:?} ({prompt_len} tokens)");

    // --- Measure prefill (full forward) ---
    let t1 = Instant::now();
    let logits = model.forward(&prompt_tokens);
    let prefill_time = t1.elapsed();
    let _ = logits.to_vec(); // force materialization
    println!("Prefill ({prompt_len} tokens): {:.2}s ({:.1} tok/s)",
        prefill_time.as_secs_f64(),
        prompt_len as f64 / prefill_time.as_secs_f64());

    // --- Measure single-token decode (pre-allocated KV cache path) ---
    let max_seq = prompt_len + 50;
    let mut cache = model.new_llama_kv_cache(max_seq);
    // Prime the cache with prompt tokens one-by-one
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        model.forward_with_llama_cache(tok, pos, &mut cache);
    }

    // Now measure decode speed
    let n_decode = 10;
    let t2 = Instant::now();
    let mut last_token = 279; // " the"
    for i in 0..n_decode {
        let logits = model.forward_with_llama_cache(last_token, prompt_len + i, &mut cache);
        let v = logits.to_vec();
        // Greedy
        last_token = v.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0;
    }
    let decode_time = t2.elapsed();
    let tok_per_sec = n_decode as f64 / decode_time.as_secs_f64();
    println!("Decode ({n_decode} tokens): {:.2}s ({:.2} tok/s)",
        decode_time.as_secs_f64(), tok_per_sec);
    println!("Per-token latency: {:.0}ms", decode_time.as_millis() as f64 / n_decode as f64);

    // --- Full generate() measurement ---
    let gen_config = SamplingConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        max_tokens: 20,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let t3 = Instant::now();
    let generated = generate(&model, &prompt_tokens, &gen_config, &mut rng);
    let gen_time = t3.elapsed();
    let total_tokens = prompt_len + generated.len();
    println!("\ngenerate() ({prompt_len} prompt + {} gen = {total_tokens} total): {:.2}s",
        generated.len(), gen_time.as_secs_f64());
    println!("  Overall: {:.2} tok/s (prompt+gen)",
        total_tokens as f64 / gen_time.as_secs_f64());
    println!("  Effective decode: {:.2} tok/s",
        generated.len() as f64 / gen_time.as_secs_f64());

    let all_tokens: Vec<usize> = prompt_tokens.iter().copied().chain(generated.iter().copied()).collect();
    let output = tokenizer.decode(&all_tokens);
    println!("  Output: {output:?}");
}
