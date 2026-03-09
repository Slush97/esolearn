//! Memory footprint + cold start benchmark for scry-llm.
//! Run: cargo test -p scry-llm --features "safetensors,tokenizer,cuda,bf16" --test bench_scry_memory --release -- --ignored --nocapture --test-threads=1

#[test]
#[ignore = "requires downloaded Llama 3.2 1B weights + CUDA GPU"]
#[cfg(all(feature = "safetensors", feature = "tokenizer", feature = "cuda", feature = "bf16"))]
fn scry_llm_memory_footprint() {
    use scry_llm::backend::cuda::{init_gpu_bf16, CudaBackend};
    use scry_llm::generate::{generate, SamplingConfig};
    use scry_llm::nn::llama::{LlamaConfig, LlamaModel};
    use scry_llm::tokenizer::HfTokenizer;
    use std::path::PathBuf;
    use std::time::Instant;

    // RSS helper (Linux /proc/self/statm, field 1 = RSS in pages)
    fn rss_mb() -> f64 {
        let statm = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
        let pages: u64 = statm
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        (pages * 4096) as f64 / 1e6
    }

    // nvidia-smi VRAM helper
    fn gpu_vram_mb() -> f64 {
        let output = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"])
            .output()
            .ok();
        output
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<f64>().ok())
            .unwrap_or(0.0)
    }

    let rss_before = rss_mb();
    let vram_before = gpu_vram_mb();

    // Init CUDA
    init_gpu_bf16(0);
    type Gpu = CudaBackend;

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/llama-3.2-1b");
    if !dir.join("config.json").exists() {
        println!("Skipping: model not found");
        return;
    }

    // --- Model load ---
    let t_load = Instant::now();
    let config_str = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let config_json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = LlamaConfig::from_hf_config(&config_json).unwrap();

    let mut shard_paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    shard_paths.sort();
    let shard_refs: Vec<&std::path::Path> = shard_paths.iter().map(PathBuf::as_path).collect();
    let model = LlamaModel::<Gpu>::from_safetensors(config.clone(), &shard_refs).unwrap();
    CudaBackend::synchronize();
    let load_time = t_load.elapsed();

    let rss_after = rss_mb();
    let vram_after = gpu_vram_mb();

    let tokenizer = HfTokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
    let prompt = "The capital of France is";
    let bos = tokenizer.bos_id().unwrap();
    let mut prompt_tokens = vec![bos];
    prompt_tokens.extend(tokenizer.encode(prompt));
    let prompt_len = prompt_tokens.len();

    // --- Time to first token ---
    let t_ttft = Instant::now();
    let logits = model.forward(&prompt_tokens);
    CudaBackend::synchronize();
    let _ = std::hint::black_box(logits.to_vec());
    let ttft = t_ttft.elapsed();

    let vram_peak = gpu_vram_mb();

    // --- Decode throughput ---
    let max_seq = prompt_len + 50;
    let mut cache = model.new_llama_kv_cache(max_seq);
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        model.forward_with_llama_cache(tok, pos, &mut cache);
    }
    CudaBackend::synchronize();

    let n_decode = 20;
    let t_dec = Instant::now();
    let mut last_token = 279;
    for i in 0..n_decode {
        let logits = model.forward_with_llama_cache(last_token, prompt_len + i, &mut cache);
        let v = std::hint::black_box(logits.to_vec());
        last_token = v.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
    }
    CudaBackend::synchronize();
    let decode_time = t_dec.elapsed();
    let tok_s = n_decode as f64 / decode_time.as_secs_f64();

    let rss_peak = rss_mb();

    // --- Full generate ---
    let gen_config = SamplingConfig { temperature: 0.0, top_k: 0, top_p: 1.0, max_tokens: 30 };
    let mut rng = fastrand::Rng::with_seed(42);
    let generated = generate(&model, &prompt_tokens, &gen_config, &mut rng);
    CudaBackend::synchronize();
    let all_tokens: Vec<usize> = prompt_tokens.iter().copied().chain(generated.iter().copied()).collect();
    let output = tokenizer.decode(&all_tokens);

    // --- Print results ---
    println!("\n======================================================================");
    println!("  scry-llm Memory Footprint (Llama 3.2 1B, BF16)");
    println!("======================================================================");
    println!("  Model load:          {:.2}s", load_time.as_secs_f64());
    println!("  RSS before load:     {:.0} MB", rss_before);
    println!("  RSS after load:      {:.0} MB", rss_after);
    println!("  RSS delta:           {:.0} MB", rss_after - rss_before);
    println!("  RSS peak:            {:.0} MB", rss_peak);
    println!("  VRAM before:         {:.0} MB", vram_before);
    println!("  VRAM after load:     {:.0} MB", vram_after);
    println!("  VRAM peak inference: {:.0} MB", vram_peak);
    println!("  Time to 1st token:   {:.3}s", ttft.as_secs_f64());
    println!("  Decode throughput:   {:.1} tok/s", tok_s);
    println!("  Decode latency:      {:.2} ms/tok", decode_time.as_millis() as f64 / n_decode as f64);
    println!("  Params:              {:?}", model.n_params());
    println!("  Output:              {output:?}");
    println!("======================================================================");
}
