//! CUDA Llama 3 inference tests — correctness + throughput.
//!
//! Run with:
//!   cargo test -p scry-llm --features "safetensors,tokenizer,cuda" --test llama_cuda --release -- --ignored --nocapture
//!
//! BF16 mode:
//!   cargo test -p scry-llm --features "safetensors,tokenizer,cuda,bf16" --test llama_cuda --release -- --ignored --nocapture

#[cfg(feature = "cuda")]
mod cuda_tests {
    use std::path::PathBuf;

    fn model_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/llama-3.2-1b")
    }

    fn model_available() -> bool {
        let dir = model_dir();
        dir.join("config.json").exists() && dir.join("tokenizer.json").exists()
    }

    #[cfg(all(feature = "safetensors", feature = "tokenizer"))]
    fn load_model_and_tokenizer<B: scry_llm::backend::MathBackend>(
    ) -> Option<(
        scry_llm::nn::llama::LlamaModel<B>,
        scry_llm::nn::llama::LlamaConfig,
        scry_llm::tokenizer::HfTokenizer,
    )> {
        use scry_llm::nn::llama::{LlamaConfig, LlamaModel};
        use scry_llm::tokenizer::HfTokenizer;

        let dir = model_dir();
        if !model_available() {
            println!("  Skipping: model files not found at {}", dir.display());
            return None;
        }

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

        let shard_refs: Vec<&std::path::Path> =
            shard_paths.iter().map(PathBuf::as_path).collect();

        let t0 = std::time::Instant::now();
        let model = LlamaModel::<B>::from_safetensors(config.clone(), &shard_refs).unwrap();
        println!("  Model load: {:.2}s", t0.elapsed().as_secs_f64());

        let tokenizer = HfTokenizer::from_file(&dir.join("tokenizer.json")).unwrap();

        Some((model, config, tokenizer))
    }

    /// CUDA f32 correctness: verify GPU logits match CPU reference (top-5 contains "Paris").
    #[test]
    #[ignore = "requires downloaded Llama 3.2 1B weights + CUDA GPU"]
    #[cfg(all(feature = "safetensors", feature = "tokenizer"))]
    fn llama_cuda_correctness() {
        use scry_llm::backend::cuda::{init_gpu, CudaBackend};

        init_gpu(0);
        type Gpu = CudaBackend;

        let Some((model, _config, tokenizer)) = load_model_and_tokenizer::<Gpu>() else {
            return;
        };

        let prompt = "The capital of France is";
        let bos = tokenizer.bos_id().unwrap();
        let mut token_ids = vec![bos];
        token_ids.extend(tokenizer.encode(prompt));
        println!("  Prompt: {prompt:?} -> {token_ids:?}");

        // Full forward pass on GPU
        let t0 = std::time::Instant::now();
        let logits = model.forward(&token_ids);
        CudaBackend::synchronize();
        let fwd_time = t0.elapsed();
        println!("  GPU forward: {:.3}s", fwd_time.as_secs_f64());

        let logits_vec = logits.to_vec();
        let seq_len = token_ids.len();
        let vocab_size = model.config.vocab_size;
        assert_eq!(logits_vec.len(), seq_len * vocab_size);

        // All logits finite
        assert!(
            logits_vec.iter().all(|v| v.is_finite()),
            "Some logits are NaN or Inf"
        );

        // Top-10 at last position
        let last_pos = &logits_vec[(seq_len - 1) * vocab_size..seq_len * vocab_size];
        let mut indexed: Vec<(usize, f32)> = last_pos.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("  Top-10 GPU predictions for '{prompt}' -> next:");
        for (rank, &(idx, val)) in indexed.iter().take(10).enumerate() {
            let token_str = tokenizer.decode(&[idx]);
            println!("    #{rank}: token={idx} ({token_str:?}), logit={val:.4}");
        }

        // "Paris" must be in top-5
        let top5_strs: Vec<String> = indexed
            .iter()
            .take(5)
            .map(|&(idx, _)| tokenizer.decode(&[idx]))
            .collect();
        let paris_in_top5 = top5_strs
            .iter()
            .any(|s| s.trim().to_lowercase().contains("paris"));
        assert!(
            paris_in_top5,
            "Expected 'Paris' in top-5, got {top5_strs:?}"
        );
        println!("  PASS: 'Paris' in top-5 on GPU");
    }

    /// CUDA f32 throughput: measure prefill + decode tok/s.
    #[test]
    #[ignore = "requires downloaded Llama 3.2 1B weights + CUDA GPU"]
    #[cfg(all(feature = "safetensors", feature = "tokenizer"))]
    fn llama_cuda_throughput() {
        use scry_llm::backend::cuda::{init_gpu, CudaBackend};
        use scry_llm::generate::{generate, SamplingConfig};
        use std::time::Instant;

        init_gpu(0);
        type Gpu = CudaBackend;

        let Some((model, _config, tokenizer)) = load_model_and_tokenizer::<Gpu>() else {
            return;
        };

        let prompt = "The capital of France is";
        let bos = tokenizer.bos_id().unwrap();
        let mut prompt_tokens = vec![bos];
        prompt_tokens.extend(tokenizer.encode(prompt));
        let prompt_len = prompt_tokens.len();
        println!("  Prompt: {prompt:?} ({prompt_len} tokens)");

        // --- Prefill throughput ---
        let t1 = Instant::now();
        let logits = model.forward(&prompt_tokens);
        CudaBackend::synchronize();
        let prefill_time = t1.elapsed();
        let _ = std::hint::black_box(logits.to_vec());
        println!(
            "  Prefill ({prompt_len} tokens): {:.3}s ({:.1} tok/s)",
            prefill_time.as_secs_f64(),
            prompt_len as f64 / prefill_time.as_secs_f64()
        );

        // --- Decode throughput (pre-allocated contiguous KV cache) ---
        let max_seq = prompt_len + 50;
        let mut cache = model.new_llama_kv_cache(max_seq);
        // Prime cache with prompt
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            model.forward_with_llama_cache(tok, pos, &mut cache);
        }
        CudaBackend::synchronize();

        let n_decode = 20;
        let t2 = Instant::now();
        let mut last_token = 279; // " the"
        for i in 0..n_decode {
            let logits = model.forward_with_llama_cache(last_token, prompt_len + i, &mut cache);
            let v = std::hint::black_box(logits.to_vec());
            last_token = v
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
        }
        CudaBackend::synchronize();
        let decode_time = t2.elapsed();
        let tok_per_sec = n_decode as f64 / decode_time.as_secs_f64();
        println!(
            "  Decode ({n_decode} tokens): {:.3}s ({:.2} tok/s)",
            decode_time.as_secs_f64(),
            tok_per_sec
        );
        println!(
            "  Per-token latency: {:.1}ms",
            decode_time.as_millis() as f64 / n_decode as f64
        );

        // --- Full generate() ---
        let gen_config = SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            max_tokens: 30,
        };
        let mut rng = fastrand::Rng::with_seed(42);
        let t3 = Instant::now();
        let generated = generate(&model, &prompt_tokens, &gen_config, &mut rng);
        CudaBackend::synchronize();
        let gen_time = t3.elapsed();

        let total_tokens = prompt_len + generated.len();
        println!(
            "\n  generate() ({prompt_len} prompt + {} gen = {total_tokens} total): {:.3}s",
            generated.len(),
            gen_time.as_secs_f64()
        );
        println!(
            "  Overall: {:.2} tok/s",
            total_tokens as f64 / gen_time.as_secs_f64()
        );
        println!(
            "  Effective decode: {:.2} tok/s",
            generated.len() as f64 / gen_time.as_secs_f64()
        );

        let all_tokens: Vec<usize> = prompt_tokens
            .iter()
            .copied()
            .chain(generated.iter().copied())
            .collect();
        let output = tokenizer.decode(&all_tokens);
        println!("  Output: {output:?}");

        // Sanity: output should mention Paris
        assert!(
            output.to_lowercase().contains("paris"),
            "Expected output to mention 'paris', got: {output:?}"
        );
    }

    /// CUDA BF16 correctness: verify bf16 mixed-precision gives same top-5.
    #[test]
    #[ignore = "requires downloaded Llama 3.2 1B weights + CUDA GPU with BF16"]
    #[cfg(all(feature = "safetensors", feature = "tokenizer", feature = "bf16"))]
    fn llama_cuda_bf16_correctness() {
        use scry_llm::backend::cuda::{init_gpu_bf16, CudaBackend};

        init_gpu_bf16(0);
        type Gpu = CudaBackend;

        let Some((model, _config, tokenizer)) = load_model_and_tokenizer::<Gpu>() else {
            return;
        };

        let prompt = "The capital of France is";
        let bos = tokenizer.bos_id().unwrap();
        let mut token_ids = vec![bos];
        token_ids.extend(tokenizer.encode(prompt));
        println!("  Prompt: {prompt:?} -> {token_ids:?}");

        let t0 = std::time::Instant::now();
        let logits = model.forward(&token_ids);
        CudaBackend::synchronize();
        let fwd_time = t0.elapsed();
        println!("  GPU BF16 forward: {:.3}s", fwd_time.as_secs_f64());

        let logits_vec = logits.to_vec();
        let vocab_size = model.config.vocab_size;
        let seq_len = token_ids.len();

        assert!(
            logits_vec.iter().all(|v| v.is_finite()),
            "BF16: some logits are NaN or Inf"
        );

        let last_pos = &logits_vec[(seq_len - 1) * vocab_size..seq_len * vocab_size];
        let mut indexed: Vec<(usize, f32)> = last_pos.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("  Top-10 BF16 predictions:");
        for (rank, &(idx, val)) in indexed.iter().take(10).enumerate() {
            let token_str = tokenizer.decode(&[idx]);
            println!("    #{rank}: token={idx} ({token_str:?}), logit={val:.4}");
        }

        let top5_strs: Vec<String> = indexed
            .iter()
            .take(5)
            .map(|&(idx, _)| tokenizer.decode(&[idx]))
            .collect();
        let paris_in_top5 = top5_strs
            .iter()
            .any(|s| s.trim().to_lowercase().contains("paris"));
        assert!(
            paris_in_top5,
            "BF16: Expected 'Paris' in top-5, got {top5_strs:?}"
        );
        println!("  PASS: BF16 'Paris' in top-5");
    }

    /// CUDA BF16 throughput measurement.
    #[test]
    #[ignore = "requires downloaded Llama 3.2 1B weights + CUDA GPU with BF16"]
    #[cfg(all(feature = "safetensors", feature = "tokenizer", feature = "bf16"))]
    fn llama_cuda_bf16_throughput() {
        use scry_llm::backend::cuda::{init_gpu_bf16, CudaBackend};
        use scry_llm::generate::{generate, SamplingConfig};
        use std::time::Instant;

        init_gpu_bf16(0);
        type Gpu = CudaBackend;

        let Some((model, _config, tokenizer)) = load_model_and_tokenizer::<Gpu>() else {
            return;
        };

        let prompt = "The capital of France is";
        let bos = tokenizer.bos_id().unwrap();
        let mut prompt_tokens = vec![bos];
        prompt_tokens.extend(tokenizer.encode(prompt));
        let prompt_len = prompt_tokens.len();

        // Prefill
        let t1 = Instant::now();
        let logits = model.forward(&prompt_tokens);
        CudaBackend::synchronize();
        let prefill_time = t1.elapsed();
        let _ = std::hint::black_box(logits.to_vec());
        println!(
            "  BF16 Prefill ({prompt_len} tokens): {:.3}s ({:.1} tok/s)",
            prefill_time.as_secs_f64(),
            prompt_len as f64 / prefill_time.as_secs_f64()
        );

        // Decode (pre-allocated contiguous KV cache)
        let max_seq = prompt_len + 50;
        let mut cache = model.new_llama_kv_cache(max_seq);
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            model.forward_with_llama_cache(tok, pos, &mut cache);
        }
        CudaBackend::synchronize();

        let n_decode = 20;
        let t2 = Instant::now();
        let mut last_token = 279;
        for i in 0..n_decode {
            let logits = model.forward_with_llama_cache(last_token, prompt_len + i, &mut cache);
            let v = std::hint::black_box(logits.to_vec());
            last_token = v
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
        }
        CudaBackend::synchronize();
        let decode_time = t2.elapsed();
        println!(
            "  BF16 Decode ({n_decode} tokens): {:.3}s ({:.2} tok/s)",
            decode_time.as_secs_f64(),
            n_decode as f64 / decode_time.as_secs_f64()
        );
        println!(
            "  BF16 Per-token latency: {:.1}ms",
            decode_time.as_millis() as f64 / n_decode as f64
        );

        // Full generate
        let gen_config = SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            max_tokens: 30,
        };
        let mut rng = fastrand::Rng::with_seed(42);
        let t3 = Instant::now();
        let generated = generate(&model, &prompt_tokens, &gen_config, &mut rng);
        CudaBackend::synchronize();
        let gen_time = t3.elapsed();

        println!(
            "  BF16 generate() ({} gen tokens): {:.3}s ({:.2} tok/s effective decode)",
            generated.len(),
            gen_time.as_secs_f64(),
            generated.len() as f64 / gen_time.as_secs_f64()
        );

        let all_tokens: Vec<usize> = prompt_tokens
            .iter()
            .copied()
            .chain(generated.iter().copied())
            .collect();
        println!("  Output: {:?}", tokenizer.decode(&all_tokens));
    }
}
