// SPDX-License-Identifier: MIT OR Apache-2.0
//! Load a real CLIP ViT-B/32 and compute image embeddings.
//!
//! cargo run -p scry-vision --features safetensors --example embed_clip --release
//!
//! Requires: testdata/clip_vit_b32.safetensors
//!   (from laion/CLIP-ViT-B-32-laion2B-s34B-b79K on HuggingFace)

use std::path::Path;
use std::time::Instant;

use scry_llm::backend::cpu::CpuBackend;

use scry_vision::models::{ClipConfig, ClipEmbedder};
use scry_vision::pipeline::Embed;
use scry_vision::postprocess::embedding::cosine_similarity;

fn main() {
    let model_path = std::env::var("CLIP_MODEL_PATH").unwrap_or_else(|_| {
        format!("{}/testdata/clip_vit_b32.safetensors", env!("CARGO_MANIFEST_DIR"))
    });
    let path = Path::new(&model_path);

    if !path.exists() {
        eprintln!("Model not found at {}", path.display());
        eprintln!("Download with:");
        eprintln!("  curl -L https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_model.safetensors \\");
        eprintln!("    -o crates/scry-vision/testdata/clip_vit_b32.safetensors");
        std::process::exit(1);
    }

    // Load model
    println!("Loading CLIP ViT-B/32 from {}...", path.display());
    let t0 = Instant::now();
    let config = ClipConfig::vit_b32();
    let embedder = ClipEmbedder::<CpuBackend>::from_safetensors(config, path).unwrap();
    println!("  Loaded in {:.0}ms\n", t0.elapsed().as_secs_f64() * 1000.0);

    // Create test images (64x64 synthetic)
    let red = make_solid(64, 64, [220, 50, 30]);
    let orange = make_solid(64, 64, [230, 120, 30]);
    let blue = make_solid(64, 64, [30, 60, 200]);
    let green = make_solid(64, 64, [30, 180, 50]);
    let stripes = make_stripes(64, 64);

    // Compute embeddings
    println!("Computing embeddings...");
    let images: Vec<(&str, &[u8])> = vec![
        ("red", &red),
        ("orange", &orange),
        ("blue", &blue),
        ("green", &green),
        ("stripes", &stripes),
    ];

    let mut embeddings = Vec::new();
    for (name, data) in &images {
        let t0 = Instant::now();
        let emb = embedder.embed(data, 64, 64).unwrap();
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("  {name:>8}: dim={}, norm={:.4}, ({:.0}ms)", emb.len(), norm, elapsed);
        embeddings.push((*name, emb));
    }

    // Similarity matrix
    println!("\n=== Cosine similarity matrix ===\n");
    print!("{:>10}", "");
    for (name, _) in &embeddings {
        print!("{:>10}", name);
    }
    println!();

    for (i, (name_a, emb_a)) in embeddings.iter().enumerate() {
        print!("{:>10}", name_a);
        for (j, (_, emb_b)) in embeddings.iter().enumerate() {
            let sim = cosine_similarity(emb_a, emb_b);
            if i == j {
                print!("{:>10}", "1.000");
            } else {
                print!("{:>10.3}", sim);
            }
        }
        println!();
    }

    // Verify embeddings are meaningful
    println!("\n=== Sanity checks ===\n");
    let sim_red_orange = cosine_similarity(&embeddings[0].1, &embeddings[1].1);
    let sim_red_blue = cosine_similarity(&embeddings[0].1, &embeddings[2].1);
    let sim_red_stripes = cosine_similarity(&embeddings[0].1, &embeddings[4].1);

    println!("  red vs orange:  {:.4} (should be high — warm colors)", sim_red_orange);
    println!("  red vs blue:    {:.4} (should be lower — opposite colors)", sim_red_blue);
    println!("  red vs stripes: {:.4} (should be different — different texture)", sim_red_stripes);

    let all_same = embeddings.windows(2).all(|w| {
        cosine_similarity(&w[0].1, &w[1].1) > 0.999
    });
    if all_same {
        println!("\n  WARNING: all embeddings are nearly identical — model may not be working");
    } else {
        println!("\n  Embeddings are discriminative — CLIP is working!");
    }
}

fn make_solid(w: u32, h: u32, rgb: [u8; 3]) -> Vec<u8> {
    let mut data = vec![0u8; (w * h * 3) as usize];
    for i in 0..(w * h) as usize {
        data[i * 3] = rgb[0];
        data[i * 3 + 1] = rgb[1];
        data[i * 3 + 2] = rgb[2];
    }
    data
}

fn make_stripes(w: u32, h: u32) -> Vec<u8> {
    let mut data = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize * 3;
            if (y / 8) % 2 == 0 {
                data[i] = 200; data[i + 1] = 180; data[i + 2] = 50;
            } else {
                data[i] = 30; data[i + 1] = 60; data[i + 2] = 150;
            }
        }
    }
    data
}
