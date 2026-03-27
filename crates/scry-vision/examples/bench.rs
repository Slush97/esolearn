// SPDX-License-Identifier: MIT OR Apache-2.0
//! Benchmark inference throughput for ResNet-18 and CLIP ViT-B/32.
//!
//! cargo run -p scry-vision --features safetensors --example bench --release
//!
//! Requires testdata/resnet18.safetensors and testdata/clip_vit_b32.safetensors

use std::path::Path;
use std::time::Instant;

use scry_llm::backend::cpu::CpuBackend;

use scry_vision::models::{ClipConfig, ClipEmbedder, ResNetClassifier, ResNetConfig};
use scry_vision::pipeline::{Classify, Embed};

fn main() {
    println!("=== scry-vision inference benchmark ===\n");
    println!("Backend: CPU (scry-llm CpuBackend)");
    println!();

    // ── ResNet-18 ──
    let resnet_path = format!("{}/testdata/resnet18.safetensors", env!("CARGO_MANIFEST_DIR"));
    if Path::new(&resnet_path).exists() {
        println!("--- ResNet-18 (ImageNet 1000-class) ---");

        let t0 = Instant::now();
        let config = ResNetConfig::resnet18(1000);
        let classifier =
            ResNetClassifier::<CpuBackend>::from_safetensors(config, Path::new(&resnet_path))
                .unwrap();
        println!("  Load:      {:.0}ms", t0.elapsed().as_secs_f64() * 1000.0);

        let image = vec![128u8; 224 * 224 * 3];

        // Warmup
        let _ = classifier.classify(&image, 224, 224, 5).unwrap();

        // Benchmark
        let n = 10;
        let t0 = Instant::now();
        for _ in 0..n {
            let _ = classifier.classify(&image, 224, 224, 5).unwrap();
        }
        let total = t0.elapsed().as_secs_f64();
        let per_image = total / n as f64 * 1000.0;
        println!("  Inference: {:.1}ms/image ({n} runs, 224x224 RGB)", per_image);
        println!("  Throughput: {:.1} images/sec\n", n as f64 / total);
    } else {
        println!("--- ResNet-18: SKIPPED (testdata/resnet18.safetensors not found) ---\n");
    }

    // ── CLIP ViT-B/32 ──
    let clip_path = format!("{}/testdata/clip_vit_b32.safetensors", env!("CARGO_MANIFEST_DIR"));
    if Path::new(&clip_path).exists() {
        println!("--- CLIP ViT-B/32 (512-dim embedding) ---");

        let t0 = Instant::now();
        let config = ClipConfig::vit_b32();
        let embedder =
            ClipEmbedder::<CpuBackend>::from_safetensors(config, Path::new(&clip_path)).unwrap();
        println!("  Load:      {:.0}ms", t0.elapsed().as_secs_f64() * 1000.0);

        let image = vec![128u8; 224 * 224 * 3];

        // Warmup
        let _ = embedder.embed(&image, 224, 224).unwrap();

        // Benchmark
        let n = 5;
        let t0 = Instant::now();
        for _ in 0..n {
            let _ = embedder.embed(&image, 224, 224).unwrap();
        }
        let total = t0.elapsed().as_secs_f64();
        let per_image = total / n as f64 * 1000.0;
        println!("  Inference: {:.0}ms/image ({n} runs, 224x224 RGB)", per_image);
        println!("  Throughput: {:.2} images/sec\n", n as f64 / total);
    } else {
        println!("--- CLIP ViT-B/32: SKIPPED (testdata/clip_vit_b32.safetensors not found) ---\n");
    }

    // ── Context ──
    println!("--- For reference ---");
    println!("  PyTorch ResNet-18 CPU:        ~30-50ms/image (with MKL)");
    println!("  ONNX Runtime ResNet-18 CPU:   ~10-20ms/image");
    println!("  PyTorch CLIP ViT-B/32 CPU:    ~200-400ms/image");
    println!("  ONNX Runtime CLIP ViT-B/32:   ~100-200ms/image");
}
