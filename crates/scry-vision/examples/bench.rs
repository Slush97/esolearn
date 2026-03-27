// SPDX-License-Identifier: MIT OR Apache-2.0
//! Benchmark inference throughput for ResNet-18 and CLIP ViT-B/32.
//!
//! Best results with BLAS enabled and thread contention avoided:
//!
//!   OPENBLAS_NUM_THREADS=1 cargo run -p scry-vision \
//!     --features safetensors,blas --example bench --release
//!
//! Requires testdata/resnet18.safetensors and testdata/clip_vit_b32.safetensors

use std::path::Path;
use std::time::Instant;

use scry_llm::backend::cpu::CpuBackend;

use scry_vision::models::{ClipConfig, ClipEmbedder, ResNetClassifier, ResNetConfig};
use scry_vision::pipeline::{Classify, Embed};

fn bench<F: FnMut()>(warmup: usize, runs: usize, mut f: F) -> (f64, f64) {
    for _ in 0..warmup {
        f();
    }
    // Take the median of `runs` individual timings for stability.
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[runs / 2];
    let per_call = median * 1000.0;
    let throughput = 1.0 / median;
    (per_call, throughput)
}

fn main() {
    let blas = cfg!(any(feature = "blas", feature = "mkl"));
    let blas_label = if cfg!(feature = "mkl") {
        "MKL"
    } else if cfg!(feature = "blas") {
        "OpenBLAS"
    } else {
        "none (pure Rust)"
    };

    println!("=== scry-vision inference benchmark ===\n");
    println!("BLAS:    {blas_label}");
    println!("Threads: OPENBLAS_NUM_THREADS={}, RAYON_NUM_THREADS={}",
        std::env::var("OPENBLAS_NUM_THREADS").unwrap_or_else(|_| "auto".into()),
        rayon::current_num_threads());
    if blas {
        println!("Tip:     Set OPENBLAS_NUM_THREADS=1 to avoid rayon/BLAS contention");
    } else {
        println!("Tip:     Enable --features blas for 5-17x speedup via OpenBLAS");
    }
    println!();

    // ── ResNet-18 ──
    let resnet_path = format!("{}/testdata/resnet18.safetensors", env!("CARGO_MANIFEST_DIR"));
    if Path::new(&resnet_path).exists() {
        println!("--- ResNet-18 (ImageNet 1000-class, 224x224) ---");

        let t0 = Instant::now();
        let config = ResNetConfig::resnet18(1000);
        let classifier =
            ResNetClassifier::<CpuBackend>::from_safetensors(config, Path::new(&resnet_path))
                .unwrap();
        println!("  Load:       {:.0}ms", t0.elapsed().as_secs_f64() * 1000.0);

        let image = vec![128u8; 224 * 224 * 3];
        let (ms, ips) = bench(2, 10, || {
            let _ = classifier.classify(&image, 224, 224, 5).unwrap();
        });
        println!("  Inference:  {:.1}ms/image", ms);
        println!("  Throughput: {:.1} images/sec\n", ips);
    } else {
        println!("--- ResNet-18: SKIPPED (testdata/resnet18.safetensors not found) ---\n");
    }

    // ── CLIP ViT-B/32 ──
    let clip_path = format!("{}/testdata/clip_vit_b32.safetensors", env!("CARGO_MANIFEST_DIR"));
    if Path::new(&clip_path).exists() {
        println!("--- CLIP ViT-B/32 (512-dim embedding, 224x224) ---");

        let t0 = Instant::now();
        let config = ClipConfig::vit_b32();
        let embedder =
            ClipEmbedder::<CpuBackend>::from_safetensors(config, Path::new(&clip_path)).unwrap();
        println!("  Load:       {:.0}ms", t0.elapsed().as_secs_f64() * 1000.0);

        let image = vec![128u8; 224 * 224 * 3];
        let (ms, ips) = bench(2, 10, || {
            let _ = embedder.embed(&image, 224, 224).unwrap();
        });
        println!("  Inference:  {:.1}ms/image", ms);
        println!("  Throughput: {:.2} images/sec\n", ips);
    } else {
        println!("--- CLIP ViT-B/32: SKIPPED (testdata/clip_vit_b32.safetensors not found) ---\n");
    }

    // ── Context ──
    println!("--- External references (approximate) ---");
    println!("  PyTorch + MKL   ResNet-18 CPU:     ~30-50ms");
    println!("  ONNX Runtime    ResNet-18 CPU:     ~10-20ms");
    println!("  PyTorch + MKL   CLIP ViT-B/32 CPU: ~200-400ms");
    println!("  ONNX Runtime    CLIP ViT-B/32 CPU: ~100-200ms");
}
