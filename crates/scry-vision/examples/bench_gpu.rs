// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end ResNet forward-pass benchmark across `CpuBackend` and
//! `ScryGpuBackend`.
//!
//! Models are zero-init (no checkpoint needed) — we only care about kernel
//! dispatch time, not classification accuracy. BatchNorm defaults
//! (`weight=1`, `bias=0`, `running_mean=0`, `running_var=1`) keep activations
//! finite so every kernel actually runs through to completion.
//!
//! Run with:
//!
//! ```text
//! CUDARC_CUDA_VERSION=13010 cargo run -p scry-vision \
//!     --features scry-gpu-cuda --example bench_gpu --release
//! ```
//!
//! The CUDA features pull `scry-gpu-cuda` through the dependency chain so
//! `ScryGpuBackend` actually engages the GPU; without them every op falls
//! back to CPU and the two columns will report identical numbers.

use std::time::Instant;

use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::scry_gpu::ScryGpuBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;
use scry_vision::models::{ResNet, ResNetConfig};

fn time_ms<F: FnMut()>(warmup: usize, runs: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[runs / 2]
}

/// Cheap LCG so the bench doesn't pull in fastrand as a dependency.
fn random_input(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            // Map 32 high bits to [-1, 1).
            ((state >> 32) as i32 as f32) / (i32::MAX as f32)
        })
        .collect()
}

fn run_resnet18(input: &[f32]) {
    println!("--- ResNet-18 (1000-class, 3×224×224) ---");

    let cpu_model = ResNet::<CpuBackend>::new(ResNetConfig::resnet18(1000));
    let cpu_input = Tensor::<CpuBackend>::from_vec(input.to_vec(), Shape::new(&[3, 224, 224]));
    let cpu_ms = time_ms(1, 3, || {
        let _ = std::hint::black_box(cpu_model.forward(&cpu_input));
    });
    println!("  CpuBackend     : {cpu_ms:7.1} ms/image");

    let gpu_model = ResNet::<ScryGpuBackend>::new(ResNetConfig::resnet18(1000));
    let gpu_input = Tensor::<ScryGpuBackend>::from_vec(input.to_vec(), Shape::new(&[3, 224, 224]));
    let gpu_ms = time_ms(2, 5, || {
        let _ = std::hint::black_box(gpu_model.forward(&gpu_input));
    });
    println!("  ScryGpuBackend : {gpu_ms:7.1} ms/image  ({:.2}× CPU)", cpu_ms / gpu_ms);
}

fn run_resnet50(input: &[f32]) {
    println!("\n--- ResNet-50 (1000-class, 3×224×224) ---");

    let cpu_model = ResNet::<CpuBackend>::new(ResNetConfig::resnet50(1000));
    let cpu_input = Tensor::<CpuBackend>::from_vec(input.to_vec(), Shape::new(&[3, 224, 224]));
    let cpu_ms = time_ms(1, 3, || {
        let _ = std::hint::black_box(cpu_model.forward(&cpu_input));
    });
    println!("  CpuBackend     : {cpu_ms:7.1} ms/image");

    let gpu_model = ResNet::<ScryGpuBackend>::new(ResNetConfig::resnet50(1000));
    let gpu_input = Tensor::<ScryGpuBackend>::from_vec(input.to_vec(), Shape::new(&[3, 224, 224]));
    let gpu_ms = time_ms(2, 5, || {
        let _ = std::hint::black_box(gpu_model.forward(&gpu_input));
    });
    println!("  ScryGpuBackend : {gpu_ms:7.1} ms/image  ({:.2}× CPU)", cpu_ms / gpu_ms);
}

fn main() {
    println!("=== scry-vision end-to-end ResNet bench ===");
    println!("(zero-init weights — kernel dispatch timing only)\n");

    let scry_gpu_cuda = cfg!(feature = "scry-gpu-cuda");
    println!("Features: scry-gpu-cuda={scry_gpu_cuda}");
    println!(
        "Threads:  RAYON_NUM_THREADS={}",
        rayon::current_num_threads()
    );
    println!();

    let input = random_input(0xdead_beef, 3 * 224 * 224);

    run_resnet18(&input);
    run_resnet50(&input);
}
