// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end ViT-B/16 forward-pass benchmark across `CpuBackend` and
//! `ScryGpuBackend`. Companion to `bench_gpu.rs` (ResNet) — exercises the
//! transformer kernels: layernorm, scaled softmax, GELU, and the still-
//! CPU-bouncing attention reshape ops.
//!
//! Models are zero-init; we only care about kernel dispatch timing, not
//! classification accuracy. Default `gamma=1`, `beta=0` for layer norms keeps
//! activations finite.
//!
//! Run with:
//!
//! ```text
//! CUDARC_CUDA_VERSION=13010 cargo run -p scry-vision \
//!     --features scry-gpu-cuda --example bench_vit --release
//! ```
//!
//! Adding `scry-gpu-bf16` enables a bf16-matmul row stacked on top of the
//! GPU-resident path.

use std::time::Instant;

use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::scry_gpu::ScryGpuBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;
use scry_vision::models::vit::{Vit, VitConfig};

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

fn random_input(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((state >> 32) as i32 as f32) / (i32::MAX as f32)
        })
        .collect()
}

fn run_vit_b16(input: &[f32]) {
    println!("--- ViT-B/16 (768 dim, 12 layers, 12 heads, 197 tokens) ---");

    let cpu_model = Vit::<CpuBackend>::new(VitConfig::vit_b16());
    let cpu_input = Tensor::<CpuBackend>::from_vec(input.to_vec(), Shape::new(&[3, 224, 224]));
    // CpuBackend ViT is much heavier than ResNet — fewer runs to keep the
    // wall-clock reasonable.
    let cpu_ms = time_ms(1, 3, || {
        let _ = std::hint::black_box(cpu_model.forward(&cpu_input));
    });
    println!("  CpuBackend                  : {cpu_ms:7.1} ms/image");

    // Lazy: weights live in ScryGpuStorage::Cpu, every forward re-uploads them.
    let gpu_model_lazy = Vit::<ScryGpuBackend>::new(VitConfig::vit_b16());
    let gpu_input = Tensor::<ScryGpuBackend>::from_vec(input.to_vec(), Shape::new(&[3, 224, 224]));
    let gpu_lazy_ms = time_ms(2, 5, || {
        let _ = std::hint::black_box(gpu_model_lazy.forward(&gpu_input));
        ScryGpuBackend::synchronize().expect("synchronize");
    });
    println!(
        "  ScryGpuBackend (lazy)       : {gpu_lazy_ms:7.1} ms/image  ({:.2}× CPU)",
        cpu_ms / gpu_lazy_ms
    );

    #[cfg(feature = "scry-gpu-bf16")]
    {
        ScryGpuBackend::set_bf16_matmul(true).expect("toggle bf16");
        let gpu_bf16_ms = time_ms(2, 5, || {
            let _ = std::hint::black_box(gpu_model_lazy.forward(&gpu_input));
            ScryGpuBackend::synchronize().expect("synchronize");
        });
        ScryGpuBackend::set_bf16_matmul(false).expect("toggle bf16");
        println!(
            "  ScryGpuBackend (bf16 matmul): {gpu_bf16_ms:7.1} ms/image  ({:.2}× CPU, {:.2}× lazy)",
            cpu_ms / gpu_bf16_ms,
            gpu_lazy_ms / gpu_bf16_ms,
        );
    }
}

fn main() {
    println!("=== scry-vision ViT bench ===");
    println!("(zero-init weights — kernel dispatch timing only)\n");

    let scry_gpu_cuda = cfg!(feature = "scry-gpu-cuda");
    let scry_gpu_bf16 = cfg!(feature = "scry-gpu-bf16");
    let scry_gpu_cudnn = cfg!(feature = "scry-gpu-cudnn");
    println!(
        "Features: scry-gpu-cuda={scry_gpu_cuda}, scry-gpu-bf16={scry_gpu_bf16}, scry-gpu-cudnn={scry_gpu_cudnn}"
    );
    println!(
        "Threads:  RAYON_NUM_THREADS={}",
        rayon::current_num_threads()
    );
    println!();

    let input = random_input(0xdead_beef, 3 * 224 * 224);
    run_vit_b16(&input);
}
