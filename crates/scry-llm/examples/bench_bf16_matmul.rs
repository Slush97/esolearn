// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pure-matmul fp32 vs bf16 microbench at ResNet-realistic shapes.
//!
//! The end-to-end ResNet-50 bench moved 1.3 → 1.3 ms when we flipped on the
//! bf16 GemmEx path — barely a win. This bench probes whether the underlying
//! GEMM is actually faster (so cast HBM passes are eating the gain) or
//! whether tensor cores aren't engaging at these shapes.
//!
//! Usage:
//! ```text
//! CUDARC_CUDA_VERSION=13010 cargo run -p scry-llm \
//!     --features scry-gpu-bf16 --example bench_bf16_matmul --release
//! ```

use std::time::Instant;

use scry_llm::backend::scry_gpu::{ScryGpuBackend, ScryGpuStorage};

fn time_us<F: FnMut()>(warmup: usize, runs: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        f();
        ScryGpuBackend::synchronize().expect("sync");
        times.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[runs / 2]
}

fn random_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 32) as i32 as f32) / (i32::MAX as f32) * 0.1
        })
        .collect()
}

fn bench_shape(label: &str, m: usize, k: usize, n: usize) {
    let a = random_vec(0xa11ce, m * k);
    let b = random_vec(0xb0b, k * n);

    let a_storage = ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(a)).expect("upload a");
    let b_storage = ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(b)).expect("upload b");

    // fp32 cuBLAS sgemm
    let f32_us = time_us(3, 10, || {
        let _ = std::hint::black_box(
            ScryGpuBackend::matmul_force_gpu_for_bench(&a_storage, &b_storage, m, k, n)
                .expect("f32 matmul"),
        );
    });

    // bf16 cuBLAS GemmEx (cast→GemmEx→cast). The cast HBM passes are
    // included in this measurement — that's the cost a real model pays.
    let bf16_us = time_us(3, 10, || {
        let _ = std::hint::black_box(
            ScryGpuBackend::matmul_force_bf16_for_bench(&a_storage, &b_storage, m, k, n)
                .expect("bf16 matmul"),
        );
    });

    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let f32_tflops = flops / (f32_us * 1e6);
    let bf16_tflops = flops / (bf16_us * 1e6);
    let speedup = f32_us / bf16_us;

    println!(
        "  {label:<28}  fp32 {f32_us:7.1} µs ({f32_tflops:5.1} TF) | bf16 {bf16_us:7.1} µs ({bf16_tflops:5.1} TF) | {speedup:.2}× speedup",
    );
}

fn main() {
    println!("=== fp32 vs bf16 matmul microbench (RTX 5070 Ti, cast inclusive) ===\n");

    // ResNet-50 1×1 conv shapes (matmul-equivalent: M=Cout, K=Cin, N=H*W)
    println!("ResNet-50 1×1 convolutions (matmul as M×K × K×N):");
    bench_shape("stage1 1x1: 64×64×3136", 64, 64, 3136);
    bench_shape("stage1 1x1: 256×64×3136", 256, 64, 3136);
    bench_shape("stage2 1x1: 512×256×784", 512, 256, 784);
    bench_shape("stage3 1x1: 1024×512×196", 1024, 512, 196);
    bench_shape("stage4 1x1: 2048×1024×49", 2048, 1024, 49);

    // ResNet-50 3×3 conv (im2col): M=Cout, K=Cin*9, N=H*W
    println!("\nResNet-50 3×3 convolutions (im2col-lowered):");
    bench_shape("stage1 3x3: 64×576×3136", 64, 576, 3136);
    bench_shape("stage2 3x3: 128×1152×784", 128, 1152, 784);
    bench_shape("stage3 3x3: 256×2304×196", 256, 2304, 196);
    bench_shape("stage4 3x3: 512×4608×49", 512, 4608, 49);

    // Square shapes for reference (compute-bound, all 16-aligned)
    println!("\nSquare shapes (best-case for tensor cores):");
    bench_shape("256×256×256", 256, 256, 256);
    bench_shape("512×512×512", 512, 512, 512);
    bench_shape("1024×1024×1024", 1024, 1024, 1024);
    bench_shape("2048×2048×2048", 2048, 2048, 2048);
}
