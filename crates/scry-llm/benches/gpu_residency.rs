//! Benchmark whether keeping intermediates on the GPU actually wins.
//!
//! Compares three implementations of `GELU(A @ B)` end-to-end:
//!
//! - `cpu_chain` — `CpuBackend` start to finish (baseline).
//! - `gpu_persistent` — `ScryGpuBackend` with the data flowing through
//!   `ScryGpuStorage::Gpu` between matmul and GELU. This is the path
//!   that M1+M2 unlock.
//! - `gpu_materialized` — `ScryGpuBackend` but forced to download to CPU
//!   between ops via `to_cpu`/`to_gpu`. Proxy for the "before M1"
//!   round-tripping behavior to quantify the savings.
//!
//! Run with: `cargo bench -p scry-llm --features scry-gpu --bench gpu_residency`.
//!
//! The `gpu_*` cases gracefully degrade to CPU when no Vulkan device is
//! present — in that environment they should track `cpu_chain`.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::scry_gpu::{ScryGpuBackend, ScryGpuStorage};
use scry_llm::backend::{DeviceBackend, MathBackend};

/// Workload sizes for `C = GELU(A @ B)`. Each tuple is (M, K, N).
/// Picked to bracket the matmul GPU threshold (65,536 elements):
///  - small_64: just at threshold.
///  - medium_256: 33M elements, well above.
///  - large_512: 268M elements, exercises memory-bound behavior.
const SHAPES: &[(usize, usize, usize)] = &[
    (64, 64, 64),     // 262_144 element output (matmul itself = 262k FMAs)
    (256, 512, 256),  // 33M FMAs
    (512, 1024, 512), // 268M FMAs
];

fn make_random(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

fn cpu_chain(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let av = a.to_vec();
    let bv = b.to_vec();
    let c = CpuBackend::matmul(&av, &bv, m, k, n, false, false);
    CpuBackend::gelu(&c)
}

fn gpu_persistent(
    a: &ScryGpuStorage,
    b: &ScryGpuStorage,
    m: usize,
    k: usize,
    n: usize,
) -> ScryGpuStorage {
    let c = ScryGpuBackend::matmul(a, b, m, k, n, false, false);
    ScryGpuBackend::gelu(&c)
}

fn gpu_materialized(
    a: &ScryGpuStorage,
    b: &ScryGpuStorage,
    m: usize,
    k: usize,
    n: usize,
) -> ScryGpuStorage {
    let c = ScryGpuBackend::matmul(a, b, m, k, n, false, false);
    // Force a download/upload round trip between ops to simulate the
    // pre-M1 behavior where every op materialized through CPU.
    let c_cpu = ScryGpuBackend::to_cpu(&c);
    let c_back_on_gpu = ScryGpuBackend::to_gpu(&c_cpu).unwrap_or(c_cpu);
    ScryGpuBackend::gelu(&c_back_on_gpu)
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_then_gelu");

    for &(m, k, n) in SHAPES {
        let a_vec = make_random(m * k, 0xA);
        let b_vec = make_random(k * n, 0xB);
        let label = format!("m{m}_k{k}_n{n}");

        // CPU baseline — no GPU involvement.
        group.bench_function(BenchmarkId::new("cpu", &label), |b_| {
            b_.iter(|| {
                let out = cpu_chain(black_box(&a_vec), black_box(&b_vec), m, k, n);
                black_box(out);
            });
        });

        // Pre-upload inputs once so we measure the chain, not the upload.
        let a_storage = ScryGpuStorage::Cpu(a_vec.clone());
        let b_storage = ScryGpuStorage::Cpu(b_vec.clone());
        let a_gpu = ScryGpuBackend::to_gpu(&a_storage).unwrap_or(a_storage);
        let b_gpu = ScryGpuBackend::to_gpu(&b_storage).unwrap_or(b_storage);

        // Sanity: ensure inputs are actually on GPU when we expect them to be.
        let on_gpu = a_gpu.is_gpu() && b_gpu.is_gpu();
        let suffix = if on_gpu { "" } else { "_no_gpu" };

        group.bench_function(
            BenchmarkId::new(format!("gpu_persistent{suffix}"), &label),
            |b_| {
                b_.iter(|| {
                    let out = gpu_persistent(&a_gpu, &b_gpu, m, k, n);
                    // Ensure result is consumed; don't download — that would
                    // contaminate the measurement with a transfer.
                    black_box(&out);
                });
            },
        );

        group.bench_function(
            BenchmarkId::new(format!("gpu_materialized{suffix}"), &label),
            |b_| {
                b_.iter(|| {
                    let out = gpu_materialized(&a_gpu, &b_gpu, m, k, n);
                    black_box(&out);
                });
            },
        );

        // Sanity-check correctness across all three at this size before
        // declaring the benchmark valid. Tolerances are generous because
        // f32 accumulation order differs.
        let cpu = cpu_chain(&a_vec, &b_vec, m, k, n);
        let gpu_p = ScryGpuBackend::to_vec(&gpu_persistent(&a_gpu, &b_gpu, m, k, n));
        let gpu_m = ScryGpuBackend::to_vec(&gpu_materialized(&a_gpu, &b_gpu, m, k, n));
        assert_eq!(cpu.len(), gpu_p.len());
        for (i, (cv, gv)) in cpu.iter().zip(gpu_p.iter()).enumerate() {
            let tol = 1e-3_f32 * cv.abs().max(1.0);
            assert!(
                (cv - gv).abs() < tol,
                "persistent mismatch at {i}: cpu={cv} gpu={gv} (size={label})"
            );
        }
        for (i, (cv, gv)) in cpu.iter().zip(gpu_m.iter()).enumerate() {
            let tol = 1e-3_f32 * cv.abs().max(1.0);
            assert!(
                (cv - gv).abs() < tol,
                "materialized mismatch at {i}: cpu={cv} gpu={gv} (size={label})"
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
