//! Threshold sweep — find the size where CPU matmul/GELU stops being
//! faster than the GPU path, so [`GPU_MIN_ELEMENTS`] and
//! [`GPU_ELEMENTWISE_MIN`] can be retuned.
//!
//! The current cutoffs (65 536 FMAs for matmul, 16 384 elements for
//! elementwise) were picked when small-matmul ran at ~0.18 TFLOPS on the
//! WGSL coarse path. After cuBLAS + async dispatch + uninit alloc, the
//! same shape clears 0.7+ TFLOPS, so the cutoffs are almost certainly too
//! conservative. This bench probes square matmuls from M=K=N=16 to 256
//! and elementwise sizes from 1 KiB to 1 Mi elements, comparing:
//!
//! * **GPU resident**: inputs already on device, output forced to GPU,
//!   plus a single explicit sync at the end so the timer captures real
//!   GPU work (under async dispatch the launch returns before compute
//!   finishes).
//! * **CPU**: `CpuBackend::matmul` / `CpuBackend::gelu` on host data.
//!
//! The crossover is the smallest size where GPU < CPU. That sets the
//! new thresholds — once a single op pays for itself in GPU mode, any
//! chain of ops at that size or larger is unambiguously a GPU win.
//!
//! Run with:
//!   CUDARC_CUDA_VERSION=13010 cargo bench -p scry-llm \
//!       --features scry-gpu-cuda --bench threshold_sweep

use std::hint::black_box;
use std::time::{Duration, Instant};

use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::scry_gpu::{ScryGpuBackend, ScryGpuStorage};
use scry_llm::backend::{DeviceBackend, MathBackend};

const WARMUP: usize = 8;
const ITERS: usize = 80;

/// Square matmul sizes (M = K = N). FMA count = side³.
/// Range covers the regime around the current threshold (40³ ≈ 64K).
const MATMUL_SIDES: &[usize] = &[16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 160, 192, 256];

/// Elementwise sizes — span 1 KiB to 1 Mi elements, doubling.
const ELEMENTWISE_SIZES: &[usize] = &[
    1 << 10,
    1 << 11,
    1 << 12,
    1 << 13,
    1 << 14,
    1 << 15,
    1 << 16,
    1 << 17,
    1 << 18,
    1 << 19,
    1 << 20,
];

fn make_random(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

fn percentile(samples: &mut [Duration], p: f64) -> Duration {
    samples.sort_unstable();
    let len = samples.len();
    let idx = (((len as f64) * p) as usize).min(len - 1);
    samples[idx]
}

fn fmt_us(d: Duration) -> String {
    format!("{:>9.2}", d.as_secs_f64() * 1e6)
}

/// Force a sync on a GPU storage by reading one element back.
/// Cheap relative to the full download but blocks until the kernel finishes
/// (cuBLAS / dispatch_cuda_async return immediately under async dispatch).
fn sync_gpu(s: &ScryGpuStorage) {
    let v = ScryGpuBackend::to_vec(s);
    black_box(v);
}

fn main() {
    println!("=== scry-llm threshold sweep ===");
    println!("({WARMUP} warmup + {ITERS} iters per size)\n");

    if !gpu_available() {
        eprintln!("scry-gpu unavailable; aborting sweep.");
        return;
    }

    sweep_matmul();
    println!();
    sweep_elementwise();
}

fn gpu_available() -> bool {
    let probe = make_random(64, 1);
    let probe_cpu = ScryGpuStorage::Cpu(probe);
    matches!(ScryGpuBackend::to_gpu(&probe_cpu), Ok(s) if s.is_gpu())
}

fn sweep_matmul() {
    println!("--- matmul sweep (square M = K = N) ---");
    println!(
        "{:>5} {:>12} {:>12} {:>12} {:>10} {:>8}",
        "side", "FMAs", "cpu µs", "gpu µs", "speedup", "winner"
    );
    println!(
        "{:->5} {:->12} {:->12} {:->12} {:->10} {:->8}",
        "", "", "", "", "", ""
    );

    for &side in MATMUL_SIDES {
        let m = side;
        let k = side;
        let n = side;
        let fmas = m * k * n;

        let a_host = make_random(m * k, 0xA + side as u64);
        let b_host = make_random(k * n, 0xB + side as u64);

        // CPU baseline — matmul on host data.
        let mut t_cpu = Vec::with_capacity(ITERS);
        for _ in 0..WARMUP {
            black_box(CpuBackend::matmul(&a_host, &b_host, m, k, n, false, false));
        }
        for _ in 0..ITERS {
            let t = Instant::now();
            let out = CpuBackend::matmul(&a_host, &b_host, m, k, n, false, false);
            t_cpu.push(t.elapsed());
            black_box(out);
        }

        // GPU resident — inputs on device, kernel dispatched, sync at end.
        let a_gpu = ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(a_host.clone())).unwrap();
        let b_gpu = ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(b_host.clone())).unwrap();

        let mut t_gpu = Vec::with_capacity(ITERS);
        for _ in 0..WARMUP {
            let c = ScryGpuBackend::matmul_force_gpu_for_bench(&a_gpu, &b_gpu, m, k, n).unwrap();
            sync_gpu(&c);
        }
        for _ in 0..ITERS {
            let t = Instant::now();
            let c = ScryGpuBackend::matmul_force_gpu_for_bench(&a_gpu, &b_gpu, m, k, n).unwrap();
            sync_gpu(&c);
            t_gpu.push(t.elapsed());
        }

        let med_cpu = percentile(&mut t_cpu, 0.5);
        let med_gpu = percentile(&mut t_gpu, 0.5);
        let speedup = med_cpu.as_secs_f64() / med_gpu.as_secs_f64();
        let winner = if speedup >= 1.0 { "gpu" } else { "cpu" };
        println!(
            "{side:>5} {fmas:>12} {} {} {speedup:>9.2}× {winner:>8}",
            fmt_us(med_cpu),
            fmt_us(med_gpu),
        );
    }

    println!("\nCrossover = smallest side where speedup ≥ 1×.");
    println!("Set GPU_MIN_ELEMENTS to that side³ (or just under).");
}

fn sweep_elementwise() {
    println!("--- elementwise (GELU) sweep ---");
    println!(
        "{:>10} {:>12} {:>12} {:>10} {:>8}",
        "n", "cpu µs", "gpu µs", "speedup", "winner"
    );
    println!("{:->10} {:->12} {:->12} {:->10} {:->8}", "", "", "", "", "");

    for &n in ELEMENTWISE_SIZES {
        let host = make_random(n, 0xC + n as u64);

        // CPU baseline.
        let mut t_cpu = Vec::with_capacity(ITERS);
        for _ in 0..WARMUP {
            black_box(CpuBackend::gelu(&host));
        }
        for _ in 0..ITERS {
            let t = Instant::now();
            let out = CpuBackend::gelu(&host);
            t_cpu.push(t.elapsed());
            black_box(out);
        }

        // GPU — input already on device, sync after dispatch.
        let gpu_in = ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(host.clone())).unwrap();

        let mut t_gpu = Vec::with_capacity(ITERS);
        for _ in 0..WARMUP {
            let g = ScryGpuBackend::gelu_force_gpu_for_bench(&gpu_in).unwrap();
            sync_gpu(&g);
        }
        for _ in 0..ITERS {
            let t = Instant::now();
            let g = ScryGpuBackend::gelu_force_gpu_for_bench(&gpu_in).unwrap();
            sync_gpu(&g);
            t_gpu.push(t.elapsed());
        }

        let med_cpu = percentile(&mut t_cpu, 0.5);
        let med_gpu = percentile(&mut t_gpu, 0.5);
        let speedup = med_cpu.as_secs_f64() / med_gpu.as_secs_f64();
        let winner = if speedup >= 1.0 { "gpu" } else { "cpu" };
        println!(
            "{n:>10} {} {} {speedup:>9.2}× {winner:>8}",
            fmt_us(med_cpu),
            fmt_us(med_gpu),
        );
    }

    println!("\nNote: GPU times include a `to_vec` sync. In a chain this cost is paid once at the");
    println!("end of the chain, not per op — the actual chain crossover is lower than this table.");
}
