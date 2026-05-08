//! POC: does batching `matmul + gelu` into a single `Device::batch()` actually
//! save a fence wait's worth of wall-clock time?
//!
//! Tests the hypothesis from `HACKING_GPU_BREAKDOWN.md` that the next
//! investment is batched command buffers, NOT matmul kernel optimization.
//! Compares two paths at the same workload sizes:
//!
//! - **unbatched**: `MathBackend::matmul` then `MathBackend::gelu` —
//!   two `run_configured` calls, each a complete record/submit/fence-wait.
//! - **batched**: `ScryGpuBackend::matmul_then_gelu_batched_for_bench` —
//!   one `Device::batch()` with both kernels and one fence wait.
//!
//! If batching wins ~20–30 µs at small sizes (consistent with the
//! breakdown bench's per-op overhead estimate), porting Phase-4-style
//! batching into `ScryGpuBackend`'s main dispatch is justified. If it
//! wins under 5 µs, the savings are inside-noise and we should look
//! elsewhere first.
//!
//! Run with:
//!   cargo bench -p scry-llm --features scry-gpu --bench gpu_batched_poc

use std::hint::black_box;
use std::time::{Duration, Instant};

use scry_llm::backend::scry_gpu::{ScryGpuBackend, ScryGpuStorage};
use scry_llm::backend::{DeviceBackend, MathBackend};

const SHAPES: &[(usize, usize, usize, &str)] = &[
    (128, 256, 128, "small  (M=128 K=256 N=128)"),
    (256, 512, 256, "medium (M=256 K=512 N=256)"),
    (1024, 1024, 1024, "large  (M=1024 K=1024 N=1024)"),
];

const WARMUP: usize = 10;
const ITERS: usize = 100;

fn make_random(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

fn percentile(samples: &mut [Duration], p: f64) -> Duration {
    samples.sort_unstable();
    let idx = (((samples.len() as f64) * p) as usize).min(samples.len() - 1);
    samples[idx]
}

fn fmt_us(d: Duration) -> String {
    format!("{:>9.2} µs", d.as_secs_f64() * 1e6)
}

fn main() {
    println!("=== scry-llm batched-dispatch POC ===");
    println!("({WARMUP} warmup + {ITERS} iters per size)\n");

    println!(
        "{:<40} {:>14} {:>14} {:>14} {:>10}",
        "size", "unbatched med", "batched med", "saved µs", "saved %"
    );
    println!(
        "{:-<40} {:->14} {:->14} {:->14} {:->10}",
        "", "", "", "", ""
    );

    for &(m, k, n, label) in SHAPES {
        run_one(m, k, n, label);
    }

    println!(
        "\nSaved column = unbatched_median − batched_median.\n\
         Hypothesis from HACKING_GPU_BREAKDOWN.md: batching saves one full\n\
         per-op fence wait (~20–50 µs). If saved % is in that ballpark,\n\
         port the batched path into ScryGpuBackend. If under 5 µs, the\n\
         fence wait was already partially hidden by GPU execution time."
    );
}

fn run_one(m: usize, k: usize, n: usize, label: &str) {
    let a_vec = make_random(m * k, 0xA);
    let b_vec = make_random(k * n, 0xB);
    let a_cpu = ScryGpuStorage::Cpu(a_vec.clone());
    let b_cpu = ScryGpuStorage::Cpu(b_vec.clone());

    let a_gpu = match ScryGpuBackend::to_gpu(&a_cpu) {
        Ok(s) if s.is_gpu() => s,
        _ => {
            println!("{label}: scry-gpu unavailable; skipping.");
            return;
        }
    };
    let b_gpu = match ScryGpuBackend::to_gpu(&b_cpu) {
        Ok(s) if s.is_gpu() => s,
        _ => {
            println!("{label}: scry-gpu unavailable; skipping.");
            return;
        }
    };

    // Correctness check before timing — confirm the batched POC produces
    // numerically equivalent output to the unbatched path.
    let unbatched_ref = {
        let c = ScryGpuBackend::matmul(&a_gpu, &b_gpu, m, k, n, false, false);
        ScryGpuBackend::to_vec(&ScryGpuBackend::gelu(&c))
    };
    let Some(out) = ScryGpuBackend::matmul_then_gelu_batched_for_bench(&a_gpu, &b_gpu, m, k, n)
    else {
        println!("{label}: batched path returned None (cuBLAS or sub-threshold); skipping.");
        return;
    };
    let batched_ref = ScryGpuBackend::to_vec(&out);
    assert_eq!(unbatched_ref.len(), batched_ref.len());
    for (i, (u, b_)) in unbatched_ref.iter().zip(batched_ref.iter()).enumerate() {
        let tol = 1e-3_f32 * u.abs().max(1.0);
        assert!(
            (u - b_).abs() < tol,
            "{label}: batched mismatch at {i}: unbatched={u} batched={b_}"
        );
    }

    // Warmup both paths so the comparison isn't biased by first-touch costs.
    for _ in 0..WARMUP {
        let c = ScryGpuBackend::matmul(&a_gpu, &b_gpu, m, k, n, false, false);
        let g = ScryGpuBackend::gelu(&c);
        black_box(&g);
        let g2 = ScryGpuBackend::matmul_then_gelu_batched_for_bench(&a_gpu, &b_gpu, m, k, n)
            .expect("batched path");
        black_box(&g2);
    }

    let mut t_unbatched = Vec::with_capacity(ITERS);
    let mut t_batched = Vec::with_capacity(ITERS);

    // Interleave the two paths so any time-of-day GPU occupancy fluctuation
    // affects both equally.
    for _ in 0..ITERS {
        let t = Instant::now();
        let c = ScryGpuBackend::matmul(&a_gpu, &b_gpu, m, k, n, false, false);
        let g = ScryGpuBackend::gelu(&c);
        t_unbatched.push(t.elapsed());
        black_box(&g);

        let t = Instant::now();
        let g2 = ScryGpuBackend::matmul_then_gelu_batched_for_bench(&a_gpu, &b_gpu, m, k, n)
            .expect("batched path");
        t_batched.push(t.elapsed());
        black_box(&g2);
    }

    let med_un = percentile(&mut t_unbatched, 0.5);
    let med_bt = percentile(&mut t_batched, 0.5);
    let saved_us = med_un.as_secs_f64() * 1e6 - med_bt.as_secs_f64() * 1e6;
    let saved_pct = saved_us / (med_un.as_secs_f64() * 1e6) * 100.0;
    println!(
        "{:<40} {:>14} {:>14} {:>+13.2} {:>9.1}%",
        label,
        fmt_us(med_un),
        fmt_us(med_bt),
        saved_us,
        saved_pct
    );
}
