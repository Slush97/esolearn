//! Manual stage-by-stage timing breakdown for the GPU-resident path.
//!
//! Times each stage of `C = GELU(A @ B)` separately at three workload
//! sizes: upload, matmul, GELU, download. Reports median + p95 per stage
//! so we can see where time goes — kernel compute, dispatch+fence
//! overhead, or transfers — and decide what to invest in next.
//!
//! The size sweep matters: at small sizes a fixed per-dispatch fence
//! cost dominates; at large sizes actual compute amortizes. Watching
//! TFLOPS climb as size grows is direct evidence for which regime we're in.
//!
//! Each `Device::run_configured` call does record → submit → fence-wait
//! synchronously, so per-op timings include both kernel compute and
//! dispatch overhead. Splitting those further requires GPU timestamp
//! queries (vkCmdWriteTimestamp), which is out of scope here.
//!
//! Run with:
//!   cargo bench -p scry-llm --features scry-gpu --bench gpu_breakdown

use std::hint::black_box;
use std::time::{Duration, Instant};

use scry_llm::backend::scry_gpu::{ScryGpuBackend, ScryGpuStorage};
use scry_llm::backend::{DeviceBackend, MathBackend};

/// (M, K, N, label). Sized to bracket the dispatch-vs-compute regime:
///  - small: just over the GPU threshold; expect overhead-dominated.
///  - medium: 33M FMAs; mixed regime.
///  - large: 1B FMAs; expect compute-dominated.
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
    let len = samples.len();
    let idx = (((len as f64) * p) as usize).min(len - 1);
    samples[idx]
}

fn fmt_us(d: Duration) -> String {
    format!("{:>9.2} µs", d.as_secs_f64() * 1e6)
}

fn main() {
    println!("=== scry-llm GPU breakdown ===");
    println!("({WARMUP} warmup + {ITERS} iters per size)\n");

    let mut tflops_table: Vec<(&str, f64)> = Vec::new();

    for &(m, k, n, label) in SHAPES {
        match run_one(m, k, n, label) {
            Some(tflops) => tflops_table.push((label, tflops)),
            None => return,
        }
    }

    println!("\n=== matmul TFLOPS scaling ===");
    println!("{:<40} {:>14}", "size", "TFLOPS@median");
    println!("{:-<40} {:->14}", "", "");
    for (label, tf) in &tflops_table {
        println!("{label:<40} {tf:>11.3}   ");
    }
    println!(
        "\nIf TFLOPS climbs sharply with size, dispatch+fence overhead dominates at small sizes\n\
         and compute amortizes at larger sizes. If TFLOPS plateaus, the kernel itself is the bottleneck.\n\
         (RTX 5070 Ti f32 peak ≈ 55 TFLOPS; prior memo notes ~28–30% peak on standalone matmul.)"
    );
}

fn run_one(m: usize, k: usize, n: usize, label: &str) -> Option<f64> {
    println!("--- {label} ---");

    let a_vec = make_random(m * k, 0xA);
    let b_vec = make_random(k * n, 0xB);
    let a_cpu = ScryGpuStorage::Cpu(a_vec.clone());
    let b_cpu = ScryGpuStorage::Cpu(b_vec.clone());

    let a_gpu = match ScryGpuBackend::to_gpu(&a_cpu) {
        Ok(s) if s.is_gpu() => s,
        _ => {
            println!("scry-gpu unavailable; aborting breakdown.");
            return None;
        }
    };
    let b_gpu = match ScryGpuBackend::to_gpu(&b_cpu) {
        Ok(s) if s.is_gpu() => s,
        _ => {
            println!("scry-gpu unavailable; aborting breakdown.");
            return None;
        }
    };

    // Warmup so kernel compilation / pipeline cache / first-touch costs
    // don't contaminate the steady-state numbers.
    for _ in 0..WARMUP {
        let c = ScryGpuBackend::matmul(&a_gpu, &b_gpu, m, k, n, false, false);
        let g = ScryGpuBackend::gelu(&c);
        black_box(ScryGpuBackend::to_vec(&g));
    }

    let mut t_upload = Vec::with_capacity(ITERS);
    let mut t_matmul = Vec::with_capacity(ITERS);
    let mut t_gelu = Vec::with_capacity(ITERS);
    let mut t_download = Vec::with_capacity(ITERS);
    let mut t_chain = Vec::with_capacity(ITERS);

    for _ in 0..ITERS {
        // Fresh CPU→GPU each time — measures actual upload, not the
        // already-resident no-op path. We measure B (size K*N).
        let t = Instant::now();
        let uploaded = ScryGpuBackend::to_gpu(&b_cpu).unwrap();
        t_upload.push(t.elapsed());
        black_box(&uploaded);
        drop(uploaded);

        let t = Instant::now();
        let c = ScryGpuBackend::matmul(&a_gpu, &b_gpu, m, k, n, false, false);
        t_matmul.push(t.elapsed());

        let t = Instant::now();
        let g = ScryGpuBackend::gelu(&c);
        t_gelu.push(t.elapsed());

        let t = Instant::now();
        let v = ScryGpuBackend::to_vec(&g);
        t_download.push(t.elapsed());
        black_box(v);

        // Cross-check: re-run the whole sequence inside one timer; the
        // delta vs sum-of-stages tells us whether stage isolation hides
        // any setup cost.
        let t = Instant::now();
        let _u = ScryGpuBackend::to_gpu(&b_cpu).unwrap();
        let c2 = ScryGpuBackend::matmul(&a_gpu, &b_gpu, m, k, n, false, false);
        let g2 = ScryGpuBackend::gelu(&c2);
        let v2 = ScryGpuBackend::to_vec(&g2);
        t_chain.push(t.elapsed());
        black_box(v2);
    }

    println!(
        "{:<32} {:>14} {:>14} {:>14}",
        "stage", "median", "p95", "work (elems)"
    );
    println!("{:-<32} {:->14} {:->14} {:->14}", "", "", "", "");
    print_row("upload B (cpu→gpu)", &mut t_upload, k * n);
    print_row("matmul (persistent)", &mut t_matmul, 2 * m * k * n);
    print_row("gelu (persistent)", &mut t_gelu, m * n);
    print_row("download C (gpu→cpu)", &mut t_download, m * n);
    print_row("chained total", &mut t_chain, 0);

    let med_upload = percentile(&mut t_upload.clone(), 0.5);
    let med_matmul = percentile(&mut t_matmul.clone(), 0.5);
    let med_gelu = percentile(&mut t_gelu.clone(), 0.5);
    let med_download = percentile(&mut t_download.clone(), 0.5);
    let med_chain = percentile(&mut t_chain.clone(), 0.5);
    let stage_sum = med_upload + med_matmul + med_gelu + med_download;
    println!(
        "stage-sum median = {}   chained = {}   delta = {:+.2} µs",
        fmt_us(stage_sum),
        fmt_us(med_chain),
        (med_chain.as_secs_f64() - stage_sum.as_secs_f64()) * 1e6
    );

    let flops = 2.0 * (m as f64) * (k as f64) * (n as f64);
    let tflops = flops / med_matmul.as_secs_f64() / 1e12;
    println!(
        "matmul: {:.3} TFLOPS  ({:.1} GFLOPS, {} FMAs)\n",
        tflops,
        flops / med_matmul.as_secs_f64() / 1e9,
        m * k * n
    );

    Some(tflops)
}

fn print_row(name: &str, samples: &mut [Duration], work: usize) {
    let median = percentile(samples, 0.5);
    let p95 = percentile(samples, 0.95);
    if work == 0 {
        println!(
            "{:<32} {:>14} {:>14} {:>14}",
            name,
            fmt_us(median),
            fmt_us(p95),
            "—"
        );
    } else {
        println!(
            "{:<32} {:>14} {:>14} {:>14}",
            name,
            fmt_us(median),
            fmt_us(p95),
            work
        );
    }
}
