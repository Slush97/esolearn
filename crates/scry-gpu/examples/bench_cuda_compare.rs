//! Cross-backend benchmark: Vulkan (WGSL shaders) vs CUDA (cuBLAS + NVRTC).
//!
//! Compares the same workloads on both backends:
//!   1. **Upload/download roundtrip** — raw transfer throughput
//!   2. **Vector scale** — simple element-wise kernel (bandwidth-bound)
//!   3. **Matrix multiply** — compute-bound (Vulkan tiled shaders vs cuBLAS)
//!
//! Run with:
//!
//! ```sh
//! cargo run -p scry-gpu --example bench_cuda_compare --features cuda --release
//! ```

use std::time::Instant;

use scry_gpu::shaders::matmul::{COARSE_64X64, COARSE_8X8, TILED_16X16};
use scry_gpu::{BackendKind, Device};

fn main() {
    let vk = Device::with_backend(BackendKind::Vulkan).expect("Vulkan unavailable");
    let cu = Device::with_backend(BackendKind::Cuda).expect("CUDA unavailable");

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║          scry-gpu: Vulkan vs CUDA comparison            ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "  Vulkan: {} ({} MB)",
        vk.name(),
        vk.memory() / (1024 * 1024)
    );
    println!(
        "  CUDA:   {} ({} MB)",
        cu.name(),
        cu.memory() / (1024 * 1024)
    );
    println!();

    bench_transfer(&vk, &cu);
    bench_vector_scale(&vk, &cu);
    bench_matmul(&vk, &cu);
}

// ── Transfer throughput ─────────────────────────────────────────────────────

fn bench_transfer(vk: &Device, cu: &Device) {
    println!("═══ Upload + Download Roundtrip ═══");
    println!(
        "  {:>10}  {:>12} {:>12} {:>10}",
        "size", "Vulkan", "CUDA", "ratio"
    );

    for &n in &[100_000u32, 1_000_000, 4_000_000, 16_000_000] {
        let data: Vec<f32> = (0..n as usize).map(|i| i as f32).collect();
        let bytes = (n as f64) * 4.0;
        let iters = if n >= 4_000_000 { 20 } else { 50 };

        // warmup
        let _ = vk.upload(&data).unwrap().download().unwrap();
        let _ = cu.upload(&data).unwrap().download().unwrap();

        let vk_time = bench_transfer_inner(vk, &data, iters);
        let cu_time = bench_transfer_inner(cu, &data, iters);

        let vk_gbps = bytes / vk_time.as_secs_f64() / 1e9;
        let cu_gbps = bytes / cu_time.as_secs_f64() / 1e9;
        let ratio = cu_time.as_secs_f64() / vk_time.as_secs_f64();

        println!(
            "  {:>10}  {:>8.2} GB/s {:>8.2} GB/s {:>9.2}x",
            fmt_count(n),
            vk_gbps,
            cu_gbps,
            1.0 / ratio
        );
    }
    println!();
}

fn bench_transfer_inner(gpu: &Device, data: &[f32], iters: u32) -> std::time::Duration {
    let start = Instant::now();
    for _ in 0..iters {
        let buf = gpu.upload(data).unwrap();
        let _ = buf.download().unwrap();
    }
    start.elapsed() / iters
}

// ── Vector scale (bandwidth-bound) ──────────────────────────────────────────

const SCALE_WGSL: &str = "\
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&input) {
        output[i] = input[i] * 2.0;
    }
}";

const SCALE_CUDA: &str = r#"
extern "C" __global__ void scale(const float* input, float* output, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[i] * 2.0f;
    }
}
"#;

fn bench_vector_scale(vk: &Device, cu: &Device) {
    println!("═══ Vector Scale — out[i] = in[i] × 2 ═══");
    println!(
        "  {:>10}  {:>12} {:>12} {:>10}",
        "size", "Vulkan", "CUDA", "CUDA/VK"
    );

    let vk_kernel = vk.compile(SCALE_WGSL).expect("compile Vulkan scale");
    let cu_kernel = cu
        .compile_cuda(SCALE_CUDA, "scale", 2, [256, 1, 1])
        .expect("compile CUDA scale");

    for &n in &[100_000u32, 1_000_000, 4_000_000, 16_000_000] {
        let iters = if n >= 4_000_000 { 50 } else { 200 };
        let data: Vec<f32> = (0..n as usize).map(|i| i as f32).collect();
        let bytes = (n as f64) * 4.0 * 2.0; // read + write

        let vk_in = vk.upload(&data).unwrap();
        let vk_out = vk.alloc::<f32>(n as usize).unwrap();
        let cu_in = cu.upload(&data).unwrap();
        let cu_out = cu.alloc::<f32>(n as usize).unwrap();

        // warmup
        vk.run(&vk_kernel, &[&vk_in, &vk_out], n).unwrap();
        let pc_n = n;
        cu.run_with_push_constants(&cu_kernel, &[&cu_in, &cu_out], n, bytemuck::bytes_of(&pc_n))
            .unwrap();

        let vk_start = Instant::now();
        for _ in 0..iters {
            vk.run(&vk_kernel, &[&vk_in, &vk_out], n).unwrap();
        }
        let vk_per = vk_start.elapsed() / iters;

        let cu_start = Instant::now();
        for _ in 0..iters {
            cu.run_with_push_constants(
                &cu_kernel,
                &[&cu_in, &cu_out],
                n,
                bytemuck::bytes_of(&pc_n),
            )
            .unwrap();
        }
        let cu_per = cu_start.elapsed() / iters;

        let vk_gbps = bytes / vk_per.as_secs_f64() / 1e9;
        let cu_gbps = bytes / cu_per.as_secs_f64() / 1e9;
        let speedup = cu_per.as_secs_f64() / vk_per.as_secs_f64();

        println!(
            "  {:>10}  {:>8.1} GB/s {:>8.1} GB/s {:>9.2}x",
            fmt_count(n),
            vk_gbps,
            cu_gbps,
            1.0 / speedup
        );
    }
    println!();
}

// ── Matrix multiply (compute-bound) ─────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulDims {
    m: u32,
    n: u32,
    k: u32,
}

fn bench_matmul(vk: &Device, cu: &Device) {
    println!("═══ Matrix Multiply — C = A × B (square, f32) ═══");
    println!();

    // Vulkan WGSL variants
    let vk_tiled = vk.compile(TILED_16X16).expect("compile Vulkan tiled 16×16");
    let vk_coarse = vk.compile(COARSE_64X64).expect("compile Vulkan coarse 4×4");
    let vk_8x8 = vk.compile(COARSE_8X8).expect("compile Vulkan coarse 8×8");

    // CUDA variants
    let cu_tiled = cu
        .compile_cuda(
            scry_gpu::shaders::matmul::TILED_16X16_CUDA,
            "matmul_tiled_16x16",
            3,
            [16, 16, 1],
        )
        .expect("compile CUDA tiled 16×16");

    for &n in &[512u32, 1024, 2048, 4096] {
        let iters: u32 = if n >= 4096 {
            5
        } else if n >= 2048 {
            20
        } else {
            50
        };
        let elems = (n * n) as usize;
        let flops = 2.0 * (n as f64).powi(3);

        let a_data: Vec<f32> = (0..elems).map(|i| (i % 100) as f32 * 0.01).collect();
        let b_data: Vec<f32> = (0..elems).map(|i| (i % 100) as f32 * 0.01).collect();

        let dims = MatmulDims { m: n, n, k: n };
        let pc = bytemuck::bytes_of(&dims);

        // Pre-upload for each backend
        let vk_a = vk.upload(&a_data).unwrap();
        let vk_b = vk.upload(&b_data).unwrap();
        let cu_a = cu.upload(&a_data).unwrap();
        let cu_b = cu.upload(&b_data).unwrap();

        println!("  {n}×{n}  ({iters} iters)");
        println!(
            "    {:<8} {:<16} {:>12} {:>10}",
            "backend", "variant", "time/iter", "GFLOPS"
        );

        // Vulkan tiled 16×16
        {
            let c = vk.alloc::<f32>(elems).unwrap();
            let wg = n.div_ceil(16);
            let dur = bench_kernel_iters(iters, || {
                vk.run_configured(&vk_tiled, &[&vk_a, &vk_b, &c], [wg, wg, 1], Some(pc))
                    .unwrap();
            });
            print_matmul_row("Vulkan", "tiled 16×16", dur, flops);
        }

        // Vulkan coarse 4×4
        {
            let c = vk.alloc::<f32>(elems).unwrap();
            let wg = n.div_ceil(64);
            let dur = bench_kernel_iters(iters, || {
                vk.run_configured(&vk_coarse, &[&vk_a, &vk_b, &c], [wg, wg, 1], Some(pc))
                    .unwrap();
            });
            print_matmul_row("Vulkan", "coarse 4×4", dur, flops);
        }

        // Vulkan coarse 8×8 (skip for small matrices)
        if n >= 1024 {
            let c = vk.alloc::<f32>(elems).unwrap();
            let wg = n.div_ceil(128);
            let dur = bench_kernel_iters(iters, || {
                vk.run_configured(&vk_8x8, &[&vk_a, &vk_b, &c], [wg, wg, 1], Some(pc))
                    .unwrap();
            });
            print_matmul_row("Vulkan", "coarse 8×8", dur, flops);
        }

        // CUDA tiled 16×16
        {
            let c = cu.alloc::<f32>(elems).unwrap();
            let wg = n.div_ceil(16);
            let dur = bench_kernel_iters(iters, || {
                cu.run_configured(&cu_tiled, &[&cu_a, &cu_b, &c], [wg, wg, 1], Some(pc))
                    .unwrap();
            });
            print_matmul_row("CUDA", "tiled 16×16", dur, flops);
        }

        // CUDA cuBLAS sgemm
        {
            let mut c = cu.alloc::<f32>(elems).unwrap();
            let dur = bench_kernel_iters(iters, || {
                cu.cublas_matmul(&cu_a, &cu_b, &mut c, n, n, n).unwrap();
            });
            print_matmul_row("CUDA", "cuBLAS sgemm", dur, flops);
        }

        println!();
    }
}

/// Run `f` once as warmup, then `iters` times, returning the per-iteration duration.
fn bench_kernel_iters(iters: u32, mut f: impl FnMut()) -> std::time::Duration {
    f(); // warmup
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed() / iters
}

fn print_matmul_row(backend: &str, variant: &str, dur: std::time::Duration, flops: f64) {
    let gflops = flops / dur.as_secs_f64() / 1e9;
    println!("    {backend:<8} {variant:<16} {dur:>9.2?}/it {gflops:>8.1}");
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn fmt_count(n: u32) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        n.to_string()
    }
}
