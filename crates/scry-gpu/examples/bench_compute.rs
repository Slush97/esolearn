//! Compute benchmarks: SAXPY, reduction, and matrix multiply.
//!
//! Measures steady-state dispatch throughput using cached kernels.
//! Run with: `cargo run -p scry-gpu --example bench_compute --release`

use std::time::Instant;

use scry_gpu::shaders::matmul::{COARSE_64X64, TILED_16X16};
use scry_gpu::{Device, GpuBuf, Kernel};

// ── Shaders ─────────────────────────────────────────────────────────────────

const SAXPY_SHADER: &str = "\
struct Params { alpha: f32 }
var<push_constant> params: Params;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&a) {
        out[i] = params.alpha * a[i] + b[i];
    }
}";

/// Vec4 SAXPY: each thread processes 4 elements via vec4<f32> loads/stores.
/// Buffer layout is identical (contiguous f32s) — the shader just reinterprets
/// them as vec4<f32>, issuing 128-bit loads instead of 32-bit.
/// Dispatch n/4 invocations (all benchmark sizes are divisible by 4).
const SAXPY_VEC4_SHADER: &str = "\
struct Params { alpha: f32 }
var<push_constant> params: Params;

@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&a) {
        out[i] = params.alpha * a[i] + b[i];
    }
}";

/// Subgroup-accelerated reduction: subgroupAdd within warps, then shared
/// memory to collect warp partial sums, then one more subgroupAdd.
/// Eliminates the log2(256) shared-memory reduction iterations.
const REDUCE_SUBGROUP_SHADER: &str = "\
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> partials: array<f32, 32>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
    @builtin(subgroup_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let val = select(0.0, input[gid.x], gid.x < arrayLength(&input));

    // Warp-level reduction (no shared memory needed)
    let warp_sum = subgroupAdd(val);

    // Lane 0 of each subgroup writes partial sum
    if lane == 0u {
        partials[sg_id] = warp_sum;
    }
    workgroupBarrier();

    // First subgroup reduces the collected partial sums
    let num_subgroups = 256u / sg_size;
    if sg_id == 0u {
        let p = select(0.0, partials[lane], lane < num_subgroups);
        let total = subgroupAdd(p);
        if lane == 0u {
            output[wid.x] = total;
        }
    }
}";

const REDUCE_SHADER: &str = "\
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let i = gid.x;
    scratch[lid.x] = select(0.0, input[i], i < arrayLength(&input));
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if lid.x < stride {
            scratch[lid.x] += scratch[lid.x + stride];
        }
        workgroupBarrier();
    }

    if lid.x == 0u {
        output[wid.x] = scratch[0];
    }
}";

const MATMUL_NAIVE_SHADER: &str = "\
struct Dims { M: u32, N: u32, K: u32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let row = idx / dims.N;
    let col = idx % dims.N;
    if row >= dims.M || col >= dims.N { return; }

    var sum = 0.0;
    for (var k = 0u; k < dims.K; k++) {
        sum += A[row * dims.K + k] * B[k * dims.N + col];
    }
    C[row * dims.N + col] = sum;
}";

// TILED_16X16 moved to scry_gpu::shaders::matmul::TILED_16X16

// COARSE_64X64 moved to scry_gpu::shaders::matmul::COARSE_64X64

/// Thread-coarsened matmul with L2 cache tiling: same 64×64 tile and 4×4
/// thread coarsening as `COARSE_64X64`, but workgroup IDs are
/// swizzled into 4-wide super-tiles so that adjacent workgroups share
/// A-tile rows, keeping them hot in L2 cache. Within each super-tile
/// (4 columns × grid_m rows), column index varies fastest.
const MATMUL_COARSE_L2_SHADER: &str = "\
struct Dims { M: u32, N: u32, K: u32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

var<workgroup> sa: array<f32, 1088>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    // L2 cache tiling: remap wid into 4-wide super-tiles.
    let grid_n = (dims.N + 63u) / 64u;
    let grid_m = (dims.M + 63u) / 64u;
    let linear = wid.x + wid.y * grid_n;
    let super_col = linear / (4u * grid_m);
    let local = linear % (4u * grid_m);
    let w = min(4u, grid_n - super_col * 4u);
    let block_row = (local / w) * 64u;
    let block_col = (super_col * 4u + local % w) * 64u;
    let tr = lid.y * 4u;
    let tc = lid.x * 4u;

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }

    let num_k_tiles = (dims.K + 15u) / 16u;

    for (var kt = 0u; kt < num_k_tiles; kt++) {
        for (var x = 0u; x < 4u; x++) {
            let flat = li * 4u + x;
            let r = flat / 16u;
            let c = flat % 16u;
            let gr = block_row + r;
            let gc = kt * 16u + c;
            if gr < dims.M && gc < dims.K {
                sa[r * 17u + c] = A[gr * dims.K + gc];
            } else {
                sa[r * 17u + c] = 0.0;
            }
        }

        for (var x = 0u; x < 4u; x++) {
            let flat = li * 4u + x;
            let r = flat / 64u;
            let c = flat % 64u;
            let gr = kt * 16u + r;
            let gc = block_col + c;
            if gr < dims.K && gc < dims.N {
                sb[flat] = B[gr * dims.N + gc];
            } else {
                sb[flat] = 0.0;
            }
        }

        workgroupBarrier();

        for (var k = 0u; k < 16u; k++) {
            for (var i = 0u; i < 4u; i++) {
                let a_val = sa[(tr + i) * 17u + k];
                for (var j = 0u; j < 4u; j++) {
                    acc[i * 4u + j] += a_val * sb[k * 64u + tc + j];
                }
            }
        }

        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            let gr = block_row + tr + i;
            let gc = block_col + tc + j;
            if gr < dims.M && gc < dims.N {
                C[gr * dims.N + gc] = acc[i * 4u + j];
            }
        }
    }
}";

/// Large-tile matmul: 128×128 output tile, 32×32 workgroup, 4×4 per thread.
///
/// Same 16-accumulator register budget as the 64×64 coarsened kernel, but
/// 4× the threads (1024 vs 256) gives 4× the output tile area.
/// Arithmetic intensity: 32 FLOP/byte (2× over 64×64).
/// K-tile = 16 to match the 64×64 kernel's barrier cadence.
/// A[128×16] padded stride 17 + B[16×128] = ~16.6 KB shared memory.
const MATMUL_LARGE_SHADER: &str = "\
struct Dims { M: u32, N: u32, K: u32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

var<workgroup> sa: array<f32, 2176>;
var<workgroup> sb: array<f32, 2048>;

@compute @workgroup_size(32, 32)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let block_row = wid.y * 128u;
    let block_col = wid.x * 128u;
    let tr = lid.y * 4u;
    let tc = lid.x * 4u;

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }

    let num_k_tiles = (dims.K + 15u) / 16u;

    for (var kt = 0u; kt < num_k_tiles; kt++) {
        // Load A[128×16]: 2048 elements, 2 per thread, padded stride 17
        for (var x = 0u; x < 2u; x++) {
            let flat = li * 2u + x;
            let r = flat / 16u;
            let c = flat % 16u;
            let gr = block_row + r;
            let gc = kt * 16u + c;
            if gr < dims.M && gc < dims.K {
                sa[r * 17u + c] = A[gr * dims.K + gc];
            } else {
                sa[r * 17u + c] = 0.0;
            }
        }

        // Load B[16×128]: 2048 elements, 2 per thread
        for (var x = 0u; x < 2u; x++) {
            let flat = li * 2u + x;
            let r = flat / 128u;
            let c = flat % 128u;
            let gr = kt * 16u + r;
            let gc = block_col + c;
            if gr < dims.K && gc < dims.N {
                sb[flat] = B[gr * dims.N + gc];
            } else {
                sb[flat] = 0.0;
            }
        }

        workgroupBarrier();

        for (var k = 0u; k < 16u; k++) {
            for (var i = 0u; i < 4u; i++) {
                let a_val = sa[(tr + i) * 17u + k];
                for (var j = 0u; j < 4u; j++) {
                    acc[i * 4u + j] += a_val * sb[k * 128u + tc + j];
                }
            }
        }

        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            let gr = block_row + tr + i;
            let gc = block_col + tc + j;
            if gr < dims.M && gc < dims.N {
                C[gr * dims.N + gc] = acc[i * 4u + j];
            }
        }
    }
}";

fn main() {
    let gpu = Device::auto().expect("no GPU found");
    println!(
        "Device: {} ({} MB)\n",
        gpu.name(),
        gpu.memory() / (1024 * 1024)
    );

    bench_saxpy(&gpu);
    bench_reduce(&gpu);
    bench_matmul(&gpu);
}

// ── SAXPY (bandwidth-bound) ─────────────────────────────────────────────────

fn bench_saxpy(gpu: &Device) {
    println!("═══ SAXPY: out = α·a + b (bandwidth-bound) ═══");

    let scalar = gpu.compile(SAXPY_SHADER).expect("compile scalar");
    let vec4 = gpu.compile(SAXPY_VEC4_SHADER).expect("compile vec4");
    let alpha: f32 = 2.0;
    let pc = bytemuck::bytes_of(&alpha);

    for &n in &[1_000_000u32, 4_000_000, 16_000_000] {
        let iters: u32 = 200;
        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();

        let a = gpu.upload(&a_data).expect("upload");
        let b = gpu.upload(&b_data).expect("upload");
        let out = gpu.alloc::<f32>(n as usize).expect("alloc");

        // 2 reads + 1 write = 12 bytes per element
        let bytes = 3.0 * n as f64 * 4.0;

        println!("  n = {:>3}", fmt_count(n));

        for (name, kernel, invocations) in [
            ("scalar", &scalar, n),
            ("vec4  ", &vec4, n / 4),
        ] {
            // warmup
            gpu.run_with_push_constants(kernel, &[&a, &b, &out], invocations, pc)
                .expect("warmup");

            // ── Per-dispatch (sync fence each time) ──
            let start = Instant::now();
            for _ in 0..iters {
                gpu.run_with_push_constants(kernel, &[&a, &b, &out], invocations, pc)
                    .expect("run");
            }
            let sync_per = start.elapsed() / iters;
            let sync_gbps = bytes / sync_per.as_secs_f64() / 1e9;

            // ── Batched (one fence for all dispatches) ──
            let start = Instant::now();
            let mut batch = gpu.batch().expect("batch");
            for _ in 0..iters {
                batch
                    .run_with_push_constants(kernel, &[&a, &b, &out], invocations, pc)
                    .expect("batch run");
            }
            batch.submit().expect("batch submit");
            let batch_per = start.elapsed() / iters;
            let batch_gbps = bytes / batch_per.as_secs_f64() / 1e9;

            let speedup = sync_per.as_nanos() as f64 / batch_per.as_nanos() as f64;

            println!(
                "    {name}  sync {:>7.2?} {:>5.0} GB/s  │  batch {:>7.2?} {:>5.0} GB/s  ({speedup:.1}x)",
                sync_per, sync_gbps, batch_per, batch_gbps
            );
        }
    }
    println!();
}

// ── Reduction (multi-pass, kernel reuse) ────────────────────────────────────

fn bench_reduce(gpu: &Device) {
    println!("═══ Reduction (sum) — shared-memory vs subgroup ═══");
    println!("  subgroup_size = {}", gpu.subgroup_size());

    let shmem = gpu.compile(REDUCE_SHADER).expect("compile shmem reduce");
    let subgroup = gpu.compile(REDUCE_SUBGROUP_SHADER).expect("compile subgroup reduce");

    for &n in &[1_000_000u32, 4_000_000, 16_000_000] {
        let iters: u32 = 100;
        let data: Vec<f32> = vec![1.0; n as usize];
        let input = gpu.upload(&data).expect("upload");
        let expected = n as f32;

        // correctness check for both variants
        let result_shmem = reduce_sum_batched(gpu, &shmem, &input, n);
        assert!(
            (result_shmem - expected).abs() / expected < 1e-3,
            "shmem reduction: got {result_shmem}, expected {expected}"
        );
        let result_sg = reduce_sum_batched(gpu, &subgroup, &input, n);
        assert!(
            (result_sg - expected).abs() / expected < 1e-3,
            "subgroup reduction: got {result_sg}, expected {expected}"
        );

        let passes = count_passes(n);
        let bytes = n as f64 * 4.0;

        println!("  n = {:>3}  ({passes} passes)", fmt_count(n));

        for (name, kernel) in [("shmem   ", &shmem), ("subgroup", &subgroup)] {
            // warmup
            reduce_sum_batched(gpu, kernel, &input, n);

            // ── Sync (fence per pass) ──
            let start = Instant::now();
            for _ in 0..iters {
                reduce_sum_sync(gpu, kernel, &input, n);
            }
            let sync_per = start.elapsed() / iters;
            let sync_gbps = bytes / sync_per.as_secs_f64() / 1e9;

            // ── Batched (one fence for all passes) ──
            let start = Instant::now();
            for _ in 0..iters {
                reduce_sum_batched(gpu, kernel, &input, n);
            }
            let batch_per = start.elapsed() / iters;
            let batch_gbps = bytes / batch_per.as_secs_f64() / 1e9;

            let speedup = sync_per.as_nanos() as f64 / batch_per.as_nanos() as f64;

            println!(
                "    {name}  sync {:>7.2?} {:>5.1} GB/s  │  batch {:>7.2?} {:>5.1} GB/s  ({speedup:.1}x)",
                sync_per, sync_gbps, batch_per, batch_gbps
            );
        }
    }
    println!();
}

fn reduce_sum_sync(gpu: &Device, kernel: &Kernel, input: &dyn GpuBuf, n: u32) -> f32 {
    let out_n = n.div_ceil(256);
    let out = gpu.alloc::<f32>(out_n as usize).expect("alloc");
    gpu.run(kernel, &[input, &out], n).expect("run");

    if out_n == 1 {
        return out.download().expect("download")[0];
    }
    reduce_sum_sync(gpu, kernel, &out, out_n)
}

fn reduce_sum_batched(gpu: &Device, kernel: &Kernel, input: &dyn GpuBuf, n: u32) -> f32 {
    // Pre-allocate all intermediate buffers
    let mut sizes = vec![n];
    let mut s = n;
    while s > 1 {
        s = s.div_ceil(256);
        sizes.push(s);
    }

    let intermediates: Vec<scry_gpu::Buffer<f32>> = sizes[1..]
        .iter()
        .map(|&s| gpu.alloc::<f32>(s as usize).expect("alloc"))
        .collect();

    // Record all passes into one batch with barriers
    let mut batch = gpu.batch().expect("batch");
    let mut prev: &dyn GpuBuf = input;
    let mut len = n;
    for out in &intermediates {
        batch.run(kernel, &[prev, out], len).expect("batch run");
        batch.barrier();
        len = len.div_ceil(256);
        prev = out;
    }
    batch.submit().expect("submit");

    // Read back the final scalar
    intermediates.last().unwrap().download().expect("download")[0]
}

fn count_passes(mut n: u32) -> u32 {
    let mut passes = 0;
    while n > 1 {
        n = n.div_ceil(256);
        passes += 1;
    }
    passes
}

// ── Matrix multiply (compute-bound) ─────────────────────────────────────────

struct MatmulKernel {
    name: &'static str,
    kernel: Kernel,
    tile_m: u32, // output tile rows (0 = 1D dispatch)
    tile_n: u32, // output tile cols (0 = 1D dispatch)
}

fn bench_matmul(gpu: &Device) {
    println!("═══ Matrix Multiply — C = A × B ═══");

    let variants = [
        MatmulKernel {
            name: "naive",
            kernel: gpu.compile(MATMUL_NAIVE_SHADER).expect("compile naive"),
            tile_m: 0,
            tile_n: 0,
        },
        MatmulKernel {
            name: "tiled 16×16",
            kernel: gpu.compile(TILED_16X16).expect("compile tiled"),
            tile_m: 16,
            tile_n: 16,
        },
        MatmulKernel {
            name: "coarse 4×4",
            kernel: gpu.compile(COARSE_64X64).expect("compile coarse"),
            tile_m: 64,
            tile_n: 64,
        },
        MatmulKernel {
            name: "coarse+L2",
            kernel: gpu.compile(MATMUL_COARSE_L2_SHADER).expect("compile coarse+L2"),
            tile_m: 64,
            tile_n: 64,
        },
        MatmulKernel {
            name: "large 128×128",
            kernel: gpu.compile(MATMUL_LARGE_SHADER).expect("compile large"),
            tile_m: 128,
            tile_n: 128,
        },
    ];

    for &n in &[512u32, 1024, 2048, 4096] {
        let iters: u32 = if n >= 4096 { 5 } else if n >= 2048 { 20 } else { 50 };
        let elems = (n * n) as usize;

        let a_data: Vec<f32> = (0..elems).map(|i| (i % 100) as f32 * 0.01).collect();
        let b_data: Vec<f32> = (0..elems).map(|i| (i % 100) as f32 * 0.01).collect();

        let a = gpu.upload(&a_data).expect("upload");
        let b = gpu.upload(&b_data).expect("upload");

        let dims: [u32; 3] = [n, n, n];
        let pc = bytemuck::bytes_of(&dims);
        let flops = 2.0 * (n as f64).powi(3);

        println!("  {n}×{n}  ({iters} iters)");

        for v in &variants {
            // Skip naive at 4096 (exceeds 1D dispatch limit of 65535 workgroups)
            if v.tile_m == 0 && n > 2048 {
                continue;
            }
            // Skip large tiles for small matrices
            if v.tile_m >= 128 && n < 1024 {
                continue;
            }

            let c = gpu.alloc::<f32>(elems).expect("alloc");

            // warmup
            dispatch_matmul(gpu, v, &a, &b, &c, n, pc);

            let start = Instant::now();
            for _ in 0..iters {
                dispatch_matmul(gpu, v, &a, &b, &c, n, pc);
            }
            let per_iter = start.elapsed() / iters;
            let gflops = flops / per_iter.as_secs_f64() / 1e9;

            println!(
                "    {:<14} {:>9.2?}/iter  {:>7.1} GFLOPS",
                v.name, per_iter, gflops
            );
        }
    }
    println!();
}

fn dispatch_matmul(
    gpu: &Device,
    v: &MatmulKernel,
    a: &scry_gpu::Buffer<f32>,
    b: &scry_gpu::Buffer<f32>,
    c: &scry_gpu::Buffer<f32>,
    n: u32,
    pc: &[u8],
) {
    if v.tile_m == 0 {
        // 1D dispatch (naive)
        gpu.run_with_push_constants(&v.kernel, &[a, b, c], n * n, pc)
            .expect("run");
    } else {
        // 2D dispatch (tiled/coarsened)
        let wg_y = n.div_ceil(v.tile_m);
        let wg_x = n.div_ceil(v.tile_n);
        gpu.run_configured(&v.kernel, &[a, b, c], [wg_x, wg_y, 1], Some(pc))
            .expect("run");
    }
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
