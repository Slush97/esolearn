//! Backend comparison: scry-gpu (Vulkan/ash) vs wgpu.
//!
//! Quantifies two things:
//! 1. **Backend overhead** — same tiled 16×16 shader, different backends
//! 2. **Total migration benefit** — wgpu tiled vs scry-gpu coarsened 64×64
//!
//! Run with:
//!   cargo run -p scry-gpu --example bench_wgpu_compare --features bench-wgpu --release

use std::time::Instant;

use scry_gpu::{Device, Kernel};
use wgpu::util::DeviceExt;

// ── Shaders ─────────────────────────────────────────────────────────────────

/// Exact shader from scry-llm and scry-learn: tiled 16×16, uniform buffer dims.
const WGPU_TILED_SHADER: &str = "\
struct Dimensions { M: u32, K: u32, N: u32, _pad: u32 }

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

const TILE: u32 = 16u;
var<workgroup> tileA: array<array<f32, TILE>, TILE>;
var<workgroup> tileB: array<array<f32, TILE>, TILE>;

@compute @workgroup_size(TILE, TILE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    var sum: f32 = 0.0;
    let numTiles = (dims.K + TILE - 1u) / TILE;
    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        let a_col = t * TILE + lid.x;
        if row < dims.M && a_col < dims.K {
            tileA[lid.y][lid.x] = A[row * dims.K + a_col];
        } else { tileA[lid.y][lid.x] = 0.0; }
        let b_row = t * TILE + lid.y;
        if b_row < dims.K && col < dims.N {
            tileB[lid.y][lid.x] = B[b_row * dims.N + col];
        } else { tileB[lid.y][lid.x] = 0.0; }
        workgroupBarrier();
        for (var k: u32 = 0u; k < TILE; k = k + 1u) {
            sum = sum + tileA[lid.y][k] * tileB[k][lid.x];
        }
        workgroupBarrier();
    }
    if row < dims.M && col < dims.N {
        C[row * dims.N + col] = sum;
    }
}";

/// scry-gpu tiled 16×16 — same algorithm, push constants for dims.
const SCRY_TILED_SHADER: &str = "\
struct Dims { M: u32, N: u32, K: u32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.y * 16u + lid.y;
    let col = wid.x * 16u + lid.x;
    let lr = lid.y;
    let lc = lid.x;
    var sum = 0.0;
    let num_tiles = (dims.K + 15u) / 16u;
    for (var t = 0u; t < num_tiles; t++) {
        let a_col = t * 16u + lc;
        if row < dims.M && a_col < dims.K {
            tile_a[lr * 16u + lc] = A[row * dims.K + a_col];
        } else { tile_a[lr * 16u + lc] = 0.0; }
        let b_row = t * 16u + lr;
        if b_row < dims.K && col < dims.N {
            tile_b[lr * 16u + lc] = B[b_row * dims.N + col];
        } else { tile_b[lr * 16u + lc] = 0.0; }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k++) {
            sum += tile_a[lr * 16u + k] * tile_b[k * 16u + lc];
        }
        workgroupBarrier();
    }
    if row < dims.M && col < dims.N {
        C[row * dims.N + col] = sum;
    }
}";

/// scry-gpu coarse 4×4: 64×64 output tile, 16 accumulators per thread.
/// Bank-conflict-free A tile (padded stride 17). 16 FLOP/byte.
const SCRY_COARSE_SHADER: &str = "\
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
    let block_row = wid.y * 64u;
    let block_col = wid.x * 64u;
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
            } else { sa[r * 17u + c] = 0.0; }
        }
        for (var x = 0u; x < 4u; x++) {
            let flat = li * 4u + x;
            let r = flat / 64u;
            let c = flat % 64u;
            let gr = kt * 16u + r;
            let gc = block_col + c;
            if gr < dims.K && gc < dims.N {
                sb[flat] = B[gr * dims.N + gc];
            } else { sb[flat] = 0.0; }
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

// ── wgpu backend ────────────────────────────────────────────────────────────

struct WgpuCtx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

fn init_wgpu() -> WgpuCtx {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("no wgpu adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("bench"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .expect("wgpu device failed");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(WGPU_TILED_SHADER.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            bgl_entry(0, wgpu::BufferBindingType::Uniform),
            bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
            bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
            bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
        ],
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    WgpuCtx { device, queue, pipeline, bgl }
}

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Dispatch with pre-allocated wgpu buffers (pure dispatch overhead).
fn wgpu_dispatch(ctx: &WgpuCtx, bg: &wgpu::BindGroup, wg_x: u32, wg_y: u32) {
    let mut enc =
        ctx.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    ctx.queue.submit(std::iter::once(enc.finish()));
    ctx.device.poll(wgpu::Maintain::Wait);
}

/// Full end-to-end wgpu matmul: alloc + upload + dispatch + readback.
/// Mirrors scry-llm's `gpu_matmul()` exactly.
fn wgpu_matmul_e2e(ctx: &WgpuCtx, a: &[f32], b: &[f32], n: u32) -> Vec<f32> {
    let elems = (n * n) as usize;
    let dims = [n, n, n, 0u32];

    let dims_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let a_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(a),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let b_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(b),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let c_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (elems * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (elems * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &ctx.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: a_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: c_buf.as_entire_binding() },
        ],
    });

    let mut enc =
        ctx.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n.div_ceil(16), n.div_ceil(16), 1);
    }
    enc.copy_buffer_to_buffer(&c_buf, 0, &readback, 0, (elems * 4) as u64);
    ctx.queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let mapped = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    result
}

/// Full end-to-end scry-gpu matmul: upload + dispatch + download.
fn scry_matmul_e2e(dev: &Device, kernel: &Kernel, a: &[f32], b: &[f32], n: u32, tile: u32) -> Vec<f32> {
    let sa = dev.upload(a).unwrap();
    let sb = dev.upload(b).unwrap();
    let sc = dev.alloc::<f32>((n * n) as usize).unwrap();
    let dims: [u32; 3] = [n, n, n];
    dev.run_configured(
        kernel,
        &[&sa, &sb, &sc],
        [n.div_ceil(tile), n.div_ceil(tile), 1],
        Some(bytemuck::bytes_of(&dims)),
    )
    .unwrap();
    sc.download().unwrap()
}

// ── Benchmarks ──────────────────────────────────────────────────────────────

fn main() {
    let scry = Device::auto().expect("no GPU");
    let wctx = init_wgpu();

    println!(
        "Device: {} ({} MB)\n",
        scry.name(),
        scry.memory() / (1024 * 1024)
    );

    let scry_tiled = scry.compile(SCRY_TILED_SHADER).expect("compile tiled");
    let scry_coarse = scry.compile(SCRY_COARSE_SHADER).expect("compile coarse");

    // ── Section 1: Dispatch throughput ──────────────────────────────────
    println!("═══ Dispatch throughput (pre-allocated buffers) ═══");

    for &n in &[256u32, 512, 1024, 2048, 4096] {
        let iters: u32 = if n >= 4096 { 10 } else if n >= 2048 { 20 } else { 50 };
        let elems = (n * n) as usize;
        let flops = 2.0 * (n as f64).powi(3);

        let a_data: Vec<f32> = (0..elems).map(|i| (i % 100) as f32 * 0.01).collect();
        let b_data: Vec<f32> = (0..elems).map(|i| (i % 100) as f32 * 0.01).collect();

        // wgpu setup
        let wgpu_dims = [n, n, n, 0u32];
        let wgpu_dims_buf = wctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&wgpu_dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let wgpu_a = wctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&a_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let wgpu_b = wctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&b_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let wgpu_c = wctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let wgpu_bg = wctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &wctx.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu_dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu_c.as_entire_binding() },
            ],
        });
        let wg = n.div_ceil(16);

        // scry-gpu setup
        let scry_a = scry.upload(&a_data).expect("upload");
        let scry_b = scry.upload(&b_data).expect("upload");
        let scry_c = scry.alloc::<f32>(elems).expect("alloc");
        let pc_dims: [u32; 3] = [n, n, n];
        let pc = bytemuck::bytes_of(&pc_dims);

        println!("  {n}×{n}  ({iters} iters)");

        // wgpu tiled 16×16
        wgpu_dispatch(&wctx, &wgpu_bg, wg, wg);
        let start = Instant::now();
        for _ in 0..iters {
            wgpu_dispatch(&wctx, &wgpu_bg, wg, wg);
        }
        let wgpu_t = start.elapsed() / iters;
        let wgpu_gf = flops / wgpu_t.as_secs_f64() / 1e9;

        // scry-gpu tiled 16×16
        scry.run_configured(
            &scry_tiled, &[&scry_a, &scry_b, &scry_c],
            [n.div_ceil(16), n.div_ceil(16), 1], Some(pc),
        ).unwrap();
        let start = Instant::now();
        for _ in 0..iters {
            scry.run_configured(
                &scry_tiled, &[&scry_a, &scry_b, &scry_c],
                [n.div_ceil(16), n.div_ceil(16), 1], Some(pc),
            ).unwrap();
        }
        let scry_tiled_t = start.elapsed() / iters;
        let scry_tiled_gf = flops / scry_tiled_t.as_secs_f64() / 1e9;

        // scry-gpu coarse 4×4
        scry.run_configured(
            &scry_coarse, &[&scry_a, &scry_b, &scry_c],
            [n.div_ceil(64), n.div_ceil(64), 1], Some(pc),
        ).unwrap();
        let start = Instant::now();
        for _ in 0..iters {
            scry.run_configured(
                &scry_coarse, &[&scry_a, &scry_b, &scry_c],
                [n.div_ceil(64), n.div_ceil(64), 1], Some(pc),
            ).unwrap();
        }
        let scry_coarse_t = start.elapsed() / iters;
        let scry_coarse_gf = flops / scry_coarse_t.as_secs_f64() / 1e9;

        let backend_x = wgpu_t.as_nanos() as f64 / scry_tiled_t.as_nanos() as f64;
        let total_x = wgpu_t.as_nanos() as f64 / scry_coarse_t.as_nanos() as f64;

        println!(
            "    wgpu tiled      {:>9.2?}  {:>7.1} GFLOPS",
            wgpu_t, wgpu_gf
        );
        println!(
            "    scry tiled      {:>9.2?}  {:>7.1} GFLOPS  (backend: {backend_x:.2}x)",
            scry_tiled_t, scry_tiled_gf
        );
        println!(
            "    scry coarse     {:>9.2?}  {:>7.1} GFLOPS  (total: {total_x:.1}x)",
            scry_coarse_t, scry_coarse_gf
        );
    }

    // ── Section 2: End-to-end ──────────────────────────────────────────
    println!("\n═══ End-to-end (alloc + upload + dispatch + readback) ═══");
    println!("  simulates scry-llm's per-call pattern\n");

    for &n in &[256u32, 512, 1024, 2048] {
        let iters: u32 = if n >= 2048 { 10 } else if n >= 1024 { 20 } else { 50 };
        let elems = (n * n) as usize;
        let flops = 2.0 * (n as f64).powi(3);

        let a_data: Vec<f32> = (0..elems).map(|i| (i % 100) as f32 * 0.01).collect();
        let b_data: Vec<f32> = (0..elems).map(|i| (i % 100) as f32 * 0.01).collect();

        println!("  {n}×{n}  ({iters} iters)");

        // wgpu end-to-end (warmup + timed)
        let _ = wgpu_matmul_e2e(&wctx, &a_data, &b_data, n);
        let start = Instant::now();
        for _ in 0..iters {
            let _ = wgpu_matmul_e2e(&wctx, &a_data, &b_data, n);
        }
        let wgpu_t = start.elapsed() / iters;
        let wgpu_gf = flops / wgpu_t.as_secs_f64() / 1e9;

        // scry-gpu end-to-end (warmup + timed)
        let _ = scry_matmul_e2e(&scry, &scry_coarse, &a_data, &b_data, n, 64);
        let start = Instant::now();
        for _ in 0..iters {
            let _ = scry_matmul_e2e(&scry, &scry_coarse, &a_data, &b_data, n, 64);
        }
        let scry_t = start.elapsed() / iters;
        let scry_gf = flops / scry_t.as_secs_f64() / 1e9;

        let speedup = wgpu_t.as_nanos() as f64 / scry_t.as_nanos() as f64;

        println!(
            "    wgpu (scry-llm)  {:>9.2?}  {:>7.1} GFLOPS",
            wgpu_t, wgpu_gf
        );
        println!(
            "    scry-gpu         {:>9.2?}  {:>7.1} GFLOPS  ({speedup:.1}x)",
            scry_t, scry_gf
        );
    }

    println!();
}
