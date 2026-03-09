//! WGPU compute backend — GPU-accelerated matmul via compute shaders.
//!
//! Only `matmul` runs on GPU; all other `MathBackend` methods delegate to
//! `CpuBackend` for simplicity.
//!
//! Because `MathBackend` trait methods are static (no `&self`), we store
//! the GPU context in a `OnceLock` initialized on first use.

use std::sync::OnceLock;

use crate::backend::cpu::CpuBackend;
use crate::backend::{DeviceBackend, MathBackend};
use crate::tensor::shape::Shape;

/// Minimum M*K*N product before engaging GPU (below this, CPU/BLAS is faster
/// due to wgpu's per-dispatch overhead: buffer creation, submission, readback).
const GPU_MIN_ELEMENTS: usize = 65_536;

/// Maximum GPU buffer size in bytes (128 MiB).
const MAX_GPU_BUFFER_BYTES: u64 = 128 * 1024 * 1024;

// ---------------------------------------------------------------------------
// GPU context — cached device, queue, and compiled pipeline
// ---------------------------------------------------------------------------

struct WgpuContext {
    device: ::wgpu::Device,
    queue: ::wgpu::Queue,
    matmul_pipeline: ::wgpu::ComputePipeline,
    matmul_bgl: ::wgpu::BindGroupLayout,
}

// Safety: wgpu device/queue are Send+Sync
unsafe impl Send for WgpuContext {}
unsafe impl Sync for WgpuContext {}

/// Global GPU context, initialized on first matmul call.
static GPU_CTX: OnceLock<Option<WgpuContext>> = OnceLock::new();

fn get_gpu_ctx() -> Option<&'static WgpuContext> {
    GPU_CTX
        .get_or_init(|| {
            match init_wgpu_context() {
                Ok(ctx) => Some(ctx),
                Err(e) => {
                    eprintln!("[scry-llm] WGPU init failed, falling back to CPU: {e}");
                    None
                }
            }
        })
        .as_ref()
}

fn init_wgpu_context() -> Result<WgpuContext, String> {
    let instance = ::wgpu::Instance::new(&::wgpu::InstanceDescriptor {
        backends: ::wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&::wgpu::RequestAdapterOptions {
        power_preference: ::wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or("wgpu: no compatible GPU adapter found")?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &::wgpu::DeviceDescriptor {
            label: Some("scry-llm-compute"),
            required_features: ::wgpu::Features::empty(),
            required_limits: ::wgpu::Limits::default(),
            memory_hints: ::wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .map_err(|e| format!("wgpu: device creation failed: {e}"))?;

    let shader = device.create_shader_module(::wgpu::ShaderModuleDescriptor {
        label: Some("matmul_shader"),
        source: ::wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul.wgsl").into()),
    });

    let matmul_bgl = device.create_bind_group_layout(&::wgpu::BindGroupLayoutDescriptor {
        label: Some("matmul_bgl"),
        entries: &[
            bgl_entry(0, ::wgpu::BufferBindingType::Uniform),
            bgl_entry(1, ::wgpu::BufferBindingType::Storage { read_only: true }),
            bgl_entry(2, ::wgpu::BufferBindingType::Storage { read_only: true }),
            bgl_entry(3, ::wgpu::BufferBindingType::Storage { read_only: false }),
        ],
    });

    let layout = device.create_pipeline_layout(&::wgpu::PipelineLayoutDescriptor {
        label: Some("matmul_layout"),
        bind_group_layouts: &[&matmul_bgl],
        push_constant_ranges: &[],
    });

    let matmul_pipeline = device.create_compute_pipeline(&::wgpu::ComputePipelineDescriptor {
        label: Some("matmul_pipeline"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: ::wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    Ok(WgpuContext {
        device,
        queue,
        matmul_pipeline,
        matmul_bgl,
    })
}

fn bgl_entry(binding: u32, ty: ::wgpu::BufferBindingType) -> ::wgpu::BindGroupLayoutEntry {
    ::wgpu::BindGroupLayoutEntry {
        binding,
        visibility: ::wgpu::ShaderStages::COMPUTE,
        ty: ::wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// ---------------------------------------------------------------------------
// GPU matmul dispatch
// ---------------------------------------------------------------------------

use ::wgpu::util::DeviceExt;

/// Run a single matmul on GPU. Returns None if GPU is unavailable.
fn gpu_matmul(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Option<Vec<f32>> {
    let ctx = get_gpu_ctx()?;

    let c_size = m * n;
    let dims = [m as u32, k as u32, n as u32, 0u32];

    let device = &ctx.device;
    let queue = &ctx.queue;

    let dims_buf = device.create_buffer_init(&::wgpu::util::BufferInitDescriptor {
        label: Some("mm_dims"),
        contents: bytemuck::cast_slice(&dims),
        usage: ::wgpu::BufferUsages::UNIFORM,
    });

    let a_buf = device.create_buffer_init(&::wgpu::util::BufferInitDescriptor {
        label: Some("mm_a"),
        contents: bytemuck::cast_slice(a),
        usage: ::wgpu::BufferUsages::STORAGE,
    });

    let b_buf = device.create_buffer_init(&::wgpu::util::BufferInitDescriptor {
        label: Some("mm_b"),
        contents: bytemuck::cast_slice(b),
        usage: ::wgpu::BufferUsages::STORAGE,
    });

    let c_buf = device.create_buffer(&::wgpu::BufferDescriptor {
        label: Some("mm_c"),
        size: (c_size * 4) as u64,
        usage: ::wgpu::BufferUsages::STORAGE | ::wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buf = device.create_buffer(&::wgpu::BufferDescriptor {
        label: Some("mm_readback"),
        size: (c_size * 4) as u64,
        usage: ::wgpu::BufferUsages::MAP_READ | ::wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&::wgpu::BindGroupDescriptor {
        label: Some("mm_bg"),
        layout: &ctx.matmul_bgl,
        entries: &[
            ::wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() },
            ::wgpu::BindGroupEntry { binding: 1, resource: a_buf.as_entire_binding() },
            ::wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() },
            ::wgpu::BindGroupEntry { binding: 3, resource: c_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&::wgpu::CommandEncoderDescriptor {
        label: Some("mm_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&::wgpu::ComputePassDescriptor {
            label: Some("mm_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.matmul_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let wg_x = n.div_ceil(16) as u32;
        let wg_y = m.div_ceil(16) as u32;
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    encoder.copy_buffer_to_buffer(&c_buf, 0, &readback_buf, 0, (c_size * 4) as u64);
    queue.submit(std::iter::once(encoder.finish()));

    // Readback
    let slice = readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(::wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(::wgpu::Maintain::Wait);

    if rx.recv().unwrap_or(Err(::wgpu::BufferAsyncError)).is_err() {
        return None;
    }

    let mapped = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    readback_buf.unmap();

    Some(result[..c_size].to_vec())
}

/// Transpose [rows × cols] → [cols × rows] on CPU.
fn transpose_cpu(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Check if a matmul is worth sending to GPU.
fn should_use_gpu(m: usize, k: usize, n: usize) -> bool {
    if m * k * n < GPU_MIN_ELEMENTS {
        return false;
    }
    let a_bytes = (m * k * 4) as u64;
    let b_bytes = (k * n * 4) as u64;
    let c_bytes = (m * n * 4) as u64;
    a_bytes <= MAX_GPU_BUFFER_BYTES
        && b_bytes <= MAX_GPU_BUFFER_BYTES
        && c_bytes <= MAX_GPU_BUFFER_BYTES
}

/// Matmul with GPU acceleration: handles transpose and size thresholds.
fn matmul_gpu_or_cpu(
    a: &Vec<f32>,
    b: &Vec<f32>,
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Vec<f32> {
    if !should_use_gpu(m, k, n) {
        return CpuBackend::matmul(a, b, m, k, n, trans_a, trans_b);
    }

    // Handle transposes: the shader expects row-major A[M×K] × B[K×N]
    let a_rm;
    let a_data: &[f32] = if trans_a {
        a_rm = transpose_cpu(a, k, m);
        &a_rm
    } else {
        a
    };

    let b_rm;
    let b_data: &[f32] = if trans_b {
        b_rm = transpose_cpu(b, n, k);
        &b_rm
    } else {
        b
    };

    gpu_matmul(a_data, b_data, m, k, n)
        .unwrap_or_else(|| CpuBackend::matmul(a, b, m, k, n, trans_a, trans_b))
}

// ---------------------------------------------------------------------------
// WgpuBackend — public type
// ---------------------------------------------------------------------------

/// GPU-accelerated backend for scry-llm using wgpu compute shaders.
///
/// Matmul dispatches to the GPU via a global `OnceLock` context;
/// all other ops use `CpuBackend`. Storage is `Vec<f32>` (CPU-resident).
pub struct WgpuBackend;

impl DeviceBackend for WgpuBackend {
    type Storage = Vec<f32>;
    type Stream = ();
    #[cfg(feature = "quantize")]
    type I8Storage = Vec<i8>;

    #[cfg(feature = "quantize")]
    fn i8_from_vec(data: Vec<i8>) -> Vec<i8> { data }
    #[cfg(feature = "quantize")]
    fn i8_to_vec(storage: &Vec<i8>) -> Vec<i8> { storage.clone() }

    fn zeros(shape: &Shape) -> Vec<f32> { CpuBackend::zeros(shape) }
    fn ones(shape: &Shape) -> Vec<f32> { CpuBackend::ones(shape) }
    fn from_vec(data: Vec<f32>, shape: &Shape) -> Vec<f32> { CpuBackend::from_vec(data, shape) }
    fn to_vec(storage: &Vec<f32>) -> Vec<f32> { CpuBackend::to_vec(storage) }
    fn clone_storage(storage: &Vec<f32>) -> Vec<f32> { CpuBackend::clone_storage(storage) }
}

impl MathBackend for WgpuBackend {
    fn matmul(
        a: &Vec<f32>,
        b: &Vec<f32>,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Vec<f32> {
        matmul_gpu_or_cpu(a, b, m, k, n, trans_a, trans_b)
    }

    fn add(a: &Vec<f32>, b: &Vec<f32>, a_shape: &Shape, b_shape: &Shape, out_shape: &Shape) -> Vec<f32> {
        CpuBackend::add(a, b, a_shape, b_shape, out_shape)
    }

    fn softmax(input: &Vec<f32>, shape: &Shape) -> Vec<f32> {
        CpuBackend::softmax(input, shape)
    }

    fn layernorm(input: &Vec<f32>, gamma: &Vec<f32>, beta: &Vec<f32>, shape: &Shape, eps: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        CpuBackend::layernorm(input, gamma, beta, shape, eps)
    }

    fn gelu(input: &Vec<f32>) -> Vec<f32> {
        CpuBackend::gelu(input)
    }

    fn embedding(weight: &Vec<f32>, indices: &[usize], vocab: usize, dim: usize) -> Vec<f32> {
        CpuBackend::embedding(weight, indices, vocab, dim)
    }

    fn sum(input: &Vec<f32>) -> f32 {
        CpuBackend::sum(input)
    }

    fn mul_elementwise(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
        CpuBackend::mul_elementwise(a, b)
    }

    fn scale(a: &Vec<f32>, scalar: f32) -> Vec<f32> {
        CpuBackend::scale(a, scalar)
    }

    fn concat_rows(a: &Vec<f32>, b: &Vec<f32>, a_rows: usize, b_rows: usize, cols: usize) -> Vec<f32> {
        CpuBackend::concat_rows(a, b, a_rows, b_rows, cols)
    }

    fn rmsnorm(input: &Vec<f32>, weight: &Vec<f32>, shape: &Shape, eps: f32) -> Vec<f32> {
        CpuBackend::rmsnorm(input, weight, shape, eps)
    }

    fn rope(input: &Vec<f32>, shape: &Shape, pos: usize, head_dim: usize, theta: f32) -> Vec<f32> {
        CpuBackend::rope(input, shape, pos, head_dim, theta)
    }

    fn rope_with_freqs_preloaded(
        input: &Vec<f32>, seq: usize, n_heads: usize, head_dim: usize,
        start_pos: usize, freqs: &Vec<f32>,
    ) -> Vec<f32> {
        CpuBackend::rope_with_freqs_preloaded(input, seq, n_heads, head_dim, start_pos, freqs)
    }

    fn swiglu(gate: &Vec<f32>, up: &Vec<f32>) -> Vec<f32> {
        CpuBackend::swiglu(gate, up)
    }

    fn repeat_kv(input: &Vec<f32>, n_kv_heads: usize, n_q_heads: usize, seq: usize, d_head: usize) -> Vec<f32> {
        CpuBackend::repeat_kv(input, n_kv_heads, n_q_heads, seq, d_head)
    }
}
