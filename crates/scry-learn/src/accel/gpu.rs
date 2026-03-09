// SPDX-License-Identifier: MIT OR Apache-2.0
//! GPU compute backend via wgpu compute shaders.
//!
//! Provides accelerated matrix operations using wgpu's compute pipeline.
//! All data is converted f64 → f32 for GPU compute (sufficient for ML
//! feature values which are typically in [-1e6, 1e6] range).

use super::ComputeBackend;

/// Maximum GPU buffer size in bytes (128 MiB).
///
/// wgpu's default `max_buffer_size` is 256 MiB, but we use a conservative
/// limit so that the combined allocation of input + output buffers stays
/// well under the adapter limit.
const MAX_GPU_BUFFER_BYTES: u64 = 128 * 1024 * 1024;

/// GPU compute context — cached wgpu device, queue, and compiled pipelines.
///
/// Creating a `GpuContext` is expensive (~100ms) due to adapter/device init
/// and shader compilation. Create once and reuse via [`GpuBackend`].
struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    matmul_pipeline: wgpu::ComputePipeline,
    matmul_bgl: wgpu::BindGroupLayout,
    distance_pipeline: wgpu::ComputePipeline,
    distance_bgl: wgpu::BindGroupLayout,
}

impl GpuContext {
    fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or("wgpu: no compatible GPU adapter found")?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("scry-learn-compute"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| format!("wgpu: device creation failed: {e}"))?;

        // --- Matmul pipeline ---
        let matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul.wgsl").into()),
        });

        let matmul_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matmul_bgl"),
            entries: &[
                // dims uniform
                bgl_entry(0, wgpu::BufferBindingType::Uniform, false),
                // A storage (read)
                bgl_entry(
                    1,
                    wgpu::BufferBindingType::Storage { read_only: true },
                    false,
                ),
                // B storage (read)
                bgl_entry(
                    2,
                    wgpu::BufferBindingType::Storage { read_only: true },
                    false,
                ),
                // C storage (read_write)
                bgl_entry(
                    3,
                    wgpu::BufferBindingType::Storage { read_only: false },
                    false,
                ),
            ],
        });

        let matmul_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("matmul_layout"),
            bind_group_layouts: &[&matmul_bgl],
            push_constant_ranges: &[],
        });

        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul_pipeline"),
            layout: Some(&matmul_layout),
            module: &matmul_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // --- Distance pipeline ---
        let distance_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("distance_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/distance.wgsl").into()),
        });

        let distance_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("distance_bgl"),
            entries: &[
                bgl_entry(0, wgpu::BufferBindingType::Uniform, false),
                bgl_entry(
                    1,
                    wgpu::BufferBindingType::Storage { read_only: true },
                    false,
                ),
                bgl_entry(
                    2,
                    wgpu::BufferBindingType::Storage { read_only: true },
                    false,
                ),
                bgl_entry(
                    3,
                    wgpu::BufferBindingType::Storage { read_only: false },
                    false,
                ),
            ],
        });

        let distance_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("distance_layout"),
            bind_group_layouts: &[&distance_bgl],
            push_constant_ranges: &[],
        });

        let distance_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("distance_pipeline"),
            layout: Some(&distance_layout),
            module: &distance_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            matmul_pipeline,
            matmul_bgl,
            distance_pipeline,
            distance_bgl,
        })
    }
}

/// Helper to create a bind group layout entry for compute shaders.
fn bgl_entry(
    binding: u32,
    ty: wgpu::BufferBindingType,
    _has_dynamic_offset: bool,
) -> wgpu::BindGroupLayoutEntry {
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

// ---------------------------------------------------------------------------
// GpuBackend — public API
// ---------------------------------------------------------------------------

/// GPU-accelerated compute backend using wgpu compute shaders.
///
/// Wraps a [`GpuContext`] and implements [`ComputeBackend`] for
/// matrix multiply and pairwise distance computation.
///
/// # Precision
///
/// All GPU computation uses f32. Input f64 values are cast to f32
/// before upload and results are cast back to f64 after readback.
/// This is sufficient for ML feature values but may lose precision
/// for values > 2²⁴ ≈ 16.7M.
#[non_exhaustive]
pub struct GpuBackend {
    ctx: GpuContext,
}

impl GpuBackend {
    /// Create a new GPU backend.
    ///
    /// # Errors
    ///
    /// Returns an error string if no compatible GPU adapter is found.
    pub fn new() -> Result<Self, String> {
        let ctx = GpuContext::new()?;
        Ok(Self { ctx })
    }
}

use wgpu::util::DeviceExt;

impl ComputeBackend for GpuBackend {
    fn matmul(&self, a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), k * n);

        // Zero-dimension guard — wgpu rejects 0-size buffers.
        if m == 0 || k == 0 || n == 0 {
            return vec![0.0; m * n];
        }

        // Size threshold: GPU overhead not worth it for small matrices
        if m * k * n < 4096 {
            return super::CpuBackend.matmul(a, b, m, k, n);
        }

        // Buffer size guard — fall back to CPU if any buffer exceeds the GPU limit.
        let a_bytes = (m * k * std::mem::size_of::<f32>()) as u64;
        let b_bytes = (k * n * std::mem::size_of::<f32>()) as u64;
        let c_bytes = (m * n * std::mem::size_of::<f32>()) as u64;
        if a_bytes > MAX_GPU_BUFFER_BYTES
            || b_bytes > MAX_GPU_BUFFER_BYTES
            || c_bytes > MAX_GPU_BUFFER_BYTES
        {
            return super::CpuBackend.matmul(a, b, m, k, n);
        }

        let a_f32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
        let b_f32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
        let c_size = m * n;

        let dims = [m as u32, k as u32, n as u32, 0u32];
        let dims_bytes = bytemuck::cast_slice(&dims);

        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matmul_dims"),
            contents: dims_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matmul_a"),
            contents: bytemuck::cast_slice(&a_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matmul_b"),
            contents: bytemuck::cast_slice(&b_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let c_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matmul_c"),
            size: (c_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matmul_readback"),
            size: (c_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bg"),
            layout: &self.ctx.matmul_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dims_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: c_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ctx.matmul_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Workgroup size is 16×16, dispatch enough to cover M×N output
            let wg_x = n.div_ceil(16) as u32;
            let wg_y = m.div_ceil(16) as u32;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        encoder.copy_buffer_to_buffer(&c_buf, 0, &readback_buf, 0, (c_size * 4) as u64);
        queue.submit(std::iter::once(encoder.finish()));

        match read_buffer_f32(device, &readback_buf, c_size) {
            Ok(result) => result,
            Err(_) => super::CpuBackend.matmul(a, b, m, k, n),
        }
    }

    fn xtx_xty(&self, features: &[Vec<f64>], target: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n_samples = target.len();
        let n_features = features.len();

        // GPU is only worth it for larger matrices
        // XᵀX cost is O(n_samples * n_features²)
        if n_samples * n_features * n_features < 50_000 {
            return super::CpuBackend.xtx_xty(features, target);
        }

        // Build augmented X matrix: [1, x1, x2, ...] (row-major, n_samples × dim)
        let dim = n_features + 1;
        let mut x_f32 = Vec::with_capacity(n_samples * dim);
        for i in 0..n_samples {
            x_f32.push(1.0f32); // intercept
            for feat in features {
                x_f32.push(feat[i] as f32);
            }
        }

        // Xᵀ is dim × n_samples (transpose of X)
        let mut xt_f32 = vec![0.0f32; dim * n_samples];
        for i in 0..n_samples {
            for j in 0..dim {
                xt_f32[j * n_samples + i] = x_f32[i * dim + j];
            }
        }

        // XᵀX = matmul(Xᵀ, X) — dim×n_samples × n_samples×dim = dim×dim
        let xtx_f64 = self.matmul(
            &xt_f32.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            &x_f32.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            dim,
            n_samples,
            dim,
        );

        // Xᵀy = matmul(Xᵀ, y) — dim×n_samples × n_samples×1 = dim×1
        let xty_f64 = self.matmul(
            &xt_f32.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            target,
            dim,
            n_samples,
            1,
        );

        (xtx_f64, xty_f64)
    }

    fn pairwise_distances_squared(
        &self,
        queries: &[f64],
        train: &[f64],
        n_q: usize,
        n_t: usize,
        dim: usize,
    ) -> Vec<f64> {
        debug_assert_eq!(queries.len(), n_q * dim);
        debug_assert_eq!(train.len(), n_t * dim);

        // Zero-dimension guard — wgpu rejects 0-size buffers.
        if n_q == 0 || n_t == 0 || dim == 0 {
            return vec![0.0; n_q * n_t];
        }

        // Size threshold
        if n_q * n_t < 1024 {
            return super::CpuBackend.pairwise_distances_squared(queries, train, n_q, n_t, dim);
        }

        // Buffer size / dispatch guard — fall back to CPU for huge datasets.
        let out_size = n_q * n_t;
        let out_bytes = (out_size * std::mem::size_of::<f32>()) as u64;
        let q_bytes = (n_q * dim * std::mem::size_of::<f32>()) as u64;
        let t_bytes = (n_t * dim * std::mem::size_of::<f32>()) as u64;
        if out_bytes > MAX_GPU_BUFFER_BYTES
            || q_bytes > MAX_GPU_BUFFER_BYTES
            || t_bytes > MAX_GPU_BUFFER_BYTES
            || out_size.div_ceil(256) > u32::MAX as usize
        {
            return super::CpuBackend.pairwise_distances_squared(queries, train, n_q, n_t, dim);
        }

        let q_f32: Vec<f32> = queries.iter().map(|&v| v as f32).collect();
        let t_f32: Vec<f32> = train.iter().map(|&v| v as f32).collect();

        let dims = [n_q as u32, n_t as u32, dim as u32, 0u32];
        let dims_bytes = bytemuck::cast_slice(&dims);

        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dist_dims"),
            contents: dims_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dist_queries"),
            contents: bytemuck::cast_slice(&q_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let t_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dist_train"),
            contents: bytemuck::cast_slice(&t_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let d_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dist_output"),
            size: (out_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dist_readback"),
            size: (out_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dist_bg"),
            layout: &self.ctx.distance_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dims_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: t_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: d_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dist_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dist_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ctx.distance_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Workgroup size is 256 threads
            let wg_count = out_size.div_ceil(256) as u32;
            pass.dispatch_workgroups(wg_count, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&d_buf, 0, &readback_buf, 0, (out_size * 4) as u64);
        queue.submit(std::iter::once(encoder.finish()));

        match read_buffer_f32(device, &readback_buf, out_size) {
            Ok(result) => result,
            Err(_) => super::CpuBackend.pairwise_distances_squared(queries, train, n_q, n_t, dim),
        }
    }

    fn name(&self) -> &'static str {
        "gpu (wgpu)"
    }

    fn build_histograms(
        &self,
        binned: &[Vec<u8>],
        gradients: &[f64],
        hessians: &[f64],
        sample_indices: &[usize],
        n_features: usize,
        n_bins: usize,
    ) -> Vec<Vec<(f64, f64, f64)>> {
        // TODO: Use histogram.wgsl shader for GPU acceleration.
        // For now, use the default CPU implementation.
        // The shader is ready at shaders/histogram.wgsl.
        let mut histograms = vec![vec![(0.0_f64, 0.0_f64, 0.0_f64); n_bins]; n_features];
        for &idx in sample_indices {
            let g = gradients[idx];
            let h = hessians[idx];
            for f in 0..n_features {
                let bin = binned[f][idx] as usize;
                if bin < n_bins {
                    histograms[f][bin].0 += g;
                    histograms[f][bin].1 += h;
                    histograms[f][bin].2 += 1.0;
                }
            }
        }
        histograms
    }
}

/// Read back f32 data from a mapped GPU buffer and convert to f64.
///
/// Returns `Err` if the GPU readback fails (device lost, mapping error, etc.)
/// so the caller can fall back to the CPU path.
fn read_buffer_f32(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<f64>, String> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .unwrap_or(Err(wgpu::BufferAsyncError))
        .map_err(|_| "GPU buffer readback failed".to_string())?;

    let mapped = slice.get_mapped_range();
    let f32_data: &[f32] = bytemuck::cast_slice(&mapped);
    let result: Vec<f64> = f32_data[..count].iter().map(|&v| f64::from(v)).collect();
    drop(mapped);
    buffer.unmap();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_gpu() -> Option<GpuBackend> {
        GpuBackend::new().ok()
    }

    #[test]
    fn gpu_matmul_identity() {
        let Some(gpu) = try_gpu() else { return };

        // Force GPU path by using large enough matrices (above 4096 threshold)
        // 64×64 identity × 64×64 identity = 64×64 identity
        let n = 64;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let b = a.clone();

        let c = gpu.matmul(&a, &b, n, n, n);
        assert_eq!(c.len(), n * n);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (c[i * n + j] - expected).abs() < 1e-5,
                    "c[{i}][{j}] = {}, expected {expected}",
                    c[i * n + j]
                );
            }
        }
    }

    #[test]
    fn gpu_matmul_known_result() {
        let Some(gpu) = try_gpu() else { return };

        // Create matrices big enough to trigger GPU path
        let m = 32;
        let k = 32;
        let n = 32;
        let a: Vec<f64> = (0..m * k).map(|i| (i % 7) as f64).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i % 5) as f64).collect();

        let gpu_result = gpu.matmul(&a, &b, m, k, n);
        let cpu_result = super::super::CpuBackend.matmul(&a, &b, m, k, n);

        assert_eq!(gpu_result.len(), cpu_result.len());
        for (i, (g, c)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            assert!(
                (g - c).abs() < 1.0, // f32 precision loss
                "matmul mismatch at {i}: gpu={g}, cpu={c}"
            );
        }
    }

    #[test]
    fn gpu_pairwise_distances() {
        let Some(gpu) = try_gpu() else { return };

        // Create test data big enough to trigger GPU path
        let n_q = 50;
        let n_t = 50;
        let dim = 10;
        let queries: Vec<f64> = (0..n_q * dim).map(|i| (i % 13) as f64 * 0.1).collect();
        let train: Vec<f64> = (0..n_t * dim).map(|i| (i % 11) as f64 * 0.1).collect();

        let gpu_result = gpu.pairwise_distances_squared(&queries, &train, n_q, n_t, dim);
        let cpu_result =
            super::super::CpuBackend.pairwise_distances_squared(&queries, &train, n_q, n_t, dim);

        assert_eq!(gpu_result.len(), cpu_result.len());
        for (i, (g, c)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            assert!(
                (g - c).abs() < 0.1, // f32 accumulation error
                "distance mismatch at {i}: gpu={g}, cpu={c}"
            );
        }
    }

    #[test]
    fn gpu_backend_name() {
        let Some(gpu) = try_gpu() else { return };
        assert_eq!(gpu.name(), "gpu (wgpu)");
    }
}
