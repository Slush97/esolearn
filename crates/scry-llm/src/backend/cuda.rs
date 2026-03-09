use std::cell::RefCell;
use std::sync::Arc;

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::backend::{DeviceBackend, MathBackend};
use crate::tensor::shape::Shape;

use super::kernels::KernelCache;

// ---- BF16 mode flag ----

#[cfg(feature = "bf16")]
thread_local! {
    static BF16_MODE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(feature = "bf16")]
fn bf16_enabled() -> bool {
    BF16_MODE.with(std::cell::Cell::get)
}


/// Thread-local GPU context: device, stream, cuBLAS handle, compiled kernels.
struct GpuCtx {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    kernels: KernelCache,
}

thread_local! {
    static GPU_CTX: RefCell<Option<GpuCtx>> = const { RefCell::new(None) };
}

/// Initialize the GPU backend on the current thread.
pub fn init_gpu(device_id: usize) {
    GPU_CTX.with(|cell| {
        let ctx = CudaContext::new(device_id).expect("failed to create CUDA context");
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).expect("failed to create cuBLAS handle");
        let kernels = KernelCache::compile(&ctx);
        *cell.borrow_mut() = Some(GpuCtx {
            ctx,
            stream,
            blas,
            kernels,
        });
    });
}

/// Initialize the GPU backend in BF16 mixed-precision mode.
#[cfg(feature = "bf16")]
pub fn init_gpu_bf16(device_id: usize) {
    init_gpu(device_id);
    BF16_MODE.with(|c| c.set(true));
}

fn with_gpu<R>(f: impl FnOnce(&GpuCtx) -> R) -> R {
    GPU_CTX.with(|cell| {
        let borrow = cell.borrow();
        let gpu = borrow.as_ref().expect("GPU not initialized — call init_gpu() first");
        f(gpu)
    })
}

fn with_gpu_mut<R>(f: impl FnOnce(&mut GpuCtx) -> R) -> R {
    GPU_CTX.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let gpu = borrow.as_mut().expect("GPU not initialized — call init_gpu() first");
        f(gpu)
    })
}

/// GPU storage: a device-side f32 buffer with optional bf16 shadow.
///
/// In bf16 mode, weight tensors store a 1-element f32 stub (`f32_is_stub = true`)
/// and the real data lives in `bf16_data`. This saves ~4.9 GB VRAM for Llama 1B
/// by not duplicating every weight in both precisions.
#[derive(Debug)]
pub struct GpuStorage {
    pub(crate) inner: CudaSlice<f32>,
    pub(crate) len: usize,
    #[cfg(feature = "bf16")]
    pub(crate) bf16_data: std::cell::OnceCell<CudaSlice<half::bf16>>,
    /// When true, `inner` is a 1-element placeholder — real data is in `bf16_data`.
    #[cfg(feature = "bf16")]
    pub(crate) f32_is_stub: bool,
}

impl Clone for GpuStorage {
    fn clone(&self) -> Self {
        let cloned = self
            .inner
            .try_clone()
            .expect("failed to clone GPU storage");
        #[cfg(feature = "bf16")]
        {
            let bf16_cell = std::cell::OnceCell::new();
            if let Some(bf16) = self.bf16_data.get() {
                bf16_cell.set(bf16.try_clone().expect("failed to clone bf16 GPU storage")).ok();
            }
            Self {
                inner: cloned,
                len: self.len,
                bf16_data: bf16_cell,
                f32_is_stub: self.f32_is_stub,
            }
        }
        #[cfg(not(feature = "bf16"))]
        {
            Self {
                inner: cloned,
                len: self.len,
            }
        }
    }
}

impl GpuStorage {
    fn new(inner: CudaSlice<f32>, len: usize) -> Self {
        Self {
            inner,
            len,
            #[cfg(feature = "bf16")]
            bf16_data: std::cell::OnceCell::new(),
            #[cfg(feature = "bf16")]
            f32_is_stub: false,
        }
    }

    /// Return a reference to the bf16 GPU buffer, lazily casting from f32 on first access.
    #[cfg(feature = "bf16")]
    pub(crate) fn bf16_ref(&self) -> &CudaSlice<half::bf16> {
        self.bf16_data.get_or_init(|| {
            assert!(!self.f32_is_stub, "bf16_ref: f32 is a stub but no bf16 data set — bug");
            with_gpu(|gpu| bf16_ops::cast_f32_to_bf16(gpu, &self.inner, self.len))
        })
    }

    /// Upload bf16 data directly from host. In bf16 mode, stores only a 1-element
    /// f32 stub to save ~4x VRAM on weight tensors.
    #[cfg(feature = "bf16")]
    pub(crate) fn from_bf16_host(bf16_bytes: &[u8], f32_data: &[f32], len: usize) -> GpuStorage {
        let bf16_vec: Vec<half::bf16> = bf16_bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        with_gpu(|gpu| {
            let bf16_gpu = gpu.stream.clone_htod(&bf16_vec).unwrap();
            if bf16_enabled() {
                // Stub: 1-element f32 placeholder — real data is bf16 only
                let f32_stub = gpu.stream.alloc_zeros::<f32>(1).unwrap();
                let storage = GpuStorage {
                    inner: f32_stub,
                    len,
                    bf16_data: std::cell::OnceCell::new(),
                    f32_is_stub: true,
                };
                storage.bf16_data.set(bf16_gpu).ok();
                storage
            } else {
                let f32_gpu = gpu.stream.clone_htod(f32_data).unwrap();
                let storage = GpuStorage::new(f32_gpu, len);
                storage.bf16_data.set(bf16_gpu).ok();
                storage
            }
        })
    }
}

/// CUDA backend.
pub struct CudaBackend;

fn grid_for(n: usize, block: u32) -> u32 {
    (n as u32).div_ceil(block)
}

const BLOCK: u32 = 256;
const ROW_BLOCK: u32 = 256;

impl DeviceBackend for CudaBackend {
    type Storage = GpuStorage;
    type Stream = Arc<CudaStream>;
    #[cfg(feature = "quantize")]
    type I8Storage = Vec<i8>;

    #[cfg(feature = "quantize")]
    fn i8_from_vec(data: Vec<i8>) -> Vec<i8> { data }
    #[cfg(feature = "quantize")]
    fn i8_to_vec(storage: &Vec<i8>) -> Vec<i8> { storage.clone() }

    fn zeros(shape: &Shape) -> GpuStorage {
        let n = shape.numel();
        with_gpu(|gpu| {
            let inner = gpu.stream.alloc_zeros::<f32>(n).unwrap();
            GpuStorage::new(inner, n)
        })
    }

    fn ones(shape: &Shape) -> GpuStorage {
        let n = shape.numel();
        let data = vec![1.0f32; n];
        with_gpu(|gpu| {
            let inner = gpu.stream.clone_htod(&data).unwrap();
            GpuStorage::new(inner, n)
        })
    }

    fn from_vec(data: Vec<f32>, _shape: &Shape) -> GpuStorage {
        let len = data.len();
        with_gpu(|gpu| {
            let inner = gpu.stream.clone_htod(&data).unwrap();
            GpuStorage::new(inner, len)
        })
    }

    #[cfg(feature = "bf16")]
    fn from_vec_with_bf16(data: Vec<f32>, bf16_bytes: &[u8], _shape: &Shape) -> GpuStorage {
        let len = data.len();
        if bf16_enabled() {
            GpuStorage::from_bf16_host(bf16_bytes, &data, len)
        } else {
            with_gpu(|gpu| {
                let inner = gpu.stream.clone_htod(&data).unwrap();
                GpuStorage::new(inner, len)
            })
        }
    }

    fn to_vec(storage: &GpuStorage) -> Vec<f32> {
        #[cfg(feature = "bf16")]
        if storage.f32_is_stub {
            // f32 is a placeholder — convert from bf16
            return with_gpu(|gpu| {
                let bf16 = storage.bf16_data.get()
                    .expect("to_vec: f32 is stub but no bf16 data");
                let bf16_host: Vec<half::bf16> = gpu.stream.clone_dtoh(bf16).unwrap();
                bf16_host.iter().map(|&v| v.to_f32()).collect()
            });
        }
        with_gpu(|gpu| gpu.stream.clone_dtoh(&storage.inner).unwrap())
    }

    fn clone_storage(storage: &GpuStorage) -> GpuStorage {
        storage.clone()
    }
}

impl MathBackend for CudaBackend {
    fn matmul(
        a: &GpuStorage,
        b: &GpuStorage,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> GpuStorage {
        #[cfg(feature = "bf16")]
        if bf16_enabled() {
            return bf16_ops::matmul_bf16(a, b, m, k, n, trans_a, trans_b);
        }
        with_gpu_mut(|gpu| {
            let mut c = gpu.stream.alloc_zeros::<f32>(m * n).unwrap();

            let (transa, lda) = if trans_b {
                (cublasOperation_t::CUBLAS_OP_T, k as i32)
            } else {
                (cublasOperation_t::CUBLAS_OP_N, n as i32)
            };
            let (transb, ldb) = if trans_a {
                (cublasOperation_t::CUBLAS_OP_T, m as i32)
            } else {
                (cublasOperation_t::CUBLAS_OP_N, k as i32)
            };

            unsafe {
                gpu.blas
                    .gemm(
                        GemmConfig {
                            transa,
                            transb,
                            m: n as i32,
                            n: m as i32,
                            k: k as i32,
                            alpha: 1.0f32,
                            lda,
                            ldb,
                            beta: 0.0f32,
                            ldc: n as i32,
                        },
                        &b.inner,
                        &a.inner,
                        &mut c,
                    )
                    .expect("cuBLAS sgemm failed");
            }

            GpuStorage::new(c, m * n)
        })
    }

    fn add(
        a: &GpuStorage,
        b: &GpuStorage,
        a_shape: &Shape,
        b_shape: &Shape,
        out_shape: &Shape,
    ) -> GpuStorage {
        // Fast path: same shape
        if a_shape == b_shape {
            let n = a.len;
            return with_gpu(|gpu| {
                let mut out = gpu.stream.alloc_zeros::<f32>(n).unwrap();
                unsafe {
                    gpu.stream.memcpy_dtod(&a.inner, &mut out).unwrap();
                    gpu.stream
                        .launch_builder(&gpu.kernels.add_inplace_kernel)
                        .arg(&mut out)
                        .arg(&b.inner)
                        .arg(&n)
                        .launch(LaunchConfig {
                            grid_dim: (grid_for(n, BLOCK), 1, 1),
                            block_dim: (BLOCK, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .unwrap();
                }
                GpuStorage::new(out, n)
            });
        }

        // Common broadcast: [rows, cols] + [1, cols] or [cols]
        let out_dims = out_shape.dims();
        let b_dims = b_shape.dims();
        if out_dims.len() == 2
            && (b_dims == [1, out_dims[1]] || b_dims == [out_dims[1]])
        {
            let rows = out_dims[0];
            let cols = out_dims[1];
            let n = rows * cols;
            return with_gpu(|gpu| {
                let mut out = gpu.stream.alloc_zeros::<f32>(n).unwrap();
                unsafe {
                    gpu.stream
                        .launch_builder(&gpu.kernels.add_broadcast_2d)
                        .arg(&mut out)
                        .arg(&a.inner)
                        .arg(&b.inner)
                        .arg(&rows)
                        .arg(&cols)
                        .launch(LaunchConfig {
                            grid_dim: (grid_for(n, BLOCK), 1, 1),
                            block_dim: (BLOCK, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .unwrap();
                }
                GpuStorage::new(out, n)
            });
        }

        // General fallback
        let a_vec = Self::to_vec(a);
        let b_vec = Self::to_vec(b);
        let result =
            crate::backend::cpu::CpuBackend::add(&a_vec, &b_vec, a_shape, b_shape, out_shape);
        Self::from_vec(result, out_shape)
    }

    fn softmax(input: &GpuStorage, shape: &Shape) -> GpuStorage {
        let dims = shape.dims();
        let cols = *dims.last().unwrap();
        let rows = input.len / cols;
        with_gpu(|gpu| {
            let mut out = gpu.stream.alloc_zeros::<f32>(input.len).unwrap();
            let threads = ROW_BLOCK.min(cols.next_power_of_two() as u32);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.softmax_fwd)
                    .arg(&mut out)
                    .arg(&input.inner)
                    .arg(&rows)
                    .arg(&cols)
                    .launch(LaunchConfig {
                        grid_dim: (rows as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: threads * 4,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, input.len)
        })
    }

    fn layernorm(
        input: &GpuStorage,
        gamma: &GpuStorage,
        beta: &GpuStorage,
        shape: &Shape,
        eps: f32,
    ) -> (GpuStorage, GpuStorage, GpuStorage) {
        let dims = shape.dims();
        let d = *dims.last().unwrap();
        let rows = input.len / d;
        with_gpu(|gpu| {
            let mut out = gpu.stream.alloc_zeros::<f32>(input.len).unwrap();
            let mut mean_out = gpu.stream.alloc_zeros::<f32>(rows).unwrap();
            let mut rstd_out = gpu.stream.alloc_zeros::<f32>(rows).unwrap();
            let threads = ROW_BLOCK.min(d.next_power_of_two() as u32);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.layernorm_fwd)
                    .arg(&mut out)
                    .arg(&mut mean_out)
                    .arg(&mut rstd_out)
                    .arg(&input.inner)
                    .arg(&gamma.inner)
                    .arg(&beta.inner)
                    .arg(&rows)
                    .arg(&d)
                    .arg(&eps)
                    .launch(LaunchConfig {
                        grid_dim: (rows as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: threads * 4,
                    })
                    .unwrap();
            }
            (
                GpuStorage::new(out, input.len),
                GpuStorage::new(mean_out, rows),
                GpuStorage::new(rstd_out, rows),
            )
        })
    }

    fn gelu(input: &GpuStorage) -> GpuStorage {
        let n = input.len;
        with_gpu(|gpu| {
            let mut out = gpu.stream.alloc_zeros::<f32>(n).unwrap();
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.gelu_fwd)
                    .arg(&mut out)
                    .arg(&input.inner)
                    .arg(&n)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, n)
        })
    }

    fn embedding(
        weight: &GpuStorage,
        indices: &[usize],
        _vocab: usize,
        dim: usize,
    ) -> GpuStorage {
        let n_indices = indices.len();
        let total = n_indices * dim;
        with_gpu(|gpu| {
            let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
            let indices_dev = gpu.stream.clone_htod(&indices_u32).unwrap();
            let mut out = gpu.stream.alloc_zeros::<f32>(total).unwrap();

            #[cfg(feature = "bf16")]
            if weight.f32_is_stub {
                // Weight is bf16-only — use bf16 embedding kernel
                let bf16_weight = weight.bf16_data.get()
                    .expect("embedding: f32 is stub but no bf16 data");
                unsafe {
                    gpu.stream
                        .launch_builder(&gpu.kernels.embedding_bf16_fwd)
                        .arg(&mut out)
                        .arg(bf16_weight)
                        .arg(&indices_dev)
                        .arg(&n_indices)
                        .arg(&dim)
                        .launch(LaunchConfig {
                            grid_dim: (grid_for(total, BLOCK), 1, 1),
                            block_dim: (BLOCK, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .unwrap();
                }
                return GpuStorage::new(out, total);
            }

            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.embedding_fwd)
                    .arg(&mut out)
                    .arg(&weight.inner)
                    .arg(&indices_dev)
                    .arg(&n_indices)
                    .arg(&dim)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(total, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, total)
        })
    }

    fn sum(input: &GpuStorage) -> f32 {
        with_gpu(|gpu| {
            let mut out = gpu.stream.alloc_zeros::<f32>(1).unwrap();
            let n = input.len;
            let threads = ROW_BLOCK.min(n.next_power_of_two() as u32);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.reduce_sum)
                    .arg(&mut out)
                    .arg(&input.inner)
                    .arg(&n)
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: threads * 4,
                    })
                    .unwrap();
            }
            let result = gpu.stream.clone_dtoh(&out).unwrap();
            result[0]
        })
    }

    fn mul_elementwise(a: &GpuStorage, b: &GpuStorage) -> GpuStorage {
        let n = a.len;
        with_gpu(|gpu| {
            let mut out = gpu.stream.alloc_zeros::<f32>(n).unwrap();
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.mul_elementwise)
                    .arg(&mut out)
                    .arg(&a.inner)
                    .arg(&b.inner)
                    .arg(&n)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, n)
        })
    }

    fn scale(a: &GpuStorage, scalar: f32) -> GpuStorage {
        let n = a.len;
        with_gpu(|gpu| {
            let mut out = gpu.stream.alloc_zeros::<f32>(n).unwrap();
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.scale_kernel)
                    .arg(&mut out)
                    .arg(&a.inner)
                    .arg(&scalar)
                    .arg(&n)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, n)
        })
    }

    fn concat_rows(
        a: &GpuStorage,
        b: &GpuStorage,
        _a_rows: usize,
        _b_rows: usize,
        _cols: usize,
    ) -> GpuStorage {
        let total = a.len + b.len;
        with_gpu(|gpu| {
            let mut out = gpu.stream.alloc_zeros::<f32>(total).unwrap();
            gpu.stream
                .memcpy_dtod(&a.inner, &mut out.slice_mut(0..a.len))
                .unwrap();
            gpu.stream
                .memcpy_dtod(&b.inner, &mut out.slice_mut(a.len..total))
                .unwrap();
            GpuStorage::new(out, total)
        })
    }

    // ---- Llama-specific ops ----

    fn rmsnorm(
        input: &GpuStorage,
        weight: &GpuStorage,
        shape: &Shape,
        eps: f32,
    ) -> GpuStorage {
        let dims = shape.dims();
        let d = *dims.last().unwrap();
        let rows = input.len / d;
        with_gpu(|gpu| {
            let mut out = gpu.stream.alloc_zeros::<f32>(input.len).unwrap();
            let threads = ROW_BLOCK.min(d.next_power_of_two() as u32);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.rmsnorm_fwd)
                    .arg(&mut out)
                    .arg(&input.inner)
                    .arg(&weight.inner)
                    .arg(&rows)
                    .arg(&d)
                    .arg(&eps)
                    .launch(LaunchConfig {
                        grid_dim: (rows as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: threads * 4,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, input.len)
        })
    }

    #[cfg(feature = "bf16")]
    fn rmsnorm_with_bf16(
        input: &GpuStorage,
        weight: &GpuStorage,
        shape: &Shape,
        eps: f32,
    ) -> GpuStorage {
        if !bf16_enabled() {
            return Self::rmsnorm(input, weight, shape, eps);
        }
        let dims = shape.dims();
        let d = *dims.last().unwrap();
        let rows = input.len / d;
        with_gpu(|gpu| {
            let mut out = gpu.stream.alloc_zeros::<f32>(input.len).unwrap();
            let mut out_bf16 = gpu.stream.alloc_zeros::<half::bf16>(input.len).unwrap();
            let threads = ROW_BLOCK.min(d.next_power_of_two() as u32);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.rmsnorm_fwd_with_bf16)
                    .arg(&mut out)
                    .arg(&mut out_bf16)
                    .arg(&input.inner)
                    .arg(&weight.inner)
                    .arg(&rows)
                    .arg(&d)
                    .arg(&eps)
                    .launch(LaunchConfig {
                        grid_dim: (rows as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: threads * 4,
                    })
                    .unwrap();
            }
            let storage = GpuStorage::new(out, input.len);
            storage.bf16_data.set(out_bf16).ok();
            storage
        })
    }

    fn rope(
        input: &GpuStorage,
        shape: &Shape,
        pos: usize,
        head_dim: usize,
        theta: f32,
    ) -> GpuStorage {
        let n = input.len;
        // n_pairs = total elements / 2 (we process pairs)
        let n_pairs = n / 2;
        with_gpu(|gpu| {
            let mut out = gpu.stream.alloc_zeros::<f32>(n).unwrap();
            unsafe {
                // Copy input to output first, then apply RoPE in-place
                gpu.stream.memcpy_dtod(&input.inner, &mut out).unwrap();
                gpu.stream
                    .launch_builder(&gpu.kernels.rope_fwd)
                    .arg(&mut out)
                    .arg(&input.inner)
                    .arg(&n)
                    .arg(&pos)
                    .arg(&head_dim)
                    .arg(&theta)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n_pairs, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            let _ = shape;
            GpuStorage::new(out, n)
        })
    }

    fn rope_with_freqs_preloaded(
        input: &GpuStorage,
        seq: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
        freqs: &GpuStorage,
    ) -> GpuStorage {
        let n = input.len;
        let half_hd = head_dim / 2;
        let total_pairs = seq * n_heads * half_hd;
        with_gpu(|gpu| {
            let mut out = gpu.stream.alloc_zeros::<f32>(n).unwrap();
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.rope_with_freqs_fwd)
                    .arg(&mut out)
                    .arg(&input.inner)
                    .arg(&freqs.inner)
                    .arg(&total_pairs)
                    .arg(&n_heads)
                    .arg(&head_dim)
                    .arg(&start_pos)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(total_pairs, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, n)
        })
    }

    fn rope_with_freqs_scaled_preloaded(
        input: &GpuStorage,
        seq: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
        freqs: &GpuStorage,
        scale: f32,
    ) -> GpuStorage {
        let n = input.len;
        let half_hd = head_dim / 2;
        let total_pairs = seq * n_heads * half_hd;
        with_gpu(|gpu| {
            let mut out = gpu.stream.alloc_zeros::<f32>(n).unwrap();
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.rope_with_freqs_scaled_fwd)
                    .arg(&mut out)
                    .arg(&input.inner)
                    .arg(&freqs.inner)
                    .arg(&total_pairs)
                    .arg(&n_heads)
                    .arg(&head_dim)
                    .arg(&start_pos)
                    .arg(&scale)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(total_pairs, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, n)
        })
    }

    fn swiglu(gate: &GpuStorage, up: &GpuStorage) -> GpuStorage {
        let n = gate.len;
        with_gpu(|gpu| {
            let mut out = gpu.stream.alloc_zeros::<f32>(n).unwrap();
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.swiglu_fwd)
                    .arg(&mut out)
                    .arg(&gate.inner)
                    .arg(&up.inner)
                    .arg(&n)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, n)
        })
    }

    #[cfg(feature = "bf16")]
    fn swiglu_with_bf16(gate: &GpuStorage, up: &GpuStorage) -> GpuStorage {
        if !bf16_enabled() {
            return Self::swiglu(gate, up);
        }
        let n = gate.len;
        with_gpu(|gpu| {
            let mut out = gpu.stream.alloc_zeros::<f32>(n).unwrap();
            let mut out_bf16 = gpu.stream.alloc_zeros::<half::bf16>(n).unwrap();
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.swiglu_fwd_with_bf16)
                    .arg(&mut out)
                    .arg(&mut out_bf16)
                    .arg(&gate.inner)
                    .arg(&up.inner)
                    .arg(&n)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            let storage = GpuStorage::new(out, n);
            storage.bf16_data.set(out_bf16).ok();
            storage
        })
    }

    fn repeat_kv(
        input: &GpuStorage,
        n_kv_heads: usize,
        n_q_heads: usize,
        seq: usize,
        d_head: usize,
    ) -> GpuStorage {
        let n_rep = n_q_heads / n_kv_heads;
        if n_rep == 1 {
            return input.clone();
        }
        let total = n_q_heads * seq * d_head;
        with_gpu(|gpu| {
            let mut out = gpu.stream.alloc_zeros::<f32>(total).unwrap();
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.repeat_kv_kernel)
                    .arg(&mut out)
                    .arg(&input.inner)
                    .arg(&n_kv_heads)
                    .arg(&n_q_heads)
                    .arg(&seq)
                    .arg(&d_head)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(total, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, total)
        })
    }

    fn gather_reshape_repeat_kv(
        cache: &GpuStorage,
        _max_seq: usize,
        cached_len: usize,
        n_kv_heads: usize,
        n_q_heads: usize,
        head_dim: usize,
    ) -> GpuStorage {
        let total = n_q_heads * cached_len * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        with_gpu(|gpu| {
            let mut out = gpu.stream.alloc_zeros::<f32>(total).unwrap();
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.gather_reshape_repeat_kv)
                    .arg(&mut out)
                    .arg(&cache.inner)
                    .arg(&cached_len)
                    .arg(&n_kv_heads)
                    .arg(&n_q_heads)
                    .arg(&head_dim)
                    .arg(&kv_dim)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(total, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, total)
        })
    }

    // ---- Attention helpers (GPU-optimized) ----

    fn gather_columns(
        storage: &GpuStorage,
        rows: usize,
        total_cols: usize,
        col_start: usize,
        col_count: usize,
    ) -> GpuStorage {
        // Fast path: rows=1 means columns are contiguous — memcpy instead of kernel launch
        if rows == 1 {
            let n = col_count;
            return with_gpu(|gpu| {
                let mut out = gpu.stream.alloc_zeros::<f32>(n).unwrap();
                gpu.stream
                    .memcpy_dtod(&storage.inner.slice(col_start..col_start + n), &mut out)
                    .unwrap();
                GpuStorage::new(out, n)
            });
        }
        let n = rows * col_count;
        with_gpu(|gpu| {
            let mut out = gpu.stream.alloc_zeros::<f32>(n).unwrap();
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.gather_columns)
                    .arg(&mut out)
                    .arg(&storage.inner)
                    .arg(&rows)
                    .arg(&total_cols)
                    .arg(&col_start)
                    .arg(&col_count)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, n)
        })
    }

    fn scatter_columns(
        dst: &mut GpuStorage,
        src: &GpuStorage,
        rows: usize,
        total_cols: usize,
        col_start: usize,
        col_count: usize,
    ) {
        let n = rows * col_count;
        with_gpu(|gpu| {
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.scatter_columns)
                    .arg(&mut dst.inner)
                    .arg(&src.inner)
                    .arg(&rows)
                    .arg(&total_cols)
                    .arg(&col_start)
                    .arg(&col_count)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
        });
    }

    fn gather_rows(
        storage: &GpuStorage,
        _total_rows: usize,
        cols: usize,
        row_start: usize,
        row_count: usize,
    ) -> GpuStorage {
        let n = row_count * cols;
        let start = row_start * cols;
        with_gpu(|gpu| {
            let mut out = gpu.stream.alloc_zeros::<f32>(n).unwrap();
            gpu.stream
                .memcpy_dtod(&storage.inner.slice(start..start + n), &mut out)
                .unwrap();
            GpuStorage::new(out, n)
        })
    }

    fn scatter_rows(
        dst: &mut GpuStorage,
        src: &GpuStorage,
        _total_rows: usize,
        cols: usize,
        row_start: usize,
        row_count: usize,
    ) {
        let start = row_start * cols;
        let n = row_count * cols;
        with_gpu(|gpu| {
            gpu.stream
                .memcpy_dtod(&src.inner, &mut dst.inner.slice_mut(start..start + n))
                .unwrap();
        });
    }

    fn apply_causal_mask_and_scale(
        scores: &mut GpuStorage,
        seq_len: usize,
        scale: f32,
        mask_value: f32,
    ) {
        let n = seq_len * seq_len;
        with_gpu(|gpu| {
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.causal_mask_scale)
                    .arg(&mut scores.inner)
                    .arg(&seq_len)
                    .arg(&scale)
                    .arg(&mask_value)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
        });
    }

    fn reshape_for_heads(
        storage: &GpuStorage,
        batch: usize,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> GpuStorage {
        // When B=1, S=1 the reshape is an identity permutation — skip the kernel launch.
        if batch == 1 && seq == 1 {
            return storage.clone();
        }
        let total = batch * n_heads * seq * d_head;
        with_gpu(|gpu| {
            let mut out = gpu.stream.alloc_zeros::<f32>(total).unwrap();
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.reshape_bsh_to_bnsh)
                    .arg(&mut out)
                    .arg(&storage.inner)
                    .arg(&batch)
                    .arg(&seq)
                    .arg(&n_heads)
                    .arg(&d_head)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(total, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, total)
        })
    }

    fn reshape_from_heads(
        storage: &GpuStorage,
        batch: usize,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> GpuStorage {
        // When B=1, S=1 the reshape is an identity permutation — skip the kernel launch.
        if batch == 1 && seq == 1 {
            return storage.clone();
        }
        let total = batch * seq * n_heads * d_head;
        with_gpu(|gpu| {
            let mut out = gpu.stream.alloc_zeros::<f32>(total).unwrap();
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.reshape_bnsh_to_bsh)
                    .arg(&mut out)
                    .arg(&storage.inner)
                    .arg(&batch)
                    .arg(&seq)
                    .arg(&n_heads)
                    .arg(&d_head)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(total, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, total)
        })
    }

    fn matmul_strided_batched(
        a: &GpuStorage,
        b: &GpuStorage,
        batch_count: usize,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> GpuStorage {
        // NOTE: bf16 is intentionally NOT used here. Strided batched matmuls are
        // attention-only (Q@K^T, attn@V) where both operands are fresh activations.
        // Casting both to bf16 adds 4 kernel launches per layer with zero bandwidth
        // benefit — the f32 SGEMM path is faster for these small activation tensors.
        let total = batch_count * m * n;
        with_gpu_mut(|gpu| {
            let mut c = gpu.stream.alloc_zeros::<f32>(total).unwrap();

            let (transa, lda) = if trans_b {
                (cublasOperation_t::CUBLAS_OP_T, k as i32)
            } else {
                (cublasOperation_t::CUBLAS_OP_N, n as i32)
            };
            let (transb, ldb) = if trans_a {
                (cublasOperation_t::CUBLAS_OP_T, m as i32)
            } else {
                (cublasOperation_t::CUBLAS_OP_N, k as i32)
            };

            let stride_a = if trans_a { k * m } else { m * k };
            let stride_b = if trans_b { n * k } else { k * n };
            let stride_c = m * n;

            unsafe {
                gpu.blas
                    .gemm_strided_batched(
                        cudarc::cublas::StridedBatchedConfig {
                            gemm: cudarc::cublas::GemmConfig {
                                transa,
                                transb,
                                m: n as i32,
                                n: m as i32,
                                k: k as i32,
                                alpha: 1.0f32,
                                lda,
                                ldb,
                                beta: 0.0f32,
                                ldc: n as i32,
                            },
                            batch_size: batch_count as i32,
                            stride_a: stride_b as i64,
                            stride_b: stride_a as i64,
                            stride_c: stride_c as i64,
                        },
                        &b.inner,
                        &a.inner,
                        &mut c,
                    )
                    .expect("cuBLAS sgemm_strided_batched failed");
            }

            GpuStorage::new(c, total)
        })
    }

    fn apply_batched_causal_mask_and_scale(
        scores: &mut GpuStorage,
        num_matrices: usize,
        seq_len: usize,
        scale: f32,
        mask_value: f32,
    ) {
        let total = num_matrices * seq_len * seq_len;
        with_gpu(|gpu| {
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.batched_causal_mask_scale)
                    .arg(&mut scores.inner)
                    .arg(&num_matrices)
                    .arg(&seq_len)
                    .arg(&scale)
                    .arg(&mask_value)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(total, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
        });
    }
}

impl CudaBackend {
    /// Synchronize the GPU stream.
    pub fn synchronize() {
        with_gpu(|gpu| {
            gpu.stream.synchronize().expect("CUDA sync failed");
        });
    }
}

// ============================================================
// BF16 helpers
// ============================================================

#[cfg(feature = "bf16")]
#[allow(clippy::wildcard_imports, clippy::doc_markdown, clippy::cast_lossless)]
mod bf16_ops {
    use super::*;

    pub fn cast_f32_to_bf16(gpu: &GpuCtx, src: &CudaSlice<f32>, n: usize) -> CudaSlice<half::bf16> {
        let mut dst = gpu.stream.alloc_zeros::<half::bf16>(n).unwrap();
        unsafe {
            gpu.stream
                .launch_builder(&gpu.kernels.f32_to_bf16)
                .arg(&mut dst)
                .arg(src)
                .arg(&n)
                .launch(LaunchConfig {
                    grid_dim: (grid_for(n, BLOCK), 1, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        dst
    }

    pub fn matmul_bf16(
        a: &GpuStorage,
        b: &GpuStorage,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> GpuStorage {
        let a_bf16 = a.bf16_ref();
        let b_bf16 = b.bf16_ref();

        with_gpu_mut(|gpu| {
            let mut c_f32 = gpu.stream.alloc_zeros::<f32>(m * n).unwrap();

            let (transa, lda) = if trans_b {
                (cublasOperation_t::CUBLAS_OP_T, k as i32)
            } else {
                (cublasOperation_t::CUBLAS_OP_N, n as i32)
            };
            let (transb, ldb) = if trans_a {
                (cublasOperation_t::CUBLAS_OP_T, m as i32)
            } else {
                (cublasOperation_t::CUBLAS_OP_N, k as i32)
            };

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            unsafe {
                use cudarc::cublas::sys::{
                    cublasComputeType_t, cublasGemmAlgo_t, cudaDataType_t,
                };
                use cudarc::driver::{DevicePtr, DevicePtrMut};

                let (a_ptr, _a_rec) = a_bf16.device_ptr(&gpu.stream);
                let (b_ptr, _b_rec) = b_bf16.device_ptr(&gpu.stream);
                let (c_ptr, _c_rec) = c_f32.device_ptr_mut(&gpu.stream);

                cudarc::cublas::result::gemm_ex(
                    *gpu.blas.handle(),
                    transa,
                    transb,
                    n as i32,
                    m as i32,
                    k as i32,
                    (&alpha) as *const f32 as *const _,
                    b_ptr as *const _,
                    cudaDataType_t::CUDA_R_16BF,
                    lda,
                    a_ptr as *const _,
                    cudaDataType_t::CUDA_R_16BF,
                    ldb,
                    (&beta) as *const f32 as *const _,
                    c_ptr as *mut _,
                    cudaDataType_t::CUDA_R_32F,
                    n as i32,
                    cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
                .expect("cublasGemmEx bf16→f32 failed");
            }

            GpuStorage::new(c_f32, m * n)
        })
    }

}
