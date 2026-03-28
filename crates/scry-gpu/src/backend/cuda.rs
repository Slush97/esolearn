//! CUDA compute backend via `cudarc`.
//!
//! Compute-only CUDA path:
//! - Context → stream → cuBLAS handle
//! - `CudaSlice<u8>` buffer management
//! - NVRTC kernel compilation from CUDA C strings
//! - cuBLAS SGEMM for matmul
//!
//! SPIR-V dispatch is not supported — use [`CudaBackend::compile_cuda`] for
//! native CUDA kernels, or [`CudaBackend::cublas_matmul`] for matrix multiply.

use std::sync::{Arc, Mutex};

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::CudaBlas;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::backend::{Backend, BackendBufferOps};
use crate::error::{backend_err, BackendOp, GpuError, Result};

// ── Public types ──

/// CUDA compute backend state.
pub struct CudaBackend {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: Mutex<CudaBlas>,
    device_name: String,
    device_memory: u64,
}

/// A buffer allocated on the CUDA device.
pub struct CudaBuffer {
    pub(crate) inner: CudaSlice<u8>,
    size: u64,
    stream: Arc<CudaStream>,
}

/// A compiled CUDA kernel, ready for dispatch.
pub struct CudaKernel {
    pub(crate) function: CudaFunction,
    /// Thread block dimensions `(x, y, z)`, matching the CUDA kernel's
    /// expected `blockDim`.
    pub(crate) block_dim: (u32, u32, u32),
}

/// A batch of kernel dispatches on a dedicated CUDA stream.
///
/// Kernel launches on a single stream are automatically serialized by the
/// CUDA runtime, so barriers are no-ops. [`CudaBatch::submit`] synchronizes
/// the stream.
pub struct CudaBatch {
    stream: Arc<CudaStream>,
}

impl CudaBackend {
    /// Compile a CUDA C source string into a reusable kernel.
    ///
    /// Uses NVRTC to compile to PTX, then loads and extracts the named
    /// entry point function.
    pub fn compile_cuda(
        &self,
        source: &str,
        entry_point: &str,
        block_dim: (u32, u32, u32),
    ) -> Result<CudaKernel> {
        let opts = CompileOptions {
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(source, opts)
            .map_err(|e| backend_err(BackendOp::CompileKernel, e))?;
        let module = self
            .ctx
            .load_module(ptx)
            .map_err(|e| backend_err(BackendOp::LoadModule, e))?;
        let function = module
            .load_function(entry_point)
            .map_err(|e| backend_err(BackendOp::LoadFunction, e))?;

        Ok(CudaKernel {
            function,
            block_dim,
        })
    }

    /// Dispatch a compiled CUDA kernel.
    ///
    /// Buffer device pointers are passed as kernel arguments in order,
    /// followed by push constant bytes split into `u32` arguments.
    pub fn dispatch_cuda(
        &self,
        kernel: &CudaKernel,
        buffers: &[&CudaBuffer],
        workgroups: [u32; 3],
        push_constants: Option<&[u8]>,
    ) -> Result<()> {
        let config = LaunchConfig {
            grid_dim: (workgroups[0], workgroups[1], workgroups[2]),
            block_dim: kernel.block_dim,
            shared_mem_bytes: 0,
        };

        // Collect push constant u32 values so they outlive the builder.
        let pc_values: Vec<u32> = push_constants
            .map(|pc| {
                pc.chunks_exact(4)
                    .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            })
            .unwrap_or_default();

        unsafe {
            let mut builder = self.stream.launch_builder(&kernel.function);

            // Push buffer device pointers as kernel arguments.
            for buf in buffers {
                builder.arg(&buf.inner);
            }

            // Push constants as individual u32 kernel arguments.
            for val in &pc_values {
                builder.arg(val);
            }

            builder
                .launch(config)
                .map_err(|e| backend_err(BackendOp::LaunchKernel, e))?;
        }

        // Synchronize to match the Vulkan fence-wait semantics.
        self.stream
            .synchronize()
            .map_err(|e| backend_err(BackendOp::StreamSync, e))?;

        Ok(())
    }

    /// Run cuBLAS SGEMM: C = alpha * A * B + beta * C.
    ///
    /// Buffers must contain `f32` data. This is the recommended path for
    /// matrix multiplication on CUDA — it reaches 80%+ peak throughput
    /// without any custom kernels.
    ///
    /// Matrix layout is row-major. Dimensions: A is `m×k`, B is `k×n`,
    /// C is `m×n`.
    #[allow(clippy::many_single_char_names)]
    pub fn cublas_matmul(
        &self,
        a: &CudaBuffer,
        b: &CudaBuffer,
        c: &mut CudaBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        // cuBLAS uses column-major layout. For row-major C = A * B, we
        // compute C^T = B^T * A^T in cuBLAS terms, which gives us:
        //   sgemm(N, N, n, m, k, 1.0, B, n, A, k, 0.0, C, n)
        #[allow(clippy::cast_possible_wrap)]
        unsafe {
            let blas = self
                .blas
                .lock()
                .map_err(|_| backend_err(BackendOp::MutexPoisoned, "cublas"))?;
            let (a_ptr, _a_guard) = a.inner.device_ptr(&self.stream);
            let (b_ptr, _b_guard) = b.inner.device_ptr(&self.stream);
            let (c_ptr, _c_guard) = c.inner.device_ptr_mut(&self.stream);

            cudarc::cublas::result::sgemm(
                *blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &1.0f32,
                b_ptr as *const f32,
                n as i32,
                a_ptr as *const f32,
                k as i32,
                &0.0f32,
                c_ptr as *mut f32,
                n as i32,
            )
            .map_err(|e| backend_err(BackendOp::CuBlas, e))?;
        }

        // Sync after cuBLAS call.
        self.stream
            .synchronize()
            .map_err(|e| backend_err(BackendOp::StreamSync, e))?;

        Ok(())
    }

    /// Begin a batch dispatch session.
    ///
    /// On CUDA, kernel launches on the same stream are inherently batched
    /// (queued without GPU idle time), so this uses the default stream.
    /// [`CudaBatch::submit`] synchronizes once at the end.
    #[allow(clippy::unnecessary_wraps)]
    pub fn begin_batch(&self) -> Result<CudaBatch> {
        Ok(CudaBatch {
            stream: Arc::clone(&self.stream),
        })
    }
}

// ── Backend trait implementation ──

impl Backend for CudaBackend {
    type Buffer = CudaBuffer;
    type Pipeline = CudaKernel;

    fn create() -> Result<Self> {
        let ctx = CudaContext::new(0).map_err(|e| backend_err(BackendOp::CreateDevice, e))?;

        let device_name = ctx
            .name()
            .map_err(|e| backend_err(BackendOp::DeviceQuery, e))?;
        let device_memory =
            ctx.total_mem()
                .map_err(|e| backend_err(BackendOp::DeviceQuery, e))? as u64;

        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(|e| backend_err(BackendOp::CuBlas, e))?;

        Ok(Self {
            ctx,
            stream,
            blas: Mutex::new(blas),
            device_name,
            device_memory,
        })
    }

    fn upload(&self, data: &[u8]) -> Result<Self::Buffer> {
        let size = data.len() as u64;
        let inner = self
            .stream
            .clone_htod(data)
            .map_err(|e| backend_err(BackendOp::CopyBuffer, e))?;
        Ok(CudaBuffer {
            inner,
            size,
            stream: Arc::clone(&self.stream),
        })
    }

    fn alloc(&self, size: u64) -> Result<Self::Buffer> {
        let inner = self
            .stream
            .alloc_zeros::<u8>(size as usize)
            .map_err(|e| backend_err(BackendOp::CreateBuffer, e))?;
        Ok(CudaBuffer {
            inner,
            size,
            stream: Arc::clone(&self.stream),
        })
    }

    fn dispatch(
        &self,
        _spirv: &[u32],
        _entry_point: &str,
        _buffers: &[&Self::Buffer],
        _workgroups: [u32; 3],
        _push_constants: Option<&[u8]>,
    ) -> Result<()> {
        Err(GpuError::BackendUnavailable(
            "CUDA cannot execute SPIR-V shaders — use compile_cuda() instead".into(),
        ))
    }

    fn create_pipeline(
        &self,
        _spirv: &[u32],
        _entry_point: &str,
        _binding_count: usize,
        _push_constant_size: u32,
    ) -> Result<Self::Pipeline> {
        Err(GpuError::BackendUnavailable(
            "CUDA cannot compile SPIR-V pipelines — use compile_cuda() instead".into(),
        ))
    }

    fn dispatch_pipeline(
        &self,
        pipeline: &Self::Pipeline,
        buffers: &[&Self::Buffer],
        workgroups: [u32; 3],
        push_constants: Option<&[u8]>,
    ) -> Result<()> {
        self.dispatch_cuda(pipeline, buffers, workgroups, push_constants)
    }

    fn device_name(&self) -> &str {
        &self.device_name
    }

    fn device_memory(&self) -> u64 {
        self.device_memory
    }

    fn subgroup_size(&self) -> u32 {
        // NVIDIA warp size is always 32.
        32
    }

    fn copy_buffer(&self, src: &Self::Buffer, size: u64) -> Result<Self::Buffer> {
        let mut dst = self
            .stream
            .alloc_zeros::<u8>(size as usize)
            .map_err(|e| backend_err(BackendOp::CreateBuffer, e))?;
        self.stream
            .memcpy_dtod(&mut dst, &src.inner, size as usize)
            .map_err(|e| backend_err(BackendOp::CopyBuffer, e))?;
        Ok(CudaBuffer {
            inner: dst,
            size,
            stream: Arc::clone(&self.stream),
        })
    }
}

// ── Buffer operations ──

impl BackendBufferOps for CudaBuffer {
    fn read_back(&self) -> Result<Vec<u8>> {
        self.stream
            .clone_dtoh(&self.inner)
            .map_err(|e| backend_err(BackendOp::CopyBuffer, e))
    }

    fn byte_size(&self) -> u64 {
        self.size
    }
}

// ── Batch dispatch ──

impl CudaBatch {
    /// Record a kernel dispatch into the batch stream.
    pub fn record_dispatch(
        &mut self,
        kernel: &CudaKernel,
        buffers: &[&CudaBuffer],
        workgroups: [u32; 3],
        push_constants: Option<&[u8]>,
    ) -> Result<()> {
        let config = LaunchConfig {
            grid_dim: (workgroups[0], workgroups[1], workgroups[2]),
            block_dim: kernel.block_dim,
            shared_mem_bytes: 0,
        };

        let pc_values: Vec<u32> = push_constants
            .map(|pc| {
                pc.chunks_exact(4)
                    .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            })
            .unwrap_or_default();

        unsafe {
            let mut builder = self.stream.launch_builder(&kernel.function);

            for buf in buffers {
                builder.arg(&buf.inner);
            }

            for val in &pc_values {
                builder.arg(val);
            }

            builder
                .launch(config)
                .map_err(|e| backend_err(BackendOp::LaunchKernel, e))?;
        }

        Ok(())
    }

    /// No-op on CUDA — kernel launches on the same stream are serialized.
    #[allow(
        clippy::unused_self,
        clippy::needless_pass_by_ref_mut,
        clippy::missing_const_for_fn
    )]
    pub fn record_barrier(&mut self) {
        // CUDA streams serialize operations automatically.
    }

    /// Synchronize the batch stream, waiting for all recorded dispatches.
    pub fn submit(self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| backend_err(BackendOp::StreamSync, e))
    }
}
