// SPDX-License-Identifier: MIT OR Apache-2.0
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
#[cfg(feature = "cudnn")]
use std::sync::OnceLock;

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::CudaBlas;
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::backend::{Backend, BackendBufferOps};
#[cfg(feature = "cudnn")]
use crate::backend::cuda_cudnn::{Conv2dKey, CudnnState};
use crate::error::{backend_err, BackendOp, GpuError, Result};

// ── Public types ──

/// CUDA compute backend state.
pub struct CudaBackend {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: Mutex<CudaBlas>,
    /// Lazy-init cuDNN state. First conv call constructs it; subsequent
    /// calls reuse. Keeps `Backend::create` from failing when cuDNN is
    /// installed but unusable on this machine.
    #[cfg(feature = "cudnn")]
    cudnn: OnceLock<CudnnState>,
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

    /// Dispatch a compiled CUDA kernel and block until it completes.
    ///
    /// Matches the Vulkan fence-wait semantics — when this returns, the
    /// kernel has finished and any host-visible output is consistent.
    /// For pipelined work that doesn't need a host sync between stages,
    /// prefer [`Self::dispatch_cuda_async`].
    pub fn dispatch_cuda(
        &self,
        kernel: &CudaKernel,
        buffers: &[&CudaBuffer],
        workgroups: [u32; 3],
        push_constants: Option<&[u8]>,
    ) -> Result<()> {
        self.dispatch_cuda_async(kernel, buffers, workgroups, push_constants)?;
        self.stream
            .synchronize()
            .map_err(|e| backend_err(BackendOp::StreamSync, e))?;
        Ok(())
    }

    /// Dispatch a compiled CUDA kernel without synchronizing.
    ///
    /// Queues the kernel on this backend's stream and returns once it has
    /// been submitted to the GPU. Subsequent calls on the same stream are
    /// stream-ordered, so chained operators see consistent state. Host-side
    /// reads (`Buffer::download`) implicitly synchronize. Use this in tight
    /// pipelines where each per-call sync would otherwise dominate latency.
    pub fn dispatch_cuda_async(
        &self,
        kernel: &CudaKernel,
        buffers: &[&CudaBuffer],
        workgroups: [u32; 3],
        push_constants: Option<&[u8]>,
    ) -> Result<()> {
        launch_on_stream(&self.stream, kernel, buffers, workgroups, push_constants)
    }

    /// Run cuBLAS SGEMM and block until it completes: C = A * B.
    ///
    /// Buffers must contain `f32` data. This is the recommended path for
    /// matrix multiplication on CUDA — it reaches 80%+ peak throughput
    /// without any custom kernels.
    ///
    /// Matrix layout is row-major. Dimensions: A is `m×k`, B is `k×n`,
    /// C is `m×n`.
    ///
    /// For pipelined GPU-resident work, prefer [`Self::cublas_matmul_async`]
    /// to avoid the per-call sync.
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
        self.cublas_matmul_async(a, b, c, m, n, k)?;
        self.stream
            .synchronize()
            .map_err(|e| backend_err(BackendOp::StreamSync, e))?;
        Ok(())
    }

    /// Run cuBLAS SGEMM without synchronizing.
    ///
    /// Queues the SGEMM on this backend's stream. Subsequent stream work
    /// (kernels, further matmuls, downloads) is stream-ordered against it,
    /// so chained operators see consistent state. Use in tight pipelines
    /// where the per-call sync would otherwise dominate latency.
    #[allow(clippy::many_single_char_names)]
    pub fn cublas_matmul_async(
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
        Ok(())
    }

    /// Run cuBLAS GemmEx with bf16 inputs/outputs and fp32 accumulate, blocking
    /// until the GPU finishes.
    ///
    /// Buffers must contain `bf16` data laid out row-major (A is `m×k`, B is
    /// `k×n`, C is `m×n`). Internally calls `cublasGemmEx` with
    /// `CUDA_R_16BF` data type, `CUBLAS_COMPUTE_32F` accumulator, and
    /// `CUBLAS_GEMM_DEFAULT` algo selection — on Ampere+ the driver picks a
    /// tensor-core path automatically. Yields ~3–5× the throughput of fp32
    /// `sgemm` at ResNet-shaped sizes.
    ///
    /// For pipelined GPU-resident work, prefer
    /// [`Self::cublas_matmul_bf16_async`].
    #[cfg(feature = "bf16")]
    #[allow(clippy::many_single_char_names)]
    pub fn cublas_matmul_bf16(
        &self,
        a: &CudaBuffer,
        b: &CudaBuffer,
        c: &mut CudaBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        self.cublas_matmul_bf16_async(a, b, c, m, n, k)?;
        self.stream
            .synchronize()
            .map_err(|e| backend_err(BackendOp::StreamSync, e))?;
        Ok(())
    }

    /// Run cuBLAS GemmEx (bf16 / fp32-accumulate) without synchronizing.
    ///
    /// Same column-major-swap trick as [`Self::cublas_matmul_async`]: row-major
    /// `C = A·B` becomes column-major `Cᵀ = Bᵀ·Aᵀ`. Alpha and beta are passed
    /// as `f32` because the compute type is `CUBLAS_COMPUTE_32F`.
    #[cfg(feature = "bf16")]
    #[allow(clippy::many_single_char_names)]
    pub fn cublas_matmul_bf16_async(
        &self,
        a: &CudaBuffer,
        b: &CudaBuffer,
        c: &mut CudaBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        self.cublas_gemm_ex_async_inner(a, b, c, m, n, k, /* c_is_f32 */ false)
    }

    /// Run cuBLAS GemmEx with bf16 inputs and an **fp32** output.
    ///
    /// Same compute path as [`Self::cublas_matmul_bf16_async`] (CUDA_R_16BF
    /// data, CUBLAS_COMPUTE_32F accumulator, tensor-core algo selection),
    /// but the fp32 accumulator is written directly to `c` without rounding
    /// down to bf16. Skips the cast-back-to-f32 HBM pass that
    /// [`Self::cublas_matmul_bf16_async`] would otherwise force on the
    /// caller, and is bit-exact with "GemmEx then cast-up" up to a single
    /// rounding step.
    ///
    /// `c` must be sized for `m × n` `f32` elements (4 bytes each).
    #[cfg(feature = "bf16")]
    #[allow(clippy::many_single_char_names)]
    pub fn cublas_matmul_bf16_in_f32_out_async(
        &self,
        a: &CudaBuffer,
        b: &CudaBuffer,
        c: &mut CudaBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        self.cublas_gemm_ex_async_inner(a, b, c, m, n, k, /* c_is_f32 */ true)
    }

    /// Shared body of the two `cublas_matmul_bf16*` variants. The output
    /// data type is the only thing that varies — both compute via the same
    /// fp32 accumulator and tensor-core algo selection.
    #[cfg(feature = "bf16")]
    #[allow(clippy::many_single_char_names)]
    fn cublas_gemm_ex_async_inner(
        &self,
        a: &CudaBuffer,
        b: &CudaBuffer,
        c: &mut CudaBuffer,
        m: u32,
        n: u32,
        k: u32,
        c_is_f32: bool,
    ) -> Result<()> {
        use cudarc::cublas::sys;
        use std::ffi::c_void;

        let c_type = if c_is_f32 {
            sys::cudaDataType_t::CUDA_R_32F
        } else {
            sys::cudaDataType_t::CUDA_R_16BF
        };

        #[allow(clippy::cast_possible_wrap)]
        unsafe {
            let blas = self
                .blas
                .lock()
                .map_err(|_| backend_err(BackendOp::MutexPoisoned, "cublas"))?;
            let (a_ptr, _a_guard) = a.inner.device_ptr(&self.stream);
            let (b_ptr, _b_guard) = b.inner.device_ptr(&self.stream);
            let (c_ptr, _c_guard) = c.inner.device_ptr_mut(&self.stream);

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            cudarc::cublas::result::gemm_ex(
                *blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                std::ptr::from_ref(&alpha).cast::<c_void>(),
                b_ptr as *const c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                n as i32,
                a_ptr as *const c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                std::ptr::from_ref(&beta).cast::<c_void>(),
                c_ptr as *mut c_void,
                c_type,
                n as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(|e| backend_err(BackendOp::CuBlas, e))?;
        }
        Ok(())
    }

    /// Run cuBLAS strided batched SGEMM without synchronizing.
    ///
    /// Computes `C[i] = op_a(A[i]) · op_b(B[i])` for `i in 0..batch`, where
    /// each per-batch matrix is laid out row-major with the natural stride
    /// (`m*k` for A, `k*n` for B, `m*n` for C). Single cuBLAS launch covers
    /// all batches — replaces an N-loop over `cublas_matmul_async` (cheaper
    /// host launch latency, lets cuBLAS pick a tensor-core-friendly algo
    /// for the whole batch).
    ///
    /// Storage shapes (row-major):
    /// - A: `[batch, m, k]` if `!trans_a` else `[batch, k, m]`
    /// - B: `[batch, k, n]` if `!trans_b` else `[batch, n, k]`
    /// - C: `[batch, m, n]`
    #[allow(clippy::too_many_arguments)]
    pub fn cublas_strided_batched_matmul_async(
        &self,
        a: &CudaBuffer,
        b: &CudaBuffer,
        c: &mut CudaBuffer,
        batch: u32,
        m: u32,
        n: u32,
        k: u32,
        trans_a: bool,
        trans_b: bool,
    ) -> Result<()> {
        // Column-major swap trick: row-major `C = op_a(A) · op_b(B)` becomes
        // col-major `Cᵀ = op_b(B)ᵀ · op_a(A)ᵀ`. We pass `B_buf` as cuBLAS's
        // first matrix arg and `A_buf` as second. Op flags travel with the
        // arg they apply to: cuBLAS's op-on-first matches user's trans_b,
        // and op-on-second matches user's trans_a.
        let cu_op_for_b = if trans_b {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let cu_op_for_a = if trans_a {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        // Leading dim is the row-major row size of the buffer (= the
        // number of rows in the col-major view, which is what cuBLAS
        // consumes for `lda`/`ldb` regardless of op flag).
        let row_stride_b = if trans_b { k } else { n };
        let row_stride_a = if trans_a { m } else { k };
        let stride_a = (m * k) as i64;
        let stride_b = (k * n) as i64;
        let stride_c = (m * n) as i64;

        #[allow(clippy::cast_possible_wrap)]
        unsafe {
            let blas = self
                .blas
                .lock()
                .map_err(|_| backend_err(BackendOp::MutexPoisoned, "cublas"))?;
            let (a_ptr, _a_guard) = a.inner.device_ptr(&self.stream);
            let (b_ptr, _b_guard) = b.inner.device_ptr(&self.stream);
            let (c_ptr, _c_guard) = c.inner.device_ptr_mut(&self.stream);

            cudarc::cublas::result::sgemm_strided_batched(
                *blas.handle(),
                cu_op_for_b,
                cu_op_for_a,
                n as i32,
                m as i32,
                k as i32,
                &1.0f32,
                b_ptr as *const f32,
                row_stride_b as i32,
                stride_b,
                a_ptr as *const f32,
                row_stride_a as i32,
                stride_a,
                &0.0f32,
                c_ptr as *mut f32,
                n as i32,
                stride_c,
                batch as i32,
            )
            .map_err(|e| backend_err(BackendOp::CuBlas, e))?;
        }
        Ok(())
    }

    /// Run cuBLAS strided batched `GemmEx` with bf16 inputs and an `f32`
    /// output, no sync.
    ///
    /// Same shape contract as [`Self::cublas_strided_batched_matmul_async`]
    /// (row-major per-batch, natural strides, `[batch, m, k]` × `[batch, k,
    /// n]` → `[batch, m, n]` with optional transpose flags) but routed
    /// through `cublasGemmStridedBatchedEx` with `CUDA_R_16BF` data and
    /// `CUBLAS_COMPUTE_32F` accumulator. The fp32 accumulator is written
    /// straight to `c` so downstream operators see fp32 storage without a
    /// cast-up pass — mirrors [`Self::cublas_matmul_bf16_in_f32_out_async`]
    /// for the batched attention path.
    ///
    /// Unblocks `ViT` attention's two strided batched matmuls per block (Q@Kᵀ
    /// and attn@V — 24 calls/forward at 12 layers): without this they stay
    /// on the fp32 `sgemm_strided_batched` path even when bf16 matmul is on.
    #[cfg(feature = "bf16")]
    #[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
    pub fn cublas_strided_batched_matmul_bf16_in_f32_out_async(
        &self,
        a: &CudaBuffer,
        b: &CudaBuffer,
        c: &mut CudaBuffer,
        batch: u32,
        m: u32,
        n: u32,
        k: u32,
        trans_a: bool,
        trans_b: bool,
    ) -> Result<()> {
        use cudarc::cublas::sys;
        use std::ffi::c_void;

        // Same column-major-swap trick as the f32 path — see the comment in
        // `cublas_strided_batched_matmul_async`. Op flags travel with the
        // arg they apply to; lda/ldb are the natural row-major row sizes.
        let cu_op_for_b = if trans_b {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let cu_op_for_a = if trans_a {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let row_stride_b = if trans_b { k } else { n };
        let row_stride_a = if trans_a { m } else { k };
        let stride_a = (m * k) as i64;
        let stride_b = (k * n) as i64;
        let stride_c = (m * n) as i64;

        #[allow(clippy::cast_possible_wrap)]
        unsafe {
            let blas = self
                .blas
                .lock()
                .map_err(|_| backend_err(BackendOp::MutexPoisoned, "cublas"))?;
            let (a_ptr, _a_guard) = a.inner.device_ptr(&self.stream);
            let (b_ptr, _b_guard) = b.inner.device_ptr(&self.stream);
            let (c_ptr, _c_guard) = c.inner.device_ptr_mut(&self.stream);

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            cudarc::cublas::result::gemm_strided_batched_ex(
                *blas.handle(),
                cu_op_for_b,
                cu_op_for_a,
                n as i32,
                m as i32,
                k as i32,
                std::ptr::from_ref(&alpha).cast::<c_void>(),
                b_ptr as *const c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                row_stride_b as i32,
                stride_b,
                a_ptr as *const c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                row_stride_a as i32,
                stride_a,
                std::ptr::from_ref(&beta).cast::<c_void>(),
                c_ptr as *mut c_void,
                sys::cudaDataType_t::CUDA_R_32F,
                n as i32,
                stride_c,
                batch as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(|e| backend_err(BackendOp::CuBlas, e))?;
        }
        Ok(())
    }

    /// Run a cuDNN 2D convolution forward pass without synchronizing.
    ///
    /// Implicit-GEMM (or Winograd / FFT — cuDNN's heuristic picks per shape)
    /// fused conv that skips the im2col→cuBLAS round-trip the default path
    /// uses. First call for a given shape pays the descriptor + algo-pick
    /// cost; subsequent calls hit the per-backend cache.
    ///
    /// Layout is `NCHW`. Buffers must be sized:
    /// - `input`: `n*c_in*h_in*w_in` × `f32`
    /// - `filter`: `c_out*c_in*k_h*k_w` × `f32` (PyTorch / scry-llm filter layout)
    /// - `output`: `n*c_out*h_out*w_out` × `f32`, where output spatial dims
    ///   come from the standard floor formula. The method returns
    ///   `(h_out, w_out)` so the caller doesn't have to recompute.
    ///
    /// Stream-ordered against subsequent kernels and matmuls; chained
    /// operators see consistent state without a host fence.
    #[cfg(feature = "cudnn")]
    #[allow(clippy::too_many_arguments)]
    pub fn cudnn_conv2d_forward_async(
        &self,
        input: &CudaBuffer,
        filter: &CudaBuffer,
        output: &mut CudaBuffer,
        n: u32,
        c_in: u32,
        h_in: u32,
        w_in: u32,
        c_out: u32,
        k_h: u32,
        k_w: u32,
        pad_h: u32,
        pad_w: u32,
        stride_h: u32,
        stride_w: u32,
    ) -> Result<(u32, u32)> {
        let cudnn = self.cudnn_state()?;
        let key = Conv2dKey {
            n,
            c_in,
            h_in,
            w_in,
            c_out,
            k_h,
            k_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
        };
        cudnn.conv2d_forward_async(&self.stream, input, filter, output, key)
    }

    /// Lazy-initialize the cuDNN handle, then return a borrow to it. Stable
    /// alternative to `OnceLock::get_or_try_init` (which is still nightly):
    /// peek with `get`, fall back to creating + `set`.
    #[cfg(feature = "cudnn")]
    fn cudnn_state(&self) -> Result<&CudnnState> {
        if let Some(state) = self.cudnn.get() {
            return Ok(state);
        }
        let state = CudnnState::create(Arc::clone(&self.stream))?;
        // Another thread may have initialized between our `get` and `set` —
        // their value wins (we drop ours), the caller still gets a valid handle.
        let _ = self.cudnn.set(state);
        Ok(self
            .cudnn
            .get()
            .expect("cuDNN handle was just initialized"))
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
            unsafe { cudarc::driver::result::device::total_mem(ctx.cu_device()) }
                .map_err(|e| backend_err(BackendOp::DeviceQuery, e))?
                as u64;

        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(|e| backend_err(BackendOp::CuBlas, e))?;

        Ok(Self {
            ctx,
            stream,
            blas: Mutex::new(blas),
            #[cfg(feature = "cudnn")]
            cudnn: OnceLock::new(),
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

    fn alloc_uninit(&self, size: u64) -> Result<Self::Buffer> {
        // SAFETY: cudarc's `Stream::alloc` returns a CudaSlice<u8> whose
        // contents are undefined. That's exactly the contract of
        // alloc_uninit — caller is responsible for fully overwriting
        // before any read.
        let inner = unsafe {
            self.stream
                .alloc::<u8>(size as usize)
                .map_err(|e| backend_err(BackendOp::CreateBuffer, e))?
        };
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
            .memcpy_dtod(&src.inner, &mut dst)
            .map_err(|e| backend_err(BackendOp::CopyBuffer, e))?;
        Ok(CudaBuffer {
            inner: dst,
            size,
            stream: Arc::clone(&self.stream),
        })
    }

    fn synchronize(&self) -> Result<()> {
        // Drain every async dispatch issued on this stream. Needed any time
        // the host needs to observe completion timing (benchmarks) or state
        // through a path that doesn't already sync (`Buffer::download` does
        // sync internally; the async dispatch helpers do not).
        self.stream
            .synchronize()
            .map_err(|e| backend_err(BackendOp::StreamSync, e))
    }
}

// ── Buffer operations ──

impl BackendBufferOps for CudaBuffer {
    fn read_back(&self) -> Result<Vec<u8>> {
        // cudarc's clone_dtoh issues a stream-ordered async memcpy to a
        // plain Vec<u8>, whose SyncOnDrop drops to a no-op — the host data
        // is not guaranteed visible on return. Sync explicitly so callers
        // observe the result of any prior async dispatches on this stream.
        let out = self
            .stream
            .clone_dtoh(&self.inner)
            .map_err(|e| backend_err(BackendOp::CopyBuffer, e))?;
        self.stream
            .synchronize()
            .map_err(|e| backend_err(BackendOp::StreamSync, e))?;
        Ok(out)
    }

    fn byte_size(&self) -> u64 {
        self.size
    }
}

// Stream-bound kernel launch shared by sync, async, and batched dispatch.
// Queues the kernel on `stream` without synchronizing — callers that need
// host-visible state must sync the stream themselves (or rely on the
// implicit sync inside `Buffer::download`).
fn launch_on_stream(
    stream: &CudaStream,
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

    // Push constants are passed as individual u32 kernel args; collect
    // them up front so they outlive the launch builder.
    let pc_values: Vec<u32> = push_constants
        .map(|pc| {
            pc.chunks_exact(4)
                .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        })
        .unwrap_or_default();

    unsafe {
        let mut builder = stream.launch_builder(&kernel.function);
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

// ── Batch dispatch ──

impl CudaBatch {
    /// Record a kernel dispatch into the batch stream.
    ///
    /// `&mut self` is kept for parity with the Vulkan batch API (which
    /// records into a command buffer and genuinely needs unique access).
    /// Records on a CUDA stream are stream-ordered without exclusive access,
    /// so the borrow is conservative on this backend.
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn record_dispatch(
        &mut self,
        kernel: &CudaKernel,
        buffers: &[&CudaBuffer],
        workgroups: [u32; 3],
        push_constants: Option<&[u8]>,
    ) -> Result<()> {
        launch_on_stream(&self.stream, kernel, buffers, workgroups, push_constants)
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

    /// Submit all recorded dispatches and return a [`CudaTicket`] for
    /// non-blocking completion tracking.
    pub fn submit_async(self) -> Result<CudaTicket> {
        let event = self
            .stream
            .record_event(None)
            .map_err(|e| backend_err(BackendOp::RecordEvent, e))?;
        Ok(CudaTicket {
            stream: self.stream,
            event,
        })
    }
}

/// In-flight GPU submission handle for the CUDA backend.
///
/// Records a [`CudaEvent`] at submit time so completion can be polled
/// without synchronizing the entire stream.
pub struct CudaTicket {
    stream: Arc<CudaStream>,
    event: CudaEvent,
}

impl CudaTicket {
    /// Block until all work preceding the recorded event completes.
    pub(crate) fn wait(self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| backend_err(BackendOp::StreamSync, e))
    }

    /// Check whether the recorded event has been reached without blocking.
    pub(crate) fn is_ready(&self) -> bool {
        self.event.is_complete()
    }
}
