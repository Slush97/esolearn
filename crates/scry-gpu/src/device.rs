// SPDX-License-Identifier: MIT OR Apache-2.0
//! Device acquisition and the primary user-facing API.

use crate::backend::{Backend, BackendBuffer, BackendKernel};
use crate::buffer::{Buffer, GpuBuf};
use crate::dispatch::{self, DispatchConfig};
use crate::error::{GpuError, Result};
use crate::kernel::Kernel;
use crate::shader;

/// A GPU compute device.
///
/// This is the main entry point for scry-gpu. A `Device` wraps a single
/// GPU and provides methods to upload data, dispatch shaders, and read
/// results back.
///
/// # Example
///
/// ```ignore
/// let gpu = Device::auto()?;
///
/// let input = gpu.upload(&[1.0f32, 2.0, 3.0, 4.0])?;
/// let output = gpu.alloc::<f32>(4)?;
///
/// gpu.dispatch(SHADER_SRC, &[&input, &output], 4)?;
///
/// let result: Vec<f32> = output.download()?;
/// ```
pub struct Device {
    inner: DeviceInner,
}

enum DeviceInner {
    #[cfg(feature = "vulkan")]
    Vulkan(crate::backend::vulkan::VulkanBackend),
    #[cfg(feature = "cuda")]
    Cuda(crate::backend::cuda::CudaBackend),
}

/// Available backend types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Vulkan (Linux, Windows, Android).
    Vulkan,
    /// CUDA (NVIDIA GPUs).
    Cuda,
    // Metal, // future
}

impl Device {
    /// Auto-select the best available GPU.
    ///
    /// Tries backends in order of preference: CUDA → Vulkan → (Metal in future).
    /// CUDA is preferred when available because it enables cuBLAS matmul and
    /// native CUDA kernel dispatch.
    pub fn auto() -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            use crate::backend::cuda::CudaBackend;
            if let Ok(backend) = CudaBackend::create() {
                return Ok(Self {
                    inner: DeviceInner::Cuda(backend),
                });
            }
        }

        #[cfg(feature = "vulkan")]
        {
            use crate::backend::vulkan::VulkanBackend;
            if let Ok(backend) = VulkanBackend::create() {
                return Ok(Self {
                    inner: DeviceInner::Vulkan(backend),
                });
            }
        }

        Err(GpuError::NoDevice)
    }

    /// Create a device with a specific backend.
    pub fn with_backend(kind: BackendKind) -> Result<Self> {
        match kind {
            BackendKind::Vulkan => {
                #[cfg(feature = "vulkan")]
                {
                    use crate::backend::vulkan::VulkanBackend;
                    let backend = VulkanBackend::create()?;
                    Ok(Self {
                        inner: DeviceInner::Vulkan(backend),
                    })
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    Err(GpuError::BackendUnavailable(
                        "vulkan feature not enabled".into(),
                    ))
                }
            }
            BackendKind::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    use crate::backend::cuda::CudaBackend;
                    let backend = CudaBackend::create()?;
                    Ok(Self {
                        inner: DeviceInner::Cuda(backend),
                    })
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(GpuError::BackendUnavailable(
                        "cuda feature not enabled".into(),
                    ))
                }
            }
        }
    }

    /// Upload a slice to GPU memory, returning a typed buffer.
    pub fn upload<T: bytemuck::Pod>(&self, data: &[T]) -> Result<Buffer<T>> {
        let bytes = bytemuck::cast_slice(data);
        let inner = self.upload_raw(bytes)?;
        Ok(Buffer {
            inner,
            len: data.len(),
            _marker: std::marker::PhantomData,
        })
    }

    /// Allocate a GPU buffer for `count` elements of type `T`.
    ///
    /// On backends that zero-initialize by default (CUDA), the buffer is
    /// zero-filled before return. Use [`Self::alloc_uninit`] in tight loops
    /// where the caller fully overwrites the buffer (matmul output, kernel
    /// dispatches that write every element).
    pub fn alloc<T: bytemuck::Pod>(&self, count: usize) -> Result<Buffer<T>> {
        let size = count.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
            GpuError::AllocationFailed {
                requested: u64::MAX,
                device_max: self.memory(),
            }
        })? as u64;
        let inner = self.alloc_raw(size)?;
        Ok(Buffer {
            inner,
            len: count,
            _marker: std::marker::PhantomData,
        })
    }

    /// Allocate a GPU buffer with **undefined** contents.
    ///
    /// Faster than [`Self::alloc`] on CUDA (skips a `cuMemsetD8Async`
    /// dispatch). Caller must overwrite every element that will later be
    /// read — otherwise downstream computation observes garbage. Use this
    /// only for outputs that the next kernel writes in full (matmul C
    /// matrix, elementwise kernels with full-coverage thread dispatch).
    pub fn alloc_uninit<T: bytemuck::Pod>(&self, count: usize) -> Result<Buffer<T>> {
        let size = count.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
            GpuError::AllocationFailed {
                requested: u64::MAX,
                device_max: self.memory(),
            }
        })? as u64;
        let inner = self.alloc_uninit_raw(size)?;
        Ok(Buffer {
            inner,
            len: count,
            _marker: std::marker::PhantomData,
        })
    }

    /// Dispatch a WGSL compute shader.
    ///
    /// Buffers are bound in order to `@binding(0)`, `@binding(1)`, etc.
    /// Workgroup dispatch dimensions are auto-calculated from `invocations`
    /// and the shader's `@workgroup_size`.
    pub fn dispatch(
        &self,
        shader_src: &str,
        buffers: &[&dyn GpuBuf],
        invocations: u32,
    ) -> Result<()> {
        let entry = "main";
        let compiled = shader::compile_wgsl(shader_src, entry)?;

        let expected = shader::binding_count(&compiled.module);
        let backend_bufs: Vec<&BackendBuffer> = buffers.iter().map(|b| b.raw()).collect();
        if expected != backend_bufs.len() {
            return Err(GpuError::BindingMismatch {
                expected,
                got: backend_bufs.len(),
            });
        }

        let wg_size = dispatch::extract_workgroup_size(&compiled.module, entry);
        let workgroups = dispatch::calc_dispatch(invocations, wg_size);

        self.dispatch_spirv(&compiled.spirv, entry, &backend_bufs, workgroups, None)
    }

    /// Dispatch with full configuration.
    pub fn dispatch_configured(
        &self,
        config: &DispatchConfig<'_>,
        buffers: &[&dyn GpuBuf],
    ) -> Result<()> {
        let entry = config.entry_point.unwrap_or("main");
        let compiled = shader::compile_wgsl(config.shader, entry)?;

        let expected = shader::binding_count(&compiled.module);
        let backend_bufs: Vec<&BackendBuffer> = buffers.iter().map(|b| b.raw()).collect();
        if expected != backend_bufs.len() {
            return Err(GpuError::BindingMismatch {
                expected,
                got: backend_bufs.len(),
            });
        }

        let workgroups = config.workgroups.unwrap_or_else(|| {
            let wg_size = dispatch::extract_workgroup_size(&compiled.module, entry);
            dispatch::calc_dispatch(config.invocations, wg_size)
        });

        self.dispatch_spirv(
            &compiled.spirv,
            entry,
            &backend_bufs,
            workgroups,
            config.push_constants,
        )
    }

    /// Compile a WGSL compute shader into a reusable [`Kernel`].
    ///
    /// The returned kernel holds all GPU objects (pipeline, layouts,
    /// shader module) and can be dispatched many times via [`Device::run`].
    ///
    /// Uses `"main"` as the entry point. See [`Device::compile_named`]
    /// for a custom entry point.
    pub fn compile(&self, shader_src: &str) -> Result<Kernel> {
        self.compile_named(shader_src, "main")
    }

    /// Compile a WGSL shader with a specific entry point name.
    pub fn compile_named(&self, shader_src: &str, entry_point: &str) -> Result<Kernel> {
        let compiled = shader::compile_wgsl(shader_src, entry_point)?;
        let binding_count = shader::binding_count(&compiled.module);
        let workgroup_size = dispatch::extract_workgroup_size(&compiled.module, entry_point);
        let push_constant_size = shader::push_constant_size(&compiled.module);

        let inner = self.create_pipeline(
            &compiled.spirv,
            entry_point,
            binding_count,
            push_constant_size,
        )?;

        Ok(Kernel {
            inner,
            binding_count,
            workgroup_size,
            entry_point: entry_point.to_string(),
        })
    }

    /// Dispatch a precompiled kernel.
    ///
    /// Buffers are bound in order to `@binding(0)`, `@binding(1)`, etc.
    /// Workgroup dispatch dimensions are auto-calculated from `invocations`
    /// and the kernel's compiled `@workgroup_size`.
    pub fn run(&self, kernel: &Kernel, buffers: &[&dyn GpuBuf], invocations: u32) -> Result<()> {
        let backend_bufs: Vec<&BackendBuffer> = buffers.iter().map(|b| b.raw()).collect();
        if kernel.binding_count != backend_bufs.len() {
            return Err(GpuError::BindingMismatch {
                expected: kernel.binding_count,
                got: backend_bufs.len(),
            });
        }

        let workgroups = dispatch::calc_dispatch(invocations, kernel.workgroup_size);
        self.run_pipeline(kernel, &backend_bufs, workgroups, None)
    }

    /// Dispatch a precompiled kernel with push constants.
    pub fn run_with_push_constants(
        &self,
        kernel: &Kernel,
        buffers: &[&dyn GpuBuf],
        invocations: u32,
        push_constants: &[u8],
    ) -> Result<()> {
        let backend_bufs: Vec<&BackendBuffer> = buffers.iter().map(|b| b.raw()).collect();
        if kernel.binding_count != backend_bufs.len() {
            return Err(GpuError::BindingMismatch {
                expected: kernel.binding_count,
                got: backend_bufs.len(),
            });
        }

        let workgroups = dispatch::calc_dispatch(invocations, kernel.workgroup_size);
        self.run_pipeline(kernel, &backend_bufs, workgroups, Some(push_constants))
    }

    /// Dispatch a precompiled kernel with explicit workgroup dimensions.
    ///
    /// Use this for 2D/3D dispatches or when you need precise control over
    /// workgroup counts. For simple 1D dispatches, prefer [`Device::run`].
    pub fn run_configured(
        &self,
        kernel: &Kernel,
        buffers: &[&dyn GpuBuf],
        workgroups: [u32; 3],
        push_constants: Option<&[u8]>,
    ) -> Result<()> {
        let backend_bufs: Vec<&BackendBuffer> = buffers.iter().map(|b| b.raw()).collect();
        if kernel.binding_count != backend_bufs.len() {
            return Err(GpuError::BindingMismatch {
                expected: kernel.binding_count,
                got: backend_bufs.len(),
            });
        }

        self.run_pipeline(kernel, &backend_bufs, workgroups, push_constants)
    }

    /// Create a GPU-to-GPU copy of a buffer.
    ///
    /// Allocates a new buffer on the same device and copies the contents
    /// of `src` into it. The copy is synchronous (blocks until complete).
    pub fn copy_buffer<T: bytemuck::Pod>(&self, src: &Buffer<T>) -> Result<Buffer<T>> {
        let size = src.byte_size();
        let inner = self.copy_buffer_raw(&src.inner, size)?;
        Ok(Buffer {
            inner,
            len: src.len,
            _marker: std::marker::PhantomData,
        })
    }

    /// Begin a batched dispatch session.
    ///
    /// Records multiple dispatches into a single command buffer, submitted
    /// with one fence wait via [`Batch::submit`].
    pub fn batch(&self) -> Result<crate::batch::Batch> {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            DeviceInner::Vulkan(b) => {
                let vk_batch = b.begin_batch()?;
                Ok(crate::batch::Batch::new_vulkan(vk_batch))
            }
            #[cfg(feature = "cuda")]
            DeviceInner::Cuda(b) => {
                let cuda_batch = b.begin_batch()?;
                Ok(crate::batch::Batch::new_cuda(cuda_batch))
            }
        }
    }

    /// Block until every previously-issued dispatch has completed.
    ///
    /// Async paths ([`Self::run_configured_async`],
    /// [`Self::cublas_matmul_async`]) elide the per-call host sync to keep
    /// chained dispatches cheap, but a benchmark needs to wait for the GPU
    /// to actually finish before stopping the timer. This is the explicit
    /// drain. On the synchronous Vulkan path it's a no-op since each
    /// dispatch already waits for its own fence.
    pub fn synchronize(&self) -> Result<()> {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            DeviceInner::Vulkan(b) => b.synchronize(),
            #[cfg(feature = "cuda")]
            DeviceInner::Cuda(b) => b.synchronize(),
        }
    }

    /// Device name (for diagnostics / logging).
    pub fn name(&self) -> &str {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            DeviceInner::Vulkan(b) => b.device_name(),
            #[cfg(feature = "cuda")]
            DeviceInner::Cuda(b) => b.device_name(),
        }
    }

    /// Total device memory in bytes.
    pub fn memory(&self) -> u64 {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            DeviceInner::Vulkan(b) => b.device_memory(),
            #[cfg(feature = "cuda")]
            DeviceInner::Cuda(b) => b.device_memory(),
        }
    }

    /// Subgroup (warp/wavefront) size.
    ///
    /// Typically 32 on NVIDIA, 64 on AMD, 32 on Intel.
    /// Useful for sizing subgroup-aware shaders.
    pub fn subgroup_size(&self) -> u32 {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            DeviceInner::Vulkan(b) => b.subgroup_size(),
            #[cfg(feature = "cuda")]
            DeviceInner::Cuda(b) => b.subgroup_size(),
        }
    }

    /// Which backend this device is using.
    pub const fn backend_kind(&self) -> BackendKind {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            DeviceInner::Vulkan(_) => BackendKind::Vulkan,
            #[cfg(feature = "cuda")]
            DeviceInner::Cuda(_) => BackendKind::Cuda,
        }
    }

    // ── private helpers ──

    fn upload_raw(&self, data: &[u8]) -> Result<BackendBuffer> {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            DeviceInner::Vulkan(b) => {
                let buf = b.upload(data)?;
                Ok(BackendBuffer::Vulkan(buf))
            }
            #[cfg(feature = "cuda")]
            DeviceInner::Cuda(b) => {
                let buf = b.upload(data)?;
                Ok(BackendBuffer::Cuda(buf))
            }
        }
    }

    fn copy_buffer_raw(&self, src: &BackendBuffer, size: u64) -> Result<BackendBuffer> {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            DeviceInner::Vulkan(b) => {
                #[allow(irrefutable_let_patterns)]
                let BackendBuffer::Vulkan(vk_src) = src
                else {
                    return Err(GpuError::BackendUnavailable(
                        "buffer/backend mismatch: expected Vulkan buffer".into(),
                    ));
                };
                let buf = b.copy_buffer(vk_src, size)?;
                Ok(BackendBuffer::Vulkan(buf))
            }
            #[cfg(feature = "cuda")]
            DeviceInner::Cuda(b) => {
                #[allow(irrefutable_let_patterns)]
                let BackendBuffer::Cuda(cuda_src) = src
                else {
                    return Err(GpuError::BackendUnavailable(
                        "buffer/backend mismatch: expected CUDA buffer".into(),
                    ));
                };
                let buf = b.copy_buffer(cuda_src, size)?;
                Ok(BackendBuffer::Cuda(buf))
            }
        }
    }

    fn alloc_raw(&self, size: u64) -> Result<BackendBuffer> {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            DeviceInner::Vulkan(b) => {
                let buf = b.alloc(size)?;
                Ok(BackendBuffer::Vulkan(buf))
            }
            #[cfg(feature = "cuda")]
            DeviceInner::Cuda(b) => {
                let buf = b.alloc(size)?;
                Ok(BackendBuffer::Cuda(buf))
            }
        }
    }

    fn alloc_uninit_raw(&self, size: u64) -> Result<BackendBuffer> {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            DeviceInner::Vulkan(b) => {
                let buf = b.alloc_uninit(size)?;
                Ok(BackendBuffer::Vulkan(buf))
            }
            #[cfg(feature = "cuda")]
            DeviceInner::Cuda(b) => {
                let buf = b.alloc_uninit(size)?;
                Ok(BackendBuffer::Cuda(buf))
            }
        }
    }

    fn dispatch_spirv(
        &self,
        spirv: &[u32],
        entry_point: &str,
        buffers: &[&BackendBuffer],
        workgroups: [u32; 3],
        push_constants: Option<&[u8]>,
    ) -> Result<()> {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            DeviceInner::Vulkan(b) => {
                let vk_bufs: Vec<&crate::backend::vulkan::VulkanBuffer> = buffers
                    .iter()
                    .map(|buf| match buf {
                        BackendBuffer::Vulkan(vb) => Ok(vb),
                        #[cfg(feature = "cuda")]
                        _ => Err(GpuError::BackendUnavailable(
                            "buffer/backend mismatch: expected Vulkan buffer".into(),
                        )),
                    })
                    .collect::<Result<Vec<_>>>()?;
                b.dispatch(spirv, entry_point, &vk_bufs, workgroups, push_constants)
            }
            #[cfg(feature = "cuda")]
            DeviceInner::Cuda(b) => {
                let cuda_bufs: Vec<&crate::backend::cuda::CudaBuffer> = buffers
                    .iter()
                    .map(|buf| match buf {
                        BackendBuffer::Cuda(cb) => Ok(cb),
                        #[cfg(feature = "vulkan")]
                        _ => Err(GpuError::BackendUnavailable(
                            "buffer/backend mismatch: expected CUDA buffer".into(),
                        )),
                    })
                    .collect::<Result<Vec<_>>>()?;
                b.dispatch(spirv, entry_point, &cuda_bufs, workgroups, push_constants)
            }
        }
    }

    fn create_pipeline(
        &self,
        spirv: &[u32],
        entry_point: &str,
        binding_count: usize,
        push_constant_size: u32,
    ) -> Result<BackendKernel> {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            DeviceInner::Vulkan(b) => {
                let kernel =
                    b.create_pipeline(spirv, entry_point, binding_count, push_constant_size)?;
                Ok(BackendKernel::Vulkan(kernel))
            }
            #[cfg(feature = "cuda")]
            DeviceInner::Cuda(b) => {
                let kernel =
                    b.create_pipeline(spirv, entry_point, binding_count, push_constant_size)?;
                Ok(BackendKernel::Cuda(kernel))
            }
        }
    }

    fn run_pipeline(
        &self,
        kernel: &Kernel,
        buffers: &[&BackendBuffer],
        workgroups: [u32; 3],
        push_constants: Option<&[u8]>,
    ) -> Result<()> {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            DeviceInner::Vulkan(b) => {
                #[allow(irrefutable_let_patterns)]
                let BackendKernel::Vulkan(vk_kernel) = &kernel.inner
                else {
                    return Err(GpuError::BackendUnavailable(
                        "kernel was not compiled for Vulkan".into(),
                    ));
                };
                let vk_bufs: Vec<&crate::backend::vulkan::VulkanBuffer> = buffers
                    .iter()
                    .map(|buf| match buf {
                        BackendBuffer::Vulkan(vb) => Ok(vb),
                        #[cfg(feature = "cuda")]
                        _ => Err(GpuError::BackendUnavailable(
                            "buffer/backend mismatch: expected Vulkan buffer".into(),
                        )),
                    })
                    .collect::<Result<Vec<_>>>()?;
                b.dispatch_pipeline(vk_kernel, &vk_bufs, workgroups, push_constants)
            }
            #[cfg(feature = "cuda")]
            DeviceInner::Cuda(b) => {
                let BackendKernel::Cuda(cuda_kernel) = &kernel.inner else {
                    return Err(GpuError::BackendUnavailable(
                        "kernel was not compiled for CUDA".into(),
                    ));
                };
                let cuda_bufs: Vec<&crate::backend::cuda::CudaBuffer> = buffers
                    .iter()
                    .map(|buf| match buf {
                        BackendBuffer::Cuda(cb) => Ok(cb),
                        #[cfg(feature = "vulkan")]
                        _ => Err(GpuError::BackendUnavailable(
                            "buffer/backend mismatch: expected CUDA buffer".into(),
                        )),
                    })
                    .collect::<Result<Vec<_>>>()?;
                b.dispatch_pipeline(cuda_kernel, &cuda_bufs, workgroups, push_constants)
            }
        }
    }
}

// ── CUDA-specific methods ──

#[cfg(feature = "cuda")]
impl Device {
    /// Compile a CUDA C kernel source into a reusable [`Kernel`].
    ///
    /// Only available on the CUDA backend. Uses NVRTC for compilation.
    ///
    /// Unlike [`Device::compile`] (which uses WGSL→SPIR-V), this accepts
    /// native CUDA C source. Because CUDA kernels don't embed metadata
    /// like WGSL's `@workgroup_size` and `@binding`, you must provide
    /// `binding_count` and `workgroup_size` explicitly.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::BackendUnavailable`] if the device is not using
    /// the CUDA backend.
    pub fn compile_cuda(
        &self,
        source: &str,
        entry_point: &str,
        binding_count: usize,
        workgroup_size: [u32; 3],
    ) -> Result<Kernel> {
        self.compile_cuda_with_arch(
            source,
            entry_point,
            binding_count,
            workgroup_size,
            None,
            &[],
        )
    }

    /// Compile a CUDA C kernel source into a reusable [`Kernel`] with an
    /// explicit virtual architecture and include paths.
    ///
    /// Forwards `arch` to NVRTC as `--gpu-architecture=<arch>` and each
    /// `include_paths` entry as `--include-path=<dir>`. Required for kernels
    /// that pull in CUDA headers (`<mma.h>`, `<cuda_bf16.h>`) or use Ampere+
    /// instructions (bf16 WMMA needs `compute_80+`). [`cuda_include_path`]
    /// discovers the toolkit install dir at runtime.
    ///
    /// When `arch` is `None` and `include_paths` is empty, this is equivalent
    /// to [`Self::compile_cuda`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::compile_cuda`].
    pub fn compile_cuda_with_arch(
        &self,
        source: &str,
        entry_point: &str,
        binding_count: usize,
        workgroup_size: [u32; 3],
        arch: Option<&'static str>,
        include_paths: &[&str],
    ) -> Result<Kernel> {
        match &self.inner {
            DeviceInner::Cuda(b) => {
                let block_dim = (workgroup_size[0], workgroup_size[1], workgroup_size[2]);
                let cuda_kernel =
                    b.compile_cuda_with_arch(source, entry_point, block_dim, arch, include_paths)?;
                Ok(Kernel {
                    inner: BackendKernel::Cuda(cuda_kernel),
                    binding_count,
                    workgroup_size,
                    entry_point: entry_point.to_string(),
                })
            }
            #[cfg(feature = "vulkan")]
            _ => Err(GpuError::BackendUnavailable(
                "compile_cuda requires CUDA backend".into(),
            )),
        }
    }

    /// Discover the CUDA toolkit's `include/` directory at runtime.
    ///
    /// NVRTC has no built-in search list for headers like `<mma.h>` or
    /// `<cuda_bf16.h>`, so kernels that pull them in must pass the toolkit
    /// include dir via [`Self::compile_cuda_with_arch`]. Tries the standard
    /// env vars (`CUDA_PATH`, `CUDA_HOME`) first, then the two well-known
    /// install prefixes — `/opt/cuda` (Arch convention) and
    /// `/usr/local/cuda` (NVIDIA installer convention).
    ///
    /// Returns `None` if no candidate has an `include/` subdirectory; the
    /// caller can fall back to a kernel that doesn't need CUDA headers, or
    /// surface a clearer error than NVRTC's "no directories in search list".
    #[must_use]
    pub fn cuda_include_path() -> Option<String> {
        for var in ["CUDA_PATH", "CUDA_HOME"] {
            if let Ok(p) = std::env::var(var) {
                let inc = std::path::Path::new(&p).join("include");
                if inc.is_dir() {
                    return Some(inc.to_string_lossy().into_owned());
                }
            }
        }
        for prefix in ["/opt/cuda", "/usr/local/cuda"] {
            let inc = std::path::Path::new(prefix).join("include");
            if inc.is_dir() {
                return Some(inc.to_string_lossy().into_owned());
            }
        }
        None
    }

    /// Run cuBLAS SGEMM: `C = A × B` (row-major `f32` matrices).
    ///
    /// Dimensions: A is `m×k`, B is `k×n`, C is `m×n`.
    ///
    /// This is the recommended matmul path on CUDA — it reaches 80%+ peak
    /// throughput without any custom kernels. Blocks until the GPU finishes;
    /// for chained GPU-resident pipelines, prefer
    /// [`Self::cublas_matmul_async`].
    #[allow(clippy::many_single_char_names)]
    pub fn cublas_matmul(
        &self,
        a: &Buffer<f32>,
        b: &Buffer<f32>,
        c: &mut Buffer<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let backend = self.cuda_backend()?;
        let (a_buf, b_buf, c_buf) = unwrap_cuda_matmul_buffers(a, b, c)?;
        backend.cublas_matmul(a_buf, b_buf, c_buf, m, n, k)
    }

    /// Run cuBLAS SGEMM without synchronizing.
    ///
    /// Queues the SGEMM on the CUDA stream and returns once it has been
    /// submitted. Subsequent stream work is stream-ordered against it
    /// (kernels, further matmuls, downloads), so chained operators see
    /// consistent state. Use this in tight pipelines where the per-call
    /// sync of [`Self::cublas_matmul`] would dominate latency.
    ///
    /// Host-visible reads (via [`Buffer::download`](crate::Buffer::download))
    /// implicitly sync, so no explicit fence is needed at storage boundaries.
    #[allow(clippy::many_single_char_names)]
    pub fn cublas_matmul_async(
        &self,
        a: &Buffer<f32>,
        b: &Buffer<f32>,
        c: &mut Buffer<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let backend = self.cuda_backend()?;
        let (a_buf, b_buf, c_buf) = unwrap_cuda_matmul_buffers(a, b, c)?;
        backend.cublas_matmul_async(a_buf, b_buf, c_buf, m, n, k)
    }

    /// Run cuBLAS GemmEx with bf16 inputs/outputs and fp32 accumulate.
    ///
    /// Row-major `C = A × B`, A is `m×k`, B is `k×n`, C is `m×n`. Mirrors
    /// [`Self::cublas_matmul`] but routes through `cublasGemmEx` so the driver
    /// can pick a tensor-core path. Blocks until the GPU finishes; for chained
    /// pipelines prefer [`Self::cublas_matmul_bf16_async`].
    #[cfg(feature = "bf16")]
    #[allow(clippy::many_single_char_names)]
    pub fn cublas_matmul_bf16(
        &self,
        a: &Buffer<half::bf16>,
        b: &Buffer<half::bf16>,
        c: &mut Buffer<half::bf16>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let backend = self.cuda_backend()?;
        let (a_buf, b_buf, c_buf) = unwrap_cuda_matmul_buffers(a, b, c)?;
        backend.cublas_matmul_bf16(a_buf, b_buf, c_buf, m, n, k)
    }

    /// Run cuBLAS GemmEx (bf16 / fp32-accumulate) without synchronizing.
    ///
    /// Stream-ordered against subsequent kernels and matmuls — chained
    /// operators see consistent state without a host fence.
    #[cfg(feature = "bf16")]
    #[allow(clippy::many_single_char_names)]
    pub fn cublas_matmul_bf16_async(
        &self,
        a: &Buffer<half::bf16>,
        b: &Buffer<half::bf16>,
        c: &mut Buffer<half::bf16>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let backend = self.cuda_backend()?;
        let (a_buf, b_buf, c_buf) = unwrap_cuda_matmul_buffers(a, b, c)?;
        backend.cublas_matmul_bf16_async(a_buf, b_buf, c_buf, m, n, k)
    }

    /// Run cuBLAS GemmEx with bf16 inputs and an `f32` output, no sync.
    ///
    /// Same compute path as [`Self::cublas_matmul_bf16_async`] (tensor cores,
    /// fp32 accumulate) but writes the fp32 accumulator straight to `c`,
    /// skipping the cast-back-to-fp32 HBM pass that pure-bf16 callers pay.
    /// Use this when downstream operators want fp32 anyway.
    #[cfg(feature = "bf16")]
    #[allow(clippy::many_single_char_names, clippy::type_complexity)]
    pub fn cublas_matmul_bf16_in_f32_out_async(
        &self,
        a: &Buffer<half::bf16>,
        b: &Buffer<half::bf16>,
        c: &mut Buffer<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let backend = self.cuda_backend()?;
        let BackendBuffer::Cuda(a_buf) = &a.inner else {
            return Err(GpuError::BackendUnavailable(
                "buffer not from CUDA backend".into(),
            ));
        };
        let BackendBuffer::Cuda(b_buf) = &b.inner else {
            return Err(GpuError::BackendUnavailable(
                "buffer not from CUDA backend".into(),
            ));
        };
        let BackendBuffer::Cuda(c_buf) = &mut c.inner else {
            return Err(GpuError::BackendUnavailable(
                "buffer not from CUDA backend".into(),
            ));
        };
        backend.cublas_matmul_bf16_in_f32_out_async(a_buf, b_buf, c_buf, m, n, k)
    }

    /// Run cuBLAS strided batched SGEMM without synchronizing.
    ///
    /// Computes `C[i] = op_a(A[i]) · op_b(B[i])` for `i in 0..batch`. Single
    /// cuBLAS launch covers all batches; row-major buffers with the natural
    /// strides (`m*k`, `k*n`, `m*n`). Storage shapes:
    /// - A: `[batch, m, k]` if `!trans_a` else `[batch, k, m]`
    /// - B: `[batch, k, n]` if `!trans_b` else `[batch, n, k]`
    /// - C: `[batch, m, n]`
    ///
    /// Replaces an N-loop over [`Self::cublas_matmul_async`] for transformer
    /// attention paths where each "batch" is a head and N is small (12–16).
    /// Stream-ordered against subsequent stream work — chained operators
    /// see consistent state without a fence.
    #[allow(clippy::too_many_arguments)]
    pub fn cublas_strided_batched_matmul_async(
        &self,
        a: &Buffer<f32>,
        b: &Buffer<f32>,
        c: &mut Buffer<f32>,
        batch: u32,
        m: u32,
        n: u32,
        k: u32,
        trans_a: bool,
        trans_b: bool,
    ) -> Result<()> {
        let backend = self.cuda_backend()?;
        let (a_buf, b_buf, c_buf) = unwrap_cuda_matmul_buffers(a, b, c)?;
        backend.cublas_strided_batched_matmul_async(
            a_buf, b_buf, c_buf, batch, m, n, k, trans_a, trans_b,
        )
    }

    /// Run cuBLAS strided batched `GemmEx` with bf16 inputs and an `f32`
    /// output, no sync.
    ///
    /// Same shape contract as
    /// [`Self::cublas_strided_batched_matmul_async`] but routes the
    /// per-batch matmul through `cublasGemmStridedBatchedEx` with bf16 data
    /// and an fp32 accumulator written straight to `c`. Mirrors
    /// [`Self::cublas_matmul_bf16_in_f32_out_async`] for transformer
    /// attention's strided batched matmuls (Q@Kᵀ and attn@V) — without this
    /// they stay on `sgemm_strided_batched` even when bf16 matmul is on,
    /// which is the dominant gap on `ViT`-B/16 bf16.
    #[cfg(feature = "bf16")]
    #[allow(
        clippy::too_many_arguments,
        clippy::type_complexity,
        clippy::many_single_char_names
    )]
    pub fn cublas_strided_batched_matmul_bf16_in_f32_out_async(
        &self,
        a: &Buffer<half::bf16>,
        b: &Buffer<half::bf16>,
        c: &mut Buffer<f32>,
        batch: u32,
        m: u32,
        n: u32,
        k: u32,
        trans_a: bool,
        trans_b: bool,
    ) -> Result<()> {
        let backend = self.cuda_backend()?;
        let BackendBuffer::Cuda(a_buf) = &a.inner else {
            return Err(GpuError::BackendUnavailable(
                "buffer not from CUDA backend".into(),
            ));
        };
        let BackendBuffer::Cuda(b_buf) = &b.inner else {
            return Err(GpuError::BackendUnavailable(
                "buffer not from CUDA backend".into(),
            ));
        };
        let BackendBuffer::Cuda(c_buf) = &mut c.inner else {
            return Err(GpuError::BackendUnavailable(
                "buffer not from CUDA backend".into(),
            ));
        };
        backend.cublas_strided_batched_matmul_bf16_in_f32_out_async(
            a_buf, b_buf, c_buf, batch, m, n, k, trans_a, trans_b,
        )
    }

    /// Run a cuDNN 2D convolution forward pass without synchronizing.
    ///
    /// Implicit-GEMM (or Winograd/FFT — cuDNN heuristic picks per shape) fused
    /// conv. Skips the im2col→cuBLAS round-trip that the default path uses.
    /// Layout is `NCHW`; filters are `[c_out, c_in, k_h, k_w]`. The output
    /// spatial dims are computed from the standard floor formula and returned.
    ///
    /// Stream-ordered against subsequent kernels and matmuls — chained
    /// operators see consistent state without a host fence.
    #[cfg(feature = "cudnn")]
    #[allow(clippy::too_many_arguments)]
    pub fn cudnn_conv2d_forward_async(
        &self,
        input: &Buffer<f32>,
        filter: &Buffer<f32>,
        output: &mut Buffer<f32>,
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
        let backend = self.cuda_backend()?;
        let BackendBuffer::Cuda(in_buf) = &input.inner else {
            return Err(GpuError::BackendUnavailable(
                "buffer not from CUDA backend".into(),
            ));
        };
        let BackendBuffer::Cuda(filt_buf) = &filter.inner else {
            return Err(GpuError::BackendUnavailable(
                "buffer not from CUDA backend".into(),
            ));
        };
        let BackendBuffer::Cuda(out_buf) = &mut output.inner else {
            return Err(GpuError::BackendUnavailable(
                "buffer not from CUDA backend".into(),
            ));
        };
        backend.cudnn_conv2d_forward_async(
            in_buf, filt_buf, out_buf, n, c_in, h_in, w_in, c_out, k_h, k_w, pad_h, pad_w,
            stride_h, stride_w,
        )
    }

    /// Dispatch a precompiled CUDA kernel without synchronizing.
    ///
    /// CUDA-only counterpart to [`Self::run_configured`] that skips the
    /// per-call `stream.synchronize()` so a chain of dispatches incurs only
    /// one host-side sync (at the next [`Buffer::download`](crate::Buffer::download)
    /// or explicit boundary).
    ///
    /// Returns [`GpuError::BackendUnavailable`] on non-CUDA backends or if
    /// the kernel was not compiled for CUDA.
    pub fn run_configured_async(
        &self,
        kernel: &Kernel,
        buffers: &[&dyn GpuBuf],
        workgroups: [u32; 3],
        push_constants: Option<&[u8]>,
    ) -> Result<()> {
        let backend = self.cuda_backend()?;
        #[allow(irrefutable_let_patterns)]
        let BackendKernel::Cuda(cuda_kernel) = &kernel.inner
        else {
            return Err(GpuError::BackendUnavailable(
                "kernel was not compiled for CUDA".into(),
            ));
        };

        let backend_bufs: Vec<&BackendBuffer> = buffers.iter().map(|b| b.raw()).collect();
        if kernel.binding_count != backend_bufs.len() {
            return Err(GpuError::BindingMismatch {
                expected: kernel.binding_count,
                got: backend_bufs.len(),
            });
        }

        let cuda_bufs: Vec<&crate::backend::cuda::CudaBuffer> = backend_bufs
            .iter()
            .map(|buf| match buf {
                BackendBuffer::Cuda(cb) => Ok(cb),
                #[cfg(feature = "vulkan")]
                _ => Err(GpuError::BackendUnavailable(
                    "buffer/backend mismatch: expected CUDA buffer".into(),
                )),
            })
            .collect::<Result<Vec<_>>>()?;

        backend.dispatch_cuda_async(cuda_kernel, &cuda_bufs, workgroups, push_constants)
    }

    fn cuda_backend(&self) -> Result<&crate::backend::cuda::CudaBackend> {
        match &self.inner {
            DeviceInner::Cuda(b) => Ok(b),
            #[cfg(feature = "vulkan")]
            _ => Err(GpuError::BackendUnavailable(
                "operation requires CUDA backend".into(),
            )),
        }
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity, irrefutable_let_patterns)]
fn unwrap_cuda_matmul_buffers<'a, T: bytemuck::Pod>(
    a: &'a Buffer<T>,
    b: &'a Buffer<T>,
    c: &'a mut Buffer<T>,
) -> Result<(
    &'a crate::backend::cuda::CudaBuffer,
    &'a crate::backend::cuda::CudaBuffer,
    &'a mut crate::backend::cuda::CudaBuffer,
)> {
    let BackendBuffer::Cuda(a_buf) = &a.inner else {
        return Err(GpuError::BackendUnavailable(
            "buffer not from CUDA backend".into(),
        ));
    };
    let BackendBuffer::Cuda(b_buf) = &b.inner else {
        return Err(GpuError::BackendUnavailable(
            "buffer not from CUDA backend".into(),
        ));
    };
    let BackendBuffer::Cuda(c_buf) = &mut c.inner else {
        return Err(GpuError::BackendUnavailable(
            "buffer not from CUDA backend".into(),
        ));
    };
    Ok((a_buf, b_buf, c_buf))
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("name", &self.name())
            .field("memory_mb", &(self.memory() / (1024 * 1024)))
            .finish()
    }
}
