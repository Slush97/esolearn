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

    /// Allocate an uninitialized GPU buffer for `count` elements of type `T`.
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
                        BackendBuffer::Vulkan(vb) => vb,
                        #[cfg(feature = "cuda")]
                        _ => unreachable!("Vulkan backend received non-Vulkan buffer"),
                    })
                    .collect();
                b.dispatch(spirv, entry_point, &vk_bufs, workgroups, push_constants)
            }
            #[cfg(feature = "cuda")]
            DeviceInner::Cuda(b) => {
                let cuda_bufs: Vec<&crate::backend::cuda::CudaBuffer> = buffers
                    .iter()
                    .map(|buf| match buf {
                        BackendBuffer::Cuda(cb) => cb,
                        #[cfg(feature = "vulkan")]
                        _ => unreachable!("CUDA backend received non-CUDA buffer"),
                    })
                    .collect();
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
                let BackendKernel::Vulkan(vk_kernel) = &kernel.inner else {
                    return Err(GpuError::BackendUnavailable(
                        "kernel was not compiled for Vulkan".into(),
                    ));
                };
                let vk_bufs: Vec<&crate::backend::vulkan::VulkanBuffer> = buffers
                    .iter()
                    .map(|buf| match buf {
                        BackendBuffer::Vulkan(vb) => vb,
                        #[cfg(feature = "cuda")]
                        _ => unreachable!("Vulkan backend received non-Vulkan buffer"),
                    })
                    .collect();
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
                        BackendBuffer::Cuda(cb) => cb,
                        #[cfg(feature = "vulkan")]
                        _ => unreachable!("CUDA backend received non-CUDA buffer"),
                    })
                    .collect();
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
        match &self.inner {
            DeviceInner::Cuda(b) => {
                let block_dim = (workgroup_size[0], workgroup_size[1], workgroup_size[2]);
                let cuda_kernel = b.compile_cuda(source, entry_point, block_dim)?;
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

    /// Run cuBLAS SGEMM: `C = A × B` (row-major `f32` matrices).
    ///
    /// Dimensions: A is `m×k`, B is `k×n`, C is `m×n`.
    ///
    /// This is the recommended matmul path on CUDA — it reaches 80%+ peak
    /// throughput without any custom kernels.
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
        match &self.inner {
            DeviceInner::Cuda(backend) => {
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
                backend.cublas_matmul(a_buf, b_buf, c_buf, m, n, k)
            }
            #[cfg(feature = "vulkan")]
            _ => Err(GpuError::BackendUnavailable(
                "cublas_matmul requires CUDA backend".into(),
            )),
        }
    }
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("name", &self.name())
            .field("memory_mb", &(self.memory() / (1024 * 1024)))
            .finish()
    }
}
