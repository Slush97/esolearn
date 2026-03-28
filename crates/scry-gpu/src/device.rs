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
}

/// Available backend types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Vulkan (Linux, Windows, Android).
    Vulkan,
    // Metal, // future
}

impl Device {
    /// Auto-select the best available GPU.
    ///
    /// Tries backends in order of preference: Vulkan → (Metal in future).
    pub fn auto() -> Result<Self> {
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
        let size = count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| GpuError::AllocationFailed {
                requested: u64::MAX,
                device_max: self.memory(),
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
    pub fn run(
        &self,
        kernel: &Kernel,
        buffers: &[&dyn GpuBuf],
        invocations: u32,
    ) -> Result<()> {
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
        }
    }

    /// Device name (for diagnostics / logging).
    pub fn name(&self) -> &str {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            DeviceInner::Vulkan(b) => b.device_name(),
        }
    }

    /// Total device memory in bytes.
    pub fn memory(&self) -> u64 {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            DeviceInner::Vulkan(b) => b.device_memory(),
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
        }
    }

    fn alloc_raw(&self, size: u64) -> Result<BackendBuffer> {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            DeviceInner::Vulkan(b) => {
                let buf = b.alloc(size)?;
                Ok(BackendBuffer::Vulkan(buf))
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
                    })
                    .collect();
                b.dispatch(spirv, entry_point, &vk_bufs, workgroups, push_constants)
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
                let kernel = b.create_pipeline(
                    spirv,
                    entry_point,
                    binding_count,
                    push_constant_size,
                )?;
                Ok(BackendKernel::Vulkan(kernel))
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
                let BackendKernel::Vulkan(vk_kernel) = &kernel.inner;
                let vk_bufs: Vec<&crate::backend::vulkan::VulkanBuffer> = buffers
                    .iter()
                    .map(|buf| match buf {
                        BackendBuffer::Vulkan(vb) => vb,
                    })
                    .collect();
                b.dispatch_pipeline(vk_kernel, &vk_bufs, workgroups, push_constants)
            }
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
