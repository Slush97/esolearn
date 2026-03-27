//! Device acquisition and the primary user-facing API.

use crate::backend::{Backend, BackendBuffer};
use crate::buffer::{Buffer, GpuBuf};
use crate::dispatch::{self, DispatchConfig};
use crate::error::{GpuError, Result};
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
        let size = (count * std::mem::size_of::<T>()) as u64;
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
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("name", &self.name())
            .field("memory_mb", &(self.memory() / (1024 * 1024)))
            .finish()
    }
}
