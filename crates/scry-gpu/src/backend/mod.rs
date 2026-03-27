//! Backend abstraction layer.
//!
//! Each backend (Vulkan, Metal, ...) implements the [`Backend`] trait,
//! providing device creation, buffer management, and compute dispatch.

#[cfg(feature = "vulkan")]
pub mod vulkan;

use crate::error::Result;

/// Trait implemented by each GPU backend.
pub trait Backend: Sized {
    /// Backend-specific buffer handle.
    type Buffer: BackendBufferOps;

    /// Backend-specific compiled pipeline handle.
    type Pipeline;

    /// Create a backend, selecting the best available device.
    fn create() -> Result<Self>;

    /// Allocate a GPU buffer and upload `data` into it.
    fn upload(&self, data: &[u8]) -> Result<Self::Buffer>;

    /// Allocate an uninitialized GPU buffer of `size` bytes.
    fn alloc(&self, size: u64) -> Result<Self::Buffer>;

    /// Compile a SPIR-V shader module and dispatch it.
    fn dispatch(
        &self,
        spirv: &[u32],
        entry_point: &str,
        buffers: &[&Self::Buffer],
        workgroups: [u32; 3],
        push_constants: Option<&[u8]>,
    ) -> Result<()>;

    /// Compile a SPIR-V shader into a reusable pipeline.
    fn create_pipeline(
        &self,
        spirv: &[u32],
        entry_point: &str,
        binding_count: usize,
        push_constant_size: u32,
    ) -> Result<Self::Pipeline>;

    /// Dispatch a precompiled pipeline.
    fn dispatch_pipeline(
        &self,
        pipeline: &Self::Pipeline,
        buffers: &[&Self::Buffer],
        workgroups: [u32; 3],
        push_constants: Option<&[u8]>,
    ) -> Result<()>;

    /// Device name for diagnostics.
    fn device_name(&self) -> &str;

    /// Total device memory in bytes (best estimate).
    fn device_memory(&self) -> u64;

    /// Subgroup (warp/wavefront) size.
    ///
    /// Typically 32 on NVIDIA, 64 on AMD, 32 on Intel.
    fn subgroup_size(&self) -> u32;
}

/// Operations available on a backend buffer.
pub trait BackendBufferOps {
    /// Read buffer contents back to CPU.
    fn read_back(&self) -> Result<Vec<u8>>;

    /// Size in bytes.
    #[allow(dead_code)]
    fn byte_size(&self) -> u64;
}

// ── Opaque handle exposed to the public API ──

/// Type-erased buffer handle used by [`Buffer<T>`](crate::Buffer).
pub enum BackendBuffer {
    #[cfg(feature = "vulkan")]
    Vulkan(vulkan::VulkanBuffer),
}

/// Type-erased pipeline handle used by [`Kernel`](crate::Kernel).
pub enum BackendKernel {
    #[cfg(feature = "vulkan")]
    Vulkan(vulkan::VulkanKernel),
}

impl BackendBufferOps for BackendBuffer {
    fn read_back(&self) -> Result<Vec<u8>> {
        match self {
            #[cfg(feature = "vulkan")]
            Self::Vulkan(b) => b.read_back(),
        }
    }

    fn byte_size(&self) -> u64 {
        match self {
            #[cfg(feature = "vulkan")]
            Self::Vulkan(b) => b.byte_size(),
        }
    }
}
