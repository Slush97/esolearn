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

    /// Device name for diagnostics.
    fn device_name(&self) -> &str;

    /// Total device memory in bytes (best estimate).
    fn device_memory(&self) -> u64;
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
