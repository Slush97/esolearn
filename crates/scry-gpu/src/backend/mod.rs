// SPDX-License-Identifier: MIT OR Apache-2.0
//! Backend abstraction layer.
//!
//! Each backend (Vulkan, Metal, ...) implements the [`Backend`] trait,
//! providing device creation, buffer management, and compute dispatch.

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "cudnn")]
pub(crate) mod cuda_cudnn;

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

    /// Allocate a GPU buffer of `size` bytes.
    ///
    /// On backends that zero-initialize by default (CUDA), this method does
    /// the zero-fill. Use [`Self::alloc_uninit`] when the caller will
    /// overwrite the buffer before any read.
    fn alloc(&self, size: u64) -> Result<Self::Buffer>;

    /// Allocate a GPU buffer with **undefined** contents.
    ///
    /// Faster than [`Self::alloc`] on backends that would otherwise zero-fill
    /// (CUDA dispatches a `cuMemsetD8Async` kernel under the default path).
    /// The caller must overwrite every byte that will subsequently be read;
    /// otherwise downstream computation observes garbage.
    ///
    /// Default implementation falls back to [`Self::alloc`].
    fn alloc_uninit(&self, size: u64) -> Result<Self::Buffer> {
        self.alloc(size)
    }

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

    /// GPU-to-GPU buffer copy.
    ///
    /// Allocates a new buffer and copies `size` bytes from `src` into it.
    /// The copy is synchronous (blocks until complete).
    fn copy_buffer(&self, src: &Self::Buffer, size: u64) -> Result<Self::Buffer>;

    /// Block until all previously-issued work on the backend's stream/queue
    /// has completed.
    ///
    /// Needed by benchmarks (the host needs to wait for async dispatches to
    /// finish before stopping the timer) and any caller that observes GPU
    /// state through a non-`Buffer::download` path. The default impl is a
    /// no-op for backends whose dispatch path is already synchronous.
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
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
    #[cfg(feature = "cuda")]
    Cuda(cuda::CudaBuffer),
}

/// Type-erased pipeline handle used by [`Kernel`](crate::Kernel).
pub enum BackendKernel {
    #[cfg(feature = "vulkan")]
    Vulkan(vulkan::VulkanKernel),
    #[cfg(feature = "cuda")]
    Cuda(cuda::CudaKernel),
}

impl BackendBufferOps for BackendBuffer {
    fn read_back(&self) -> Result<Vec<u8>> {
        match self {
            #[cfg(feature = "vulkan")]
            Self::Vulkan(b) => b.read_back(),
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b.read_back(),
        }
    }

    fn byte_size(&self) -> u64 {
        match self {
            #[cfg(feature = "vulkan")]
            Self::Vulkan(b) => b.byte_size(),
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b.byte_size(),
        }
    }
}
