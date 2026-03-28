//! GPU buffer abstraction.
//!
//! Buffers are the primary way to move data between CPU and GPU.
//! They wrap backend-specific storage and expose a typed, safe interface.

use crate::backend::{BackendBuffer, BackendBufferOps};
use crate::error::Result;

/// Trait for types that can be bound as a GPU buffer in a dispatch call.
///
/// This is object-safe so that `&[&dyn GpuBuf]` can hold mixed buffer types.
#[allow(private_interfaces)]
pub trait GpuBuf {
    #[doc(hidden)]
    fn raw(&self) -> &BackendBuffer;
}

/// A typed GPU buffer.
///
/// `Buffer<T>` owns a region of GPU memory containing `len` elements of type `T`.
/// Data is moved to the GPU on creation ([`Device::upload`]) and read back on demand
/// ([`Buffer::download`]).
///
/// [`Device::upload`]: crate::Device::upload
pub struct Buffer<T: bytemuck::Pod> {
    pub(crate) inner: BackendBuffer,
    pub(crate) len: usize,
    pub(crate) _marker: std::marker::PhantomData<T>,
}

impl<T: bytemuck::Pod> Buffer<T> {
    /// Number of `T` elements in this buffer.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Size in bytes.
    pub const fn byte_size(&self) -> u64 {
        match self.len.checked_mul(std::mem::size_of::<T>()) {
            Some(s) => s as u64,
            None => u64::MAX,
        }
    }

    /// Copy buffer contents back to the CPU.
    ///
    /// This blocks until the transfer is complete.
    pub fn download(&self) -> Result<Vec<T>> {
        let bytes = self.inner.read_back()?;
        let elements = bytemuck::cast_slice(&bytes).to_vec();
        Ok(elements)
    }
}

#[allow(private_interfaces)]
impl<T: bytemuck::Pod> GpuBuf for Buffer<T> {
    fn raw(&self) -> &BackendBuffer {
        &self.inner
    }
}

impl<T: bytemuck::Pod> std::fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("len", &self.len)
            .field("byte_size", &self.byte_size())
            .field("type", &std::any::type_name::<T>())
            .finish_non_exhaustive()
    }
}
