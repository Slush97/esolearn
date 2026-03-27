//! Batched dispatch — multiple dispatches in a single GPU submission.
//!
//! A [`Batch`] records multiple kernel dispatches into one command buffer,
//! then submits them all with a single fence wait. This eliminates the
//! per-dispatch synchronization overhead that dominates bandwidth-bound
//! workloads.
//!
//! # Example
//!
//! ```ignore
//! let mut batch = gpu.batch()?;
//! batch.run(&kernel, &[&input, &pass1], n)?;
//! batch.barrier();  // ensure pass1 finishes before pass2 reads it
//! batch.run(&kernel, &[&pass1, &pass2], pass1_n)?;
//! batch.submit()?;
//! ```

use crate::backend::BackendBuffer;
use crate::buffer::GpuBuf;
use crate::dispatch;
use crate::error::{GpuError, Result};
use crate::kernel::Kernel;

/// A batch of dispatches recorded into a single command buffer.
///
/// Created via [`Device::batch`](crate::Device::batch).
/// Use [`barrier`](Batch::barrier) between dispatches that have data
/// dependencies (where one dispatch reads from another's output).
pub struct Batch {
    inner: BatchInner,
}

enum BatchInner {
    #[cfg(feature = "vulkan")]
    Vulkan(crate::backend::vulkan::VulkanBatch),
}

impl Batch {
    #[cfg(feature = "vulkan")]
    pub(crate) fn new_vulkan(vk_batch: crate::backend::vulkan::VulkanBatch) -> Self {
        Self {
            inner: BatchInner::Vulkan(vk_batch),
        }
    }

    /// Record a kernel dispatch with auto-calculated workgroups.
    pub fn run(
        &mut self,
        kernel: &Kernel,
        buffers: &[&dyn GpuBuf],
        invocations: u32,
    ) -> Result<&mut Self> {
        let workgroups = dispatch::calc_dispatch(invocations, kernel.workgroup_size);
        self.run_configured(kernel, buffers, workgroups, None)
    }

    /// Record a kernel dispatch with push constants.
    pub fn run_with_push_constants(
        &mut self,
        kernel: &Kernel,
        buffers: &[&dyn GpuBuf],
        invocations: u32,
        push_constants: &[u8],
    ) -> Result<&mut Self> {
        let workgroups = dispatch::calc_dispatch(invocations, kernel.workgroup_size);
        self.run_configured(kernel, buffers, workgroups, Some(push_constants))
    }

    /// Record a kernel dispatch with explicit workgroups and optional push constants.
    pub fn run_configured(
        &mut self,
        kernel: &Kernel,
        buffers: &[&dyn GpuBuf],
        workgroups: [u32; 3],
        push_constants: Option<&[u8]>,
    ) -> Result<&mut Self> {
        let backend_bufs: Vec<&BackendBuffer> = buffers.iter().map(|b| b.raw()).collect();
        if kernel.binding_count != backend_bufs.len() {
            return Err(GpuError::BindingMismatch {
                expected: kernel.binding_count,
                got: backend_bufs.len(),
            });
        }

        match &mut self.inner {
            #[cfg(feature = "vulkan")]
            BatchInner::Vulkan(vk_batch) => {
                let crate::backend::BackendKernel::Vulkan(vk_kernel) = &kernel.inner;
                let vk_bufs: Vec<&crate::backend::vulkan::VulkanBuffer> = backend_bufs
                    .iter()
                    .map(|buf| match buf {
                        BackendBuffer::Vulkan(vb) => vb,
                    })
                    .collect();
                vk_batch.record_dispatch(vk_kernel, &vk_bufs, workgroups, push_constants)?;
            }
        }

        Ok(self)
    }

    /// Insert a compute-to-compute barrier.
    ///
    /// Use this between dispatches where a later dispatch reads from an
    /// earlier dispatch's output buffer. Without a barrier, the GPU may
    /// execute dispatches out of order or overlap writes with reads.
    pub fn barrier(&mut self) -> &mut Self {
        match &mut self.inner {
            #[cfg(feature = "vulkan")]
            BatchInner::Vulkan(vk_batch) => vk_batch.record_barrier(),
        }
        self
    }

    /// Submit all recorded dispatches and wait for completion.
    ///
    /// All dispatches execute in a single command buffer with one fence wait,
    /// eliminating per-dispatch synchronization overhead.
    pub fn submit(self) -> Result<()> {
        match self.inner {
            #[cfg(feature = "vulkan")]
            BatchInner::Vulkan(vk_batch) => vk_batch.submit(),
        }
    }
}
