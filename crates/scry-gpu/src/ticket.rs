//! Non-blocking GPU submission handles.
//!
//! A [`Ticket`] represents an in-flight GPU submission. It is created by
//! [`Batch::submit_async`](crate::Batch::submit_async) and allows the caller
//! to overlap CPU work with GPU execution.
//!
//! # Example
//!
//! ```ignore
//! let mut batch = gpu.batch()?;
//! batch.run(&kernel, &[&input, &output], n)?;
//! let ticket = batch.submit_async()?;
//!
//! // CPU work while GPU runs...
//!
//! ticket.wait()?;
//! let result: Vec<f32> = output.download()?;
//! ```
//!
//! # Drop guarantee
//!
//! If a `Ticket` is dropped without calling [`wait`](Ticket::wait), the
//! destructor blocks until the GPU work finishes. This prevents leaking
//! in-flight resources but may introduce unexpected stalls. Prefer calling
//! `wait()` explicitly.

use crate::error::Result;

/// A handle to an in-flight GPU submission.
///
/// Created by [`Batch::submit_async`](crate::Batch::submit_async). The GPU
/// work is already queued; this handle lets you poll for completion or block
/// until done.
pub struct Ticket {
    inner: Option<TicketInner>,
}

enum TicketInner {
    #[cfg(feature = "vulkan")]
    Vulkan(crate::backend::vulkan::VulkanTicket),
    #[cfg(feature = "cuda")]
    Cuda(crate::backend::cuda::CudaTicket),
}

impl Ticket {
    #[cfg(feature = "vulkan")]
    pub(crate) const fn new_vulkan(ticket: crate::backend::vulkan::VulkanTicket) -> Self {
        Self {
            inner: Some(TicketInner::Vulkan(ticket)),
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) const fn new_cuda(ticket: crate::backend::cuda::CudaTicket) -> Self {
        Self {
            inner: Some(TicketInner::Cuda(ticket)),
        }
    }

    /// Block until the GPU work completes.
    ///
    /// If the GPU work has already finished (e.g. [`is_ready`](Ticket::is_ready)
    /// returned `true`), this returns immediately. Consumes the ticket and
    /// recycles backend resources.
    pub fn wait(mut self) -> Result<()> {
        self.wait_inner()
    }

    /// Poll whether the GPU work has completed without blocking.
    ///
    /// Returns `Ok(true)` if all dispatches have finished, `Ok(false)` if
    /// still in progress.
    ///
    /// # Panics
    ///
    /// Panics if called after [`wait`](Ticket::wait) (which consumes `self`,
    /// so this can only happen via internal misuse).
    pub fn is_ready(&self) -> Result<bool> {
        match self.inner.as_ref().expect("ticket already consumed") {
            #[cfg(feature = "vulkan")]
            TicketInner::Vulkan(t) => t.is_ready(),
            #[cfg(feature = "cuda")]
            TicketInner::Cuda(t) => Ok(t.is_ready()),
        }
    }

    /// Internal wait callable from both `wait()` and `Drop`.
    fn wait_inner(&mut self) -> Result<()> {
        if let Some(inner) = self.inner.take() {
            match inner {
                #[cfg(feature = "vulkan")]
                TicketInner::Vulkan(t) => t.wait()?,
                #[cfg(feature = "cuda")]
                TicketInner::Cuda(t) => t.wait()?,
            }
        }
        Ok(())
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        // Best-effort: block until GPU finishes so resources can be
        // safely recycled. Errors are discarded because Drop cannot
        // propagate them.
        let _ = self.wait_inner();
    }
}

impl std::fmt::Debug for Ticket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ticket")
            .field("pending", &self.inner.is_some())
            .finish_non_exhaustive()
    }
}
