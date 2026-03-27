//! Precompiled compute kernels for reuse across dispatches.

use crate::backend::BackendKernel;

/// A compiled GPU compute kernel, ready for repeated dispatch.
///
/// Created via [`Device::compile`]. Holds all GPU objects needed to
/// dispatch a shader — SPIR-V, Vulkan pipeline, layouts — so that
/// repeated dispatches skip compilation entirely.
///
/// # Example
///
/// ```ignore
/// let kernel = gpu.compile(SHADER_SRC)?;
/// for batch in &batches {
///     gpu.run(&kernel, &[&batch.input, &batch.output], batch.len)?;
/// }
/// ```
pub struct Kernel {
    pub(crate) inner: BackendKernel,
    /// Number of storage buffer bindings the shader expects.
    pub(crate) binding_count: usize,
    /// Workgroup size extracted from the shader's `@workgroup_size`.
    pub(crate) workgroup_size: [u32; 3],
    /// Entry point name.
    pub(crate) entry_point: String,
}

impl Kernel {
    /// Number of buffer bindings the shader expects.
    pub const fn binding_count(&self) -> usize {
        self.binding_count
    }

    /// The entry point name compiled into this kernel.
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    /// Workgroup size `[x, y, z]` declared in the shader.
    pub const fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup_size
    }
}

impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel")
            .field("entry_point", &self.entry_point)
            .field("binding_count", &self.binding_count)
            .field("workgroup_size", &self.workgroup_size)
            .finish_non_exhaustive()
    }
}
