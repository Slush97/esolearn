//! Shader dispatch configuration and execution.

/// Configuration for a compute dispatch.
///
/// The simple path ([`Device::dispatch`]) covers most cases.
/// Use `DispatchConfig` when you need control over workgroup sizes
/// or push constants.
///
/// [`Device::dispatch`]: crate::Device::dispatch
pub struct DispatchConfig<'a> {
    /// Shader source (WGSL).
    pub shader: &'a str,

    /// Entry point name. Defaults to `"main"` if `None`.
    pub entry_point: Option<&'a str>,

    /// Workgroup dimensions `[x, y, z]`.
    ///
    /// If `None`, the crate auto-calculates from `invocations` and the
    /// shader's declared `@workgroup_size`.
    pub workgroups: Option<[u32; 3]>,

    /// Total invocations requested. Used to auto-calculate workgroup
    /// dispatch count when `workgroups` is `None`.
    pub invocations: u32,

    /// Optional push constant data (raw bytes, must match shader layout).
    pub push_constants: Option<&'a [u8]>,
}

impl<'a> DispatchConfig<'a> {
    /// Create a minimal dispatch config.
    pub const fn new(shader: &'a str, invocations: u32) -> Self {
        Self {
            shader,
            entry_point: None,
            workgroups: None,
            invocations,
            push_constants: None,
        }
    }

    /// Override the entry point name (default: `"main"`).
    pub const fn entry_point(mut self, name: &'a str) -> Self {
        self.entry_point = Some(name);
        self
    }

    /// Set explicit workgroup dispatch dimensions.
    pub const fn workgroups(mut self, dims: [u32; 3]) -> Self {
        self.workgroups = Some(dims);
        self
    }

    /// Attach push constant data.
    pub const fn push_constants(mut self, data: &'a [u8]) -> Self {
        self.push_constants = Some(data);
        self
    }
}

/// Extract `@workgroup_size` from a parsed naga module's entry point.
///
/// Returns `[x, y, z]` or a default of `[64, 1, 1]` if the shader
/// doesn't declare one.
pub fn extract_workgroup_size(module: &naga::Module, entry: &str) -> [u32; 3] {
    for ep in &module.entry_points {
        if ep.name == entry {
            let s = ep.workgroup_size;
            return [s[0], s[1], s[2]];
        }
    }
    [64, 1, 1]
}

/// Calculate dispatch dimensions given total invocations and per-workgroup size.
///
/// Applies `ceil(invocations / workgroup_size)` and clamps to the Vulkan
/// `maxComputeWorkGroupCount` limit (65535 per axis).
pub fn calc_dispatch(invocations: u32, workgroup_size: [u32; 3]) -> [u32; 3] {
    let ceil_div = |a: u32, b: u32| a.div_ceil(b);

    [
        ceil_div(invocations, workgroup_size[0]).min(65535),
        1,
        1,
    ]
}
