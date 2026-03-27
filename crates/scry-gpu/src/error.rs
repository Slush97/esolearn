//! Error types for scry-gpu.

/// Errors that can occur during GPU operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GpuError {
    /// No suitable GPU device was found on this system.
    #[error("no suitable GPU device found")]
    NoDevice,

    /// The requested backend is not available (e.g. Vulkan not installed).
    #[error("backend not available: {0}")]
    BackendUnavailable(String),

    /// Buffer allocation failed — typically the device ran out of memory.
    #[error("buffer allocation failed: requested {requested} bytes (device max: {device_max})")]
    AllocationFailed {
        /// Bytes requested.
        requested: u64,
        /// Device maximum allocation size.
        device_max: u64,
    },

    /// Shader compilation failed.
    #[error("shader compilation error: {0}")]
    ShaderCompilation(String),

    /// Shader is missing an entry point with the expected name.
    #[error("entry point \"{name}\" not found in shader")]
    MissingEntryPoint {
        /// The entry point name that was expected.
        name: String,
    },

    /// A dispatch or readback operation failed.
    #[error("dispatch failed: {0}")]
    Dispatch(String),

    /// Readback from GPU buffer to CPU timed out.
    #[error("readback timed out after {ms}ms")]
    ReadbackTimeout {
        /// Timeout duration in milliseconds.
        ms: u64,
    },

    /// The provided buffer count does not match the shader's binding count.
    #[error("binding mismatch: shader expects {expected} buffer(s), got {got}")]
    BindingMismatch {
        /// Bindings declared in the shader.
        expected: usize,
        /// Buffers provided by the caller.
        got: usize,
    },

    /// Internal backend error (Vulkan, Metal, etc.).
    #[error("backend error: {0}")]
    Backend(String),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, GpuError>;
