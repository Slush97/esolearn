//! # scry-gpu
//!
//! Lightweight GPU compute for Rust — dispatch shaders without the graphics baggage.
//!
//! scry-gpu is a compute-only GPU abstraction. No render passes, no swapchains,
//! no framebuffers. Upload data, dispatch a WGSL shader, read results back.
//!
//! ## Quick start
//!
//! ```ignore
//! use scry_gpu::Device;
//!
//! let gpu = Device::auto()?;
//!
//! let input = gpu.upload(&[1.0f32, 2.0, 3.0, 4.0])?;
//! let output = gpu.alloc::<f32>(4)?;
//!
//! gpu.dispatch("@group(0) @binding(0) var<storage, read> input: array<f32>;
//!               @group(0) @binding(1) var<storage, read_write> output: array<f32>;
//!               @compute @workgroup_size(64)
//!               fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
//!                   let i = gid.x;
//!                   if i < arrayLength(&input) {
//!                       output[i] = input[i] * 2.0;
//!                   }
//!               }", &[&input, &output], 4)?;
//!
//! let result: Vec<f32> = output.download()?;
//! assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
//! ```
//!
//! ## Design principles
//!
//! - **Compute only.** No graphics API surface. This keeps the dependency tree
//!   small and the API surface minimal.
//! - **Auto-dispatch.** Workgroup dimensions are calculated from your invocation
//!   count and the shader's `@workgroup_size`. No manual `ceil(n / 64)`.
//! - **Typed buffers.** `Buffer<f32>` uploads `&[f32]` and downloads `Vec<f32>`.
//!   Staging, alignment, and synchronization are handled internally.
//! - **Backend abstraction.** Vulkan today, Metal tomorrow. The public API
//!   doesn't change.

mod backend;
mod batch;
mod buffer;
mod device;
mod dispatch;
mod error;
mod kernel;
mod shader;
pub mod shaders;

pub use batch::Batch;
pub use buffer::{Buffer, GpuBuf};
pub use device::{BackendKind, Device};
pub use dispatch::DispatchConfig;
pub use error::{BackendOp, GpuError, Result};
pub use kernel::Kernel;
