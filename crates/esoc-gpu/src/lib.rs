// SPDX-License-Identifier: MIT OR Apache-2.0
//! wgpu GPU rendering backend for esoc-scene.
//!
//! Takes a [`SceneGraph`](esoc_scene::SceneGraph) and renders it via instanced
//! draw calls with SDF-based anti-aliasing.

pub mod buffer;
pub mod context;
pub mod error;
pub mod pass;
pub mod renderer;
pub mod tessellate;

pub use context::GpuContext;
pub use error::GpuError;
pub use renderer::Renderer;
