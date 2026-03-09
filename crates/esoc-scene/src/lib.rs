// SPDX-License-Identifier: MIT OR Apache-2.0
//! Arena scene graph with typed visual marks.
//!
//! This crate defines the shared visual vocabulary consumed by rendering
//! backends (esoc-gpu, esoc-gfx) and produced by chart logic (esoc-chart).

pub mod arena;
pub mod bounds;
pub mod mark;
pub mod node;
pub mod scale;
pub mod style;
pub mod transform;
pub mod transition;

pub use arena::SceneGraph;
pub use bounds::{BoundingBox, DataBounds};
pub use mark::{Mark, MarkBatch, MarkKey};
pub use node::{Node, NodeContent, NodeId};
pub use scale::Scale;
pub use style::{FillStyle, FontStyle, MarkerShape, StrokeStyle};
pub use transform::Affine2D;
