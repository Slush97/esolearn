// SPDX-License-Identifier: MIT OR Apache-2.0
//! High-level pipeline traits for common vision tasks.
//!
//! These traits define the user-facing API for vision inference. Each trait
//! takes raw image bytes and returns structured outputs. Concrete
//! implementations compose transforms, model inference, and postprocessing.
//!
//! Output types:
//! - [`Detection`] / [`BBox`] — from [`crate::postprocess::nms`]
//! - [`Classification`] — from [`crate::postprocess::classify`]
//! - [`Mask`] — segmentation mask (defined here)

use crate::error::Result;

// Re-export core postprocess types as pipeline outputs
pub use crate::postprocess::classify::Classification;
pub use crate::postprocess::nms::{BBox, Detection};

/// Segmentation prompt — how to specify what to segment.
#[derive(Clone, Debug)]
pub enum SegmentPrompt {
    /// Segment at a specific point `(x, y)`.
    Point { x: f32, y: f32 },
    /// Segment within a bounding box.
    Box(BBox),
}

/// A binary segmentation mask.
#[derive(Clone, Debug)]
pub struct Mask {
    /// Binary mask data (0 = background, 255 = foreground).
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Object detection pipeline.
///
/// Takes raw RGB image bytes and returns bounding boxes with class IDs and
/// confidence scores.
pub trait Detect {
    fn detect(&self, image: &[u8], width: u32, height: u32, conf_threshold: f32) -> Result<Vec<Detection>>;
}

/// Image classification pipeline.
///
/// Takes raw RGB image bytes and returns top-k `(class_id, score)` predictions.
pub trait Classify {
    fn classify(&self, image: &[u8], width: u32, height: u32, top_k: usize) -> Result<Vec<Classification>>;
}

/// Image embedding pipeline.
///
/// Takes raw RGB image bytes and returns a normalized embedding vector.
pub trait Embed {
    fn embed(&self, image: &[u8], width: u32, height: u32) -> Result<Vec<f32>>;
}

/// Image segmentation pipeline.
///
/// Takes raw RGB image bytes and a prompt, returns a binary mask.
pub trait Segment {
    fn segment(&self, image: &[u8], width: u32, height: u32, prompt: SegmentPrompt) -> Result<Mask>;
}
