// SPDX-License-Identifier: MIT OR Apache-2.0
//! # scry-vision
//!
//! High-level vision inference toolkit for Rust.
//!
//! Provides image preprocessing, model loading, postprocessing, and pre-built
//! pipelines for common vision tasks (detection, classification, embedding,
//! segmentation).
//!
//! Built on [`scry_llm`]'s tensor and backend infrastructure.

#[cfg(feature = "safetensors")]
pub mod checkpoint;
pub mod error;
pub mod image;
pub mod model;
pub mod models;
pub mod nn;
pub mod pipeline;
pub mod postprocess;
pub mod transform;

pub use error::{Result, VisionError};
pub use image::ImageBuffer;
