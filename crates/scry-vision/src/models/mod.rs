// SPDX-License-Identifier: MIT OR Apache-2.0
//! Concrete model implementations (pre-built pipelines).
//!
//! Each module bundles preprocessing, model inference, and postprocessing into
//! a struct that implements one of the [`crate::pipeline`] traits.
//!
//! Native architectures (use scry-llm tensors):
//! - [`resnet`] — ResNet-18/34/50/101 backbone
//! - [`vit`] — Vision Transformer backbone
//! - [`clip`] — CLIP visual encoder (ViT + projection)
//!
//! ONNX-based pipelines:
//! - [`yolo`] — YOLO object detector (v8/v11)
//! - [`scrfd`] — SCRFD face detector
//! - [`arcface`] — ArcFace face embedder

pub mod arcface;
pub mod clip;
pub mod resnet;
pub mod scrfd;
pub mod vit;
pub mod yolo;

pub use arcface::ArcFaceEmbedder;
pub use clip::{ClipConfig, ClipEmbedder, ClipVisual};
pub use resnet::{ResNet, ResNetClassifier, ResNetConfig};
pub use scrfd::ScrfdDetector;
pub use vit::{Vit, VitConfig};
pub use yolo::YoloDetector;
