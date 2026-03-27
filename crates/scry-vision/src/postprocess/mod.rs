// SPDX-License-Identifier: MIT OR Apache-2.0
//! Postprocessing utilities for vision model outputs.
//!
//! - [`nms`] — non-maximum suppression (standard, soft, class-agnostic)
//! - [`boxes`] — YOLO bounding box decoding (anchor-free and anchor-based)
//! - [`classify`] — top-k classification from logits
//! - [`embedding`] — L2 normalization for embedding vectors

pub mod boxes;
pub mod classify;
pub mod embedding;
pub mod nms;

pub use boxes::{decode_anchor_based, decode_anchor_free, rescale_detections, Anchor, GridCell};
pub use classify::{softmax, top_k_from_scores, top_k_softmax, Classification};
pub use embedding::{cosine_similarity, l2_norm, l2_normalize, l2_normalized};
pub use nms::{nms, nms_class_agnostic, soft_nms, BBox, Detection, SoftNmsMethod};
