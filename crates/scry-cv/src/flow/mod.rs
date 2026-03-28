// SPDX-License-Identifier: MIT OR Apache-2.0
//! Optical flow estimation: dense (Farneback) and sparse (Lucas-Kanade).

pub mod farneback;
pub mod flow_field;
pub mod lucas_kanade;
pub mod poly_expansion;

pub use farneback::Farneback;
pub use flow_field::{FlowField, SparseFlowResult};
pub use lucas_kanade::LucasKanade;

use crate::error::Result;
use crate::image::{Gray, ImageBuf};

/// Trait for dense optical flow algorithms.
pub trait DenseOpticalFlow {
    /// Compute dense flow between two consecutive grayscale frames.
    fn calc(&mut self, prev: &ImageBuf<f32, Gray>, next: &ImageBuf<f32, Gray>)
        -> Result<FlowField>;
}

/// Trait for sparse optical flow algorithms.
pub trait SparseOpticalFlow {
    /// Track a set of points from `prev` to `next`.
    fn calc(
        &mut self,
        prev: &ImageBuf<f32, Gray>,
        next: &ImageBuf<f32, Gray>,
        prev_pts: &[(f32, f32)],
    ) -> Result<SparseFlowResult>;
}
