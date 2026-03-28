// SPDX-License-Identifier: MIT OR Apache-2.0
//! Background subtraction algorithms: MOG2 and KNN.

pub mod knn;
pub mod mog2;

pub use knn::KnnBackground;
pub use mog2::Mog2;

use crate::error::Result;
use crate::image::{Gray, ImageBuf};

/// Trait for background subtraction algorithms.
pub trait BackgroundSubtractor {
    /// Process a new frame and return the foreground mask.
    ///
    /// - `frame`: grayscale f32 input frame (pixel values in [0, 1]).
    /// - `learning_rate`: adaptation rate (0 = no update, 1 = full replace,
    ///   negative = use algorithm default).
    ///
    /// Returns a binary mask: 255 = foreground, 127 = shadow (if detected), 0 = background.
    fn apply(
        &mut self,
        frame: &ImageBuf<f32, Gray>,
        learning_rate: f64,
    ) -> Result<ImageBuf<u8, Gray>>;

    /// Get the current background model as a grayscale f32 image.
    fn background(&self) -> Result<ImageBuf<f32, Gray>>;
}
