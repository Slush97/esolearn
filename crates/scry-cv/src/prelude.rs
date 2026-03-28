// SPDX-License-Identifier: MIT OR Apache-2.0
//! Convenience re-exports for common types and traits.

pub use crate::error::{Result, ScryVisionError};
pub use crate::image::buf::ImageBuf;
pub use crate::image::convert::gray_from_u8_slice;
pub use crate::image::pixel::{ChannelLayout, Gray, Pixel, Rgb, Rgba};
pub use crate::image::view::{ImageView, ImageViewMut};
pub use crate::integral::IntegralImage;

// Filtering & edge detection
pub use crate::filter::canny::{canny, canny_with_sigma};

// Connected components & contours
pub use crate::components::{
    connected_components, find_contours, ComponentStats, ConnectedComponents, Connectivity, Contour,
};

// Hough transforms
pub use crate::hough::{hough_circles, hough_lines, HoughCircle, HoughLine};

// Feature detection & matching
pub use crate::features::{BinaryDescriptor, FloatDescriptor, KeyPoint, Orb};
pub use crate::matching::{
    find_best_match, knn_match_binary, match_binary, match_float, match_template, ratio_test,
    DMatch, TemplateMatchMethod,
};
pub use crate::registration::{
    find_fundamental, find_homography, EpipolarPair, FundamentalMatrix, Homography, PointPair,
    RansacConfig,
};

// Feature-gated re-exports
#[cfg(feature = "stereo")]
pub use crate::stereo::SgbmStereo;

#[cfg(feature = "flow")]
pub use crate::flow::{
    DenseOpticalFlow, Farneback, FlowField, LucasKanade, SparseFlowResult, SparseOpticalFlow,
};

#[cfg(feature = "background")]
pub use crate::background::{BackgroundSubtractor, KnnBackground, Mog2};

// Type aliases for the most common image types.

/// Grayscale u8 image.
pub type GrayImage = ImageBuf<u8, Gray>;
/// Grayscale f32 image (internal workhorse for vision algorithms).
pub type GrayImageF = ImageBuf<f32, Gray>;
/// RGB u8 image.
pub type RgbImage = ImageBuf<u8, Rgb>;
/// RGBA u8 image.
pub type RgbaImage = ImageBuf<u8, Rgba>;
/// Grayscale u8 view.
pub type GrayView<'a> = ImageView<'a, u8, Gray>;
/// Grayscale f32 view.
pub type GrayViewF<'a> = ImageView<'a, f32, Gray>;
