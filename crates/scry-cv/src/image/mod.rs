// SPDX-License-Identifier: MIT OR Apache-2.0
//! Core image types: owned buffers, borrowed views, and conversions.

pub mod buf;
pub mod convert;
pub mod pixel;
pub mod view;

#[cfg(any(
    feature = "image-interop",
    feature = "skia-interop",
    feature = "ndarray-interop"
))]
pub mod interop;

// Re-export the main types at module level.
pub use buf::ImageBuf;
pub use pixel::{ChannelLayout, Gray, Pixel, Rgb, Rgba};
pub use view::{ImageView, ImageViewMut};
