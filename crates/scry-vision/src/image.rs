// SPDX-License-Identifier: MIT OR Apache-2.0
//! Lightweight owned image representation.
//!
//! [`ImageBuffer`] stores pixel data as HWC (height × width × channels) row-major
//! `u8` values. It is deliberately simple — no dependency on the `image` crate by
//! default. Enable the `decode` feature for JPEG/PNG decoding from file bytes.

use crate::error::{Result, VisionError};

/// A raw image in memory: height × width × channels, row-major, u8.
///
/// Channel layout is interleaved (HWC): for a 3-channel RGB image, pixel (x, y)
/// occupies bytes `[(y * width + x) * 3 .. (y * width + x) * 3 + 3]`.
#[derive(Clone, Debug)]
pub struct ImageBuffer {
    /// Raw pixel data in HWC order.
    pub data: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Number of channels (1=gray, 3=RGB, 4=RGBA).
    pub channels: u8,
}

impl ImageBuffer {
    /// Create an `ImageBuffer` from raw pixel data.
    ///
    /// The caller must ensure `data.len() == width * height * channels`.
    pub fn from_raw(data: Vec<u8>, width: u32, height: u32, channels: u8) -> Result<Self> {
        let expected = width as usize * height as usize * channels as usize;
        if data.len() != expected {
            return Err(VisionError::InvalidDimensions {
                width,
                height,
                channels,
            });
        }
        Ok(Self {
            data,
            width,
            height,
            channels,
        })
    }

    /// Create a zero-filled image with the given dimensions.
    #[must_use]
    pub fn zeros(width: u32, height: u32, channels: u8) -> Self {
        let len = width as usize * height as usize * channels as usize;
        Self {
            data: vec![0; len],
            width,
            height,
            channels,
        }
    }

    /// Total number of pixels (width × height).
    #[must_use]
    pub fn num_pixels(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Total byte length of the pixel data.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    /// Get the pixel value at (x, y) for a given channel.
    ///
    /// Returns `None` if coordinates are out of bounds.
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32, channel: u8) -> Option<u8> {
        if x >= self.width || y >= self.height || channel >= self.channels {
            return None;
        }
        let idx =
            (y as usize * self.width as usize + x as usize) * self.channels as usize
                + channel as usize;
        Some(self.data[idx])
    }

    /// Set the pixel value at (x, y) for a given channel.
    pub fn set_pixel(&mut self, x: u32, y: u32, channel: u8, value: u8) {
        let idx =
            (y as usize * self.width as usize + x as usize) * self.channels as usize
                + channel as usize;
        self.data[idx] = value;
    }

    /// Row stride in bytes.
    #[must_use]
    pub fn row_stride(&self) -> usize {
        self.width as usize * self.channels as usize
    }

    /// Decode an image from JPEG/PNG bytes.
    ///
    /// Requires the `decode` feature. The output is always RGB (3 channels).
    #[cfg(feature = "decode")]
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        use image::io::Reader as ImageReader;
        use std::io::Cursor;

        let reader = ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|e| VisionError::Decode(e.to_string()))?;
        let img = reader
            .decode()
            .map_err(|e| VisionError::Decode(e.to_string()))?;
        let rgb = img.to_rgb8();
        let (w, h) = rgb.dimensions();
        Ok(Self {
            data: rgb.into_raw(),
            width: w,
            height: h,
            channels: 3,
        })
    }
}

#[cfg(feature = "decode")]
impl From<image::DynamicImage> for ImageBuffer {
    fn from(img: image::DynamicImage) -> Self {
        let rgb = img.to_rgb8();
        let (w, h) = rgb.dimensions();
        Self {
            data: rgb.into_raw(),
            width: w,
            height: h,
            channels: 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_raw_valid() {
        let data = vec![0u8; 4 * 3 * 3]; // 4×3, RGB
        let img = ImageBuffer::from_raw(data, 4, 3, 3).unwrap();
        assert_eq!(img.num_pixels(), 12);
        assert_eq!(img.byte_len(), 36);
        assert_eq!(img.row_stride(), 12);
    }

    #[test]
    fn from_raw_invalid_size() {
        let data = vec![0u8; 10]; // wrong size
        let err = ImageBuffer::from_raw(data, 4, 3, 3);
        assert!(err.is_err());
    }

    #[test]
    fn pixel_access() {
        // 2×2 RGB: pixel (0,0)=[1,2,3], (1,0)=[4,5,6], (0,1)=[7,8,9], (1,1)=[10,11,12]
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let img = ImageBuffer::from_raw(data, 2, 2, 3).unwrap();
        assert_eq!(img.pixel(0, 0, 0), Some(1));
        assert_eq!(img.pixel(0, 0, 2), Some(3));
        assert_eq!(img.pixel(1, 1, 1), Some(11));
        assert_eq!(img.pixel(2, 0, 0), None); // out of bounds
    }

    #[test]
    fn zeros_creates_blank() {
        let img = ImageBuffer::zeros(10, 10, 3);
        assert_eq!(img.byte_len(), 300);
        assert!(img.data.iter().all(|&b| b == 0));
    }
}
