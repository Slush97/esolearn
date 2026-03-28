// SPDX-License-Identifier: MIT OR Apache-2.0
//! Owned image buffer.

use std::marker::PhantomData;

use crate::error::{Result, ScryVisionError};
use crate::image::pixel::{ChannelLayout, Pixel};
use crate::image::view::{ImageView, ImageViewMut};

/// Owned, row-major image buffer generic over pixel scalar and channel layout.
///
/// # Layout
///
/// Data is stored as a contiguous `Vec<T>` in row-major order with interleaved
/// channels: `[R,G,B, R,G,B, ...]` for RGB, `[v, v, v, ...]` for grayscale.
/// Length is always `width * height * C::CHANNELS`.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImageBuf<T: Pixel, C: ChannelLayout> {
    data: Vec<T>,
    width: u32,
    height: u32,
    _channel: PhantomData<C>,
}

impl<T: Pixel, C: ChannelLayout> ImageBuf<T, C> {
    /// Create a new image filled with `T::ZERO`.
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let len = Self::required_len(width, height)?;
        Ok(Self {
            data: vec![T::ZERO; len],
            width,
            height,
            _channel: PhantomData,
        })
    }

    /// Create an image from an existing data vector.
    pub fn from_vec(data: Vec<T>, width: u32, height: u32) -> Result<Self> {
        let expected = Self::required_len(width, height)?;
        if data.len() != expected {
            return Err(ScryVisionError::BufferSizeMismatch {
                expected,
                got: data.len(),
            });
        }
        Ok(Self {
            data,
            width,
            height,
            _channel: PhantomData,
        })
    }

    /// Create an image by calling `f(x, y)` for each pixel.
    pub fn from_fn(
        width: u32,
        height: u32,
        mut f: impl FnMut(u32, u32) -> Vec<T>,
    ) -> Result<Self> {
        let len = Self::required_len(width, height)?;
        let mut data = Vec::with_capacity(len);
        for y in 0..height {
            for x in 0..width {
                let px = f(x, y);
                debug_assert_eq!(px.len(), C::CHANNELS);
                data.extend_from_slice(&px);
            }
        }
        Ok(Self {
            data,
            width,
            height,
            _channel: PhantomData,
        })
    }

    /// Image width in pixels.
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// `(width, height)` tuple.
    #[inline]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Total number of scalar elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the buffer is empty (zero-area image).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Raw slice of all pixel data.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutable raw slice of all pixel data.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consume the buffer, returning the underlying `Vec<T>`.
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Row stride in elements (= `width * CHANNELS`).
    #[inline]
    pub fn stride(&self) -> usize {
        self.width as usize * C::CHANNELS
    }

    /// Slice of one row's data (all channels interleaved).
    #[inline]
    pub fn row(&self, y: u32) -> &[T] {
        let s = self.stride();
        let start = y as usize * s;
        &self.data[start..start + s]
    }

    /// Mutable slice of one row.
    #[inline]
    pub fn row_mut(&mut self, y: u32) -> &mut [T] {
        let s = self.stride();
        let start = y as usize * s;
        &mut self.data[start..start + s]
    }

    /// Slice of channel values at pixel `(x, y)`.
    #[inline]
    pub fn pixel(&self, x: u32, y: u32) -> &[T] {
        let idx = (y as usize * self.width as usize + x as usize) * C::CHANNELS;
        &self.data[idx..idx + C::CHANNELS]
    }

    /// Mutable slice of channel values at pixel `(x, y)`.
    #[inline]
    pub fn pixel_mut(&mut self, x: u32, y: u32) -> &mut [T] {
        let idx = (y as usize * self.width as usize + x as usize) * C::CHANNELS;
        &mut self.data[idx..idx + C::CHANNELS]
    }

    /// Immutable view over the entire image.
    #[inline]
    pub fn view(&self) -> ImageView<'_, T, C> {
        ImageView::new(&self.data, self.width, self.height, self.stride())
    }

    /// Mutable view over the entire image.
    #[inline]
    pub fn view_mut(&mut self) -> ImageViewMut<'_, T, C> {
        let stride = self.stride();
        ImageViewMut::new(&mut self.data, self.width, self.height, stride)
    }

    /// Zero-copy sub-image view.
    pub fn sub_image(&self, x: u32, y: u32, w: u32, h: u32) -> Result<ImageView<'_, T, C>> {
        Self::check_bounds(self.width, self.height, x, y, w, h)?;
        let stride = self.stride();
        let start = (y as usize * stride) + (x as usize * C::CHANNELS);
        let end = ((y + h - 1) as usize * stride) + ((x + w) as usize * C::CHANNELS);
        Ok(ImageView::new(&self.data[start..end], w, h, stride))
    }

    /// Zero-copy mutable sub-image view.
    pub fn sub_image_mut(
        &mut self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<ImageViewMut<'_, T, C>> {
        Self::check_bounds(self.width, self.height, x, y, w, h)?;
        let stride = self.stride();
        let start = (y as usize * stride) + (x as usize * C::CHANNELS);
        let end = ((y + h - 1) as usize * stride) + ((x + w) as usize * C::CHANNELS);
        Ok(ImageViewMut::new(&mut self.data[start..end], w, h, stride))
    }

    /// Convert pixel scalar type (e.g. u8 → f32).
    pub fn convert_type<U: Pixel>(&self) -> ImageBuf<U, C> {
        let data: Vec<U> = self.data.iter().map(|&v| U::from_f32(v.to_f32())).collect();
        ImageBuf {
            data,
            width: self.width,
            height: self.height,
            _channel: PhantomData,
        }
    }

    // ── Internal helpers ──

    fn required_len(width: u32, height: u32) -> Result<usize> {
        let pixels = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| {
                ScryVisionError::InvalidDimensions(format!("{width}x{height} overflows"))
            })?;
        pixels.checked_mul(C::CHANNELS).ok_or_else(|| {
            ScryVisionError::InvalidDimensions(format!(
                "{width}x{height}x{} overflows",
                C::CHANNELS
            ))
        })
    }

    fn check_bounds(
        img_w: u32,
        img_h: u32,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<()> {
        if x + w > img_w || y + h > img_h || w == 0 || h == 0 {
            return Err(ScryVisionError::OutOfBounds(format!(
                "sub-image ({x},{y})+({w},{h}) exceeds {img_w}x{img_h}"
            )));
        }
        Ok(())
    }
}

impl<T: Pixel, C: ChannelLayout> std::fmt::Debug for ImageBuf<T, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageBuf")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("channels", &C::NAME)
            .field("len", &self.data.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::pixel::{Gray, Rgb, Rgba};

    #[test]
    fn new_zero_filled() {
        let img = ImageBuf::<u8, Gray>::new(4, 3).unwrap();
        assert_eq!(img.width(), 4);
        assert_eq!(img.height(), 3);
        assert_eq!(img.len(), 12);
        assert!(img.as_slice().iter().all(|&v| v == 0));
    }

    #[test]
    fn from_vec_rgb() {
        let data = vec![0u8; 3 * 2 * 2];
        let img = ImageBuf::<u8, Rgb>::from_vec(data, 2, 2).unwrap();
        assert_eq!(img.stride(), 6);
    }

    #[test]
    fn from_vec_wrong_size() {
        let data = vec![0u8; 10];
        let err = ImageBuf::<u8, Rgba>::from_vec(data, 2, 2);
        assert!(err.is_err());
    }

    #[test]
    fn pixel_access() {
        let data = vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let img = ImageBuf::<u8, Rgb>::from_vec(data, 2, 2).unwrap();
        assert_eq!(img.pixel(0, 0), &[10, 20, 30]);
        assert_eq!(img.pixel(1, 0), &[40, 50, 60]);
        assert_eq!(img.pixel(0, 1), &[70, 80, 90]);
        assert_eq!(img.pixel(1, 1), &[100, 110, 120]);
    }

    #[test]
    fn convert_u8_to_f32() {
        let data = vec![0u8, 128, 255];
        let img = ImageBuf::<u8, Gray>::from_vec(data, 3, 1).unwrap();
        let fimg = img.convert_type::<f32>();
        assert!((fimg.pixel(0, 0)[0]).abs() < 1e-5);
        assert!((fimg.pixel(1, 0)[0] - 128.0 / 255.0).abs() < 0.01);
        assert!((fimg.pixel(2, 0)[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn sub_image_view() {
        let mut img = ImageBuf::<u8, Gray>::new(4, 4).unwrap();
        img.pixel_mut(2, 2)[0] = 42;
        let view = img.sub_image(1, 1, 3, 3).unwrap();
        assert_eq!(view.width(), 3);
        assert_eq!(view.height(), 3);
        // pixel (2,2) in full image = pixel (1,1) in sub-image
        assert_eq!(view.pixel(1, 1), &[42]);
    }
}
