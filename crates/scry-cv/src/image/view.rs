// SPDX-License-Identifier: MIT OR Apache-2.0
//! Borrowed image views for zero-copy sub-image access.

use std::marker::PhantomData;

use crate::image::pixel::{ChannelLayout, Pixel};

/// Immutable borrowed view into an image buffer.
///
/// Supports non-contiguous rows via an explicit stride, enabling
/// zero-copy sub-image extraction from an [`ImageBuf`](super::ImageBuf).
pub struct ImageView<'a, T: Pixel, C: ChannelLayout> {
    data: &'a [T],
    width: u32,
    height: u32,
    stride: usize,
    _channel: PhantomData<C>,
}

impl<'a, T: Pixel, C: ChannelLayout> ImageView<'a, T, C> {
    /// Create a view from raw parts.
    ///
    /// `stride` is measured in elements of `T`, not bytes.
    #[inline]
    pub fn new(data: &'a [T], width: u32, height: u32, stride: usize) -> Self {
        Self {
            data,
            width,
            height,
            stride,
            _channel: PhantomData,
        }
    }

    /// Image width.
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// `(width, height)`.
    #[inline]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Row stride in elements.
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Slice of one row's data (only the view's columns).
    #[inline]
    pub fn row(&self, y: u32) -> &[T] {
        let start = y as usize * self.stride;
        let row_len = self.width as usize * C::CHANNELS;
        &self.data[start..start + row_len]
    }

    /// Pixel channel slice at `(x, y)`.
    #[inline]
    pub fn pixel(&self, x: u32, y: u32) -> &[T] {
        let start = y as usize * self.stride + x as usize * C::CHANNELS;
        &self.data[start..start + C::CHANNELS]
    }

    /// Copy this view into a new owned `ImageBuf`.
    pub fn to_owned_buf(&self) -> super::ImageBuf<T, C> {
        let row_len = self.width as usize * C::CHANNELS;
        let mut data = Vec::with_capacity(row_len * self.height as usize);
        for y in 0..self.height {
            data.extend_from_slice(self.row(y));
        }
        // Safety: we just constructed the correct length
        super::ImageBuf::from_vec(data, self.width, self.height)
            .expect("view to_owned_buf: dimensions match by construction")
    }
}

// derive(Clone, Copy) doesn't work with generic bounds on lifetime types,
// so we implement manually.
impl<T: Pixel, C: ChannelLayout> Copy for ImageView<'_, T, C> {}
#[allow(clippy::expl_impl_clone_on_copy)]
impl<T: Pixel, C: ChannelLayout> Clone for ImageView<'_, T, C> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Mutable borrowed view into an image buffer.
pub struct ImageViewMut<'a, T: Pixel, C: ChannelLayout> {
    data: &'a mut [T],
    width: u32,
    height: u32,
    stride: usize,
    _channel: PhantomData<C>,
}

impl<'a, T: Pixel, C: ChannelLayout> ImageViewMut<'a, T, C> {
    /// Create a mutable view from raw parts.
    #[inline]
    pub fn new(data: &'a mut [T], width: u32, height: u32, stride: usize) -> Self {
        Self {
            data,
            width,
            height,
            stride,
            _channel: PhantomData,
        }
    }

    /// Image width.
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Row stride in elements.
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Slice of one row.
    #[inline]
    pub fn row(&self, y: u32) -> &[T] {
        let start = y as usize * self.stride;
        let row_len = self.width as usize * C::CHANNELS;
        &self.data[start..start + row_len]
    }

    /// Mutable slice of one row.
    #[inline]
    pub fn row_mut(&mut self, y: u32) -> &mut [T] {
        let start = y as usize * self.stride;
        let row_len = self.width as usize * C::CHANNELS;
        &mut self.data[start..start + row_len]
    }

    /// Pixel channel slice at `(x, y)`.
    #[inline]
    pub fn pixel(&self, x: u32, y: u32) -> &[T] {
        let start = y as usize * self.stride + x as usize * C::CHANNELS;
        &self.data[start..start + C::CHANNELS]
    }

    /// Mutable pixel channel slice at `(x, y)`.
    #[inline]
    pub fn pixel_mut(&mut self, x: u32, y: u32) -> &mut [T] {
        let start = y as usize * self.stride + x as usize * C::CHANNELS;
        &mut self.data[start..start + C::CHANNELS]
    }
}
