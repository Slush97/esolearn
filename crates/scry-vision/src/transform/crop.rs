// SPDX-License-Identifier: MIT OR Apache-2.0
//! Crop transforms.

use crate::error::{Result, VisionError};
use crate::image::ImageBuffer;
use crate::transform::ImageTransform;

/// Crop a centered region from the image.
///
/// If the image is smaller than the requested crop size, an error is returned.
#[derive(Clone, Debug)]
pub struct CenterCrop {
    pub width: u32,
    pub height: u32,
}

impl CenterCrop {
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Square center crop.
    #[must_use]
    pub fn square(size: u32) -> Self {
        Self::new(size, size)
    }
}

impl ImageTransform for CenterCrop {
    fn apply(&self, image: &ImageBuffer) -> Result<ImageBuffer> {
        if self.width > image.width || self.height > image.height {
            return Err(VisionError::Transform(format!(
                "CenterCrop {}x{} exceeds image {}x{}",
                self.width, self.height, image.width, image.height
            )));
        }

        let x_offset = (image.width - self.width) / 2;
        let y_offset = (image.height - self.height) / 2;

        crop_region(image, x_offset, y_offset, self.width, self.height)
    }
}

/// Crop an arbitrary rectangle from the image.
#[derive(Clone, Debug)]
pub struct Crop {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Crop {
    #[must_use]
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

impl ImageTransform for Crop {
    fn apply(&self, image: &ImageBuffer) -> Result<ImageBuffer> {
        if self.x + self.width > image.width || self.y + self.height > image.height {
            return Err(VisionError::Transform(format!(
                "Crop region ({},{})..({},{}) exceeds image {}x{}",
                self.x,
                self.y,
                self.x + self.width,
                self.y + self.height,
                image.width,
                image.height
            )));
        }
        crop_region(image, self.x, self.y, self.width, self.height)
    }
}

fn crop_region(
    image: &ImageBuffer,
    x_offset: u32,
    y_offset: u32,
    crop_w: u32,
    crop_h: u32,
) -> Result<ImageBuffer> {
    let ch = image.channels as usize;
    let src_stride = image.row_stride();
    let dst_stride = crop_w as usize * ch;
    let mut data = vec![0u8; crop_h as usize * dst_stride];

    for dy in 0..crop_h as usize {
        let sy = y_offset as usize + dy;
        let src_start = sy * src_stride + x_offset as usize * ch;
        let dst_start = dy * dst_stride;
        data[dst_start..dst_start + dst_stride]
            .copy_from_slice(&image.data[src_start..src_start + dst_stride]);
    }

    ImageBuffer::from_raw(data, crop_w, crop_h, image.channels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_4x4_rgb() -> ImageBuffer {
        // 4×4 image with pixel value = x + y*4 in channel 0
        let mut data = vec![0u8; 4 * 4 * 3];
        for y in 0..4u32 {
            for x in 0..4u32 {
                let idx = (y * 4 + x) as usize * 3;
                data[idx] = (x + y * 4) as u8;
                data[idx + 1] = 0;
                data[idx + 2] = 0;
            }
        }
        ImageBuffer::from_raw(data, 4, 4, 3).unwrap()
    }

    #[test]
    fn center_crop_identity() {
        let img = make_4x4_rgb();
        let cropped = CenterCrop::new(4, 4).apply(&img).unwrap();
        assert_eq!(cropped.data, img.data);
    }

    #[test]
    fn center_crop_2x2() {
        let img = make_4x4_rgb();
        let cropped = CenterCrop::new(2, 2).apply(&img).unwrap();
        assert_eq!(cropped.width, 2);
        assert_eq!(cropped.height, 2);
        // Offset: x=1, y=1. Pixels at (1,1),(2,1),(1,2),(2,2) → values 5,6,9,10
        assert_eq!(cropped.pixel(0, 0, 0), Some(5));
        assert_eq!(cropped.pixel(1, 0, 0), Some(6));
        assert_eq!(cropped.pixel(0, 1, 0), Some(9));
        assert_eq!(cropped.pixel(1, 1, 0), Some(10));
    }

    #[test]
    fn center_crop_too_large() {
        let img = make_4x4_rgb();
        assert!(CenterCrop::new(5, 5).apply(&img).is_err());
    }

    #[test]
    fn crop_region_top_left() {
        let img = make_4x4_rgb();
        let cropped = Crop::new(0, 0, 2, 2).apply(&img).unwrap();
        assert_eq!(cropped.pixel(0, 0, 0), Some(0));
        assert_eq!(cropped.pixel(1, 0, 0), Some(1));
        assert_eq!(cropped.pixel(0, 1, 0), Some(4));
        assert_eq!(cropped.pixel(1, 1, 0), Some(5));
    }

    #[test]
    fn crop_region_out_of_bounds() {
        let img = make_4x4_rgb();
        assert!(Crop::new(3, 3, 2, 2).apply(&img).is_err());
    }

    #[test]
    fn center_crop_square() {
        let img = make_4x4_rgb();
        let cropped = CenterCrop::square(2).apply(&img).unwrap();
        assert_eq!(cropped.width, 2);
        assert_eq!(cropped.height, 2);
    }
}
