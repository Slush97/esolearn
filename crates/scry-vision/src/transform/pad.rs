// SPDX-License-Identifier: MIT OR Apache-2.0
//! Padding transforms.

use crate::error::Result;
use crate::image::ImageBuffer;
use crate::transform::ImageTransform;

/// Padding mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadMode {
    /// Fill with a constant value.
    Constant(u8),
    /// Reflect pixels at the border (e.g., `dcba|abcd|dcba`).
    Reflect,
    /// Replicate edge pixels (e.g., `aaaa|abcd|dddd`).
    Edge,
}

/// Pad an image on all four sides.
#[derive(Clone, Debug)]
pub struct Pad {
    pub top: u32,
    pub bottom: u32,
    pub left: u32,
    pub right: u32,
    pub mode: PadMode,
}

impl Pad {
    /// Uniform padding on all sides.
    #[must_use]
    pub fn uniform(amount: u32, mode: PadMode) -> Self {
        Self {
            top: amount,
            bottom: amount,
            left: amount,
            right: amount,
            mode,
        }
    }

    /// Asymmetric padding.
    #[must_use]
    pub fn new(top: u32, bottom: u32, left: u32, right: u32, mode: PadMode) -> Self {
        Self {
            top,
            bottom,
            left,
            right,
            mode,
        }
    }
}

impl ImageTransform for Pad {
    fn apply(&self, image: &ImageBuffer) -> Result<ImageBuffer> {
        let new_w = image.width + self.left + self.right;
        let new_h = image.height + self.top + self.bottom;
        let ch = image.channels as usize;
        let mut data = vec![0u8; new_w as usize * new_h as usize * ch];

        for dy in 0..new_h {
            for dx in 0..new_w {
                let dst_idx = (dy as usize * new_w as usize + dx as usize) * ch;

                // Map output pixel to source pixel
                let (sx, sy) = self.map_pixel(dx, dy, image.width, image.height);

                let src_idx = (sy as usize * image.width as usize + sx as usize) * ch;
                data[dst_idx..dst_idx + ch]
                    .copy_from_slice(&image.data[src_idx..src_idx + ch]);
            }
        }

        // If constant mode, overwrite padded pixels with the constant value
        if let PadMode::Constant(val) = self.mode {
            for dy in 0..new_h {
                for dx in 0..new_w {
                    if dy < self.top
                        || dy >= self.top + image.height
                        || dx < self.left
                        || dx >= self.left + image.width
                    {
                        let idx = (dy as usize * new_w as usize + dx as usize) * ch;
                        for c in 0..ch {
                            data[idx + c] = val;
                        }
                    }
                }
            }
        }

        ImageBuffer::from_raw(data, new_w, new_h, image.channels)
    }
}

impl Pad {
    /// Map an output coordinate to a source coordinate based on padding mode.
    fn map_pixel(&self, dx: u32, dy: u32, src_w: u32, src_h: u32) -> (u32, u32) {
        let sx = self.map_coord(dx, self.left, src_w);
        let sy = self.map_coord(dy, self.top, src_h);
        (sx, sy)
    }

    fn map_coord(&self, dst: u32, pad_before: u32, src_len: u32) -> u32 {
        let rel = dst as i64 - pad_before as i64;
        match self.mode {
            PadMode::Constant(_) => {
                // Clamp — the outer loop handles overwriting with the constant
                rel.clamp(0, src_len as i64 - 1) as u32
            }
            PadMode::Edge => rel.clamp(0, src_len as i64 - 1) as u32,
            PadMode::Reflect => reflect(rel, src_len),
        }
    }
}

/// Reflect coordinate into [0, len-1].
fn reflect(coord: i64, len: u32) -> u32 {
    if len <= 1 {
        return 0;
    }
    let len = len as i64;
    // Bring into range [0, 2*(len-1))
    let period = 2 * (len - 1);
    let mut c = coord % period;
    if c < 0 {
        c += period;
    }
    // Fold
    if c >= len {
        c = period - c;
    }
    c as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_3x3_gray() -> ImageBuffer {
        // 3×3 gray: pixel value = x + y*3
        let data: Vec<u8> = (0..9).collect();
        ImageBuffer::from_raw(data, 3, 3, 1).unwrap()
    }

    #[test]
    fn pad_constant_uniform() {
        let img = make_3x3_gray();
        let padded = Pad::uniform(1, PadMode::Constant(128)).apply(&img).unwrap();
        assert_eq!(padded.width, 5);
        assert_eq!(padded.height, 5);
        // Corners should be 128
        assert_eq!(padded.pixel(0, 0, 0), Some(128));
        assert_eq!(padded.pixel(4, 4, 0), Some(128));
        // Original (0,0) is now at (1,1)
        assert_eq!(padded.pixel(1, 1, 0), Some(0));
        assert_eq!(padded.pixel(2, 1, 0), Some(1));
        assert_eq!(padded.pixel(1, 2, 0), Some(3));
    }

    #[test]
    fn pad_edge() {
        let img = make_3x3_gray();
        let padded = Pad::uniform(1, PadMode::Edge).apply(&img).unwrap();
        assert_eq!(padded.width, 5);
        // Top-left corner should replicate pixel (0,0) = 0
        assert_eq!(padded.pixel(0, 0, 0), Some(0));
        // Top-right corner should replicate pixel (2,0) = 2
        assert_eq!(padded.pixel(4, 0, 0), Some(2));
        // Bottom-left should replicate pixel (0,2) = 6
        assert_eq!(padded.pixel(0, 4, 0), Some(6));
    }

    #[test]
    fn pad_reflect() {
        let img = make_3x3_gray();
        let padded = Pad::uniform(1, PadMode::Reflect).apply(&img).unwrap();
        assert_eq!(padded.width, 5);
        // Reflect: col -1 maps to col 1, row -1 maps to row 1
        // So padded(0,0) = original(1,1) = 4
        assert_eq!(padded.pixel(0, 0, 0), Some(4));
        // padded(1,0) = original(0,1) = 3
        assert_eq!(padded.pixel(1, 0, 0), Some(3));
    }

    #[test]
    fn pad_zero_is_identity() {
        let img = make_3x3_gray();
        let padded = Pad::uniform(0, PadMode::Constant(0)).apply(&img).unwrap();
        assert_eq!(padded.data, img.data);
    }

    #[test]
    fn pad_asymmetric() {
        let img = make_3x3_gray();
        let padded = Pad::new(1, 2, 0, 0, PadMode::Constant(255))
            .apply(&img)
            .unwrap();
        assert_eq!(padded.width, 3);
        assert_eq!(padded.height, 6); // 3 + 1 + 2
        // First row should be padding
        assert_eq!(padded.pixel(0, 0, 0), Some(255));
        // Original row 0 at y=1
        assert_eq!(padded.pixel(0, 1, 0), Some(0));
    }

    #[test]
    fn reflect_coord_basic() {
        assert_eq!(reflect(0, 5), 0);
        assert_eq!(reflect(4, 5), 4);
        assert_eq!(reflect(-1, 5), 1);
        assert_eq!(reflect(-2, 5), 2);
        assert_eq!(reflect(5, 5), 3);
        assert_eq!(reflect(6, 5), 2);
    }
}
