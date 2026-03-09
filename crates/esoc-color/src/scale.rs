// SPDX-License-Identifier: MIT OR Apache-2.0
//! Continuous `[0,1]` → Color mapping for GPU 1D texture lookups.

use crate::palette::Palette;
use crate::Color;

/// A continuous color scale mapping `[0, 1]` → Color.
#[derive(Clone, Debug)]
pub struct ColorScale {
    palette: Palette,
}

impl ColorScale {
    /// Create a color scale from a palette.
    pub fn new(palette: Palette) -> Self {
        Self { palette }
    }

    /// Viridis color scale.
    pub fn viridis() -> Self {
        Self::new(Palette::viridis())
    }

    /// Red-Blue diverging scale.
    pub fn rdbu() -> Self {
        Self::new(Palette::rdbu())
    }

    /// Map a value in `[0, 1]` to a color.
    pub fn map(&self, t: f32) -> Color {
        self.palette.sample(t)
    }

    /// Generate texture data (RGBA8 sRGB) for GPU 1D lookup.
    ///
    /// Returns `width × 4` bytes of sRGB-encoded RGBA.
    pub fn to_texture_data(&self, width: u32) -> Vec<u8> {
        let mut data = Vec::with_capacity(width as usize * 4);
        for i in 0..width {
            let t = if width <= 1 {
                0.5
            } else {
                i as f32 / (width - 1) as f32
            };
            let c = self.map(t);
            let [r, g, b, a] = c.to_srgb8();
            data.extend_from_slice(&[r, g, b, a]);
        }
        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_data_length() {
        let scale = ColorScale::viridis();
        let data = scale.to_texture_data(256);
        assert_eq!(data.len(), 256 * 4);
    }

    #[test]
    fn endpoints() {
        let scale = ColorScale::viridis();
        let start = scale.map(0.0);
        let end = scale.map(1.0);
        // Viridis: dark purple → yellow
        assert!(start.r < 0.1); // dark
        assert!(end.r > 0.5); // bright
    }
}
