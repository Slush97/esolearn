// SPDX-License-Identifier: MIT OR Apache-2.0
//! Color palettes for data visualization.

use crate::color::Color;

/// A palette is an ordered list of colors for cycling through data series.
#[derive(Clone, Debug)]
pub struct Palette {
    colors: Vec<Color>,
}

impl Palette {
    /// Create a palette from a list of colors.
    pub fn new(colors: Vec<Color>) -> Self {
        assert!(!colors.is_empty(), "palette must have at least one color");
        Self { colors }
    }

    /// Get the color at the given index, cycling if needed.
    pub fn get(&self, index: usize) -> Color {
        self.colors[index % self.colors.len()]
    }

    /// Number of colors in this palette.
    pub fn len(&self) -> usize {
        self.colors.len()
    }

    /// Whether the palette is empty (always false after construction).
    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }

    /// Tableau 10 categorical palette (default for chart series).
    pub fn tab10() -> Self {
        Self::new(vec![
            Color::from_rgb8(0x1f, 0x77, 0xb4), // blue
            Color::from_rgb8(0xff, 0x7f, 0x0e), // orange
            Color::from_rgb8(0x2c, 0xa0, 0x2c), // green
            Color::from_rgb8(0xd6, 0x27, 0x28), // red
            Color::from_rgb8(0x94, 0x67, 0xbd), // purple
            Color::from_rgb8(0x8c, 0x56, 0x4b), // brown
            Color::from_rgb8(0xe3, 0x77, 0xc2), // pink
            Color::from_rgb8(0x7f, 0x7f, 0x7f), // gray
            Color::from_rgb8(0xbc, 0xbd, 0x22), // olive
            Color::from_rgb8(0x17, 0xbe, 0xcf), // cyan
        ])
    }

    /// Generate a sequential palette by interpolating between two colors.
    pub fn sequential(start: Color, end: Color, n: usize) -> Self {
        let n = n.max(2);
        let colors = (0..n)
            .map(|i| start.lerp(end, i as f64 / (n - 1) as f64))
            .collect();
        Self::new(colors)
    }

    /// Viridis-like sequential palette (5 key stops).
    pub fn viridis() -> Self {
        Self::new(vec![
            Color::from_rgb8(0x44, 0x01, 0x54), // dark purple
            Color::from_rgb8(0x31, 0x68, 0x8e), // blue
            Color::from_rgb8(0x35, 0xb7, 0x79), // teal
            Color::from_rgb8(0x90, 0xd7, 0x43), // green
            Color::from_rgb8(0xfd, 0xe7, 0x25), // yellow
        ])
    }

    /// Red-Blue diverging palette.
    pub fn rdbu() -> Self {
        Self::new(vec![
            Color::from_rgb8(0xb2, 0x18, 0x2b), // dark red
            Color::from_rgb8(0xef, 0x8a, 0x62), // light red
            Color::from_rgb8(0xf7, 0xf7, 0xf7), // white
            Color::from_rgb8(0x67, 0xa9, 0xcf), // light blue
            Color::from_rgb8(0x21, 0x66, 0xac), // dark blue
        ])
    }

    /// Interpolate across this palette's color stops to get a color at `t` in `[0, 1]`.
    pub fn sample(&self, t: f64) -> Color {
        let t = t.clamp(0.0, 1.0);
        if self.colors.len() == 1 {
            return self.colors[0];
        }
        let max_idx = self.colors.len() - 1;
        let scaled = t * max_idx as f64;
        let lo = (scaled.floor() as usize).min(max_idx - 1);
        let frac = scaled - lo as f64;
        self.colors[lo].lerp(self.colors[lo + 1], frac)
    }
}

impl Default for Palette {
    fn default() -> Self {
        Self::tab10()
    }
}
