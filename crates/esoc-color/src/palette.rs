// SPDX-License-Identifier: MIT OR Apache-2.0
//! Color palettes for data visualization.

use crate::oklab::OkLch;
use crate::Color;

/// An ordered list of colors for data visualization.
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

    /// Number of colors.
    pub fn len(&self) -> usize {
        self.colors.len()
    }

    /// Whether the palette is empty.
    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }

    /// Iterate over colors.
    pub fn iter(&self) -> impl Iterator<Item = &Color> {
        self.colors.iter()
    }

    /// Interpolate across stops to get a color at `t` in `[0, 1]`.
    pub fn sample(&self, t: f32) -> Color {
        let t = t.clamp(0.0, 1.0);
        if self.colors.len() == 1 {
            return self.colors[0];
        }
        let max_idx = self.colors.len() - 1;
        let scaled = t * max_idx as f32;
        let lo = (scaled.floor() as usize).min(max_idx - 1);
        let frac = scaled - lo as f32;
        self.colors[lo].lerp_oklab(self.colors[lo + 1], frac)
    }

    // ── Built-in palettes ───────────────────────────────────────────

    /// Tableau 10 categorical palette.
    pub fn tab10() -> Self {
        Self::new(vec![
            Color::from_srgb8(0x1f, 0x77, 0xb4),
            Color::from_srgb8(0xff, 0x7f, 0x0e),
            Color::from_srgb8(0x2c, 0xa0, 0x2c),
            Color::from_srgb8(0xd6, 0x27, 0x28),
            Color::from_srgb8(0x94, 0x67, 0xbd),
            Color::from_srgb8(0x8c, 0x56, 0x4b),
            Color::from_srgb8(0xe3, 0x77, 0xc2),
            Color::from_srgb8(0x7f, 0x7f, 0x7f),
            Color::from_srgb8(0xbc, 0xbd, 0x22),
            Color::from_srgb8(0x17, 0xbe, 0xcf),
        ])
    }

    /// Viridis sequential palette (5 key stops).
    pub fn viridis() -> Self {
        Self::new(vec![
            Color::from_srgb8(0x44, 0x01, 0x54),
            Color::from_srgb8(0x31, 0x68, 0x8e),
            Color::from_srgb8(0x35, 0xb7, 0x79),
            Color::from_srgb8(0x90, 0xd7, 0x43),
            Color::from_srgb8(0xfd, 0xe7, 0x25),
        ])
    }

    /// Red-Blue diverging palette.
    pub fn rdbu() -> Self {
        Self::new(vec![
            Color::from_srgb8(0xb2, 0x18, 0x2b),
            Color::from_srgb8(0xef, 0x8a, 0x62),
            Color::from_srgb8(0xf7, 0xf7, 0xf7),
            Color::from_srgb8(0x67, 0xa9, 0xcf),
            Color::from_srgb8(0x21, 0x66, 0xac),
        ])
    }

    /// Generate a sequential palette by interpolating in `OKLab`.
    pub fn sequential(start: Color, end: Color, n: usize) -> Self {
        let n = n.max(2);
        let colors = (0..n)
            .map(|i| start.lerp_oklab(end, i as f32 / (n - 1) as f32))
            .collect();
        Self::new(colors)
    }

    /// Generate a diverging palette with a neutral midpoint.
    pub fn diverging(low: Color, mid: Color, high: Color, n: usize) -> Self {
        let n = n.max(3) | 1; // force odd
        let half = n / 2;
        let mut colors = Vec::with_capacity(n);
        for i in 0..half {
            colors.push(low.lerp_oklab(mid, i as f32 / half as f32));
        }
        colors.push(mid);
        for i in 1..=half {
            colors.push(mid.lerp_oklab(high, i as f32 / half as f32));
        }
        Self::new(colors)
    }

    /// Generate `n` evenly-spaced categorical colors in OKLCH.
    pub fn categorical(n: usize) -> Self {
        let n = n.max(1);
        let colors = (0..n)
            .map(|i| {
                let lch = OkLch::new(0.7, 0.15, (i as f32 / n as f32) * 360.0);
                Color::from_oklch(lch)
            })
            .collect();
        Self::new(colors)
    }
}

impl Default for Palette {
    fn default() -> Self {
        Self::tab10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tab10_has_10_colors() {
        assert_eq!(Palette::tab10().len(), 10);
    }

    #[test]
    fn sample_endpoints() {
        let p = Palette::viridis();
        let first = p.sample(0.0);
        let last = p.sample(1.0);
        assert!((first.r - p.get(0).r).abs() < 1e-3);
        let end = p.get(p.len() - 1);
        assert!((last.r - end.r).abs() < 0.05);
    }

    #[test]
    fn categorical_distinct() {
        let p = Palette::categorical(6);
        assert_eq!(p.len(), 6);
        // Colors should all be different
        for i in 0..5 {
            let a = p.get(i);
            let b = p.get(i + 1);
            let diff = (a.r - b.r).abs() + (a.g - b.g).abs() + (a.b - b.b).abs();
            assert!(diff > 0.01, "colors {i} and {} are too similar", i + 1);
        }
    }

    #[test]
    fn cycling() {
        let p = Palette::tab10();
        assert_eq!(p.get(0).to_hex(), p.get(10).to_hex());
    }
}
