// SPDX-License-Identifier: MIT OR Apache-2.0
//! Text measurement heuristics for chart label layout.

/// Trait for measuring text dimensions.
pub trait TextMeasurer {
    /// Estimate the width of a string in pixels at the given font size.
    fn measure_width(&self, text: &str, font_size: f64) -> f64;

    /// Estimate the height of a line of text at the given font size.
    fn measure_height(&self, font_size: f64) -> f64;
}

/// Heuristic text measurer using character-class widths.
///
/// Good enough for chart labels without requiring a real font engine.
/// Character classes: narrow (iIl1|!.:;,'), wide (mMwW@), average (everything else).
#[derive(Clone, Debug, Default)]
pub struct HeuristicTextMeasurer;

impl HeuristicTextMeasurer {
    /// Average character width as a fraction of font size.
    const AVG_WIDTH: f64 = 0.6;
    /// Narrow character width fraction.
    const NARROW_WIDTH: f64 = 0.35;
    /// Wide character width fraction.
    const WIDE_WIDTH: f64 = 0.75;

    fn char_width_factor(c: char) -> f64 {
        match c {
            'i' | 'I' | 'l' | '1' | '|' | '!' | '.' | ':' | ';' | ',' | '\'' | ' ' => {
                Self::NARROW_WIDTH
            }
            'm' | 'M' | 'w' | 'W' | '@' => Self::WIDE_WIDTH,
            _ => Self::AVG_WIDTH,
        }
    }
}

impl TextMeasurer for HeuristicTextMeasurer {
    fn measure_width(&self, text: &str, font_size: f64) -> f64 {
        text.chars()
            .map(|c| Self::char_width_factor(c) * font_size)
            .sum()
    }

    fn measure_height(&self, font_size: f64) -> f64 {
        font_size * 1.2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heuristic_measurer() {
        let m = HeuristicTextMeasurer;
        let w = m.measure_width("Hello", 12.0);
        assert!(w > 0.0);
        // "iii" should be narrower than "MMM"
        let narrow = m.measure_width("iii", 12.0);
        let wide = m.measure_width("MMM", 12.0);
        assert!(narrow < wide);
    }
}
