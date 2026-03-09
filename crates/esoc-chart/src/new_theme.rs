// SPDX-License-Identifier: MIT OR Apache-2.0
//! Enhanced theme for the grammar-of-graphics chart API.

use esoc_color::{Color, Palette};

/// Theme controlling visual appearance of grammar-based charts.
#[derive(Clone, Debug)]
pub struct NewTheme {
    /// Background color.
    pub background: Color,
    /// Foreground color (text, axes, ticks).
    pub foreground: Color,
    /// Muted foreground color (subtitles, secondary text).
    pub muted_foreground: Color,
    /// Color palette for data series.
    pub palette: Palette,
    /// Grid line color.
    pub grid_color: Color,
    /// Grid line width.
    pub grid_width: f32,
    /// Whether to show grid lines.
    pub show_grid: bool,
    /// Base font size from which all others are derived.
    pub base_font_size: f32,
    /// Title font size (base * 1.2).
    pub title_font_size: f32,
    /// Subtitle font size (base * 1.1).
    pub subtitle_font_size: f32,
    /// Axis label font size (base * 1.0).
    pub label_font_size: f32,
    /// Tick label font size (base * 0.9).
    pub tick_font_size: f32,
    /// Legend font size (base * 0.9).
    pub legend_font_size: f32,
    /// Font family.
    pub font_family: String,
    /// Axis line width.
    pub axis_width: f32,
    /// Default data line width.
    pub line_width: f32,
    /// Default scatter point size (area in px²).
    pub point_size: f32,
}

impl NewTheme {
    /// Default light theme.
    pub fn light() -> Self {
        let base = 11.0_f32;
        Self {
            background: Color::WHITE,
            foreground: Color::BLACK,
            muted_foreground: Color::new(0.20, 0.20, 0.20, 1.0), // ~60% grey
            palette: Palette::tab10(),
            grid_color: Color::new(0.88, 0.88, 0.88, 1.0), // ~#E0E0E0
            grid_width: 0.5,
            show_grid: true,
            base_font_size: base,
            title_font_size: (base * 1.2).max(8.0),
            subtitle_font_size: (base * 1.1).max(8.0),
            label_font_size: (base * 1.0).max(8.0),
            tick_font_size: (base * 0.9).max(8.0),
            legend_font_size: (base * 0.9).max(8.0),
            font_family: "sans-serif".to_string(),
            axis_width: 1.0,
            line_width: 2.0,
            point_size: 30.0, // px² area (Vega default)
        }
    }

    /// Dark theme.
    pub fn dark() -> Self {
        let base = 11.0_f32;
        Self {
            background: Color::from_srgb8(0x1e, 0x1e, 0x2e),
            foreground: Color::from_srgb8(0xcd, 0xd6, 0xf4),
            muted_foreground: Color::new(0.55, 0.55, 0.65, 1.0), // muted light
            palette: Palette::tab10(),
            grid_color: Color::new(0.07, 0.07, 0.10, 1.0), // ~sRGB 0.3
            grid_width: 0.5,
            show_grid: true,
            base_font_size: base,
            title_font_size: (base * 1.2).max(8.0),
            subtitle_font_size: (base * 1.1).max(8.0),
            label_font_size: (base * 1.0).max(8.0),
            tick_font_size: (base * 0.9).max(8.0),
            legend_font_size: (base * 0.9).max(8.0),
            font_family: "sans-serif".to_string(),
            axis_width: 1.0,
            line_width: 2.0,
            point_size: 30.0,
        }
    }

    /// Publication theme — clean, minimal, for papers.
    pub fn publication() -> Self {
        let base = 10.0_f32;
        Self {
            background: Color::WHITE,
            foreground: Color::BLACK,
            muted_foreground: Color::new(0.25, 0.25, 0.25, 1.0),
            palette: Palette::tab10(),
            grid_color: Color::TRANSPARENT,
            grid_width: 0.0,
            show_grid: false,
            base_font_size: base,
            title_font_size: (base * 1.2).max(8.0),
            subtitle_font_size: (base * 1.1).max(8.0),
            label_font_size: (base * 1.0).max(8.0),
            tick_font_size: (base * 0.9).max(8.0),
            legend_font_size: (base * 0.9).max(8.0),
            font_family: "serif".to_string(),
            axis_width: 1.0,
            line_width: 1.5,
            point_size: 25.0, // slightly smaller for publication
        }
    }
}

impl Default for NewTheme {
    fn default() -> Self {
        Self::light()
    }
}
