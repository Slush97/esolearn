// SPDX-License-Identifier: MIT OR Apache-2.0
//! Chart themes controlling colors, fonts, and visual style.

use esoc_gfx::color::Color;
use esoc_gfx::palette::Palette;

/// A theme controlling the visual appearance of charts.
#[derive(Clone, Debug)]
pub struct Theme {
    /// Background color.
    pub background: Color,
    /// Foreground color (text, axes, ticks).
    pub foreground: Color,
    /// Color palette for data series.
    pub palette: Palette,
    /// Grid line color.
    pub grid_color: Color,
    /// Grid line width.
    pub grid_width: f64,
    /// Whether to show grid lines.
    pub show_grid: bool,
    /// Title font size.
    pub title_font_size: f64,
    /// Axis label font size.
    pub label_font_size: f64,
    /// Tick label font size.
    pub tick_font_size: f64,
    /// Legend font size.
    pub legend_font_size: f64,
    /// Font family.
    pub font_family: String,
    /// Axis line width.
    pub axis_width: f64,
    /// Default data line width.
    pub line_width: f64,
    /// Default scatter point radius.
    pub point_radius: f64,
}

impl Theme {
    /// Default light theme.
    pub fn light() -> Self {
        Self {
            background: Color::WHITE,
            foreground: Color::BLACK,
            palette: Palette::tab10(),
            grid_color: Color::new(0.9, 0.9, 0.9, 1.0),
            grid_width: 0.5,
            show_grid: true,
            title_font_size: 16.0,
            label_font_size: 13.0,
            tick_font_size: 11.0,
            legend_font_size: 11.0,
            font_family: "sans-serif".to_string(),
            axis_width: 1.0,
            line_width: 2.0,
            point_radius: 4.0,
        }
    }

    /// Dark theme.
    pub fn dark() -> Self {
        Self {
            background: Color::from_rgb8(0x1e, 0x1e, 0x2e),
            foreground: Color::from_rgb8(0xcd, 0xd6, 0xf4),
            palette: Palette::tab10(),
            grid_color: Color::new(0.3, 0.3, 0.35, 1.0),
            grid_width: 0.5,
            show_grid: true,
            title_font_size: 16.0,
            label_font_size: 13.0,
            tick_font_size: 11.0,
            legend_font_size: 11.0,
            font_family: "sans-serif".to_string(),
            axis_width: 1.0,
            line_width: 2.0,
            point_radius: 4.0,
        }
    }

    /// Minimal theme — no grid, thin axes.
    pub fn minimal() -> Self {
        Self {
            background: Color::WHITE,
            foreground: Color::from_rgb8(0x33, 0x33, 0x33),
            palette: Palette::tab10(),
            grid_color: Color::TRANSPARENT,
            grid_width: 0.0,
            show_grid: false,
            title_font_size: 14.0,
            label_font_size: 12.0,
            tick_font_size: 10.0,
            legend_font_size: 10.0,
            font_family: "sans-serif".to_string(),
            axis_width: 0.5,
            line_width: 1.5,
            point_radius: 3.0,
        }
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::light()
    }
}
