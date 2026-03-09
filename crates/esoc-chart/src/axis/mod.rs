// SPDX-License-Identifier: MIT OR Apache-2.0
//! Axis configuration, scales, and tick generation.

pub mod scale;
pub mod tick;

pub use scale::Scale;
pub use tick::{format_tick, nice_ticks, nice_ticks_log, Ticks};

/// Position of an axis relative to the plot area.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxisPosition {
    /// Bottom X-axis (default).
    Bottom,
    /// Top X-axis.
    Top,
    /// Left Y-axis (default).
    Left,
    /// Right Y-axis.
    Right,
}

/// Configuration for a single axis.
#[derive(Clone, Debug)]
pub struct AxisConfig {
    /// Axis label (e.g., "Epoch", "Loss").
    pub label: Option<String>,
    /// Scale type.
    pub scale: Scale,
    /// Manual range override `(min, max)`. `None` = auto-scale.
    pub range: Option<(f64, f64)>,
    /// Target number of ticks.
    pub tick_count: usize,
    /// Custom tick labels (overrides auto-generated labels).
    pub tick_labels: Option<Vec<String>>,
    /// Whether to show the axis line.
    pub visible: bool,
    /// Whether to invert the axis direction.
    pub inverted: bool,
}

impl AxisConfig {
    /// Create a default axis config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the axis label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set the scale type.
    pub fn scale(mut self, scale: Scale) -> Self {
        self.scale = scale;
        self
    }

    /// Set a manual range.
    pub fn range(mut self, min: f64, max: f64) -> Self {
        self.range = Some((min, max));
        self
    }

    /// Set the target tick count.
    pub fn tick_count(mut self, count: usize) -> Self {
        self.tick_count = count;
        self
    }

    /// Set custom tick labels.
    pub fn tick_labels(mut self, labels: Vec<String>) -> Self {
        self.tick_labels = Some(labels);
        self
    }
}

impl Default for AxisConfig {
    fn default() -> Self {
        Self {
            label: None,
            scale: Scale::Linear,
            range: None,
            tick_count: 6,
            tick_labels: None,
            visible: true,
            inverted: false,
        }
    }
}
