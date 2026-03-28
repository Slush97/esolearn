// SPDX-License-Identifier: MIT OR Apache-2.0
//! Visual mark types — the 9 primitives and batch variants.

use crate::bounds::BoundingBox;
use crate::style::{FillStyle, FontStyle, MarkerShape, StrokeStyle};

/// A key for identifying marks across scene diffs (enter/update/exit).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MarkKey {
    /// Integer key (e.g., data row index).
    Index(u64),
    /// String key (e.g., category name).
    Name(String),
}

/// Interpolation mode for line marks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Interpolation {
    /// Straight line segments.
    #[default]
    Linear,
    /// Step function (horizontal then vertical).
    StepBefore,
    /// Step function (vertical then horizontal).
    StepAfter,
    /// Monotone cubic spline.
    Monotone,
}

/// A polyline with optional interpolation.
#[derive(Clone, Debug)]
pub struct LineMark {
    /// Points along the line.
    pub points: Vec<[f32; 2]>,
    /// Line style.
    pub stroke: StrokeStyle,
    /// Interpolation mode.
    pub interpolation: Interpolation,
}

/// A rectangle (bars, heatmap cells).
#[derive(Clone, Debug)]
pub struct RectMark {
    /// Position and size.
    pub bounds: BoundingBox,
    /// Fill style.
    pub fill: FillStyle,
    /// Stroke style.
    pub stroke: StrokeStyle,
    /// Corner radius for rounded rectangles.
    pub corner_radius: f32,
}

/// A point mark (scatter markers).
#[derive(Clone, Debug)]
pub struct PointMark {
    /// Center position.
    pub center: [f32; 2],
    /// Marker size (diameter in pixels).
    pub size: f32,
    /// Marker shape.
    pub shape: MarkerShape,
    /// Fill style.
    pub fill: FillStyle,
    /// Stroke style.
    pub stroke: StrokeStyle,
}

/// A filled area between two lines.
#[derive(Clone, Debug)]
pub struct AreaMark {
    /// Upper boundary points.
    pub upper: Vec<[f32; 2]>,
    /// Lower boundary points (same x-values, different y).
    pub lower: Vec<[f32; 2]>,
    /// Fill style.
    pub fill: FillStyle,
    /// Stroke for the boundary lines.
    pub stroke: StrokeStyle,
}

/// Text anchor alignment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TextAnchor {
    /// Align to the start (left for LTR text).
    #[default]
    Start,
    /// Center alignment.
    Middle,
    /// Align to the end (right for LTR text).
    End,
}

/// A text label or annotation.
#[derive(Clone, Debug)]
pub struct TextMark {
    /// Position.
    pub position: [f32; 2],
    /// Text content.
    pub text: String,
    /// Font style.
    pub font: FontStyle,
    /// Fill color for the text.
    pub fill: FillStyle,
    /// Rotation in degrees.
    pub angle: f32,
    /// Text anchor alignment.
    pub anchor: TextAnchor,
}

/// An arc (pie/donut sector).
#[derive(Clone, Debug)]
pub struct ArcMark {
    /// Center position.
    pub center: [f32; 2],
    /// Inner radius (0 for pie, >0 for donut).
    pub inner_radius: f32,
    /// Outer radius.
    pub outer_radius: f32,
    /// Start angle in radians.
    pub start_angle: f32,
    /// End angle in radians.
    pub end_angle: f32,
    /// Fill style.
    pub fill: FillStyle,
    /// Stroke style.
    pub stroke: StrokeStyle,
}

/// A rule (grid line, reference line, whisker).
#[derive(Clone, Debug)]
pub struct RuleMark {
    /// Line segments as `(start, end)` pairs.
    pub segments: Vec<([f32; 2], [f32; 2])>,
    /// Line style.
    pub stroke: StrokeStyle,
}

/// A general path (Bezier curves, complex shapes).
#[derive(Clone, Debug)]
pub struct PathMark {
    /// SVG-like path commands.
    pub commands: Vec<PathCommand>,
    /// Fill style.
    pub fill: FillStyle,
    /// Stroke style.
    pub stroke: StrokeStyle,
}

/// SVG-like path command.
#[derive(Clone, Copy, Debug)]
pub enum PathCommand {
    /// Move to absolute position.
    MoveTo([f32; 2]),
    /// Line to absolute position.
    LineTo([f32; 2]),
    /// Cubic bezier to (control1, control2, end).
    CubicTo([f32; 2], [f32; 2], [f32; 2]),
    /// Quadratic bezier to (control, end).
    QuadTo([f32; 2], [f32; 2]),
    /// Close the path.
    Close,
}

/// An image/raster data mark.
#[derive(Clone, Debug)]
pub struct ImageMark {
    /// Position and size.
    pub bounds: BoundingBox,
    /// RGBA pixel data.
    pub data: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// A single visual mark (one of 9 primitives).
pub enum Mark {
    /// Polyline.
    Line(LineMark),
    /// Rectangle.
    Rect(RectMark),
    /// Scatter point.
    Point(PointMark),
    /// Filled area.
    Area(AreaMark),
    /// Text label.
    Text(TextMark),
    /// Arc/sector.
    Arc(ArcMark),
    /// Grid/reference lines.
    Rule(RuleMark),
    /// Bezier path.
    Path(PathMark),
    /// Raster image.
    Image(ImageMark),
}

/// Uniform or per-instance attribute.
#[derive(Clone, Debug)]
pub enum BatchAttr<T> {
    /// One value shared by all instances.
    Uniform(T),
    /// Per-instance data.
    Varying(Vec<T>),
}

impl<T: Clone> BatchAttr<T> {
    /// Get the value for a given instance index.
    pub fn get(&self, index: usize) -> &T {
        match self {
            Self::Uniform(v) => v,
            Self::Varying(v) => {
                debug_assert!(
                    index < v.len(),
                    "BatchAttr index {index} out of bounds (len {})",
                    v.len()
                );
                &v[index]
            }
        }
    }

    /// Number of instances (1 for uniform, N for varying).
    pub fn len(&self) -> usize {
        match self {
            Self::Uniform(_) => 1,
            Self::Varying(v) => v.len(),
        }
    }

    /// Whether this is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Uniform(_) => false,
            Self::Varying(v) => v.is_empty(),
        }
    }
}

/// A batch of homogeneous marks for instanced rendering.
pub enum MarkBatch {
    /// Batch of scatter points.
    Points {
        /// Center positions.
        positions: Vec<[f32; 2]>,
        /// Per-point sizes.
        sizes: BatchAttr<f32>,
        /// Per-point fill styles.
        fills: BatchAttr<FillStyle>,
        /// Marker shape (uniform for whole batch).
        shape: MarkerShape,
        /// Per-point stroke styles.
        strokes: BatchAttr<StrokeStyle>,
    },
    /// Batch of rule segments.
    Rules {
        /// Line segments.
        segments: Vec<([f32; 2], [f32; 2])>,
        /// Line style.
        stroke: StrokeStyle,
    },
    /// Batch of rectangles.
    Rects {
        /// Rectangle bounds.
        rects: Vec<BoundingBox>,
        /// Per-rect fills.
        fills: BatchAttr<FillStyle>,
        /// Per-rect strokes.
        strokes: BatchAttr<StrokeStyle>,
        /// Corner radius.
        corner_radius: f32,
    },
}

impl MarkBatch {
    /// Create a validated Points batch.
    pub fn points(
        positions: Vec<[f32; 2]>,
        sizes: BatchAttr<f32>,
        fills: BatchAttr<FillStyle>,
        shape: MarkerShape,
        strokes: BatchAttr<StrokeStyle>,
    ) -> Result<Self, String> {
        let batch = Self::Points {
            positions,
            sizes,
            fills,
            shape,
            strokes,
        };
        batch.validate()?;
        Ok(batch)
    }

    /// Create a validated Rects batch.
    pub fn rects(
        rects: Vec<BoundingBox>,
        fills: BatchAttr<FillStyle>,
        strokes: BatchAttr<StrokeStyle>,
        corner_radius: f32,
    ) -> Result<Self, String> {
        let batch = Self::Rects {
            rects,
            fills,
            strokes,
            corner_radius,
        };
        batch.validate()?;
        Ok(batch)
    }

    /// Validate that all Varying attributes match the batch size.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Points {
                positions,
                sizes,
                fills,
                strokes,
                ..
            } => {
                let n = positions.len();
                check_varying_len("sizes", sizes, n)?;
                check_varying_len("fills", fills, n)?;
                check_varying_len("strokes", strokes, n)?;
            }
            Self::Rects {
                rects,
                fills,
                strokes,
                ..
            } => {
                let n = rects.len();
                check_varying_len("fills", fills, n)?;
                check_varying_len("strokes", strokes, n)?;
            }
            Self::Rules { .. } => {}
        }
        Ok(())
    }
}

fn check_varying_len<T: Clone>(
    name: &str,
    attr: &BatchAttr<T>,
    expected: usize,
) -> Result<(), String> {
    if let BatchAttr::Varying(v) = attr {
        if v.len() != expected {
            return Err(format!(
                "BatchAttr '{name}' has {} elements, expected {expected}",
                v.len()
            ));
        }
    }
    Ok(())
}
