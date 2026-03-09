// SPDX-License-Identifier: MIT OR Apache-2.0
//! SVG path construction.

use crate::geom::Point;

/// An SVG path data string builder.
#[derive(Clone, Debug, Default)]
pub struct PathBuilder {
    commands: Vec<PathCommand>,
}

/// Individual path commands.
#[derive(Clone, Debug)]
enum PathCommand {
    MoveTo(Point),
    LineTo(Point),
    CurveTo {
        cp1: Point,
        cp2: Point,
        end: Point,
    },
    QuadTo {
        cp: Point,
        end: Point,
    },
    Arc {
        rx: f64,
        ry: f64,
        rotation: f64,
        large_arc: bool,
        sweep: bool,
        end: Point,
    },
    Close,
}

/// The result of building a path — an SVG path data string.
#[derive(Clone, Debug, Default)]
pub struct PathData {
    /// The SVG `d` attribute string.
    pub d: String,
}

impl PathBuilder {
    /// Create a new empty path builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Move to a point.
    pub fn move_to(&mut self, x: f64, y: f64) -> &mut Self {
        self.commands.push(PathCommand::MoveTo(Point::new(x, y)));
        self
    }

    /// Draw a line to a point.
    pub fn line_to(&mut self, x: f64, y: f64) -> &mut Self {
        self.commands.push(PathCommand::LineTo(Point::new(x, y)));
        self
    }

    /// Draw a cubic Bezier curve.
    pub fn curve_to(
        &mut self,
        cp1x: f64,
        cp1y: f64,
        cp2x: f64,
        cp2y: f64,
        x: f64,
        y: f64,
    ) -> &mut Self {
        self.commands.push(PathCommand::CurveTo {
            cp1: Point::new(cp1x, cp1y),
            cp2: Point::new(cp2x, cp2y),
            end: Point::new(x, y),
        });
        self
    }

    /// Draw a quadratic Bezier curve.
    pub fn quad_to(&mut self, cpx: f64, cpy: f64, x: f64, y: f64) -> &mut Self {
        self.commands.push(PathCommand::QuadTo {
            cp: Point::new(cpx, cpy),
            end: Point::new(x, y),
        });
        self
    }

    /// Draw an SVG arc.
    #[allow(clippy::too_many_arguments)]
    pub fn arc(
        &mut self,
        rx: f64,
        ry: f64,
        rotation: f64,
        large_arc: bool,
        sweep: bool,
        x: f64,
        y: f64,
    ) -> &mut Self {
        self.commands.push(PathCommand::Arc {
            rx,
            ry,
            rotation,
            large_arc,
            sweep,
            end: Point::new(x, y),
        });
        self
    }

    /// Close the path.
    pub fn close(&mut self) -> &mut Self {
        self.commands.push(PathCommand::Close);
        self
    }

    /// Build the path data string.
    pub fn build(&self) -> PathData {
        let mut d = String::new();
        for cmd in &self.commands {
            if !d.is_empty() {
                d.push(' ');
            }
            match cmd {
                PathCommand::MoveTo(p) => {
                    d.push_str(&format!("M{} {}", p.x, p.y));
                }
                PathCommand::LineTo(p) => {
                    d.push_str(&format!("L{} {}", p.x, p.y));
                }
                PathCommand::CurveTo { cp1, cp2, end } => {
                    d.push_str(&format!(
                        "C{} {},{} {},{} {}",
                        cp1.x, cp1.y, cp2.x, cp2.y, end.x, end.y
                    ));
                }
                PathCommand::QuadTo { cp, end } => {
                    d.push_str(&format!("Q{} {},{} {}", cp.x, cp.y, end.x, end.y));
                }
                PathCommand::Arc {
                    rx,
                    ry,
                    rotation,
                    large_arc,
                    sweep,
                    end,
                } => {
                    let la = u8::from(*large_arc);
                    let sw = u8::from(*sweep);
                    d.push_str(&format!(
                        "A{rx} {ry} {rotation} {la} {sw} {} {}",
                        end.x, end.y
                    ));
                }
                PathCommand::Close => d.push('Z'),
            }
        }
        PathData { d }
    }
}
