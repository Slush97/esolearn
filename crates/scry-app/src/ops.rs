// SPDX-License-Identifier: MIT OR Apache-2.0
//! Operation specification, dispatch, and overlay types.

use scry_cv::prelude::{
    canny, connected_components, find_contours, hough_circles, hough_lines, Connectivity,
    GrayImageF, Orb,
};
use serde::{Deserialize, Serialize};

use crate::generators;

// ---------------------------------------------------------------------------
// Operation specification (JSON-serializable)
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum OpSpec {
    // Sources
    SolidColor { value: f32, width: u32, height: u32 },
    Checkerboard { cell_size: u32, width: u32, height: u32 },
    Gradient { width: u32, height: u32 },
    Rectangle { width: u32, height: u32, rx: u32, ry: u32, rw: u32, rh: u32 },
    GaussianBlob { width: u32, height: u32, sigma: f32 },
    LoadFile { path: String },

    // Filters
    GaussianBlur { sigma: f32 },
    Bilateral { sigma_space: f32, sigma_color: f32 },
    Median { radius: u32 },
    BoxBlur { radius: u32 },

    // Edge detection
    Sobel,
    Canny { low: f32, high: f32 },

    // Detection (returns overlay)
    HoughLines { rho_res: f32, theta_res: f32, threshold: u32 },
    HoughCircles { center_threshold: u32, radius_threshold: u32, min_radius: u32, max_radius: u32, min_dist: f32 },

    // Features
    OrbDetect { n_features: u32, fast_threshold: f32 },

    // Analysis
    ConnectedComponents { connectivity: u8 },
    Contours,

    // Morphology
    Erode { shape: String, ksize: u32 },
    Dilate { shape: String, ksize: u32 },
    MorphOpen { shape: String, ksize: u32 },
    MorphClose { shape: String, ksize: u32 },
}

impl OpSpec {
    /// Human-readable label for the breadcrumb trail.
    pub fn label(&self) -> String {
        match self {
            Self::SolidColor { value, width, height } => format!("Solid({value:.1}) {width}x{height}"),
            Self::Checkerboard { cell_size, width, height } => format!("Checker({cell_size}) {width}x{height}"),
            Self::Gradient { width, height } => format!("Gradient {width}x{height}"),
            Self::Rectangle { width, height, .. } => format!("Rect {width}x{height}"),
            Self::GaussianBlob { width, height, sigma } => format!("Blob(s={sigma:.1}) {width}x{height}"),
            Self::LoadFile { path } => {
                let name = std::path::Path::new(path)
                    .file_name()
                    .map_or("file", |n| n.to_str().unwrap_or("file"));
                format!("File: {name}")
            }
            Self::GaussianBlur { sigma } => format!("Blur(s={sigma:.1})"),
            Self::Bilateral { sigma_space, sigma_color } => format!("Bilateral({sigma_space:.1},{sigma_color:.1})"),
            Self::Median { radius } => format!("Median(r={radius})"),
            Self::BoxBlur { radius } => format!("Box(r={radius})"),
            Self::Sobel => "Sobel".into(),
            Self::Canny { low, high } => format!("Canny({low:.2},{high:.2})"),
            Self::HoughLines { threshold, .. } => format!("Hough Lines(t={threshold})"),
            Self::HoughCircles { center_threshold, .. } => format!("Hough Circles(t={center_threshold})"),
            Self::OrbDetect { n_features, .. } => format!("ORB(n={n_features})"),
            Self::ConnectedComponents { connectivity } => format!("Components({connectivity})"),
            Self::Contours => "Contours".into(),
            Self::Erode { ksize, .. } => format!("Erode(k={ksize})"),
            Self::Dilate { ksize, .. } => format!("Dilate(k={ksize})"),
            Self::MorphOpen { ksize, .. } => format!("Open(k={ksize})"),
            Self::MorphClose { ksize, .. } => format!("Close(k={ksize})"),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn is_source(&self) -> bool {
        matches!(
            self,
            Self::SolidColor { .. }
                | Self::Checkerboard { .. }
                | Self::Gradient { .. }
                | Self::Rectangle { .. }
                | Self::GaussianBlob { .. }
                | Self::LoadFile { .. }
        )
    }
}

// ---------------------------------------------------------------------------
// Overlay types (sent to JS for canvas rendering)
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize)]
#[serde(tag = "type")]
pub enum Overlay {
    Lines { lines: Vec<OverlayLine> },
    Circles { circles: Vec<OverlayCircle> },
    Keypoints { points: Vec<OverlayKeypoint> },
    Components { labels: Vec<u32>, num_labels: u32, width: u32, height: u32 },
    ContourPaths { contours: Vec<Vec<[u32; 2]>> },
}

#[derive(Clone, Serialize)]
pub struct OverlayLine {
    pub rho: f32,
    pub theta: f32,
    pub votes: u32,
}

#[derive(Clone, Serialize)]
pub struct OverlayCircle {
    pub cx: f32,
    pub cy: f32,
    pub radius: f32,
    pub votes: u32,
}

#[derive(Clone, Serialize)]
pub struct OverlayKeypoint {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub angle: f32,
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

fn parse_shape(s: &str) -> scry_cv::morphology::StructuringElement {
    match s {
        "cross" => scry_cv::morphology::StructuringElement::Cross,
        "ellipse" => scry_cv::morphology::StructuringElement::Ellipse,
        _ => scry_cv::morphology::StructuringElement::Rect,
    }
}

/// Execute a source operation (no input required).
pub fn execute_source(op: &OpSpec) -> std::result::Result<(GrayImageF, Option<Overlay>), String> {
    match op {
        OpSpec::SolidColor { value, width, height } => {
            Ok((generators::solid_color(*value, *width, *height)?, None))
        }
        OpSpec::Checkerboard { cell_size, width, height } => {
            Ok((generators::checkerboard(*cell_size, *width, *height)?, None))
        }
        OpSpec::Gradient { width, height } => {
            Ok((generators::gradient(*width, *height)?, None))
        }
        OpSpec::Rectangle { width, height, rx, ry, rw, rh } => {
            Ok((generators::rectangle(*width, *height, *rx, *ry, *rw, *rh)?, None))
        }
        OpSpec::GaussianBlob { width, height, sigma } => {
            Ok((generators::gaussian_blob(*width, *height, *sigma)?, None))
        }
        OpSpec::LoadFile { path } => {
            Ok((generators::load_file(path)?, None))
        }
        _ => Err("not a source operation".into()),
    }
}

/// Execute a processing operation on an input image.
#[allow(clippy::too_many_lines)]
pub fn execute(input: &GrayImageF, op: &OpSpec) -> std::result::Result<(GrayImageF, Option<Overlay>), String> {
    let map_err = |e: scry_cv::error::ScryVisionError| e.to_string();

    match op {
        // Filters
        OpSpec::GaussianBlur { sigma } => {
            let out = scry_cv::filter::gaussian::gaussian_blur(input, *sigma).map_err(map_err)?;
            Ok((out, None))
        }
        OpSpec::Bilateral { sigma_space, sigma_color } => {
            let out = scry_cv::filter::bilateral::bilateral_filter(input, *sigma_space, *sigma_color).map_err(map_err)?;
            Ok((out, None))
        }
        OpSpec::Median { radius } => {
            let out = scry_cv::filter::median::median_filter(input, *radius).map_err(map_err)?;
            Ok((out, None))
        }
        OpSpec::BoxBlur { radius } => {
            let out = scry_cv::filter::box_filter::box_blur(input, *radius).map_err(map_err)?;
            Ok((out, None))
        }

        // Edge detection
        OpSpec::Sobel => {
            let gx = scry_cv::filter::sobel::sobel_x(input).map_err(map_err)?;
            let gy = scry_cv::filter::sobel::sobel_y(input).map_err(map_err)?;
            // Gradient magnitude
            let data: Vec<f32> = gx.as_slice().iter().zip(gy.as_slice())
                .map(|(&x, &y)| x.hypot(y))
                .collect();
            // Normalize to [0, 1]
            let max = data.iter().copied().fold(0.0f32, f32::max);
            let data = if max > 0.0 {
                data.iter().map(|&v| v / max).collect()
            } else {
                data
            };
            let out = GrayImageF::from_vec(data, input.width(), input.height()).map_err(map_err)?;
            Ok((out, None))
        }
        OpSpec::Canny { low, high } => {
            let out = canny(input, *low, *high).map_err(map_err)?;
            Ok((out, None))
        }

        // Detection — pass image through, return overlay
        OpSpec::HoughLines { rho_res, theta_res, threshold } => {
            let lines = hough_lines(input, *rho_res, *theta_res, *threshold).map_err(map_err)?;
            let overlay = Overlay::Lines {
                lines: lines.iter().map(|l| OverlayLine {
                    rho: l.rho,
                    theta: l.theta,
                    votes: l.votes,
                }).collect(),
            };
            Ok((input.clone(), Some(overlay)))
        }
        OpSpec::HoughCircles { center_threshold, radius_threshold, min_radius, max_radius, min_dist } => {
            let circles = hough_circles(input, *center_threshold, *radius_threshold, *min_radius, *max_radius, *min_dist).map_err(map_err)?;
            let overlay = Overlay::Circles {
                circles: circles.iter().map(|c| OverlayCircle {
                    cx: c.cx,
                    cy: c.cy,
                    radius: c.radius,
                    votes: c.votes,
                }).collect(),
            };
            Ok((input.clone(), Some(overlay)))
        }

        // Features
        OpSpec::OrbDetect { n_features, fast_threshold } => {
            let orb = Orb::new()
                .n_features(*n_features as usize)
                .fast_threshold(*fast_threshold);
            let (keypoints, _descriptors) = orb.detect_and_compute(input).map_err(map_err)?;
            let overlay = Overlay::Keypoints {
                points: keypoints.iter().map(|kp| OverlayKeypoint {
                    x: kp.x,
                    y: kp.y,
                    size: kp.scale,
                    angle: kp.angle,
                }).collect(),
            };
            Ok((input.clone(), Some(overlay)))
        }

        // Analysis
        OpSpec::ConnectedComponents { connectivity } => {
            let conn = if *connectivity == 4 { Connectivity::Four } else { Connectivity::Eight };
            let cc = connected_components(input, conn).map_err(map_err)?;
            let overlay = Overlay::Components {
                labels: cc.labels,
                num_labels: cc.num_labels,
                width: cc.width,
                height: cc.height,
            };
            Ok((input.clone(), Some(overlay)))
        }
        OpSpec::Contours => {
            let contours = find_contours(input).map_err(map_err)?;
            let overlay = Overlay::ContourPaths {
                contours: contours.iter().map(|c| {
                    c.points.iter().map(|&(x, y)| [x, y]).collect()
                }).collect(),
            };
            Ok((input.clone(), Some(overlay)))
        }

        // Morphology
        OpSpec::Erode { shape, ksize } => {
            let kernel = scry_cv::morphology::make_kernel(parse_shape(shape), *ksize);
            let out = scry_cv::morphology::erode(input, &kernel, *ksize).map_err(map_err)?;
            Ok((out, None))
        }
        OpSpec::Dilate { shape, ksize } => {
            let kernel = scry_cv::morphology::make_kernel(parse_shape(shape), *ksize);
            let out = scry_cv::morphology::dilate(input, &kernel, *ksize).map_err(map_err)?;
            Ok((out, None))
        }
        OpSpec::MorphOpen { shape, ksize } => {
            let kernel = scry_cv::morphology::make_kernel(parse_shape(shape), *ksize);
            let out = scry_cv::morphology::open(input, &kernel, *ksize).map_err(map_err)?;
            Ok((out, None))
        }
        OpSpec::MorphClose { shape, ksize } => {
            let kernel = scry_cv::morphology::make_kernel(parse_shape(shape), *ksize);
            let out = scry_cv::morphology::close(input, &kernel, *ksize).map_err(map_err)?;
            Ok((out, None))
        }

        // Sources should not be dispatched here
        _ => Err("source operations must use execute_source".to_string()),
    }
}
