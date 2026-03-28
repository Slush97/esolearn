// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tauri command handlers — bridge between frontend IPC and scry-cv operations.

use crate::ops::{self, OpSpec, Overlay};
use crate::state::{AppState, Pipeline, PipelineStep};
use scry_cv::image::buf::ImageBuf;
use scry_cv::image::pixel::Gray;

type GrayImageF = ImageBuf<f32, Gray>;
use serde::Serialize;
use tauri::State;

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct StepResult {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub overlay: Option<Overlay>,
    pub pipeline: PipelineInfo,
}

#[derive(Serialize)]
pub struct PipelineInfo {
    pub steps: Vec<StepSummary>,
    pub active_index: usize,
}

#[derive(Serialize)]
pub struct StepSummary {
    pub index: usize,
    pub label: String,
    pub op: OpSpec,
}

#[derive(Serialize)]
pub struct OpInfo {
    pub label: String,
    pub category: String,
    pub default_op: OpSpec,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn gray_to_rgba(img: &GrayImageF) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(img.as_slice().len() * 4);
    for &v in img.as_slice() {
        let byte = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        rgba.push(byte);
        rgba.push(byte);
        rgba.push(byte);
        rgba.push(255);
    }
    rgba
}

fn make_pipeline_info(pipeline: &Pipeline) -> PipelineInfo {
    PipelineInfo {
        steps: pipeline
            .steps
            .iter()
            .enumerate()
            .map(|(i, s)| StepSummary {
                index: i,
                label: s.op.label(),
                op: s.op.clone(),
            })
            .collect(),
        active_index: pipeline.active_index,
    }
}

fn make_step_result(pipeline: &Pipeline) -> Result<StepResult, String> {
    let step = pipeline
        .steps
        .get(pipeline.active_index)
        .ok_or("pipeline is empty")?;
    Ok(StepResult {
        pixels: gray_to_rgba(&step.result),
        width: step.result.width(),
        height: step.result.height(),
        overlay: step.overlay.clone(),
        pipeline: make_pipeline_info(pipeline),
    })
}

/// Re-execute pipeline from `start_index` onward.
fn reexecute_from(pipeline: &mut Pipeline, start_index: usize) -> Result<(), String> {
    for i in start_index..pipeline.steps.len() {
        let input = if i == 0 {
            // Source — execute without input
            let (result, overlay) = ops::execute_source(&pipeline.steps[i].op)?;
            pipeline.steps[i].result = result;
            pipeline.steps[i].overlay = overlay;
            continue;
        } else {
            pipeline.steps[i - 1].result.clone()
        };
        let (result, overlay) = ops::execute(&input, &pipeline.steps[i].op)?;
        pipeline.steps[i].result = result;
        pipeline.steps[i].overlay = overlay;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn list_operations() -> Vec<OpInfo> {
    vec![
        // Sources
        OpInfo { label: "Solid Color".into(), category: "Source".into(), default_op: OpSpec::SolidColor { value: 0.5, width: 256, height: 256 } },
        OpInfo { label: "Checkerboard".into(), category: "Source".into(), default_op: OpSpec::Checkerboard { cell_size: 16, width: 256, height: 256 } },
        OpInfo { label: "Gradient".into(), category: "Source".into(), default_op: OpSpec::Gradient { width: 256, height: 256 } },
        OpInfo { label: "Rectangle".into(), category: "Source".into(), default_op: OpSpec::Rectangle { width: 256, height: 256, rx: 64, ry: 64, rw: 128, rh: 128 } },
        OpInfo { label: "Gaussian Blob".into(), category: "Source".into(), default_op: OpSpec::GaussianBlob { width: 256, height: 256, sigma: 40.0 } },
        OpInfo { label: "Load File".into(), category: "Source".into(), default_op: OpSpec::LoadFile { path: String::new() } },
        // Filters
        OpInfo { label: "Gaussian Blur".into(), category: "Filter".into(), default_op: OpSpec::GaussianBlur { sigma: 1.5 } },
        OpInfo { label: "Bilateral".into(), category: "Filter".into(), default_op: OpSpec::Bilateral { sigma_space: 3.0, sigma_color: 0.1 } },
        OpInfo { label: "Median".into(), category: "Filter".into(), default_op: OpSpec::Median { radius: 2 } },
        OpInfo { label: "Box Blur".into(), category: "Filter".into(), default_op: OpSpec::BoxBlur { radius: 2 } },
        // Edge
        OpInfo { label: "Sobel".into(), category: "Edge".into(), default_op: OpSpec::Sobel },
        OpInfo { label: "Canny".into(), category: "Edge".into(), default_op: OpSpec::Canny { low: 0.05, high: 0.15 } },
        // Detection
        OpInfo { label: "Hough Lines".into(), category: "Detect".into(), default_op: OpSpec::HoughLines { rho_res: 1.0, theta_res: 0.0175, threshold: 20 } },
        OpInfo { label: "Hough Circles".into(), category: "Detect".into(), default_op: OpSpec::HoughCircles { center_threshold: 15, radius_threshold: 10, min_radius: 5, max_radius: 80, min_dist: 20.0 } },
        // Features
        OpInfo { label: "ORB Detect".into(), category: "Feature".into(), default_op: OpSpec::OrbDetect { n_features: 200, fast_threshold: 0.05 } },
        // Analysis
        OpInfo { label: "Components".into(), category: "Analysis".into(), default_op: OpSpec::ConnectedComponents { connectivity: 8 } },
        OpInfo { label: "Contours".into(), category: "Analysis".into(), default_op: OpSpec::Contours },
        // Morphology
        OpInfo { label: "Erode".into(), category: "Morphology".into(), default_op: OpSpec::Erode { shape: "rect".into(), ksize: 3 } },
        OpInfo { label: "Dilate".into(), category: "Morphology".into(), default_op: OpSpec::Dilate { shape: "rect".into(), ksize: 3 } },
        OpInfo { label: "Open".into(), category: "Morphology".into(), default_op: OpSpec::MorphOpen { shape: "rect".into(), ksize: 3 } },
        OpInfo { label: "Close".into(), category: "Morphology".into(), default_op: OpSpec::MorphClose { shape: "rect".into(), ksize: 3 } },
    ]
}

#[tauri::command]
pub fn set_source(state: State<'_, AppState>, op: OpSpec) -> Result<StepResult, String> {
    let (result, overlay) = ops::execute_source(&op)?;
    let mut pipeline = state.pipeline.lock().unwrap();
    *pipeline = Pipeline::new();
    pipeline.steps.push(PipelineStep { op, result, overlay });
    pipeline.active_index = 0;
    make_step_result(&pipeline)
}

#[tauri::command]
pub fn add_step(state: State<'_, AppState>, op: OpSpec) -> Result<StepResult, String> {
    let mut pipeline = state.pipeline.lock().unwrap();
    if pipeline.steps.is_empty() {
        return Err("set a source first".into());
    }
    let input = pipeline.steps.last().unwrap().result.clone();
    let (result, overlay) = ops::execute(&input, &op)?;
    pipeline.steps.push(PipelineStep { op, result, overlay });
    pipeline.active_index = pipeline.steps.len() - 1;
    make_step_result(&pipeline)
}

#[tauri::command]
pub fn update_step(
    state: State<'_, AppState>,
    index: usize,
    op: OpSpec,
) -> Result<StepResult, String> {
    let mut pipeline = state.pipeline.lock().unwrap();
    if index >= pipeline.steps.len() {
        return Err("step index out of range".into());
    }
    pipeline.steps[index].op = op;
    reexecute_from(&mut pipeline, index)?;
    pipeline.active_index = index;
    make_step_result(&pipeline)
}

#[tauri::command]
pub fn remove_step(state: State<'_, AppState>, index: usize) -> Result<StepResult, String> {
    let mut pipeline = state.pipeline.lock().unwrap();
    if index == 0 {
        return Err("cannot remove the source step".into());
    }
    if index >= pipeline.steps.len() {
        return Err("step index out of range".into());
    }
    pipeline.steps.remove(index);
    reexecute_from(&mut pipeline, index)?;
    pipeline.active_index = (index).min(pipeline.steps.len() - 1);
    make_step_result(&pipeline)
}

#[tauri::command]
pub fn get_step(state: State<'_, AppState>, index: usize) -> Result<StepResult, String> {
    let mut pipeline = state.pipeline.lock().unwrap();
    if index >= pipeline.steps.len() {
        return Err("step index out of range".into());
    }
    pipeline.active_index = index;
    make_step_result(&pipeline)
}

#[tauri::command]
pub fn get_pipeline(state: State<'_, AppState>) -> PipelineInfo {
    let pipeline = state.pipeline.lock().unwrap();
    make_pipeline_info(&pipeline)
}
