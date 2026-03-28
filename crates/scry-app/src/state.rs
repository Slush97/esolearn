// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pipeline state management.

use crate::ops::{OpSpec, Overlay};
use scry_cv::prelude::GrayImageF;
use std::sync::Mutex;

/// A single step in the processing pipeline.
#[derive(Clone)]
pub struct PipelineStep {
    pub op: OpSpec,
    pub result: GrayImageF,
    pub overlay: Option<Overlay>,
}

/// Ordered pipeline: step 0 is always a source generator.
pub struct Pipeline {
    pub steps: Vec<PipelineStep>,
    pub active_index: usize,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            active_index: 0,
        }
    }
}

/// Tauri-managed application state.
pub struct AppState {
    pub pipeline: Mutex<Pipeline>,
}
