// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tauri command handlers grouped by domain.

pub mod diffusion;
pub mod llm;
pub mod vision;

use serde::Serialize;

use crate::state::BACKEND_NAME;

#[derive(Serialize)]
pub struct BackendInfo {
    pub name: &'static str,
    pub onnx: bool,
    pub cuda: bool,
    pub bf16: bool,
}

#[tauri::command]
pub fn backend_info() -> BackendInfo {
    BackendInfo {
        name: BACKEND_NAME,
        onnx: cfg!(feature = "onnx"),
        cuda: cfg!(feature = "scry-gpu-cuda"),
        bf16: cfg!(feature = "scry-gpu-bf16"),
    }
}
