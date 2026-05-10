// SPDX-License-Identifier: MIT OR Apache-2.0
//! ResNet image classification and SCRFD face detection commands.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tauri::async_runtime::spawn_blocking;

use scry_vision::models::{ResNetClassifier, ResNetConfig};
use scry_vision::pipeline::Classify;
#[cfg(feature = "onnx")]
use scry_vision::pipeline::Detect;

use crate::state::{AppState, Backend, LoadedResnet};

#[derive(Deserialize)]
pub struct ResnetLoadArgs {
    pub model_path: String,
    /// One of `"resnet18"`, `"resnet34"`, `"resnet50"`, `"resnet101"`, `"resnet152"`.
    pub preset: String,
    #[serde(default)]
    pub labels_path: Option<String>,
}

#[derive(Serialize)]
pub struct ResnetLoadResult {
    pub num_classes: usize,
    pub has_labels: bool,
}

fn resnet_config(preset: &str, num_classes: usize) -> Result<ResNetConfig, String> {
    match preset {
        "resnet18" => Ok(ResNetConfig::resnet18(num_classes)),
        "resnet34" => Ok(ResNetConfig::resnet34(num_classes)),
        "resnet50" => Ok(ResNetConfig::resnet50(num_classes)),
        "resnet101" => Ok(ResNetConfig::resnet101(num_classes)),
        "resnet152" => Ok(ResNetConfig::resnet152(num_classes)),
        other => Err(format!("unknown resnet preset: {other}")),
    }
}

#[tauri::command]
pub async fn resnet_load(
    state: tauri::State<'_, AppState>,
    args: ResnetLoadArgs,
) -> Result<ResnetLoadResult, String> {
    let slot = state.resnet.clone();
    let model_path = PathBuf::from(args.model_path);
    let labels_path = args.labels_path.map(PathBuf::from);
    let preset = args.preset;

    spawn_blocking(move || -> Result<ResnetLoadResult, String> {
        let labels = match labels_path {
            Some(path) => std::fs::read_to_string(&path)
                .map_err(|e| format!("labels read failed: {e}"))?
                .lines()
                .map(str::to_owned)
                .collect::<Vec<_>>(),
            None => Vec::new(),
        };
        let num_classes = if labels.is_empty() { 1000 } else { labels.len() };
        let config = resnet_config(&preset, num_classes)?;
        let classifier = ResNetClassifier::<Backend>::from_safetensors(config, &model_path)
            .map_err(|e| format!("resnet load failed: {e}"))?;
        let has_labels = !labels.is_empty();
        *slot.lock() = Some(LoadedResnet { classifier, labels });
        Ok(ResnetLoadResult {
            num_classes,
            has_labels,
        })
    })
    .await
    .map_err(|e| format!("join error: {e}"))?
}

#[derive(Deserialize)]
pub struct ImagePayload {
    /// Raw RGB or RGBA image bytes from the frontend (any format `image` can decode).
    pub bytes: Vec<u8>,
}

#[derive(Deserialize)]
pub struct ResnetClassifyArgs {
    pub image: ImagePayload,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_top_k() -> usize {
    5
}

#[derive(Serialize)]
pub struct ResnetClassifyEntry {
    pub class_id: u32,
    pub label: Option<String>,
    pub score: f32,
}

#[derive(Serialize)]
pub struct ResnetClassifyResult {
    pub top: Vec<ResnetClassifyEntry>,
    pub elapsed_ms: u128,
}

fn decode_rgb(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
    let img = image::load_from_memory(bytes).map_err(|e| format!("image decode failed: {e}"))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    Ok((rgb.into_raw(), w, h))
}

#[tauri::command]
pub async fn resnet_classify(
    state: tauri::State<'_, AppState>,
    args: ResnetClassifyArgs,
) -> Result<ResnetClassifyResult, String> {
    let slot = state.resnet.clone();
    spawn_blocking(move || -> Result<ResnetClassifyResult, String> {
        let (rgb, w, h) = decode_rgb(&args.image.bytes)?;
        let guard = slot.lock();
        let loaded = guard.as_ref().ok_or("ResNet model not loaded")?;
        let start = std::time::Instant::now();
        let predictions = loaded
            .classifier
            .classify(&rgb, w, h, args.top_k)
            .map_err(|e| format!("classify failed: {e}"))?;
        let elapsed_ms = start.elapsed().as_millis();
        let top = predictions
            .into_iter()
            .map(|c| ResnetClassifyEntry {
                class_id: c.class_id,
                label: loaded.labels.get(c.class_id as usize).cloned(),
                score: c.score,
            })
            .collect();
        Ok(ResnetClassifyResult { top, elapsed_ms })
    })
    .await
    .map_err(|e| format!("join error: {e}"))?
}

// ---------- SCRFD (only when the `onnx` feature is enabled) ----------

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct ScrfdLoadArgs {
    pub model_path: String,
    #[serde(default = "default_input_size")]
    pub input_size: u32,
}

fn default_input_size() -> u32 {
    640
}

#[derive(Serialize)]
pub struct ScrfdLoadResult {
    pub input_size: u32,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct ScrfdDetectArgs {
    pub image: ImagePayload,
    #[serde(default = "default_conf")]
    pub conf_threshold: f32,
}

fn default_conf() -> f32 {
    0.5
}

#[derive(Serialize)]
pub struct DetectionDto {
    pub bbox: [f32; 4],
    pub class_id: u32,
    pub confidence: f32,
    pub keypoints: Option<Vec<[f32; 2]>>,
}

#[derive(Serialize)]
pub struct ScrfdDetectResult {
    pub detections: Vec<DetectionDto>,
    pub elapsed_ms: u128,
}

#[cfg(feature = "onnx")]
#[tauri::command]
pub async fn scrfd_load(
    state: tauri::State<'_, AppState>,
    args: ScrfdLoadArgs,
) -> Result<ScrfdLoadResult, String> {
    use scry_vision::models::ScrfdDetector;

    let slot = state.scrfd.clone();
    let path = PathBuf::from(args.model_path);
    let size = args.input_size;
    spawn_blocking(move || -> Result<ScrfdLoadResult, String> {
        let detector =
            ScrfdDetector::from_onnx(&path, size).map_err(|e| format!("scrfd load failed: {e}"))?;
        *slot.lock() = Some(crate::state::LoadedScrfd { detector });
        Ok(ScrfdLoadResult { input_size: size })
    })
    .await
    .map_err(|e| format!("join error: {e}"))?
}

#[cfg(not(feature = "onnx"))]
#[tauri::command]
pub async fn scrfd_load(
    _state: tauri::State<'_, AppState>,
    _args: ScrfdLoadArgs,
) -> Result<ScrfdLoadResult, String> {
    Err("scry-studio was built without the `onnx` feature; rebuild with `--features onnx` to enable SCRFD".into())
}

#[cfg(feature = "onnx")]
#[tauri::command]
pub async fn scrfd_detect(
    state: tauri::State<'_, AppState>,
    args: ScrfdDetectArgs,
) -> Result<ScrfdDetectResult, String> {
    let slot = state.scrfd.clone();
    spawn_blocking(move || -> Result<ScrfdDetectResult, String> {
        let (rgb, w, h) = decode_rgb(&args.image.bytes)?;
        let guard = slot.lock();
        let loaded = guard.as_ref().ok_or("SCRFD detector not loaded")?;
        let start = std::time::Instant::now();
        let dets = loaded
            .detector
            .detect(&rgb, w, h, args.conf_threshold)
            .map_err(|e| format!("detect failed: {e}"))?;
        let elapsed_ms = start.elapsed().as_millis();
        let detections = dets
            .into_iter()
            .map(|d| DetectionDto {
                bbox: [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                class_id: d.class_id,
                confidence: d.confidence,
                keypoints: d.keypoints,
            })
            .collect();
        Ok(ScrfdDetectResult {
            detections,
            elapsed_ms,
        })
    })
    .await
    .map_err(|e| format!("join error: {e}"))?
}

#[cfg(not(feature = "onnx"))]
#[tauri::command]
pub async fn scrfd_detect(
    _state: tauri::State<'_, AppState>,
    _args: ScrfdDetectArgs,
) -> Result<ScrfdDetectResult, String> {
    Err("scry-studio was built without the `onnx` feature".into())
}
