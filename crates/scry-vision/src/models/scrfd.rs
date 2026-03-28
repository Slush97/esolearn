// SPDX-License-Identifier: MIT OR Apache-2.0
//! SCRFD face detector (ONNX-based).
//!
//! Supports both single-output and multi-output (9-tensor) SCRFD formats.
//! Multi-output format: 3 strides × (scores, bbox offsets, keypoints).
//! Preprocessing: Letterbox → ToTensor (InsightFace normalization).
//! Postprocessing: grid-based decode → NMS → rescale.

use scry_llm::backend::cpu::CpuBackend;

use crate::error::{Result, VisionError};
use crate::image::ImageBuffer;
use crate::pipeline::Detect;
use crate::postprocess::boxes::rescale_detections;
use crate::postprocess::nms::{nms, BBox, Detection};
use crate::transform::resize::Letterbox;
use crate::transform::to_tensor::ToTensor;

/// SCRFD face detector.
///
/// Handles the standard InsightFace multi-output format (9 tensors) as well
/// as single-output anchor-free format.
pub struct ScrfdDetector {
    inner: ScrfdInner,
    input_size: u32,
    iou_threshold: f32,
}

enum ScrfdInner {
    /// Single-output model via VisionModel trait (for mocks and simple models).
    Single(Box<dyn crate::model::VisionModel>),
    /// Multi-output ONNX model (9 tensors: scores + bboxes + keypoints per stride).
    #[cfg(feature = "onnx")]
    Multi(crate::model::OnnxModel),
}

impl ScrfdDetector {
    /// Create from a single-output VisionModel (for mocks and simple models).
    pub fn new(model: Box<dyn crate::model::VisionModel>, input_size: u32) -> Self {
        Self {
            inner: ScrfdInner::Single(model),
            input_size,
            iou_threshold: 0.4,
        }
    }

    #[must_use]
    pub fn with_iou_threshold(mut self, iou_threshold: f32) -> Self {
        self.iou_threshold = iou_threshold;
        self
    }

    /// Load from an ONNX model file. Auto-detects single vs multi-output format.
    #[cfg(feature = "onnx")]
    pub fn from_onnx(
        path: impl AsRef<std::path::Path>,
        input_size: u32,
    ) -> Result<Self> {
        let model = crate::model::OnnxModel::from_file(path)?;
        Ok(Self {
            inner: ScrfdInner::Multi(model),
            input_size,
            iou_threshold: 0.4,
        })
    }

    /// Preprocess: letterbox + InsightFace normalization.
    fn preprocess(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(Vec<f32>, crate::transform::resize::LetterboxInfo)> {
        let img = ImageBuffer::from_raw(image.to_vec(), width, height, 3)?;
        let letterbox = Letterbox::new(self.input_size, self.input_size);
        let (padded, info) = letterbox.apply_with_info(&img)?;

        // InsightFace normalization: (pixel - 127.5) / 128.0
        let std_val = 128.0 / 255.0;
        let tensor =
            ToTensor::normalized([0.5, 0.5, 0.5], [std_val, std_val, std_val])
                .apply::<CpuBackend>(&padded);

        Ok((tensor.to_vec(), info))
    }
}

impl Detect for ScrfdDetector {
    fn detect(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        conf_threshold: f32,
    ) -> Result<Vec<Detection>> {
        let (input_data, info) = self.preprocess(image, width, height)?;
        let s = self.input_size as usize;
        let input_shape = [1, 3, s, s];

        let mut detections = match &self.inner {
            ScrfdInner::Single(model) => {
                let output = model.forward(&input_data, &input_shape)?;
                decode_single_output(&output, conf_threshold)?
            }
            #[cfg(feature = "onnx")]
            ScrfdInner::Multi(model) => {
                let outputs = model.forward_multi(&input_data, &input_shape)?;
                if outputs.len() >= 9 {
                    decode_multi_output(&outputs, self.input_size, conf_threshold)
                } else if outputs.len() == 1 {
                    decode_single_output(&outputs[0], conf_threshold)?
                } else {
                    return Err(VisionError::Inference(format!(
                        "unexpected SCRFD output count: {} (expected 1 or 9)",
                        outputs.len()
                    )));
                }
            }
        };

        detections = nms(&detections, self.iou_threshold, conf_threshold);

        let inv_scale = 1.0 / info.scale;
        rescale_detections(&mut detections, inv_scale, inv_scale, info.pad_x, info.pad_y);

        Ok(detections)
    }
}

/// Decode single-output format: [5, num_proposals] (cx, cy, w, h, conf).
fn decode_single_output(output: &[f32], conf_threshold: f32) -> Result<Vec<Detection>> {
    let rows = 5;
    if output.len() % rows != 0 {
        return Err(VisionError::Inference(format!(
            "output size {} is not a multiple of 5",
            output.len()
        )));
    }
    let num_proposals = output.len() / rows;
    Ok(crate::postprocess::boxes::decode_anchor_free(
        output,
        num_proposals,
        1,
        conf_threshold,
    ))
}

/// Decode multi-output (9-tensor) InsightFace SCRFD format.
///
/// Outputs layout:
/// - `[0,1,2]`: scores  `[N, 1]` per stride 8/16/32 (pre-sigmoid)
/// - `[3,4,5]`: bboxes  `[N, 4]` per stride 8/16/32 (distance offsets)
/// - `[6,7,8]`: kps     `[N, 10]` per stride 8/16/32 (keypoint offsets)
fn decode_multi_output(
    outputs: &[Vec<f32>],
    input_size: u32,
    conf_threshold: f32,
) -> Vec<Detection> {
    let strides = [8u32, 16, 32];
    let mut detections = Vec::new();
    let has_keypoints = outputs.len() >= 9;

    for (i, &stride) in strides.iter().enumerate() {
        let scores = &outputs[i];       // [N, 1]
        let bboxes = &outputs[3 + i];   // [N, 4]
        let kps = if has_keypoints { Some(&outputs[6 + i]) } else { None };

        let grid_size = input_size / stride;
        let anchors_per_cell = 2; // SCRFD uses 2 anchors per grid cell
        let num_anchors = (grid_size * grid_size * anchors_per_cell) as usize;

        if scores.len() != num_anchors || bboxes.len() != num_anchors * 4 {
            continue;
        }

        // Validate keypoint tensor size: 5 landmarks × 2 coords = 10 per anchor
        let kps = kps.filter(|k| k.len() == num_anchors * 10);

        for idx in 0..num_anchors {
            // Scores are already sigmoided in the InsightFace ONNX export
            let score = scores[idx];
            if score < conf_threshold {
                continue;
            }

            // Grid position
            let cell = idx / anchors_per_cell as usize;
            let gx = (cell % grid_size as usize) as f32;
            let gy = (cell / grid_size as usize) as f32;

            // Bbox: distance offsets [left, top, right, bottom] × stride
            let left = bboxes[idx * 4] * stride as f32;
            let top = bboxes[idx * 4 + 1] * stride as f32;
            let right = bboxes[idx * 4 + 2] * stride as f32;
            let bottom = bboxes[idx * 4 + 3] * stride as f32;

            let cx = (gx + 0.5) * stride as f32;
            let cy = (gy + 0.5) * stride as f32;

            let x1 = cx - left;
            let y1 = cy - top;
            let x2 = cx + right;
            let y2 = cy + bottom;

            // Decode 5 facial landmarks (eyes, nose, mouth corners)
            let keypoints = kps.map(|kp_data| {
                let base = idx * 10;
                (0..5)
                    .map(|k| {
                        let kp_x = cx + kp_data[base + k * 2] * stride as f32;
                        let kp_y = cy + kp_data[base + k * 2 + 1] * stride as f32;
                        [kp_x, kp_y]
                    })
                    .collect()
            });

            detections.push(Detection {
                bbox: BBox::new(x1, y1, x2, y2),
                class_id: 0,
                confidence: score,
                keypoints,
            });
        }
    }

    detections
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::mock::MockModel;

    #[test]
    fn detect_with_mock_model() {
        // Single-output format: 2 proposals, [5, 2]: cx, cy, w, h, confidence
        #[rustfmt::skip]
        let output = vec![
            320.0, 100.0, // cx
            320.0, 100.0, // cy
             80.0,  40.0, // w
             80.0,  40.0, // h
              0.95,  0.3, // confidence
        ];

        let model = MockModel { output };
        let detector = ScrfdDetector::new(Box::new(model), 640);

        let image = vec![128u8; 640 * 640 * 3];
        let dets = detector.detect(&image, 640, 640, 0.5).unwrap();

        assert_eq!(dets.len(), 1);
        assert!((dets[0].confidence - 0.95).abs() < 1e-5);
        assert!((dets[0].bbox.x1 - 280.0).abs() < 1.0);
        assert!((dets[0].bbox.x2 - 360.0).abs() < 1.0);
        assert!((dets[0].bbox.y1 - 280.0).abs() < 1.0);
        assert!((dets[0].bbox.y2 - 360.0).abs() < 1.0);
    }

    #[test]
    fn detect_with_letterbox_rescaling() {
        #[rustfmt::skip]
        let output = vec![
            320.0, 320.0, 100.0, 100.0, 0.9,
        ];

        let model = MockModel { output };
        let detector = ScrfdDetector::new(Box::new(model), 640);

        let image = vec![128u8; 1280 * 720 * 3];
        let dets = detector.detect(&image, 1280, 720, 0.5).unwrap();

        assert_eq!(dets.len(), 1);
        let d = &dets[0];
        let cx = (d.bbox.x1 + d.bbox.x2) / 2.0;
        let cy = (d.bbox.y1 + d.bbox.y2) / 2.0;
        assert!((cx - 640.0).abs() < 2.0);
        assert!((cy - 360.0).abs() < 2.0);
    }

    #[test]
    fn detect_filters_low_confidence() {
        #[rustfmt::skip]
        let output = vec![320.0, 320.0, 100.0, 100.0, 0.2];

        let model = MockModel { output };
        let detector = ScrfdDetector::new(Box::new(model), 640);

        let image = vec![128u8; 640 * 640 * 3];
        let dets = detector.detect(&image, 640, 640, 0.5).unwrap();
        assert!(dets.is_empty());
    }

    #[test]
    fn detect_nms_suppresses_overlapping() {
        #[rustfmt::skip]
        let output = vec![
            320.0, 325.0, 320.0, 325.0, 100.0, 100.0, 100.0, 100.0, 0.95, 0.85,
        ];

        let model = MockModel { output };
        let detector = ScrfdDetector::new(Box::new(model), 640);

        let image = vec![128u8; 640 * 640 * 3];
        let dets = detector.detect(&image, 640, 640, 0.5).unwrap();

        assert_eq!(dets.len(), 1);
        assert!((dets[0].confidence - 0.95).abs() < 1e-5);
    }

    #[test]
    fn detect_validates_output_format() {
        let output = vec![1.0; 7];
        let model = MockModel { output };
        let detector = ScrfdDetector::new(Box::new(model), 640);

        let image = vec![128u8; 640 * 640 * 3];
        let result = detector.detect(&image, 640, 640, 0.5);
        assert!(result.is_err());
    }
}
