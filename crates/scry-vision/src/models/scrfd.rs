// SPDX-License-Identifier: MIT OR Apache-2.0
//! SCRFD face detector (ONNX-based).
//!
//! Supports both single-output and multi-output (9-tensor) SCRFD formats.
//! Multi-output format: 3 strides × (scores, bbox offsets, keypoints).
//! Preprocessing: Letterbox → `ToTensor` (`InsightFace` normalization).
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
/// Handles the standard `InsightFace` multi-output format (9 tensors) as well
/// as single-output anchor-free format.
pub struct ScrfdDetector {
    inner: ScrfdInner,
    input_size: u32,
    iou_threshold: f32,
}

enum ScrfdInner {
    /// Single-output model via `VisionModel` trait (for mocks and simple models).
    Single(Box<dyn crate::model::VisionModel>),
    /// Multi-output ONNX model (9 tensors: scores + bboxes + keypoints per stride).
    #[cfg(feature = "onnx")]
    Multi(crate::model::OnnxModel),
}

impl ScrfdDetector {
    /// Create from a single-output `VisionModel` (for mocks and simple models).
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
    pub fn from_onnx(path: impl AsRef<std::path::Path>, input_size: u32) -> Result<Self> {
        let model = crate::model::OnnxModel::from_file(path)?;
        Ok(Self {
            inner: ScrfdInner::Multi(model),
            input_size,
            iou_threshold: 0.4,
        })
    }

    /// Preprocess: letterbox + `InsightFace` normalization.
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
        let tensor = ToTensor::normalized([0.5, 0.5, 0.5], [std_val, std_val, std_val])
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
        rescale_detections(
            &mut detections,
            inv_scale,
            inv_scale,
            info.pad_x,
            info.pad_y,
        );

        Ok(detections)
    }
}

/// Decode single-output format: [5, `num_proposals`] (cx, cy, w, h, conf).
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

/// Decode multi-output (9-tensor) `InsightFace` SCRFD format.
///
/// Outputs layout (grouped by type, then by stride):
/// - `[0,1,2]`: scores  `[N, 1]` per stride 8/16/32 (pre-sigmoid in the ONNX export)
/// - `[3,4,5]`: bboxes  `[N, 4]` per stride 8/16/32 (distance offsets from anchor center)
/// - `[6,7,8]`: kps     `[N, 10]` per stride 8/16/32 (5 landmarks × 2 coords, offsets)
///
/// **Important**: this assumes the standard `InsightFace` ONNX export ordering where
/// all score tensors come first, then all bbox tensors, then all keypoint tensors.
/// If the model outputs are interleaved per-stride (`score_8`, `bbox_8`, `kps_8`, ...),
/// the caller must reorder before passing to this function.
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
fn decode_multi_output(
    outputs: &[Vec<f32>],
    input_size: u32,
    conf_threshold: f32,
) -> Vec<Detection> {
    let strides = [8u32, 16, 32];
    let mut detections = Vec::new();
    let has_keypoints = outputs.len() >= 9;

    for (i, &stride) in strides.iter().enumerate() {
        let scores = &outputs[i]; // [N, 1]
        let bboxes = &outputs[3 + i]; // [N, 4]
        let kps = if has_keypoints {
            Some(&outputs[6 + i])
        } else {
            None
        };

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
#[allow(
    clippy::too_many_arguments,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::needless_range_loop
)]
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
        let cx = f32::midpoint(d.bbox.x1, d.bbox.x2);
        let cy = f32::midpoint(d.bbox.y1, d.bbox.y2);
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

    // ── decode_multi_output tests ───────────────────────────────────────

    /// Helper: build synthetic 9-tensor SCRFD outputs for a given `input_size`.
    ///
    /// Places a single high-confidence anchor at grid cell (`target_gx`, `target_gy`)
    /// on the specified stride, with known bbox offsets and keypoint offsets.
    /// All other anchors have score 0.
    fn build_multi_output(
        input_size: u32,
        target_stride: u32,
        target_gx: u32,
        target_gy: u32,
        anchor_idx_in_cell: usize, // 0 or 1
        score: f32,
        bbox_offsets: [f32; 4], // [left, top, right, bottom] in stride units
        kp_offsets: Option<[f32; 10]>, // 5 landmarks × 2 coords, in stride units
    ) -> Vec<Vec<f32>> {
        let strides = [8u32, 16, 32];
        let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(9);

        // Build score, bbox, kps tensors for each stride
        for &stride in &strides {
            let grid_size = input_size / stride;
            let num_anchors = (grid_size * grid_size * 2) as usize;

            let mut scores = vec![0.0f32; num_anchors];
            let mut bboxes = vec![0.0f32; num_anchors * 4];
            let mut kps = vec![0.0f32; num_anchors * 10];

            if stride == target_stride {
                let cell = (target_gy * grid_size + target_gx) as usize;
                let idx = cell * 2 + anchor_idx_in_cell;

                scores[idx] = score;
                bboxes[idx * 4] = bbox_offsets[0];
                bboxes[idx * 4 + 1] = bbox_offsets[1];
                bboxes[idx * 4 + 2] = bbox_offsets[2];
                bboxes[idx * 4 + 3] = bbox_offsets[3];

                if let Some(kp) = &kp_offsets {
                    for k in 0..10 {
                        kps[idx * 10 + k] = kp[k];
                    }
                }
            }

            outputs.push(scores);
            outputs.push(bboxes);
            outputs.push(kps);
        }

        // Reorder from [s8, b8, k8, s16, b16, k16, s32, b32, k32]
        //            to [s8, s16, s32, b8, b16, b32, k8, k16, k32]
        let reordered = vec![
            outputs[0].clone(), // scores stride 8
            outputs[3].clone(), // scores stride 16
            outputs[6].clone(), // scores stride 32
            outputs[1].clone(), // bboxes stride 8
            outputs[4].clone(), // bboxes stride 16
            outputs[7].clone(), // bboxes stride 32
            outputs[2].clone(), // kps stride 8
            outputs[5].clone(), // kps stride 16
            outputs[8].clone(), // kps stride 32
        ];
        reordered
    }

    #[test]
    fn multi_output_single_detection_stride8() {
        // Place a face at grid cell (10, 5) on stride 8
        // Anchor center: cx = (10 + 0.5) * 8 = 84, cy = (5 + 0.5) * 8 = 44
        let outputs = build_multi_output(
            640,
            8, // stride
            10,
            5,                    // grid cell
            0,                    // first anchor
            0.92,                 // confidence
            [3.0, 2.0, 4.0, 5.0], // left, top, right, bottom in stride units
            None,
        );

        let dets = decode_multi_output(&outputs, 640, 0.5);

        assert_eq!(dets.len(), 1, "expected exactly 1 detection");
        let d = &dets[0];
        assert!((d.confidence - 0.92).abs() < 1e-5);

        // cx = (10 + 0.5) * 8 = 84.0
        // cy = (5 + 0.5) * 8  = 44.0
        // x1 = 84 - 3*8 = 84 - 24 = 60
        // y1 = 44 - 2*8 = 44 - 16 = 28
        // x2 = 84 + 4*8 = 84 + 32 = 116
        // y2 = 44 + 5*8 = 44 + 40 = 84
        assert!((d.bbox.x1 - 60.0).abs() < 1e-3, "x1={}", d.bbox.x1);
        assert!((d.bbox.y1 - 28.0).abs() < 1e-3, "y1={}", d.bbox.y1);
        assert!((d.bbox.x2 - 116.0).abs() < 1e-3, "x2={}", d.bbox.x2);
        assert!((d.bbox.y2 - 84.0).abs() < 1e-3, "y2={}", d.bbox.y2);
        assert_eq!(d.class_id, 0);
    }

    #[test]
    fn multi_output_single_detection_stride32() {
        // Place a face at grid cell (3, 7) on stride 32
        // Grid size = 640/32 = 20
        // Anchor center: cx = (3 + 0.5) * 32 = 112, cy = (7 + 0.5) * 32 = 240
        let outputs = build_multi_output(
            640,
            32,
            3,
            7,
            1, // second anchor in cell
            0.88,
            [2.0, 1.5, 2.5, 3.0],
            None,
        );

        let dets = decode_multi_output(&outputs, 640, 0.5);

        assert_eq!(dets.len(), 1);
        let d = &dets[0];
        assert!((d.confidence - 0.88).abs() < 1e-5);
        // x1 = 112 - 2*32 = 48, y1 = 240 - 1.5*32 = 192
        // x2 = 112 + 2.5*32 = 192, y2 = 240 + 3*32 = 336
        assert!((d.bbox.x1 - 48.0).abs() < 1e-3, "x1={}", d.bbox.x1);
        assert!((d.bbox.y1 - 192.0).abs() < 1e-3, "y1={}", d.bbox.y1);
        assert!((d.bbox.x2 - 192.0).abs() < 1e-3, "x2={}", d.bbox.x2);
        assert!((d.bbox.y2 - 336.0).abs() < 1e-3, "y2={}", d.bbox.y2);
    }

    #[test]
    fn multi_output_with_keypoints() {
        // Place face at grid (0, 0) stride 16
        // cx = 0.5 * 16 = 8, cy = 0.5 * 16 = 8
        #[rustfmt::skip]
        let kp_offsets = [
            -0.25, -0.5,   // left eye:  (8 + -0.25*16, 8 + -0.5*16) = (4, 0)
             0.25, -0.5,   // right eye: (8 + 0.25*16, 8 + -0.5*16)  = (12, 0)
             0.0,   0.0,   // nose:      (8, 8)
            -0.25,  0.5,   // left mouth: (4, 16)
             0.25,  0.5,   // right mouth: (12, 16)
        ];

        let outputs = build_multi_output(
            640,
            16,
            0,
            0,
            0,
            0.95,
            [0.5, 0.5, 0.5, 0.5],
            Some(kp_offsets),
        );

        let dets = decode_multi_output(&outputs, 640, 0.5);
        assert_eq!(dets.len(), 1);

        let kps = dets[0].keypoints.as_ref().expect("expected keypoints");
        assert_eq!(kps.len(), 5);

        assert!((kps[0][0] - 4.0).abs() < 1e-3, "left eye x");
        assert!((kps[0][1] - 0.0).abs() < 1e-3, "left eye y");
        assert!((kps[1][0] - 12.0).abs() < 1e-3, "right eye x");
        assert!((kps[1][1] - 0.0).abs() < 1e-3, "right eye y");
        assert!((kps[2][0] - 8.0).abs() < 1e-3, "nose x");
        assert!((kps[2][1] - 8.0).abs() < 1e-3, "nose y");
        assert!((kps[3][0] - 4.0).abs() < 1e-3, "left mouth x");
        assert!((kps[3][1] - 16.0).abs() < 1e-3, "left mouth y");
        assert!((kps[4][0] - 12.0).abs() < 1e-3, "right mouth x");
        assert!((kps[4][1] - 16.0).abs() < 1e-3, "right mouth y");
    }

    #[test]
    fn multi_output_confidence_filtering() {
        let strides = [8u32, 16, 32];
        let mut outputs = Vec::new();

        for &stride in &strides {
            let grid_size = 640 / stride;
            let num_anchors = (grid_size * grid_size * 2) as usize;
            // Fill all scores with 0.3 (below typical threshold)
            outputs.push(vec![0.3f32; num_anchors]);
        }
        for &stride in &strides {
            let grid_size = 640 / stride;
            let num_anchors = (grid_size * grid_size * 2) as usize;
            outputs.push(vec![1.0f32; num_anchors * 4]);
        }
        for &stride in &strides {
            let grid_size = 640 / stride;
            let num_anchors = (grid_size * grid_size * 2) as usize;
            outputs.push(vec![0.0f32; num_anchors * 10]);
        }

        let dets = decode_multi_output(&outputs, 640, 0.5);
        assert!(dets.is_empty(), "all below threshold → no detections");
    }

    #[test]
    fn multi_output_tensor_shape_validation() {
        // If score tensor has wrong size, that stride should be skipped
        let strides = [8u32, 16, 32];
        let mut outputs = Vec::new();

        // Scores: give stride 8 the WRONG size (off by 1)
        let grid8 = 640 / 8;
        let wrong_num = (grid8 * grid8 * 2 + 1) as usize; // wrong!
        outputs.push(vec![0.9f32; wrong_num]);

        // Strides 16 and 32 get correct sizes but all zeros
        for &stride in &strides[1..] {
            let gs = 640 / stride;
            outputs.push(vec![0.0f32; (gs * gs * 2) as usize]);
        }
        // Bboxes (all correct sizes)
        for &stride in &strides {
            let gs = 640 / stride;
            outputs.push(vec![1.0f32; (gs * gs * 2 * 4) as usize]);
        }
        // Kps
        for &stride in &strides {
            let gs = 640 / stride;
            outputs.push(vec![0.0f32; (gs * gs * 2 * 10) as usize]);
        }

        let dets = decode_multi_output(&outputs, 640, 0.5);
        // Stride 8 skipped due to shape mismatch, strides 16/32 have no high-conf anchors
        assert!(dets.is_empty());
    }

    #[test]
    fn multi_output_grid_cell_position_stride16() {
        // Verify grid position calculation for stride 16.
        // Grid size = 640/16 = 40. Place anchor at last cell (39, 39).
        let outputs = build_multi_output(
            640,
            16,
            39,
            39, // last grid cell
            0,
            0.99,
            [1.0, 1.0, 1.0, 1.0], // symmetric box
            None,
        );

        let dets = decode_multi_output(&outputs, 640, 0.5);
        assert_eq!(dets.len(), 1);

        // cx = (39 + 0.5) * 16 = 632, cy = (39 + 0.5) * 16 = 632
        let d = &dets[0];
        let cx = f32::midpoint(d.bbox.x1, d.bbox.x2);
        let cy = f32::midpoint(d.bbox.y1, d.bbox.y2);
        assert!((cx - 632.0).abs() < 1e-3, "cx={cx}");
        assert!((cy - 632.0).abs() < 1e-3, "cy={cy}");
    }

    #[test]
    fn multi_output_both_anchors_in_cell() {
        // Place two detections in the same grid cell (both anchors)
        let strides = [8u32, 16, 32];
        let mut score_tensors = Vec::new();
        let mut bbox_tensors = Vec::new();
        let mut kps_tensors = Vec::new();

        for &stride in &strides {
            let grid_size = 640 / stride;
            let num_anchors = (grid_size * grid_size * 2) as usize;
            let mut scores = vec![0.0f32; num_anchors];
            let mut bboxes = vec![0.0f32; num_anchors * 4];
            let kps = vec![0.0f32; num_anchors * 10];

            if stride == 8 {
                // Grid cell (5, 5), both anchors
                let cell = (5 * grid_size + 5) as usize;
                let idx0 = cell * 2;
                let idx1 = cell * 2 + 1;

                scores[idx0] = 0.90;
                scores[idx1] = 0.85;

                // Different box sizes for the two anchors
                for off in 0..4 {
                    bboxes[idx0 * 4 + off] = 2.0; // 2 stride units each direction
                    bboxes[idx1 * 4 + off] = 4.0; // 4 stride units each direction
                }
            }

            score_tensors.push(scores);
            bbox_tensors.push(bboxes);
            kps_tensors.push(kps);
        }

        let mut outputs = Vec::new();
        outputs.extend(score_tensors);
        outputs.extend(bbox_tensors);
        outputs.extend(kps_tensors);

        let dets = decode_multi_output(&outputs, 640, 0.5);
        assert_eq!(dets.len(), 2, "both anchors should produce detections");

        // Both should have the same center
        let cx0 = f32::midpoint(dets[0].bbox.x1, dets[0].bbox.x2);
        let cx1 = f32::midpoint(dets[1].bbox.x1, dets[1].bbox.x2);
        assert!((cx0 - cx1).abs() < 1e-3, "same cell → same center");

        // But different sizes
        let w0 = dets[0].bbox.x2 - dets[0].bbox.x1;
        let w1 = dets[1].bbox.x2 - dets[1].bbox.x1;
        assert!((w0 - 32.0).abs() < 1e-3, "anchor 0 width = 2*2*8 = 32");
        assert!((w1 - 64.0).abs() < 1e-3, "anchor 1 width = 4*2*8 = 64");
    }

    #[test]
    fn multi_output_empty_tensors() {
        // 9 empty tensors — no anchors match
        let outputs = vec![vec![]; 9];
        let dets = decode_multi_output(&outputs, 640, 0.5);
        assert!(dets.is_empty());
    }

    #[test]
    fn multi_output_without_keypoints_6_tensors() {
        // Only 6 tensors (scores + bboxes, no keypoints)
        let strides = [8u32, 16, 32];
        let mut outputs = Vec::new();

        // Scores
        for &stride in &strides {
            let gs = 640 / stride;
            let n = (gs * gs * 2) as usize;
            let mut scores = vec![0.0f32; n];
            if stride == 16 {
                scores[0] = 0.95; // first anchor, first cell
            }
            outputs.push(scores);
        }
        // Bboxes
        for &stride in &strides {
            let gs = 640 / stride;
            let n = (gs * gs * 2) as usize;
            let mut bboxes = vec![0.0f32; n * 4];
            if stride == 16 {
                bboxes[0] = 1.0; // left
                bboxes[1] = 1.0; // top
                bboxes[2] = 1.0; // right
                bboxes[3] = 1.0; // bottom
            }
            outputs.push(bboxes);
        }

        assert_eq!(outputs.len(), 6);
        let dets = decode_multi_output(&outputs, 640, 0.5);
        assert_eq!(dets.len(), 1);
        assert!(dets[0].keypoints.is_none(), "6 tensors → no keypoints");
    }

    #[test]
    fn multi_output_detections_across_strides() {
        // Place one detection per stride to verify all strides are decoded
        let strides = [8u32, 16, 32];
        let mut score_tensors = Vec::new();
        let mut bbox_tensors = Vec::new();
        let mut kps_tensors = Vec::new();

        for &stride in &strides {
            let gs = 640 / stride;
            let n = (gs * gs * 2) as usize;
            let mut scores = vec![0.0f32; n];
            let mut bboxes = vec![0.0f32; n * 4];
            let kps = vec![0.0f32; n * 10];

            // Place detection at first anchor of cell (0, 0) for each stride
            scores[0] = 0.8;
            for off in 0..4 {
                bboxes[off] = 0.5;
            }

            score_tensors.push(scores);
            bbox_tensors.push(bboxes);
            kps_tensors.push(kps);
        }

        let mut outputs = Vec::new();
        outputs.extend(score_tensors);
        outputs.extend(bbox_tensors);
        outputs.extend(kps_tensors);

        let dets = decode_multi_output(&outputs, 640, 0.5);
        assert_eq!(dets.len(), 3, "one detection per stride");

        // Each stride produces a detection at different center positions
        // stride 8:  cx = 0.5 * 8  = 4
        // stride 16: cx = 0.5 * 16 = 8
        // stride 32: cx = 0.5 * 32 = 16
        let mut centers: Vec<f32> = dets
            .iter()
            .map(|d| f32::midpoint(d.bbox.x1, d.bbox.x2))
            .collect();
        centers.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((centers[0] - 4.0).abs() < 1e-3);
        assert!((centers[1] - 8.0).abs() < 1e-3);
        assert!((centers[2] - 16.0).abs() < 1e-3);
    }
}
