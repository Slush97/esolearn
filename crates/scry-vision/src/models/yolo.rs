// SPDX-License-Identifier: MIT OR Apache-2.0
//! YOLO object detector (v8/v11, anchor-free).
//!
//! Composes the full detection pipeline:
//! Letterbox → ToTensor → forward → decode → NMS → rescale.

use scry_llm::backend::cpu::CpuBackend;

use crate::error::{Result, VisionError};
use crate::image::ImageBuffer;
use crate::model::VisionModel;
use crate::pipeline::Detect;
use crate::postprocess::boxes::{decode_anchor_free, rescale_detections};
use crate::postprocess::nms::{nms, Detection};
use crate::transform::resize::Letterbox;
use crate::transform::to_tensor::ToTensor;

/// YOLO object detector for anchor-free models (YOLOv8, YOLOv11).
///
/// Bundles the complete inference pipeline from raw RGB bytes to detections
/// in original image coordinates:
///
/// 1. **Letterbox** — aspect-preserving resize + pad to `input_size × input_size`
/// 2. **ToTensor** — HWC u8 → CHW f32, scaled to 0..1
/// 3. **Forward** — model inference via [`VisionModel`]
/// 4. **Decode** — anchor-free box decoding (`cx,cy,w,h` → `x1,y1,x2,y2`)
/// 5. **NMS** — per-class non-maximum suppression
/// 6. **Rescale** — map coordinates back to original image space
pub struct YoloDetector {
    model: Box<dyn VisionModel>,
    input_size: u32,
    num_classes: usize,
    iou_threshold: f32,
}

impl YoloDetector {
    /// Create a new YOLO detector.
    ///
    /// - `model` — any [`VisionModel`] (ONNX or native)
    /// - `input_size` — square input resolution (e.g., 640)
    /// - `num_classes` — number of detection classes (e.g., 80 for COCO)
    pub fn new(model: Box<dyn VisionModel>, input_size: u32, num_classes: usize) -> Self {
        Self {
            model,
            input_size,
            num_classes,
            iou_threshold: 0.45,
        }
    }

    /// Set the IoU threshold for NMS (default: 0.45).
    #[must_use]
    pub fn with_iou_threshold(mut self, iou_threshold: f32) -> Self {
        self.iou_threshold = iou_threshold;
        self
    }

    /// Load a YOLO detector from an ONNX model file.
    #[cfg(feature = "onnx")]
    pub fn from_onnx(
        path: impl AsRef<std::path::Path>,
        input_size: u32,
        num_classes: usize,
    ) -> Result<Self> {
        let model = crate::model::OnnxModel::from_file(path)?;
        Ok(Self::new(Box::new(model), input_size, num_classes))
    }
}

impl Detect for YoloDetector {
    fn detect(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        conf_threshold: f32,
    ) -> Result<Vec<Detection>> {
        let img = ImageBuffer::from_raw(image.to_vec(), width, height, 3)?;

        // Letterbox to model input size
        let letterbox = Letterbox::new(self.input_size, self.input_size);
        let (padded, info) = letterbox.apply_with_info(&img)?;

        // HWC u8 → CHW f32, scaled to 0..1
        let tensor = ToTensor::new(true).apply::<CpuBackend>(&padded);
        let input_data = tensor.to_vec();
        let input_shape = [1, 3, self.input_size as usize, self.input_size as usize];

        // Forward pass
        let output = self.model.forward(&input_data, &input_shape)?;

        // Decode anchor-free output: [1, 4+num_classes, num_proposals] flattened
        let rows = 4 + self.num_classes;
        if output.len() % rows != 0 {
            return Err(VisionError::Inference(format!(
                "output size {} is not a multiple of {} (4 + {} classes)",
                output.len(),
                rows,
                self.num_classes
            )));
        }
        let num_proposals = output.len() / rows;

        let mut detections =
            decode_anchor_free(&output, num_proposals, self.num_classes, conf_threshold);

        // NMS
        detections = nms(&detections, self.iou_threshold, conf_threshold);

        // Rescale from model input coordinates to original image coordinates
        let inv_scale = 1.0 / info.scale;
        rescale_detections(&mut detections, inv_scale, inv_scale, info.pad_x, info.pad_y);

        Ok(detections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock model that returns a fixed YOLO-format output.
    struct MockYoloModel {
        output: Vec<f32>,
    }

    impl VisionModel for MockYoloModel {
        fn forward(&self, _input: &[f32], _input_shape: &[usize]) -> Result<Vec<f32>> {
            Ok(self.output.clone())
        }

        fn output_shape(&self, _input_shape: &[usize]) -> Vec<usize> {
            vec![]
        }
    }

    #[test]
    fn detect_with_mock_model() {
        // 2 proposals, 2 classes → output shape [6, 2] row-major
        // Proposal 0: cx=320, cy=320, w=100, h=100, class0=0.9, class1=0.1
        // Proposal 1: cx=100, cy=100, w=50, h=50, class0=0.1, class1=0.8
        #[rustfmt::skip]
        let output = vec![
            320.0, 100.0,   // cx
            320.0, 100.0,   // cy
            100.0,  50.0,   // w
            100.0,  50.0,   // h
              0.9,   0.1,   // class 0
              0.1,   0.8,   // class 1
        ];

        let model = MockYoloModel { output };
        let detector = YoloDetector::new(Box::new(model), 640, 2);

        // 640×640 input → letterbox is identity (scale=1, pad=0)
        let image = vec![128u8; 640 * 640 * 3];
        let dets = detector.detect(&image, 640, 640, 0.5).unwrap();

        assert_eq!(dets.len(), 2);

        let mut sorted = dets.clone();
        sorted.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        // Best: class 0, conf 0.9, cx=320 w=100 → x1=270, x2=370
        assert_eq!(sorted[0].class_id, 0);
        assert!((sorted[0].confidence - 0.9).abs() < 1e-5);
        assert!((sorted[0].bbox.x1 - 270.0).abs() < 1.0);
        assert!((sorted[0].bbox.x2 - 370.0).abs() < 1.0);

        // Second: class 1, conf 0.8
        assert_eq!(sorted[1].class_id, 1);
        assert!((sorted[1].confidence - 0.8).abs() < 1e-5);
    }

    #[test]
    fn detect_with_letterbox_rescaling() {
        // Input: 1280×720 → Letterbox to 640×640
        //   scale = min(640/1280, 640/720) = 0.5
        //   new_w=640, new_h=360, pad_x=0, pad_y=140
        //
        // Detection at model center (320, 320) maps to original:
        //   x = (320 - 0) / 0.5 = 640
        //   y = (320 - 140) / 0.5 = 360
        #[rustfmt::skip]
        let output = vec![
            320.0,  // cx
            320.0,  // cy
            100.0,  // w
            100.0,  // h
              0.9,  // class 0
        ];

        let model = MockYoloModel { output };
        let detector = YoloDetector::new(Box::new(model), 640, 1);

        let image = vec![128u8; 1280 * 720 * 3];
        let dets = detector.detect(&image, 1280, 720, 0.5).unwrap();

        assert_eq!(dets.len(), 1);
        let d = &dets[0];

        let cx = (d.bbox.x1 + d.bbox.x2) / 2.0;
        let cy = (d.bbox.y1 + d.bbox.y2) / 2.0;
        assert!((cx - 640.0).abs() < 2.0);
        assert!((cy - 360.0).abs() < 2.0);

        // w/h in original coords: 100 / 0.5 = 200
        let w = d.bbox.x2 - d.bbox.x1;
        let h = d.bbox.y2 - d.bbox.y1;
        assert!((w - 200.0).abs() < 2.0);
        assert!((h - 200.0).abs() < 2.0);
    }

    #[test]
    fn detect_filters_low_confidence() {
        #[rustfmt::skip]
        let output = vec![
            320.0,  // cx
            320.0,  // cy
            100.0,  // w
            100.0,  // h
              0.3,  // class 0 — below threshold
        ];

        let model = MockYoloModel { output };
        let detector = YoloDetector::new(Box::new(model), 640, 1);

        let image = vec![128u8; 640 * 640 * 3];
        let dets = detector.detect(&image, 640, 640, 0.5).unwrap();
        assert!(dets.is_empty());
    }

    #[test]
    fn detect_nms_suppresses_overlapping() {
        // Two nearly identical boxes, same class → NMS keeps only the best
        #[rustfmt::skip]
        let output = vec![
            320.0, 325.0,   // cx
            320.0, 325.0,   // cy
            100.0, 100.0,   // w
            100.0, 100.0,   // h
              0.9,   0.8,   // class 0
        ];

        let model = MockYoloModel { output };
        let detector = YoloDetector::new(Box::new(model), 640, 1);

        let image = vec![128u8; 640 * 640 * 3];
        let dets = detector.detect(&image, 640, 640, 0.5).unwrap();

        assert_eq!(dets.len(), 1);
        assert!((dets[0].confidence - 0.9).abs() < 1e-5);
    }

    /// Integration test with a real YOLOv8n ONNX model.
    ///
    /// Set `YOLO_MODEL_PATH` env var or place the model at `testdata/yolov8n.onnx`.
    #[test]
    #[ignore]
    #[cfg(feature = "onnx")]
    fn detect_with_real_yolov8n() {
        let model_path = std::env::var("YOLO_MODEL_PATH").unwrap_or_else(|_| {
            format!("{}/testdata/yolov8n.onnx", env!("CARGO_MANIFEST_DIR"))
        });

        if !std::path::Path::new(&model_path).exists() {
            eprintln!("Skipping: model not found at {model_path}");
            return;
        }

        let detector = YoloDetector::from_onnx(&model_path, 640, 80).unwrap();

        // Solid-color image won't produce meaningful detections, but must not crash
        let image = vec![128u8; 640 * 480 * 3];
        let dets = detector.detect(&image, 640, 480, 0.25).unwrap();

        for d in &dets {
            assert!(d.confidence >= 0.25);
            assert!(d.bbox.x1 < d.bbox.x2);
            assert!(d.bbox.y1 < d.bbox.y2);
            assert!(d.class_id < 80);
        }
    }
}
