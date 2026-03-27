// SPDX-License-Identifier: MIT OR Apache-2.0
//! SCRFD face detector (ONNX-based).
//!
//! Wraps an SCRFD ONNX model and implements the [`Detect`] pipeline trait.
//! Preprocessing: Letterbox → ToTensor (scale to 0..1).
//! Postprocessing: anchor-free decode → NMS → rescale.

use scry_llm::backend::cpu::CpuBackend;

use crate::error::{Result, VisionError};
use crate::image::ImageBuffer;
use crate::model::VisionModel;
use crate::pipeline::Detect;
use crate::postprocess::boxes::{decode_anchor_free, rescale_detections};
use crate::postprocess::nms::{nms, Detection};
use crate::transform::resize::Letterbox;
use crate::transform::to_tensor::ToTensor;

/// SCRFD face detector.
///
/// Uses an ONNX model with single-output anchor-free format.
/// For multi-output SCRFD variants, use the ONNX session directly.
pub struct ScrfdDetector {
    model: Box<dyn VisionModel>,
    input_size: u32,
    iou_threshold: f32,
}

impl ScrfdDetector {
    pub fn new(model: Box<dyn VisionModel>, input_size: u32) -> Self {
        Self {
            model,
            input_size,
            iou_threshold: 0.4,
        }
    }

    #[must_use]
    pub fn with_iou_threshold(mut self, iou_threshold: f32) -> Self {
        self.iou_threshold = iou_threshold;
        self
    }

    #[cfg(feature = "onnx")]
    pub fn from_onnx(
        path: impl AsRef<std::path::Path>,
        input_size: u32,
    ) -> Result<Self> {
        let model = crate::model::OnnxModel::from_file(path)?;
        Ok(Self::new(Box::new(model), input_size))
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
        let img = ImageBuffer::from_raw(image.to_vec(), width, height, 3)?;

        let letterbox = Letterbox::new(self.input_size, self.input_size);
        let (padded, info) = letterbox.apply_with_info(&img)?;

        let tensor = ToTensor::new(true).apply::<CpuBackend>(&padded);
        let input_data = tensor.to_vec();
        let input_shape = [1, 3, self.input_size as usize, self.input_size as usize];

        let output = self.model.forward(&input_data, &input_shape)?;

        // SCRFD single-output format: [1, 5, num_proposals]
        // rows 0..4 = cx, cy, w, h; row 4 = face confidence (1 class)
        let rows = 5;
        if output.len() % rows != 0 {
            return Err(VisionError::Inference(format!(
                "output size {} is not a multiple of 5",
                output.len()
            )));
        }
        let num_proposals = output.len() / rows;

        let mut detections = decode_anchor_free(&output, num_proposals, 1, conf_threshold);
        detections = nms(&detections, self.iou_threshold, conf_threshold);

        let inv_scale = 1.0 / info.scale;
        rescale_detections(&mut detections, inv_scale, inv_scale, info.pad_x, info.pad_y);

        Ok(detections)
    }
}
