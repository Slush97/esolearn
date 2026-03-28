// SPDX-License-Identifier: MIT OR Apache-2.0
//! YOLO bounding box decoding.
//!
//! Converts raw model outputs to `(x1, y1, x2, y2)` bounding boxes with class
//! scores. Supports both anchor-free (YOLOv8/v11) and anchor-based (YOLOv5)
//! output formats.

use super::nms::{BBox, Detection};

/// Decode anchor-free YOLO outputs (YOLOv8, YOLOv11, YOLO-NAS).
///
/// # Input format
///
/// Raw output tensor shape: `[4 + num_classes, num_proposals]`
/// (transposed from the typical ONNX output `[1, 4 + num_classes, num_proposals]`
///  — the caller strips the batch dim).
///
/// - Rows 0..4: `cx, cy, w, h` (in input image pixels)
/// - Rows 4..: class confidence scores (already sigmoided by the model)
///
/// # Arguments
///
/// - `output` — flat f32 slice of shape `[4 + num_classes, num_proposals]`, row-major
/// - `num_proposals` — number of candidate boxes
/// - `num_classes` — number of classes
/// - `conf_threshold` — minimum confidence to keep
pub fn decode_anchor_free(
    output: &[f32],
    num_proposals: usize,
    num_classes: usize,
    conf_threshold: f32,
) -> Vec<Detection> {
    let rows = 4 + num_classes;
    assert_eq!(
        output.len(),
        rows * num_proposals,
        "output length {} != {} rows * {} proposals",
        output.len(),
        rows,
        num_proposals
    );

    let mut detections = Vec::new();

    for j in 0..num_proposals {
        let cx = output[0 * num_proposals + j];
        let cy = output[1 * num_proposals + j];
        let w = output[2 * num_proposals + j];
        let h = output[3 * num_proposals + j];

        // Find best class
        let mut best_class = 0u32;
        let mut best_score = f32::NEG_INFINITY;
        for c in 0..num_classes {
            let score = output[(4 + c) * num_proposals + j];
            if score > best_score {
                best_score = score;
                best_class = c as u32;
            }
        }

        if best_score < conf_threshold {
            continue;
        }

        let x1 = cx - w * 0.5;
        let y1 = cy - h * 0.5;
        let x2 = cx + w * 0.5;
        let y2 = cy + h * 0.5;

        detections.push(Detection {
            bbox: BBox::new(x1, y1, x2, y2),
            class_id: best_class,
            confidence: best_score,
            keypoints: None,
        });
    }

    detections
}

/// Anchor specification for YOLOv5-style decoding.
#[derive(Clone, Debug)]
pub struct Anchor {
    pub width: f32,
    pub height: f32,
}

/// Grid cell origin for anchor-based decoding.
#[derive(Clone, Debug)]
pub struct GridCell {
    /// X position of grid cell (column index).
    pub gx: f32,
    /// Y position of grid cell (row index).
    pub gy: f32,
    /// Stride at this feature level (e.g., 8, 16, 32).
    pub stride: f32,
    /// Anchor for this cell.
    pub anchor: Anchor,
}

/// Decode anchor-based YOLO outputs (YOLOv5).
///
/// # Input format
///
/// Raw output tensor shape: `[num_proposals, 5 + num_classes]` (row-major).
///
/// - Columns 0..4: `tx, ty, tw, th` (raw, pre-sigmoid/exp)
/// - Column 4: objectness score (raw, pre-sigmoid)
/// - Columns 5..: class scores (raw, pre-sigmoid)
///
/// Decoding applies the YOLOv5 formula:
/// ```text
/// bx = (2 * sigmoid(tx) - 0.5 + gx) * stride
/// by = (2 * sigmoid(ty) - 0.5 + gy) * stride
/// bw = (2 * sigmoid(tw))^2 * anchor_w
/// bh = (2 * sigmoid(th))^2 * anchor_h
/// ```
///
/// # Arguments
///
/// - `output` — flat f32 slice of shape `[num_proposals, 5 + num_classes]`, row-major
/// - `grid` — one [`GridCell`] per proposal specifying grid position, stride, and anchor
/// - `num_classes` — number of classes
/// - `conf_threshold` — minimum `objectness * class_score` to keep
pub fn decode_anchor_based(
    output: &[f32],
    grid: &[GridCell],
    num_classes: usize,
    conf_threshold: f32,
) -> Vec<Detection> {
    let cols = 5 + num_classes;
    let num_proposals = grid.len();
    assert_eq!(
        output.len(),
        num_proposals * cols,
        "output length {} != {} proposals * {} cols",
        output.len(),
        num_proposals,
        cols
    );

    let mut detections = Vec::new();

    for i in 0..num_proposals {
        let row = &output[i * cols..(i + 1) * cols];
        let obj = sigmoid(row[4]);
        if obj < conf_threshold {
            continue;
        }

        // Find best class
        let mut best_class = 0u32;
        let mut best_cls_score = f32::NEG_INFINITY;
        for c in 0..num_classes {
            let s = sigmoid(row[5 + c]);
            if s > best_cls_score {
                best_cls_score = s;
                best_class = c as u32;
            }
        }

        let confidence = obj * best_cls_score;
        if confidence < conf_threshold {
            continue;
        }

        let g = &grid[i];
        let bx = (2.0 * sigmoid(row[0]) - 0.5 + g.gx) * g.stride;
        let by = (2.0 * sigmoid(row[1]) - 0.5 + g.gy) * g.stride;
        let s_tw = 2.0 * sigmoid(row[2]);
        let s_th = 2.0 * sigmoid(row[3]);
        let bw = s_tw * s_tw * g.anchor.width;
        let bh = s_th * s_th * g.anchor.height;

        let x1 = bx - bw * 0.5;
        let y1 = by - bh * 0.5;
        let x2 = bx + bw * 0.5;
        let y2 = by + bh * 0.5;

        detections.push(Detection {
            bbox: BBox::new(x1, y1, x2, y2),
            class_id: best_class,
            confidence,
            keypoints: None,
        });
    }

    detections
}

/// Rescale detections from model input coordinates back to the original image.
///
/// Useful after letterbox/resize preprocessing where the model operates on a
/// padded/resized image and detections need to be mapped back.
///
/// # Arguments
///
/// - `detections` — detections in model input coordinates
/// - `scale_x`, `scale_y` — `original_width / model_input_width` (etc.)
/// - `pad_x`, `pad_y` — padding offset added during letterbox
pub fn rescale_detections(
    detections: &mut [Detection],
    scale_x: f32,
    scale_y: f32,
    pad_x: f32,
    pad_y: f32,
) {
    for d in detections.iter_mut() {
        d.bbox.x1 = (d.bbox.x1 - pad_x) * scale_x;
        d.bbox.y1 = (d.bbox.y1 - pad_y) * scale_y;
        d.bbox.x2 = (d.bbox.x2 - pad_x) * scale_x;
        d.bbox.y2 = (d.bbox.y2 - pad_y) * scale_y;

        if let Some(kps) = &mut d.keypoints {
            for kp in kps.iter_mut() {
                kp[0] = (kp[0] - pad_x) * scale_x;
                kp[1] = (kp[1] - pad_y) * scale_y;
            }
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_anchor_free_basic() {
        // 2 proposals, 3 classes → output shape [7, 2]
        // Proposal 0: cx=50, cy=60, w=20, h=30, class scores [0.9, 0.1, 0.2]
        // Proposal 1: cx=100, cy=100, w=40, h=40, class scores [0.1, 0.05, 0.01]
        #[rustfmt::skip]
        let output = vec![
            50.0, 100.0,   // cx
            60.0, 100.0,   // cy
            20.0, 40.0,    // w
            30.0, 40.0,    // h
            0.9, 0.1,      // class 0
            0.1, 0.05,     // class 1
            0.2, 0.01,     // class 2
        ];
        let dets = decode_anchor_free(&output, 2, 3, 0.15);
        assert_eq!(dets.len(), 1); // Only proposal 0 above threshold
        assert_eq!(dets[0].class_id, 0);
        assert!((dets[0].confidence - 0.9).abs() < 1e-6);
        // cx=50, w=20 → x1=40, x2=60
        assert!((dets[0].bbox.x1 - 40.0).abs() < 1e-6);
        assert!((dets[0].bbox.x2 - 60.0).abs() < 1e-6);
        // cy=60, h=30 → y1=45, y2=75
        assert!((dets[0].bbox.y1 - 45.0).abs() < 1e-6);
        assert!((dets[0].bbox.y2 - 75.0).abs() < 1e-6);
    }

    #[test]
    fn decode_anchor_free_all_below_threshold() {
        #[rustfmt::skip]
        let output = vec![
            50.0,  // cx
            60.0,  // cy
            20.0,  // w
            30.0,  // h
            0.1,   // class 0
        ];
        let dets = decode_anchor_free(&output, 1, 1, 0.5);
        assert!(dets.is_empty());
    }

    #[test]
    fn decode_anchor_based_basic() {
        // 1 proposal, 2 classes
        // tx=0, ty=0, tw=0, th=0, obj_raw=5.0 (sigmoid≈0.993), cls=[5.0, 0.0]
        let output = vec![0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0];
        let grid = vec![GridCell {
            gx: 0.0,
            gy: 0.0,
            stride: 8.0,
            anchor: Anchor {
                width: 10.0,
                height: 13.0,
            },
        }];
        let dets = decode_anchor_based(&output, &grid, 2, 0.5);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class_id, 0);
        // sigmoid(0) = 0.5
        // bx = (2*0.5 - 0.5 + 0.0) * 8 = 0.5 * 8 = 4.0
        // bw = (2*0.5)^2 * 10 = 1.0 * 10 = 10.0
        assert!((dets[0].bbox.x1 - (4.0 - 5.0)).abs() < 1e-4);
        assert!((dets[0].bbox.x2 - (4.0 + 5.0)).abs() < 1e-4);
    }

    #[test]
    fn decode_anchor_based_below_threshold() {
        let output = vec![0.0, 0.0, 0.0, 0.0, -10.0, 0.0]; // very low objectness
        let grid = vec![GridCell {
            gx: 0.0,
            gy: 0.0,
            stride: 8.0,
            anchor: Anchor {
                width: 10.0,
                height: 10.0,
            },
        }];
        let dets = decode_anchor_based(&output, &grid, 1, 0.5);
        assert!(dets.is_empty());
    }

    #[test]
    fn rescale_detections_identity() {
        let mut dets = vec![Detection {
            bbox: BBox::new(10.0, 20.0, 30.0, 40.0),
            class_id: 0,
            confidence: 0.9,
            keypoints: None,
        }];
        rescale_detections(&mut dets, 1.0, 1.0, 0.0, 0.0);
        assert!((dets[0].bbox.x1 - 10.0).abs() < 1e-6);
    }

    #[test]
    fn rescale_detections_with_padding_and_scale() {
        let mut dets = vec![Detection {
            bbox: BBox::new(20.0, 20.0, 120.0, 120.0),
            class_id: 0,
            confidence: 0.9,
            keypoints: None,
        }];
        // Model input was 640×640, original was 1280×960
        // scale_x = 1280/640 = 2.0, scale_y = 960/480 = 2.0
        // pad_x = 0, pad_y = 80 (letterbox padding)
        rescale_detections(&mut dets, 2.0, 2.0, 0.0, 80.0);
        assert!((dets[0].bbox.x1 - 40.0).abs() < 1e-4);
        assert!((dets[0].bbox.y1 - (-120.0)).abs() < 1e-4);
    }

    #[test]
    fn sigmoid_values() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);
    }
}
