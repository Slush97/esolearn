// SPDX-License-Identifier: MIT OR Apache-2.0
//! Non-maximum suppression for object detection.
//!
//! Provides three NMS variants:
//! - [`nms`] — standard greedy NMS (per-class)
//! - [`soft_nms`] — soft-NMS with linear or Gaussian decay
//! - [`nms_class_agnostic`] — suppresses across all classes

/// Axis-aligned bounding box in `(x1, y1, x2, y2)` format.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BBox {
    #[must_use]
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    /// Area of the bounding box.
    #[must_use]
    pub fn area(&self) -> f32 {
        (self.x2 - self.x1).max(0.0) * (self.y2 - self.y1).max(0.0)
    }

    /// Intersection-over-union with another box.
    #[must_use]
    pub fn iou(&self, other: &BBox) -> f32 {
        let inter_x1 = self.x1.max(other.x1);
        let inter_y1 = self.y1.max(other.y1);
        let inter_x2 = self.x2.min(other.x2);
        let inter_y2 = self.y2.min(other.y2);
        let inter = (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);
        let union = self.area() + other.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }
}

/// A single detection: bounding box + class + confidence + optional keypoints.
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    pub bbox: BBox,
    pub class_id: u32,
    pub confidence: f32,
    /// Optional keypoints as `[x, y]` pairs (e.g. facial landmarks from SCRFD).
    pub keypoints: Option<Vec<[f32; 2]>>,
}

/// Soft-NMS decay method.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SoftNmsMethod {
    /// Linear decay: `score *= 1 - iou` when iou > threshold.
    Linear,
    /// Gaussian decay: `score *= exp(-iou^2 / sigma)` for all overlaps.
    Gaussian {
        /// Gaussian sigma parameter (typical: 0.5).
        sigma: f32,
    },
}

/// Standard greedy NMS, applied per-class.
///
/// Detections below `score_threshold` are discarded first. Then for each class,
/// boxes are sorted by confidence descending and overlapping boxes with
/// IoU > `iou_threshold` are suppressed.
pub fn nms(detections: &[Detection], iou_threshold: f32, score_threshold: f32) -> Vec<Detection> {
    // Group by class
    let mut by_class: std::collections::HashMap<u32, Vec<&Detection>> =
        std::collections::HashMap::new();
    for d in detections {
        if d.confidence >= score_threshold {
            by_class.entry(d.class_id).or_default().push(d);
        }
    }

    let mut result = Vec::new();
    for (_, dets) in &mut by_class {
        dets.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        let mut suppressed = vec![false; dets.len()];

        for i in 0..dets.len() {
            if suppressed[i] {
                continue;
            }
            result.push(dets[i].clone());
            for j in (i + 1)..dets.len() {
                if !suppressed[j] && dets[i].bbox.iou(&dets[j].bbox) > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }
    }

    result
}

/// Class-agnostic NMS — suppresses across all classes.
///
/// Identical to [`nms`] but treats every detection as the same class.
pub fn nms_class_agnostic(
    detections: &[Detection],
    iou_threshold: f32,
    score_threshold: f32,
) -> Vec<Detection> {
    let mut dets: Vec<&Detection> = detections
        .iter()
        .filter(|d| d.confidence >= score_threshold)
        .collect();
    dets.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut suppressed = vec![false; dets.len()];
    let mut result = Vec::new();

    for i in 0..dets.len() {
        if suppressed[i] {
            continue;
        }
        result.push(dets[i].clone());
        for j in (i + 1)..dets.len() {
            if !suppressed[j] && dets[i].bbox.iou(&dets[j].bbox) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    result
}

/// Soft-NMS — decays confidence of overlapping boxes instead of hard suppression.
///
/// Returns detections with updated confidence scores. Detections whose
/// confidence falls below `score_threshold` after decay are discarded.
pub fn soft_nms(
    detections: &[Detection],
    method: SoftNmsMethod,
    iou_threshold: f32,
    score_threshold: f32,
) -> Vec<Detection> {
    let mut dets: Vec<Detection> = detections
        .iter()
        .filter(|d| d.confidence >= score_threshold)
        .cloned()
        .collect();
    dets.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut result = Vec::new();

    while !dets.is_empty() {
        // Pick the highest-confidence detection
        let best = dets.remove(0);
        result.push(best.clone());

        // Decay remaining detections
        for d in &mut dets {
            let iou = best.bbox.iou(&d.bbox);
            match method {
                SoftNmsMethod::Linear => {
                    if iou > iou_threshold {
                        d.confidence *= 1.0 - iou;
                    }
                }
                SoftNmsMethod::Gaussian { sigma } => {
                    d.confidence *= (-iou * iou / sigma).exp();
                }
            }
        }

        // Remove detections that fell below threshold
        dets.retain(|d| d.confidence >= score_threshold);
        // Re-sort since scores changed
        dets.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x1: f32, y1: f32, x2: f32, y2: f32, class_id: u32, conf: f32) -> Detection {
        Detection {
            bbox: BBox::new(x1, y1, x2, y2),
            class_id,
            confidence: conf,
            keypoints: None,
        }
    }

    #[test]
    fn iou_identical_boxes() {
        let b = BBox::new(0.0, 0.0, 10.0, 10.0);
        assert!((b.iou(&b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_no_overlap() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(20.0, 20.0, 30.0, 30.0);
        assert!((a.iou(&b)).abs() < 1e-6);
    }

    #[test]
    fn iou_partial_overlap() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(5.0, 5.0, 15.0, 15.0);
        // Intersection: 5×5 = 25, union: 100+100-25 = 175
        let expected = 25.0 / 175.0;
        assert!((a.iou(&b) - expected).abs() < 1e-6);
    }

    #[test]
    fn nms_suppresses_overlapping_same_class() {
        let dets = vec![
            det(0.0, 0.0, 10.0, 10.0, 0, 0.9),
            det(1.0, 1.0, 11.0, 11.0, 0, 0.8), // highly overlapping, lower conf
            det(50.0, 50.0, 60.0, 60.0, 0, 0.7), // no overlap
        ];
        let result = nms(&dets, 0.5, 0.0);
        assert_eq!(result.len(), 2);
        assert!((result[0].confidence - 0.9).abs() < 1e-6);
        assert!((result[1].confidence - 0.7).abs() < 1e-6);
    }

    #[test]
    fn nms_different_classes_not_suppressed() {
        let dets = vec![
            det(0.0, 0.0, 10.0, 10.0, 0, 0.9),
            det(1.0, 1.0, 11.0, 11.0, 1, 0.8), // same location, different class
        ];
        let result = nms(&dets, 0.5, 0.0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn nms_score_threshold_filters() {
        let dets = vec![
            det(0.0, 0.0, 10.0, 10.0, 0, 0.9),
            det(50.0, 50.0, 60.0, 60.0, 0, 0.1),
        ];
        let result = nms(&dets, 0.5, 0.5);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn nms_class_agnostic_suppresses_across_classes() {
        let dets = vec![
            det(0.0, 0.0, 10.0, 10.0, 0, 0.9),
            det(1.0, 1.0, 11.0, 11.0, 1, 0.8), // same location, different class
        ];
        let result = nms_class_agnostic(&dets, 0.5, 0.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].class_id, 0);
    }

    #[test]
    fn soft_nms_linear_decays_confidence() {
        let dets = vec![
            det(0.0, 0.0, 10.0, 10.0, 0, 0.9),
            det(1.0, 1.0, 11.0, 11.0, 0, 0.8),
        ];
        let result = soft_nms(&dets, SoftNmsMethod::Linear, 0.3, 0.0);
        assert_eq!(result.len(), 2);
        // First detection keeps its score
        assert!((result[0].confidence - 0.9).abs() < 1e-6);
        // Second detection has decayed confidence
        assert!(result[1].confidence < 0.8);
    }

    #[test]
    fn soft_nms_gaussian_decays_confidence() {
        let dets = vec![
            det(0.0, 0.0, 10.0, 10.0, 0, 0.9),
            det(1.0, 1.0, 11.0, 11.0, 0, 0.8),
        ];
        let result = soft_nms(
            &dets,
            SoftNmsMethod::Gaussian { sigma: 0.5 },
            0.3,
            0.0,
        );
        assert_eq!(result.len(), 2);
        assert!((result[0].confidence - 0.9).abs() < 1e-6);
        assert!(result[1].confidence < 0.8);
    }

    #[test]
    fn soft_nms_drops_below_threshold() {
        let dets = vec![
            det(0.0, 0.0, 10.0, 10.0, 0, 0.9),
            det(0.0, 0.0, 10.0, 10.0, 0, 0.3), // identical box, will decay to ~0
        ];
        let result = soft_nms(&dets, SoftNmsMethod::Linear, 0.3, 0.2);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn nms_empty_input() {
        assert!(nms(&[], 0.5, 0.0).is_empty());
        assert!(nms_class_agnostic(&[], 0.5, 0.0).is_empty());
        assert!(soft_nms(&[], SoftNmsMethod::Linear, 0.5, 0.0).is_empty());
    }
}
