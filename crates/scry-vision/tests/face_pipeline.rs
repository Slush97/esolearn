// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration test: full face recognition pipeline.
//!
//! Demonstrates the transferrable pattern for testing vision pipelines:
//!
//! 1. Inject mock models via `Box<dyn VisionModel>` (no real weights needed)
//! 2. Use real transforms (crop, affine, resize) to validate data flow
//! 3. Assert on pipeline outputs (bboxes, embedding similarity)
//!
//! This is the reference test for cloudbox-vision's face pipeline integration.

use scry_vision::error::Result;
use scry_vision::image::ImageBuffer;
use scry_vision::model::VisionModel;
use scry_vision::models::{ArcFaceEmbedder, ScrfdDetector};
use scry_vision::pipeline::{Detect, Embed};
use scry_vision::postprocess::embedding::cosine_similarity;
use scry_vision::transform::{Crop, ImageTransform};

// ── Mock model (inline — integration tests can't access pub(crate)) ──────────

struct MockModel {
    output: Vec<f32>,
}

impl VisionModel for MockModel {
    fn forward(&self, _input: &[f32], _input_shape: &[usize]) -> Result<Vec<f32>> {
        Ok(self.output.clone())
    }

    fn output_shape(&self, _input_shape: &[usize]) -> Vec<usize> {
        vec![self.output.len()]
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Create a synthetic 640×640 RGB image with two distinct "face" regions.
///
/// Region A (around 200,200): bright pixels (200,200,200)
/// Region B (around 400,400): dark pixels (50,50,50)
/// Background: mid-gray (128,128,128)
fn synthetic_image() -> (Vec<u8>, u32, u32) {
    let (w, h) = (640u32, 640u32);
    let mut data = vec![128u8; (w * h * 3) as usize];

    // Paint face region A: 160..240 x 160..240
    for y in 160..240u32 {
        for x in 160..240u32 {
            let idx = ((y * w + x) * 3) as usize;
            data[idx] = 200;
            data[idx + 1] = 200;
            data[idx + 2] = 200;
        }
    }

    // Paint face region B: 360..440 x 360..440
    for y in 360..440u32 {
        for x in 360..440u32 {
            let idx = ((y * w + x) * 3) as usize;
            data[idx] = 50;
            data[idx + 1] = 50;
            data[idx + 2] = 50;
        }
    }

    (data, w, h)
}

/// SCRFD mock output with two detected faces.
/// Format: [5, 2] row-major (cx, cy, w, h, confidence).
fn scrfd_two_faces_output() -> Vec<f32> {
    #[rustfmt::skip]
    let output = vec![
        200.0, 400.0, // cx
        200.0, 400.0, // cy
         80.0,  80.0, // w
         80.0,  80.0, // h
          0.95,  0.90, // confidence
    ];
    output
}

/// Create an ArcFace mock that returns a known 512-dim embedding.
fn arcface_mock(embedding: Vec<f32>) -> ArcFaceEmbedder {
    ArcFaceEmbedder::new(Box::new(MockModel { output: embedding }), 112, 512)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[test]
fn full_pipeline_detect_crop_embed_compare() {
    let (image_data, w, h) = synthetic_image();

    // ── Step 1: Detect faces ─────────────────────────────────────────────
    let scrfd_model = MockModel {
        output: scrfd_two_faces_output(),
    };
    let detector = ScrfdDetector::new(Box::new(scrfd_model), 640);
    let detections = detector.detect(&image_data, w, h, 0.5).unwrap();

    assert_eq!(detections.len(), 2, "expected 2 face detections");

    // Sort by x position for deterministic ordering
    let mut dets = detections;
    dets.sort_by(|a, b| a.bbox.x1.partial_cmp(&b.bbox.x1).unwrap());

    // Verify bbox geometry for face A (centered at 200, 200)
    let face_a = &dets[0];
    let cx_a = (face_a.bbox.x1 + face_a.bbox.x2) / 2.0;
    let cy_a = (face_a.bbox.y1 + face_a.bbox.y2) / 2.0;
    assert!((cx_a - 200.0).abs() < 2.0, "face A cx={cx_a}, expected ~200");
    assert!((cy_a - 200.0).abs() < 2.0, "face A cy={cy_a}, expected ~200");

    // ── Step 2: Crop face regions ────────────────────────────────────────
    let img = ImageBuffer::from_raw(image_data.clone(), w, h, 3).unwrap();

    let crop_a = crop_from_bbox(&img, face_a);
    assert!(crop_a.width > 0 && crop_a.height > 0);

    let face_b = &dets[1];
    let crop_b = crop_from_bbox(&img, face_b);
    assert!(crop_b.width > 0 && crop_b.height > 0);

    // ── Step 3: Embed each face ──────────────────────────────────────────
    //
    // Use distinct mock embeddings to simulate what a real ArcFace would
    // produce: same person → similar direction, different person → different.

    // Embedding for "person A"
    let mut emb_a = vec![0.0f32; 512];
    emb_a[0] = 1.0;
    emb_a[1] = 0.5;
    emb_a[2] = 0.3;

    // Embedding for "person B" (orthogonal direction)
    let mut emb_b = vec![0.0f32; 512];
    emb_b[100] = 1.0;
    emb_b[101] = 0.5;
    emb_b[102] = 0.3;

    // "Same person A" — slight variation
    let mut emb_a2 = vec![0.0f32; 512];
    emb_a2[0] = 0.98;
    emb_a2[1] = 0.52;
    emb_a2[2] = 0.31;

    let embedder_a = arcface_mock(emb_a);
    let embedder_b = arcface_mock(emb_b);
    let embedder_a2 = arcface_mock(emb_a2);

    let result_a = embedder_a
        .embed(&crop_a.data, crop_a.width, crop_a.height)
        .unwrap();
    let result_b = embedder_b
        .embed(&crop_b.data, crop_b.width, crop_b.height)
        .unwrap();
    let result_a2 = embedder_a2
        .embed(&crop_a.data, crop_a.width, crop_a.height)
        .unwrap();

    // ── Step 4: Verify similarity ────────────────────────────────────────
    let same_person = cosine_similarity(&result_a, &result_a2);
    let diff_person = cosine_similarity(&result_a, &result_b);

    assert!(
        same_person > 0.95,
        "same person similarity={same_person}, expected > 0.95"
    );
    assert!(
        diff_person < 0.05,
        "different person similarity={diff_person}, expected < 0.05"
    );
}

#[test]
fn pipeline_no_faces_detected() {
    // SCRFD returns nothing above threshold
    #[rustfmt::skip]
    let output = vec![
        320.0, // cx
        320.0, // cy
        100.0, // w
        100.0, // h
          0.1, // confidence — below threshold
    ];

    let detector = ScrfdDetector::new(Box::new(MockModel { output }), 640);
    let image = vec![128u8; 640 * 640 * 3];
    let dets = detector.detect(&image, 640, 640, 0.5).unwrap();

    assert!(dets.is_empty(), "should skip embedding when no faces found");
    // In a real pipeline (cloudbox), this is where you'd short-circuit:
    // if dets.is_empty() { return Ok(()); }
}

#[test]
fn embedding_normalization_invariant() {
    // Regardless of the raw magnitude, output must be unit-norm.
    // This is critical for cosine similarity to work correctly.
    let magnitudes = [0.001, 1.0, 100.0, 100_000.0];

    for &mag in &magnitudes {
        let mut raw = vec![0.0f32; 512];
        raw[0] = mag;
        raw[1] = mag * 0.5;

        let embedder = arcface_mock(raw);
        let face = vec![128u8; 112 * 112 * 3];
        let emb = embedder.embed(&face, 112, 112).unwrap();

        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "magnitude={mag} → norm={norm}, expected 1.0"
        );
    }
}

// ── Utility ──────────────────────────────────────────────────────────────────

/// Crop a face region from an image using a detection bbox.
///
/// Clamps to image bounds — the same logic cloudbox-vision would use.
fn crop_from_bbox(
    img: &ImageBuffer,
    det: &scry_vision::pipeline::Detection,
) -> ImageBuffer {
    let x = (det.bbox.x1.max(0.0) as u32).min(img.width - 1);
    let y = (det.bbox.y1.max(0.0) as u32).min(img.height - 1);
    let x2 = (det.bbox.x2.max(0.0) as u32).min(img.width);
    let y2 = (det.bbox.y2.max(0.0) as u32).min(img.height);
    let w = x2.saturating_sub(x).max(1);
    let h = y2.saturating_sub(y).max(1);

    Crop::new(x, y, w, h).apply(img).unwrap()
}
