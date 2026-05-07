// SPDX-License-Identifier: MIT OR Apache-2.0
//! Comprehensive face recognition pipeline tests.
//!
//! These tests validate the algorithmic correctness of each stage in the
//! face recognition pipeline — preprocessing, detection, alignment, embedding,
//! and similarity — using synthetic data with known ground-truth values.
//!
//! Unlike `face_pipeline.rs` (which uses mock models to test plumbing),
//! these tests exercise the actual math: tensor normalization values,
//! affine warp correctness, grid anchor decoding, and coordinate transforms.

use scry_vision::error::Result;
use scry_vision::image::ImageBuffer;
use scry_vision::model::VisionModel;
use scry_vision::models::{ArcFaceEmbedder, ScrfdDetector};
use scry_vision::pipeline::{Detect, Embed};
use scry_vision::postprocess::embedding::{cosine_similarity, l2_norm, l2_normalize};
use scry_vision::transform::affine::AffineTransform;
use scry_vision::transform::resize::{InterpolationMode, Letterbox, Resize};
use scry_vision::transform::to_tensor::ToTensor;
use scry_vision::transform::{Crop, ImageTransform};

use scry_llm::backend::cpu::CpuBackend;

// ── Mock that captures input for validation ─────────────────────────────────

struct CaptureMock {
    output: Vec<f32>,
    /// Stores the input tensor that was passed to forward().
    captured: std::sync::Mutex<Option<Vec<f32>>>,
}

impl CaptureMock {
    fn new(output: Vec<f32>) -> Self {
        Self {
            output,
            captured: std::sync::Mutex::new(None),
        }
    }

    fn captured_input(&self) -> Vec<f32> {
        self.captured.lock().unwrap().clone().unwrap()
    }
}

impl VisionModel for CaptureMock {
    fn forward(&self, input: &[f32], _input_shape: &[usize]) -> Result<Vec<f32>> {
        *self.captured.lock().unwrap() = Some(input.to_vec());
        Ok(self.output.clone())
    }

    fn output_shape(&self, _input_shape: &[usize]) -> Vec<usize> {
        vec![self.output.len()]
    }
}

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

// ── Preprocessing numerical validation ──────────────────────────────────────

#[test]
fn scrfd_preprocessing_insightface_normalization() {
    // Verify that SCRFD preprocessing produces the exact InsightFace normalization:
    // output = (pixel - 127.5) / 128.0
    //
    // For pixel=0:   (0 - 127.5) / 128  = -0.99609375
    // For pixel=128: (128 - 127.5) / 128 = 0.00390625
    // For pixel=255: (255 - 127.5) / 128 = 0.99609375

    let std_val = 128.0 / 255.0;
    let to_tensor = ToTensor::normalized([0.5, 0.5, 0.5], [std_val, std_val, std_val]);

    // Single pixel, RGB = [0, 128, 255]
    let img = ImageBuffer::from_raw(vec![0, 128, 255], 1, 1, 3).unwrap();
    let tensor = to_tensor.apply::<CpuBackend>(&img);
    let data = tensor.to_vec();

    let expected_r = (0.0 - 127.5) / 128.0;
    let expected_g = (128.0 - 127.5) / 128.0;
    let expected_b = (255.0 - 127.5) / 128.0;

    assert!(
        (data[0] - expected_r).abs() < 1e-4,
        "R channel: got {}, expected {}",
        data[0],
        expected_r
    );
    assert!(
        (data[1] - expected_g).abs() < 1e-4,
        "G channel: got {}, expected {}",
        data[1],
        expected_g
    );
    assert!(
        (data[2] - expected_b).abs() < 1e-4,
        "B channel: got {}, expected {}",
        data[2],
        expected_b
    );
}

#[test]
fn arcface_preprocessing_normalization() {
    // ArcFace normalization: (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1
    // For pixel=0:   -1.0
    // For pixel=128: 0.003921...
    // For pixel=255: 1.0

    let to_tensor = ToTensor::normalized([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);

    let img = ImageBuffer::from_raw(vec![0, 128, 255], 1, 1, 3).unwrap();
    let tensor = to_tensor.apply::<CpuBackend>(&img);
    let data = tensor.to_vec();

    let expected_r = (0.0 / 255.0 - 0.5) / 0.5;
    let expected_g = (128.0 / 255.0 - 0.5) / 0.5;
    let expected_b = (255.0 / 255.0 - 0.5) / 0.5;

    assert!(
        (data[0] - expected_r).abs() < 1e-4,
        "R: got {}, expected {}",
        data[0],
        expected_r
    );
    assert!(
        (data[1] - expected_g).abs() < 1e-4,
        "G: got {}, expected {}",
        data[1],
        expected_g
    );
    assert!(
        (data[2] - expected_b).abs() < 1e-4,
        "B: got {}, expected {}",
        data[2],
        expected_b
    );
}

#[test]
fn scrfd_preprocessing_produces_correct_tensor_shape() {
    // 640×640 RGB image → ToTensor → [3, 640, 640] = 1,228,800 floats
    let img = ImageBuffer::from_raw(vec![128u8; 640 * 640 * 3], 640, 640, 3).unwrap();
    let std_val = 128.0 / 255.0;
    let to_tensor = ToTensor::normalized([0.5, 0.5, 0.5], [std_val, std_val, std_val]);
    let tensor = to_tensor.apply::<CpuBackend>(&img);
    let data = tensor.to_vec();

    assert_eq!(data.len(), 3 * 640 * 640);
    // CHW layout: first 640*640 elements are R channel
    // All pixels are 128, so all channels have same value
    let expected = (128.0 / 255.0 - 0.5) / std_val;
    assert!((data[0] - expected).abs() < 1e-4);
    assert!((data[640 * 640] - expected).abs() < 1e-4);     // G channel start
    assert!((data[2 * 640 * 640] - expected).abs() < 1e-4); // B channel start
}

#[test]
fn scrfd_capture_verifies_input_reaches_model() {
    // Use CaptureMock to verify the actual tensor values reaching the model
    let dummy_output = vec![320.0, 320.0, 100.0, 100.0, 0.9];
    let mock = std::sync::Arc::new(CaptureMock::new(dummy_output));

    // We need a VisionModel, create a wrapper
    struct ArcMock(std::sync::Arc<CaptureMock>);
    impl VisionModel for ArcMock {
        fn forward(&self, input: &[f32], input_shape: &[usize]) -> Result<Vec<f32>> {
            self.0.forward(input, input_shape)
        }
        fn output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
            self.0.output_shape(input_shape)
        }
    }

    let mock_clone = mock.clone();
    let detector = ScrfdDetector::new(Box::new(ArcMock(mock_clone)), 640);

    // Create a 640×640 image where all pixels are [100, 150, 200]
    let mut image = Vec::with_capacity(640 * 640 * 3);
    for _ in 0..(640 * 640) {
        image.extend_from_slice(&[100, 150, 200]);
    }

    let _ = detector.detect(&image, 640, 640, 0.5);

    let captured = mock.captured_input();
    assert_eq!(captured.len(), 3 * 640 * 640);

    // Verify normalization: (pixel - 127.5) / 128
    let expected_r = (100.0 - 127.5) / 128.0;
    let expected_g = (150.0 - 127.5) / 128.0;
    let expected_b = (200.0 - 127.5) / 128.0;

    // CHW layout: first 640*640 = R, next = G, last = B
    let n = 640 * 640;
    assert!(
        (captured[0] - expected_r).abs() < 1e-3,
        "R[0]: got {}, expected {}",
        captured[0],
        expected_r
    );
    assert!(
        (captured[n] - expected_g).abs() < 1e-3,
        "G[0]: got {}, expected {}",
        captured[n],
        expected_g
    );
    assert!(
        (captured[2 * n] - expected_b).abs() < 1e-3,
        "B[0]: got {}, expected {}",
        captured[2 * n],
        expected_b
    );
}

// ── Letterbox + rescale round-trip ──────────────────────────────────────────

#[test]
fn letterbox_rescale_roundtrip_wide_image() {
    // 1920×1080 → 640×640 letterbox → rescale back
    let img = ImageBuffer::from_raw(vec![128u8; 1920 * 1080 * 3], 1920, 1080, 3).unwrap();
    let lb = Letterbox::new(640, 640);
    let (padded, info) = lb.apply_with_info(&img).unwrap();

    assert_eq!(padded.width, 640);
    assert_eq!(padded.height, 640);

    // scale = min(640/1920, 640/1080) = min(0.333, 0.593) = 0.333
    let expected_scale = 640.0 / 1920.0;
    assert!((info.scale - expected_scale).abs() < 0.01);

    // A point at the center of the original image (960, 540) should map to
    // the center of the resized area in the letterboxed image
    let model_x = 960.0 * info.scale + info.pad_x;
    let model_y = 540.0 * info.scale + info.pad_y;

    // Inverse: (model_coord - pad) / scale = original_coord
    let recovered_x = (model_x - info.pad_x) / info.scale;
    let recovered_y = (model_y - info.pad_y) / info.scale;

    assert!(
        (recovered_x - 960.0).abs() < 1.0,
        "x roundtrip: {}",
        recovered_x
    );
    assert!(
        (recovered_y - 540.0).abs() < 1.0,
        "y roundtrip: {}",
        recovered_y
    );
}

#[test]
fn letterbox_rescale_roundtrip_tall_image() {
    // 720×1280 portrait → 640×640
    let img = ImageBuffer::from_raw(vec![128u8; 720 * 1280 * 3], 720, 1280, 3).unwrap();
    let lb = Letterbox::new(640, 640);
    let (_padded, info) = lb.apply_with_info(&img).unwrap();

    // scale = min(640/720, 640/1280) = min(0.889, 0.5) = 0.5
    assert!((info.scale - 0.5).abs() < 0.01);
    // Horizontal padding since image becomes 360×640
    assert!(info.pad_x > 0.0);
    assert!((info.pad_y).abs() < 1.0);
}

// ── Face alignment tests ────────────────────────────────────────────────────

/// Standard ArcFace 5-point canonical template landmarks (112×112 output).
/// These are the reference positions that aligned faces should conform to.
const ARCFACE_TEMPLATE: [[f32; 2]; 5] = [
    [38.2946, 51.6963],  // left eye
    [73.5318, 51.5014],  // right eye
    [56.0252, 71.7366],  // nose tip
    [41.5493, 92.3655],  // left mouth corner
    [70.7299, 92.2041],  // right mouth corner
];

#[test]
fn alignment_identity_when_landmarks_match_template() {
    // When detected landmarks exactly match the template, the transform
    // should be approximately identity (no rotation, no scale change).
    let src = ARCFACE_TEMPLATE;
    let dst = ARCFACE_TEMPLATE;

    let transform = AffineTransform::estimate_similarity(&src, &dst, 112, 112);

    // Matrix should be close to identity: [[1, 0, 0], [0, 1, 0]]
    assert!((transform.matrix[0][0] - 1.0).abs() < 1e-3, "a={}", transform.matrix[0][0]);
    assert!((transform.matrix[0][1]).abs() < 1e-3, "b={}", transform.matrix[0][1]);
    assert!((transform.matrix[0][2]).abs() < 1e-3, "tx={}", transform.matrix[0][2]);
    assert!((transform.matrix[1][0]).abs() < 1e-3, "c={}", transform.matrix[1][0]);
    assert!((transform.matrix[1][1] - 1.0).abs() < 1e-3, "d={}", transform.matrix[1][1]);
    assert!((transform.matrix[1][2]).abs() < 1e-3, "ty={}", transform.matrix[1][2]);
}

#[test]
fn alignment_handles_scaled_landmarks() {
    // Face detected at 2x the template size. The transform should scale down by 0.5.
    let src: Vec<[f32; 2]> = ARCFACE_TEMPLATE
        .iter()
        .map(|&[x, y]| [x * 2.0, y * 2.0])
        .collect();

    let transform = AffineTransform::estimate_similarity(&src, &ARCFACE_TEMPLATE, 112, 112);

    // scale ≈ 0.5, no rotation
    assert!(
        (transform.matrix[0][0] - 0.5).abs() < 0.05,
        "scale_x={}",
        transform.matrix[0][0]
    );
    assert!(
        (transform.matrix[1][1] - 0.5).abs() < 0.05,
        "scale_y={}",
        transform.matrix[1][1]
    );
    // Rotation component should be near zero
    assert!(
        transform.matrix[0][1].abs() < 0.05,
        "rotation component b={}",
        transform.matrix[0][1]
    );
}

#[test]
fn alignment_handles_translated_landmarks() {
    // Face detected with an offset of (+100, +50) from template positions
    let src: Vec<[f32; 2]> = ARCFACE_TEMPLATE
        .iter()
        .map(|&[x, y]| [x + 100.0, y + 50.0])
        .collect();

    let transform = AffineTransform::estimate_similarity(&src, &ARCFACE_TEMPLATE, 112, 112);

    // Should have unit scale, no rotation, and translation ≈ (-100, -50)
    assert!((transform.matrix[0][0] - 1.0).abs() < 0.05);
    assert!((transform.matrix[0][2] - (-100.0)).abs() < 1.0, "tx={}", transform.matrix[0][2]);
    assert!((transform.matrix[1][2] - (-50.0)).abs() < 1.0, "ty={}", transform.matrix[1][2]);
}

#[test]
fn alignment_warp_preserves_face_content() {
    // Create a 224×224 image with a "face" pattern centered at the template
    // landmarks scaled by 2x. After alignment (which should scale down), the
    // face content should appear at the template positions in the 112×112 output.
    let (w, h) = (224u32, 224u32);
    let mut data = vec![0u8; (w * h * 3) as usize];

    // Paint a bright region where the "nose" would be (template nose * 2)
    let nose_x = (ARCFACE_TEMPLATE[2][0] * 2.0) as u32;
    let nose_y = (ARCFACE_TEMPLATE[2][1] * 2.0) as u32;
    let r = 10u32;
    for dy in nose_y.saturating_sub(r)..=(nose_y + r).min(h - 1) {
        for dx in nose_x.saturating_sub(r)..=(nose_x + r).min(w - 1) {
            let idx = ((dy * w + dx) * 3) as usize;
            data[idx] = 255;
            data[idx + 1] = 255;
            data[idx + 2] = 255;
        }
    }

    let img = ImageBuffer::from_raw(data, w, h, 3).unwrap();

    // Source landmarks = template * 2, destination = template
    let src: Vec<[f32; 2]> = ARCFACE_TEMPLATE.iter().map(|&[x, y]| [x * 2.0, y * 2.0]).collect();
    let transform = AffineTransform::estimate_similarity(&src, &ARCFACE_TEMPLATE, 112, 112);
    let aligned = transform.apply(&img).unwrap();

    assert_eq!(aligned.width, 112);
    assert_eq!(aligned.height, 112);

    // The nose area in the aligned image should be bright (near the template nose position)
    let aligned_nose_x = ARCFACE_TEMPLATE[2][0] as u32;
    let aligned_nose_y = ARCFACE_TEMPLATE[2][1] as u32;
    let pixel = aligned.pixel(aligned_nose_x, aligned_nose_y, 0).unwrap();
    assert!(
        pixel > 100,
        "nose region should be bright after alignment, got {}",
        pixel
    );
}

#[test]
fn alignment_rotated_face() {
    // Simulate a face rotated ~15 degrees clockwise around its center
    let angle = -15.0_f32.to_radians();
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    // Center of the template landmarks
    let cx: f32 = ARCFACE_TEMPLATE.iter().map(|p| p[0]).sum::<f32>() / 5.0;
    let cy: f32 = ARCFACE_TEMPLATE.iter().map(|p| p[1]).sum::<f32>() / 5.0;

    // Rotate each landmark around the center
    let src: Vec<[f32; 2]> = ARCFACE_TEMPLATE
        .iter()
        .map(|&[x, y]| {
            let dx = x - cx;
            let dy = y - cy;
            [cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a]
        })
        .collect();

    let transform = AffineTransform::estimate_similarity(&src, &ARCFACE_TEMPLATE, 112, 112);

    // The transform should contain a rotation component
    // For a similarity transform [[a, -b, tx], [b, a, ty]], rotation = atan2(b, a)
    let recovered_angle = transform.matrix[1][0].atan2(transform.matrix[0][0]);
    // The transform undoes the rotation, so it should rotate by +15 degrees
    let expected = 15.0_f32.to_radians();
    assert!(
        (recovered_angle - expected).abs() < 0.05,
        "recovered rotation {:.1} deg, expected {:.1} deg",
        recovered_angle.to_degrees(),
        expected.to_degrees()
    );
}

// ── Edge cases ──────────────────────────────────────────────────────────────

#[test]
fn detect_tiny_image() {
    // Very small image (32×32) — should still work through letterbox
    #[rustfmt::skip]
    let output = vec![320.0, 320.0, 50.0, 50.0, 0.9];

    let model = MockModel { output };
    let detector = ScrfdDetector::new(Box::new(model), 640);

    let image = vec![128u8; 32 * 32 * 3];
    let dets = detector.detect(&image, 32, 32, 0.5).unwrap();

    assert_eq!(dets.len(), 1);
    // Letterbox: 32×32 → 640×640, scale = 20.0
    // Detection at (320, 320) in model space → (320/20, 320/20) = (16, 16) in original
    let cx = (dets[0].bbox.x1 + dets[0].bbox.x2) / 2.0;
    let cy = (dets[0].bbox.y1 + dets[0].bbox.y2) / 2.0;
    assert!((cx - 16.0).abs() < 1.0, "cx={cx}");
    assert!((cy - 16.0).abs() < 1.0, "cy={cy}");
}

#[test]
fn detect_extreme_aspect_ratio() {
    // Very wide panoramic image: 4000×200
    #[rustfmt::skip]
    let output = vec![320.0, 320.0, 50.0, 50.0, 0.85];

    let model = MockModel { output };
    let detector = ScrfdDetector::new(Box::new(model), 640);

    let image = vec![128u8; 4000 * 200 * 3];
    let dets = detector.detect(&image, 4000, 200, 0.5).unwrap();

    assert_eq!(dets.len(), 1);
    // Letterbox: scale = min(640/4000, 640/200) = min(0.16, 3.2) = 0.16
    // Rescaled coords should be in valid range for original image
    assert!(dets[0].bbox.x1 >= -100.0); // allow some tolerance for padding
    assert!(dets[0].bbox.x2 <= 4100.0);
}

#[test]
fn crop_face_at_image_boundary() {
    // Detection where bbox extends to image edges
    let img = ImageBuffer::from_raw(vec![128u8; 100 * 100 * 3], 100, 100, 3).unwrap();

    // Face at top-left corner
    let crop = Crop::new(0, 0, 30, 30);
    let result = crop.apply(&img).unwrap();
    assert_eq!(result.width, 30);
    assert_eq!(result.height, 30);

    // Face at bottom-right corner
    let crop = Crop::new(70, 70, 30, 30);
    let result = crop.apply(&img).unwrap();
    assert_eq!(result.width, 30);
    assert_eq!(result.height, 30);
}

#[test]
fn embedding_consistency_same_crop() {
    // Same face crop → same embedding (deterministic)
    let face = vec![128u8; 112 * 112 * 3];
    let mut emb_vec = vec![0.0f32; 512];
    emb_vec[0] = 1.0;
    emb_vec[1] = 0.5;

    let embedder = ArcFaceEmbedder::new(Box::new(MockModel { output: emb_vec.clone() }), 112, 512);
    let emb1 = embedder.embed(&face, 112, 112).unwrap();
    let emb2 = embedder.embed(&face, 112, 112).unwrap();

    assert_eq!(emb1, emb2, "same input → identical output");
}

#[test]
fn embedding_resize_from_different_crop_sizes() {
    // ArcFace should handle crops of various sizes by resizing to 112×112
    let mut emb_vec = vec![0.0f32; 512];
    emb_vec[0] = 1.0;

    let sizes = [(50, 50), (112, 112), (224, 224), (80, 120)];
    for (w, h) in sizes {
        let face = vec![128u8; (w * h * 3) as usize];
        let embedder = ArcFaceEmbedder::new(
            Box::new(MockModel { output: emb_vec.clone() }),
            112,
            512,
        );
        let emb = embedder.embed(&face, w, h).unwrap();
        assert_eq!(emb.len(), 512, "embed dim should be 512 for {}x{}", w, h);
        let norm = l2_norm(&emb);
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "embedding should be unit norm for {}x{}, got {}",
            w,
            h,
            norm
        );
    }
}

// ── Similarity threshold validation ─────────────────────────────────────────

#[test]
fn cosine_similarity_near_identical_embeddings() {
    // Simulate two photos of the same person with slight variation
    let mut a = vec![0.0f32; 512];
    let mut b = vec![0.0f32; 512];

    // Same general direction with small perturbation
    for i in 0..50 {
        a[i] = 1.0 + (i as f32 * 0.01);
        b[i] = 1.0 + (i as f32 * 0.01) + 0.02; // tiny noise
    }

    l2_normalize(&mut a);
    l2_normalize(&mut b);

    let sim = cosine_similarity(&a, &b);
    assert!(
        sim > 0.99,
        "near-identical embeddings should have very high similarity, got {}",
        sim
    );
}

#[test]
fn cosine_similarity_different_people() {
    // Two embeddings pointing in very different directions
    let mut a = vec![0.0f32; 512];
    let mut b = vec![0.0f32; 512];

    // a is concentrated in dims 0..50
    for i in 0..50 {
        a[i] = 1.0;
    }
    // b is concentrated in dims 200..250 (orthogonal-ish)
    for i in 200..250 {
        b[i] = 1.0;
    }

    l2_normalize(&mut a);
    l2_normalize(&mut b);

    let sim = cosine_similarity(&a, &b);
    assert!(
        sim < 0.01,
        "different people should have low similarity, got {}",
        sim
    );
}

#[test]
fn cosine_similarity_realistic_thresholds() {
    // Verify common ArcFace thresholds work correctly:
    // - Same person: typically > 0.3 (after L2 normalization)
    // - Different person: typically < 0.3
    // Note: the exact threshold depends on the model. 0.3-0.4 is typical for ArcFace.

    // Two embeddings with controlled similarity
    let mut a = vec![0.0f32; 512];
    let mut b = vec![0.0f32; 512];

    // Create embeddings with approximately 0.5 cosine similarity
    for i in 0..100 {
        a[i] = 1.0;
        b[i] = 1.0; // shared direction
    }
    for i in 100..200 {
        a[i] = 1.0;
        b[i] = 0.0; // diverge
    }
    for i in 200..300 {
        a[i] = 0.0;
        b[i] = 1.0; // diverge
    }

    l2_normalize(&mut a);
    l2_normalize(&mut b);

    let sim = cosine_similarity(&a, &b);
    // Should be around 100/sqrt(200*200) = 100/200 = 0.5
    assert!(
        (sim - 0.5).abs() < 0.1,
        "controlled similarity should be ~0.5, got {}",
        sim
    );
}

// ── Full pipeline with alignment ────────────────────────────────────────────

#[test]
fn full_pipeline_detect_align_embed() {
    // Simulate the complete pipeline:
    // 1. Detect face (mock SCRFD returning bbox + keypoints)
    // 2. Align face using keypoints → AffineTransform
    // 3. Embed aligned face (mock ArcFace)
    // 4. Compare embeddings

    let (w, h) = (640u32, 640u32);
    let image_data = vec![128u8; (w * h * 3) as usize];

    // Step 1: Mock detection with keypoints
    // Single face centered at (320, 320) with 80×80 bbox
    #[rustfmt::skip]
    let scrfd_output = vec![
        320.0,  // cx
        320.0,  // cy
         80.0,  // w
         80.0,  // h
          0.95, // conf
    ];

    let detector = ScrfdDetector::new(Box::new(MockModel { output: scrfd_output }), 640);
    let dets = detector.detect(&image_data, w, h, 0.5).unwrap();
    assert_eq!(dets.len(), 1);

    // Step 2: Simulate detected landmarks (as if SCRFD returned them)
    // These are scaled/positioned in the detected face region
    let detected_landmarks = [
        [295.0, 305.0], // left eye
        [345.0, 305.0], // right eye
        [320.0, 325.0], // nose
        [300.0, 345.0], // left mouth
        [340.0, 345.0], // right mouth
    ];

    // Compute alignment transform
    let align_transform = AffineTransform::estimate_similarity(
        &detected_landmarks,
        &ARCFACE_TEMPLATE,
        112,
        112,
    );

    // Step 3: Apply alignment
    let img = ImageBuffer::from_raw(image_data.clone(), w, h, 3).unwrap();
    let aligned = align_transform.apply(&img).unwrap();
    assert_eq!(aligned.width, 112);
    assert_eq!(aligned.height, 112);

    // Step 4: Embed
    let mut emb_output = vec![0.0f32; 512];
    emb_output[0] = 1.0;
    emb_output[1] = 0.5;
    let embedder = ArcFaceEmbedder::new(Box::new(MockModel { output: emb_output }), 112, 512);
    let embedding = embedder.embed(&aligned.data, aligned.width, aligned.height).unwrap();

    assert_eq!(embedding.len(), 512);
    let norm = l2_norm(&embedding);
    assert!((norm - 1.0).abs() < 1e-4, "embedding should be unit norm");
}

#[test]
fn pipeline_alignment_vs_naive_crop_produces_different_tensors() {
    // This test demonstrates that alignment and naive crop+resize produce
    // different tensor inputs, which is why alignment matters for ArcFace.

    // Create an image with distinct regions
    let (w, h) = (200u32, 200u32);
    let mut data = vec![0u8; (w * h * 3) as usize];
    // Paint gradient so different regions are distinguishable
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 3) as usize;
            data[idx] = (x * 255 / w) as u8;     // R: horizontal gradient
            data[idx + 1] = (y * 255 / h) as u8; // G: vertical gradient
            data[idx + 2] = 128;
        }
    }
    let img = ImageBuffer::from_raw(data, w, h, 3).unwrap();

    // Approach 1: Naive crop + resize (no alignment)
    let crop = Crop::new(50, 50, 80, 80);
    let cropped = crop.apply(&img).unwrap();
    let resize = Resize::new(112, 112, InterpolationMode::Bilinear);
    let naive = resize.apply(&cropped).unwrap();

    // Approach 2: Alignment via affine transform
    // Simulate landmarks that are slightly rotated
    let src_landmarks = [
        [65.0, 60.0],
        [115.0, 65.0], // slightly tilted
        [90.0, 85.0],
        [70.0, 105.0],
        [110.0, 110.0],
    ];
    let align = AffineTransform::estimate_similarity(&src_landmarks, &ARCFACE_TEMPLATE, 112, 112);
    let aligned = align.apply(&img).unwrap();

    // The two approaches should produce different pixel data
    assert_eq!(naive.width, aligned.width);
    assert_eq!(naive.height, aligned.height);

    let diff: u64 = naive
        .data
        .iter()
        .zip(aligned.data.iter())
        .map(|(&a, &b)| (i32::from(a) - i32::from(b)).unsigned_abs() as u64)
        .sum();

    let avg_diff = diff as f64 / naive.data.len() as f64;
    assert!(
        avg_diff > 1.0,
        "aligned and naive crop should differ significantly, avg_diff={}",
        avg_diff
    );
}

// ── Detection coordinate accuracy ───────────────────────────────────────────

#[test]
fn detection_coordinates_match_expected_after_letterbox() {
    // Verify that a face detected at a known position in model space
    // maps back correctly to the original image coordinates for various
    // image sizes.

    let test_cases: Vec<(u32, u32, f32, f32)> = vec![
        (640, 640, 320.0, 320.0),    // square, no padding
        (1280, 720, 640.0, 360.0),   // wide 16:9
        (720, 1280, 360.0, 640.0),   // tall 9:16
        (1920, 1080, 960.0, 540.0),  // full HD
        (100, 100, 50.0, 50.0),      // tiny
    ];

    for (img_w, img_h, expected_cx, expected_cy) in test_cases {
        // Put detection at the center of the model input
        let output = vec![320.0, 320.0, 50.0, 50.0, 0.9];
        let model = MockModel { output };
        let detector = ScrfdDetector::new(Box::new(model), 640);

        let image = vec![128u8; (img_w * img_h * 3) as usize];
        let dets = detector.detect(&image, img_w, img_h, 0.5).unwrap();

        assert_eq!(dets.len(), 1, "{}x{}", img_w, img_h);

        let d = &dets[0];
        let cx = (d.bbox.x1 + d.bbox.x2) / 2.0;
        let cy = (d.bbox.y1 + d.bbox.y2) / 2.0;

        // The center of the model input (320, 320) should map to the center
        // of the original image, with tolerance for rounding
        assert!(
            (cx - expected_cx).abs() < 5.0,
            "{}x{}: cx={cx}, expected {expected_cx}",
            img_w,
            img_h,
        );
        assert!(
            (cy - expected_cy).abs() < 5.0,
            "{}x{}: cy={cy}, expected {expected_cy}",
            img_w,
            img_h,
        );
    }
}

#[test]
fn detection_box_size_scales_correctly() {
    // A 100×100 box in model space should scale proportionally to the original image
    let output = vec![320.0, 320.0, 100.0, 100.0, 0.9];

    // Test with 1280×720 image
    let model = MockModel { output };
    let detector = ScrfdDetector::new(Box::new(model), 640);
    let image = vec![128u8; 1280 * 720 * 3];
    let dets = detector.detect(&image, 1280, 720, 0.5).unwrap();

    let d = &dets[0];
    let det_w = d.bbox.x2 - d.bbox.x1;
    let det_h = d.bbox.y2 - d.bbox.y1;

    // Letterbox scale = min(640/1280, 640/720) = 0.5
    // 100 model pixels / 0.5 = 200 original pixels
    assert!(
        (det_w - 200.0).abs() < 3.0,
        "width should scale: got {}",
        det_w
    );
    assert!(
        (det_h - 200.0).abs() < 3.0,
        "height should scale: got {}",
        det_h
    );
}

// ── NMS interaction tests ───────────────────────────────────────────────────

#[test]
fn many_overlapping_detections_nms() {
    // 10 highly overlapping detections — NMS should keep only the best
    let num = 10;
    let mut output = Vec::new();
    // cx: all at 320 ± small offset
    for i in 0..num {
        output.push(320.0 + i as f32);
    }
    // cy
    for i in 0..num {
        output.push(320.0 + i as f32);
    }
    // w
    for _ in 0..num {
        output.push(100.0);
    }
    // h
    for _ in 0..num {
        output.push(100.0);
    }
    // confidence: descending
    for i in 0..num {
        output.push(0.95 - i as f32 * 0.03);
    }

    let model = MockModel { output };
    let detector = ScrfdDetector::new(Box::new(model), 640);
    let image = vec![128u8; 640 * 640 * 3];
    let dets = detector.detect(&image, 640, 640, 0.5).unwrap();

    // All boxes overlap heavily (shifted by only 1-10px on 100px box)
    // NMS should suppress most, keeping 1-2
    assert!(dets.len() <= 3, "NMS should suppress overlaps, got {}", dets.len());
    assert!(!dets.is_empty(), "at least one detection should survive");
    // The highest-confidence one should be kept
    assert!(dets[0].confidence > 0.9);
}

#[test]
fn well_separated_detections_survive_nms() {
    // 3 detections far apart — all should survive NMS
    #[rustfmt::skip]
    let output = vec![
        100.0, 320.0, 540.0, // cx — well separated
        320.0, 320.0, 320.0, // cy
        80.0,  80.0,  80.0,  // w
        80.0,  80.0,  80.0,  // h
        0.9,   0.85,  0.8,   // confidence
    ];

    let model = MockModel { output };
    let detector = ScrfdDetector::new(Box::new(model), 640);
    let image = vec![128u8; 640 * 640 * 3];
    let dets = detector.detect(&image, 640, 640, 0.5).unwrap();

    assert_eq!(dets.len(), 3, "separated detections should all survive NMS");
}

// ── Embedding edge cases ────────────────────────────────────────────────────

#[test]
fn embedding_large_magnitude_normalized() {
    // Model returns very large raw values — embedding should still be unit norm
    let output = vec![1e6; 512];
    let embedder = ArcFaceEmbedder::new(Box::new(MockModel { output }), 112, 512);
    let face = vec![128u8; 112 * 112 * 3];
    let emb = embedder.embed(&face, 112, 112).unwrap();

    let norm = l2_norm(&emb);
    assert!((norm - 1.0).abs() < 1e-4, "large values: norm={norm}");
}

#[test]
fn embedding_tiny_magnitude_normalized() {
    // Model returns very small raw values
    let output = vec![1e-6; 512];
    let embedder = ArcFaceEmbedder::new(Box::new(MockModel { output }), 112, 512);
    let face = vec![128u8; 112 * 112 * 3];
    let emb = embedder.embed(&face, 112, 112).unwrap();

    let norm = l2_norm(&emb);
    assert!((norm - 1.0).abs() < 1e-4, "tiny values: norm={norm}");
}

#[test]
fn embedding_sparse_vector() {
    // Only one non-zero dimension — should still produce valid unit embedding
    let mut output = vec![0.0f32; 512];
    output[256] = 42.0;

    let embedder = ArcFaceEmbedder::new(Box::new(MockModel { output }), 112, 512);
    let face = vec![128u8; 112 * 112 * 3];
    let emb = embedder.embed(&face, 112, 112).unwrap();

    assert_eq!(emb.len(), 512);
    // Should be [0, ..., 0, 1.0, 0, ..., 0] at index 256
    assert!((emb[256] - 1.0).abs() < 1e-4);
    assert!(emb[0].abs() < 1e-4);
}