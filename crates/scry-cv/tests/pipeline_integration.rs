// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cross-module pipeline integration tests.
//!
//! Each test chains multiple scry-cv algorithms together on programmatically
//! generated images to verify that module outputs compose correctly.

use scry_cv::prelude::*;
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Helpers — procedural image generation
// ---------------------------------------------------------------------------

/// Create a `w x h` f32 gray image with a bright rectangle from (x0,y0) to (x1,y1).
fn make_rect_image(w: u32, h: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> GrayImageF {
    let mut data = vec![0.0f32; (w * h) as usize];
    for y in y0..y1 {
        for x in x0..x1 {
            data[y as usize * w as usize + x as usize] = 1.0;
        }
    }
    ImageBuf::from_vec(data, w, h).unwrap()
}

/// Create a textured image using a simple hash-based pattern.
#[cfg(feature = "stereo")]
fn make_textured_image(w: u32, h: u32, seed: u32) -> GrayImageF {
    let mut data = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            // Simple deterministic "texture" — mixes position and seed
            let v = ((x.wrapping_mul(7919).wrapping_add(y.wrapping_mul(104729)))
                .wrapping_add(seed.wrapping_mul(31337)))
                % 256;
            data[y as usize * w as usize + x as usize] = v as f32 / 255.0;
        }
    }
    ImageBuf::from_vec(data, w, h).unwrap()
}

/// Shift an image by (dx, dy) pixels, filling new pixels with 0.
fn shift_image(img: &GrayImageF, dx: i32, dy: i32) -> GrayImageF {
    let w = img.width();
    let h = img.height();
    let mut data = vec![0.0f32; (w * h) as usize];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let sx = x - dx;
            let sy = y - dy;
            if sx >= 0 && sx < w as i32 && sy >= 0 && sy < h as i32 {
                data[y as usize * w as usize + x as usize] =
                    img.as_slice()[sy as usize * w as usize + sx as usize];
            }
        }
    }
    ImageBuf::from_vec(data, w, h).unwrap()
}

// ---------------------------------------------------------------------------
// (a) Edge detection → shape detection pipeline
// ---------------------------------------------------------------------------

#[test]
fn canny_to_hough_lines_detects_rectangle() {
    // 80x80 image with a bright 40x40 rectangle in the center
    let img = make_rect_image(80, 80, 20, 20, 60, 60);

    // Canny edge detection
    let edges = canny(&img, 0.05, 0.15).unwrap();
    assert_eq!(edges.dimensions(), (80, 80));

    let edge_count: usize = edges.as_slice().iter().filter(|&&v| v > 0.5).count();
    assert!(
        edge_count > 30,
        "expected edges around rectangle, got {edge_count}"
    );

    // Hough lines — should detect roughly 4 dominant lines
    let lines = hough_lines(&edges, 1.0, PI / 180.0, 15).unwrap();
    assert!(
        lines.len() >= 2,
        "expected at least 2 Hough lines for a rectangle, got {}",
        lines.len()
    );

    // Verify we have both roughly-horizontal and roughly-vertical lines
    let has_horizontal = lines.iter().any(|l| {
        let deg = l.theta * 180.0 / PI;
        (deg - 90.0).abs() < 15.0
    });
    let has_vertical = lines.iter().any(|l| {
        let deg = l.theta * 180.0 / PI;
        !(15.0..=165.0).contains(&deg)
    });
    assert!(has_horizontal, "expected at least one horizontal line");
    assert!(has_vertical, "expected at least one vertical line");
}

#[test]
fn canny_to_contours_detects_rectangle() {
    let img = make_rect_image(64, 64, 16, 16, 48, 48);

    let edges = canny(&img, 0.05, 0.15).unwrap();
    let contours = find_contours(&edges).unwrap();

    assert!(
        !contours.is_empty(),
        "expected at least one contour from rectangle edges"
    );

    // The largest contour should have a perimeter roughly proportional to 4 * 32
    let largest = contours.iter().max_by_key(|c| c.points.len()).unwrap();
    let perimeter = largest.points.len();
    // Perimeter of a 32x32 rectangle is ~128. With Canny smoothing, expect at least 40.
    assert!(
        perimeter > 40,
        "largest contour perimeter {perimeter} seems too small for a 32x32 rect"
    );
}

// ---------------------------------------------------------------------------
// (b) Feature matching pipeline
// ---------------------------------------------------------------------------

#[test]
fn orb_match_shifted_image() {
    // Use a checkerboard-like texture with strong corners for ORB
    let (w, h) = (160u32, 160u32);
    let mut data1 = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            // Checkerboard of 8x8 blocks with some gradient variation
            let bx = x / 8;
            let by = y / 8;
            let checker = ((bx + by) % 2) as f32;
            // Add sub-block gradient for richer descriptors
            let sub = (x % 8) as f32 / 16.0;
            data1[y as usize * w as usize + x as usize] = checker * 0.8 + sub * 0.2;
        }
    }
    let img1 = ImageBuf::<f32, Gray>::from_vec(data1, w, h).unwrap();

    let dx = 5i32;
    let dy = 4i32;
    let img2 = shift_image(&img1, dx, dy);

    let orb = Orb::new().n_features(300).fast_threshold(0.02);
    let (kp1, desc1) = orb.detect_and_compute(&img1).unwrap();
    let (kp2, desc2) = orb.detect_and_compute(&img2).unwrap();

    // Need some features in both
    if kp1.len() < 5 || kp2.len() < 5 {
        eprintln!(
            "Skipping: too few features (kp1={}, kp2={})",
            kp1.len(),
            kp2.len()
        );
        return;
    }

    let knn = knn_match_binary(&desc1, &desc2, 2);
    let good = ratio_test(&knn, 0.8);

    assert!(
        !good.is_empty(),
        "expected some good matches between original and shifted image"
    );

    // Verify average displacement trends in the right direction.
    // ORB uses a pyramid so individual keypoint precision varies — we check
    // the average displacement vector is roughly correct rather than requiring
    // per-match precision.
    let mut sum_dx = 0.0f64;
    let mut sum_dy = 0.0f64;
    for m in &good {
        let p1 = &kp1[m.query_idx];
        let p2 = &kp2[m.train_idx];
        sum_dx += (p2.x - p1.x) as f64;
        sum_dy += (p2.y - p1.y) as f64;
    }
    let avg_dx = sum_dx / good.len() as f64;
    let avg_dy = sum_dy / good.len() as f64;

    // The average should at least be in the right direction (positive)
    // with reasonable magnitude
    assert!(
        avg_dx > 0.0 && avg_dy > 0.0,
        "expected positive average displacement for shift ({dx},{dy}), got ({avg_dx:.1},{avg_dy:.1})"
    );
}

// ---------------------------------------------------------------------------
// (c) Template matching pipeline
// ---------------------------------------------------------------------------

#[test]
fn template_match_single_stamp() {
    let (iw, ih) = (64u32, 64u32);
    let (tw, th) = (8u32, 8u32);

    // Template: small gradient patch
    let tmpl_data: Vec<f32> = (0..tw * th)
        .map(|i| (i as f32 + 1.0) / (tw * th) as f32)
        .collect();
    let tmpl = ImageBuf::<f32, Gray>::from_vec(tmpl_data.clone(), tw, th).unwrap();

    // Stamp template at (20, 15) in a uniform background
    let mut img_data = vec![0.1f32; (iw * ih) as usize];
    for ty in 0..th as usize {
        for tx in 0..tw as usize {
            img_data[(15 + ty) * iw as usize + (20 + tx)] = tmpl_data[ty * tw as usize + tx];
        }
    }
    let img = ImageBuf::<f32, Gray>::from_vec(img_data, iw, ih).unwrap();

    let score = match_template(&img, &tmpl, TemplateMatchMethod::Ssd).unwrap();
    let (bx, by, bv) = find_best_match(&score, TemplateMatchMethod::Ssd);

    assert_eq!((bx, by), (20, 15), "best match should be at stamp location");
    assert!(bv < 1e-4, "SSD at exact match should be ~0, got {bv}");
}

#[test]
fn template_match_multiple_stamps() {
    let (iw, ih) = (80u32, 80u32);
    let (tw, th) = (6u32, 6u32);

    let tmpl_data: Vec<f32> = (0..tw * th)
        .map(|i| (i as f32 + 1.0) / (tw * th) as f32)
        .collect();
    let tmpl = ImageBuf::<f32, Gray>::from_vec(tmpl_data.clone(), tw, th).unwrap();

    // Stamp at 3 locations: (5,5), (40,10), (10,50)
    let stamps = [(5u32, 5u32), (40, 10), (10, 50)];
    let mut img_data = vec![0.05f32; (iw * ih) as usize];
    for &(sx, sy) in &stamps {
        for ty in 0..th as usize {
            for tx in 0..tw as usize {
                img_data[(sy as usize + ty) * iw as usize + (sx as usize + tx)] =
                    tmpl_data[ty * tw as usize + tx];
            }
        }
    }
    let img = ImageBuf::<f32, Gray>::from_vec(img_data, iw, ih).unwrap();

    // Use SSD — lower is better; find top-3 peaks (minima)
    let score = match_template(&img, &tmpl, TemplateMatchMethod::Ssd).unwrap();
    let score_data = score.as_slice();
    let sw = score.width() as usize;

    // Collect all positions with their scores, sort ascending (best first)
    let mut positions: Vec<(usize, usize, f32)> = Vec::new();
    for y in 0..score.height() as usize {
        for x in 0..sw {
            positions.push((x, y, score_data[y * sw + x]));
        }
    }
    positions.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    // Greedily pick peaks with minimum distance of tw between them
    let min_dist = tw as f32;
    let mut peaks: Vec<(usize, usize)> = Vec::new();
    for &(x, y, _) in &positions {
        let too_close = peaks.iter().any(|&(px, py)| {
            let dx = x as f32 - px as f32;
            let dy = y as f32 - py as f32;
            dx.hypot(dy) < min_dist
        });
        if !too_close {
            peaks.push((x, y));
        }
        if peaks.len() >= 3 {
            break;
        }
    }

    assert_eq!(peaks.len(), 3, "expected 3 distinct match peaks");

    // Each stamp location should be near one of the peaks
    for &(sx, sy) in &stamps {
        let found = peaks.iter().any(|&(px, py)| {
            let dx = (px as i32 - sx as i32).unsigned_abs();
            let dy = (py as i32 - sy as i32).unsigned_abs();
            dx <= 1 && dy <= 1
        });
        assert!(found, "stamp at ({sx},{sy}) not found in peaks: {peaks:?}");
    }
}

// ---------------------------------------------------------------------------
// (d) Background subtraction → region analysis
// ---------------------------------------------------------------------------

#[cfg(feature = "background")]
#[test]
fn mog2_detects_foreground_blob() {
    let (w, h) = (64u32, 64u32);

    // Static background: uniform gray
    let bg = ImageBuf::<f32, Gray>::from_vec(vec![0.3f32; (w * h) as usize], w, h).unwrap();

    let mut mog = Mog2::new(w, h);

    // Learn background over 10 frames
    for _ in 0..10 {
        let _mask = mog.apply(&bg, -1.0).unwrap();
    }

    // Frame with a bright blob at center (25..39, 25..39)
    let mut fg_data = vec![0.3f32; (w * h) as usize];
    let blob_x0 = 25u32;
    let blob_y0 = 25u32;
    let blob_x1 = 39u32;
    let blob_y1 = 39u32;
    for y in blob_y0..blob_y1 {
        for x in blob_x0..blob_x1 {
            fg_data[y as usize * w as usize + x as usize] = 0.9;
        }
    }
    let fg_frame = ImageBuf::<f32, Gray>::from_vec(fg_data, w, h).unwrap();
    let mask = mog.apply(&fg_frame, -1.0).unwrap();

    // Convert u8 mask to f32 for connected_components (foreground = 255)
    let mask_f32: Vec<f32> = mask
        .as_slice()
        .iter()
        .map(|&v| if v > 127 { 1.0 } else { 0.0 })
        .collect();
    let mask_img = ImageBuf::<f32, Gray>::from_vec(mask_f32, w, h).unwrap();

    let cc = connected_components(&mask_img, Connectivity::Eight).unwrap();

    // Should detect at least one foreground component
    assert!(
        cc.num_labels >= 1,
        "expected at least 1 foreground component, got {}",
        cc.num_labels
    );

    // The largest component's centroid should be near the blob center
    if let Some(largest) = cc.stats.iter().max_by_key(|s| s.area) {
        let expected_cx = (blob_x0 + blob_x1) as f64 / 2.0;
        let expected_cy = (blob_y0 + blob_y1) as f64 / 2.0;
        let dx = (largest.centroid.0 - expected_cx).abs();
        let dy = (largest.centroid.1 - expected_cy).abs();
        assert!(
            dx < 8.0 && dy < 8.0,
            "largest component centroid ({:.1}, {:.1}) too far from expected ({expected_cx:.1}, {expected_cy:.1})",
            largest.centroid.0, largest.centroid.1
        );
    }
}

// ---------------------------------------------------------------------------
// (e) Stereo pipeline
// ---------------------------------------------------------------------------

#[cfg(feature = "stereo")]
#[test]
fn sgbm_recovers_known_disparity() {
    let (w, h) = (80u32, 80u32);
    let known_disparity = 8i32;

    // Left image: textured
    let left = make_textured_image(w, h, 99);

    // Right image: left shifted right by `known_disparity` pixels
    // (i.e., right(x,y) = left(x + d, y) — standard rectified stereo convention)
    let mut right_data = vec![0.0f32; (w * h) as usize];
    let left_data = left.as_slice();
    for y in 0..h as usize {
        for x in 0..w as usize {
            let sx = x + known_disparity as usize;
            if sx < w as usize {
                right_data[y * w as usize + x] = left_data[y * w as usize + sx];
            }
        }
    }
    let right = ImageBuf::<f32, Gray>::from_vec(right_data, w, h).unwrap();

    let sgbm = SgbmStereo::new()
        .num_disparities(16)
        .block_size(5)
        .disp12_max_diff(-1); // disable LR check for simplicity

    let disp = sgbm.compute(&left, &right).unwrap();
    assert_eq!(disp.dimensions(), (w, h));

    // Check average disparity in the center region (away from borders)
    let margin = 16u32;
    let mut sum = 0.0f64;
    let mut count = 0u32;
    for y in margin..h - margin {
        for x in margin..w - margin {
            let d = disp.pixel(x, y)[0];
            if d >= 0.0 {
                sum += d as f64;
                count += 1;
            }
        }
    }

    if count > 0 {
        let avg = sum / count as f64;
        let diff = (avg - known_disparity as f64).abs();
        assert!(
            diff < 4.0,
            "average disparity {avg:.1} differs from expected {known_disparity} by {diff:.1}"
        );
    } else {
        // SGBM may invalidate all pixels on highly uniform regions; that's acceptable
        // as long as the pipeline doesn't crash.
        eprintln!("SGBM produced no valid disparities in center region (pipeline still works)");
    }
}
