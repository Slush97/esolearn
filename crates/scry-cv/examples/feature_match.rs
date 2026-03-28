// SPDX-License-Identifier: MIT OR Apache-2.0
//! ORB feature detection and matching on two synthetic images.
//!
//! ```sh
//! cargo run -p scry-cv --example feature_match
//! ```

use scry_cv::prelude::*;

fn main() {
    let (w, h) = (128u32, 128u32);
    let dx = 4i32;
    let dy = 3i32;

    // Generate a textured image
    let mut data1 = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let v = ((x.wrapping_mul(7919).wrapping_add(y.wrapping_mul(104729)))
                .wrapping_add(42u32.wrapping_mul(31337)))
                % 256;
            data1[y as usize * w as usize + x as usize] = v as f32 / 255.0;
        }
    }
    let img1 = ImageBuf::<f32, Gray>::from_vec(data1.clone(), w, h).unwrap();

    // Create shifted copy
    let mut data2 = vec![0.0f32; (w * h) as usize];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let sx = x - dx;
            let sy = y - dy;
            if sx >= 0 && sx < w as i32 && sy >= 0 && sy < h as i32 {
                data2[y as usize * w as usize + x as usize] =
                    data1[sy as usize * w as usize + sx as usize];
            }
        }
    }
    let img2 = ImageBuf::<f32, Gray>::from_vec(data2, w, h).unwrap();

    println!("Images: {w}x{h}, shift = ({dx}, {dy})");

    // ORB detection
    let orb = Orb::new().n_features(300).fast_threshold(0.02);
    let (kp1, desc1) = orb.detect_and_compute(&img1).unwrap();
    let (kp2, desc2) = orb.detect_and_compute(&img2).unwrap();
    println!("Features: img1={}, img2={}", kp1.len(), kp2.len());

    if desc1.is_empty() || desc2.is_empty() {
        println!("Not enough features detected for matching.");
        return;
    }

    // KNN match + ratio test
    let knn = knn_match_binary(&desc1, &desc2, 2);
    let good = ratio_test(&knn, 0.8);
    println!("Matches: {} raw KNN, {} after ratio test", knn.len(), good.len());

    if good.is_empty() {
        println!("No good matches found.");
        return;
    }

    // Compute average displacement
    let mut sum_dx = 0.0f64;
    let mut sum_dy = 0.0f64;
    let mut consistent = 0u32;
    for m in &good {
        let p1 = &kp1[m.query_idx];
        let p2 = &kp2[m.train_idx];
        let mdx = p2.x - p1.x;
        let mdy = p2.y - p1.y;
        sum_dx += mdx as f64;
        sum_dy += mdy as f64;
        if (mdx - dx as f32).abs() < 5.0 && (mdy - dy as f32).abs() < 5.0 {
            consistent += 1;
        }
    }
    let avg_dx = sum_dx / good.len() as f64;
    let avg_dy = sum_dy / good.len() as f64;

    println!("Average displacement: ({avg_dx:.2}, {avg_dy:.2})");
    println!("Expected:             ({dx}, {dy})");
    println!(
        "Spatially consistent:  {consistent}/{} ({:.0}%)",
        good.len(),
        consistent as f64 / good.len() as f64 * 100.0
    );
}
