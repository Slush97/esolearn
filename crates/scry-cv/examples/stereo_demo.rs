// SPDX-License-Identifier: MIT OR Apache-2.0
//! Synthetic stereo pair → SGBM disparity estimation.
//!
//! ```sh
//! cargo run -p scry-cv --example stereo_demo --features stereo
//! ```

#[cfg(feature = "stereo")]
fn main() {
    use scry_cv::prelude::*;

    let (w, h) = (100u32, 100u32);
    let known_disparity = 6u32;

    // Generate a textured left image
    let mut left_data = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let v = ((x.wrapping_mul(7919).wrapping_add(y.wrapping_mul(104729)))
                .wrapping_add(42u32.wrapping_mul(31337)))
                % 256;
            left_data[y as usize * w as usize + x as usize] = v as f32 / 255.0;
        }
    }
    let left = ImageBuf::<f32, Gray>::from_vec(left_data.clone(), w, h).unwrap();

    // Right image: shift left by `known_disparity`
    let mut right_data = vec![0.0f32; (w * h) as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let sx = x + known_disparity as usize;
            if sx < w as usize {
                right_data[y * w as usize + x] = left_data[y * w as usize + sx];
            }
        }
    }
    let right = ImageBuf::<f32, Gray>::from_vec(right_data, w, h).unwrap();

    println!("Stereo pair: {w}x{h}, known disparity = {known_disparity}");

    let sgbm = SgbmStereo::new()
        .num_disparities(16)
        .block_size(5)
        .disp12_max_diff(-1);

    let disp = sgbm.compute(&left, &right).unwrap();
    let disp_data = disp.as_slice();

    // Disparity statistics
    let valid: Vec<f32> = disp_data.iter().copied().filter(|&d| d >= 0.0).collect();
    if valid.is_empty() {
        println!("No valid disparities computed (all invalidated)");
        return;
    }

    let min = valid.iter().copied().fold(f32::INFINITY, f32::min);
    let max = valid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = valid.iter().copied().sum::<f32>() / valid.len() as f32;
    let invalid_count = disp_data.len() - valid.len();

    println!("Disparity map: {w}x{h}");
    println!("  valid pixels:   {}", valid.len());
    println!("  invalid pixels: {invalid_count}");
    println!("  min disparity:  {min:.2}");
    println!("  max disparity:  {max:.2}");
    println!("  mean disparity: {mean:.2}");
    println!("  expected:       {known_disparity}");

    // Center-region accuracy
    let margin = 20u32;
    let mut center_sum = 0.0f64;
    let mut center_count = 0u32;
    for y in margin..h - margin {
        for x in margin..w - margin {
            let d = disp.pixel(x, y)[0];
            if d >= 0.0 {
                center_sum += d as f64;
                center_count += 1;
            }
        }
    }
    if center_count > 0 {
        let center_mean = center_sum / center_count as f64;
        println!(
            "  center mean:    {center_mean:.2} (error: {:.2})",
            (center_mean - known_disparity as f64).abs()
        );
    }
}

#[cfg(not(feature = "stereo"))]
fn main() {
    eprintln!("This example requires the 'stereo' feature:");
    eprintln!("  cargo run -p scry-cv --example stereo_demo --features stereo");
}
