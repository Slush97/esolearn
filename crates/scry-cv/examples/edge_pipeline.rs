// SPDX-License-Identifier: MIT OR Apache-2.0
//! Edge detection → Hough lines → contour analysis on a synthetic scene.
//!
//! ```sh
//! cargo run -p scry-cv --example edge_pipeline
//! ```

use scry_cv::prelude::*;
use std::f32::consts::PI;

fn main() {
    // Generate a 100x100 image with a bright rectangle (30..70, 25..75)
    let (w, h) = (100u32, 100u32);
    let mut data = vec![0.0f32; (w * h) as usize];
    for y in 25..75u32 {
        for x in 30..70u32 {
            data[y as usize * w as usize + x as usize] = 1.0;
        }
    }
    let img = ImageBuf::<f32, Gray>::from_vec(data, w, h).unwrap();
    println!("Input: {w}x{h} image with 40x50 rectangle at (30,25)");

    // Canny edge detection
    let edges = canny(&img, 0.05, 0.15).unwrap();
    let edge_count: usize = edges.as_slice().iter().filter(|&&v| v > 0.5).count();
    println!("Canny: {edge_count} edge pixels detected");

    // Hough line detection
    let lines = hough_lines(&edges, 1.0, PI / 180.0, 15).unwrap();
    println!("Hough lines: {} lines detected", lines.len());
    for (i, line) in lines.iter().take(8).enumerate() {
        let deg = line.theta * 180.0 / PI;
        println!(
            "  line {i}: rho={:.1}, theta={:.1}°, votes={}",
            line.rho, deg, line.votes
        );
    }

    // Contour detection
    let contours = find_contours(&edges).unwrap();
    println!("Contours: {} found", contours.len());
    for (i, contour) in contours.iter().take(5).enumerate() {
        println!(
            "  contour {i}: {} points, outer={}",
            contour.points.len(),
            contour.is_outer
        );
    }

    // Connected components on the edge image
    let cc = connected_components(&edges, Connectivity::Eight).unwrap();
    println!("Connected components: {} foreground regions", cc.num_labels);
    for (i, stat) in cc.stats.iter().take(5).enumerate() {
        println!(
            "  component {}: area={}, bbox=({},{})..({},{}), centroid=({:.1},{:.1})",
            i + 1,
            stat.area,
            stat.bbox.0,
            stat.bbox.1,
            stat.bbox.2,
            stat.bbox.3,
            stat.centroid.0,
            stat.centroid.1
        );
    }
}
