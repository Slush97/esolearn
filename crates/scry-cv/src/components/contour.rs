// SPDX-License-Identifier: MIT OR Apache-2.0
//! Contour tracing on binary images using Moore neighborhood boundary following.
//!
//! Extracts the boundary pixel chains of foreground regions.

use crate::error::Result;
use crate::image::{Gray, ImageBuf};

/// A single contour (boundary) of a connected foreground region.
#[derive(Clone, Debug)]
pub struct Contour {
    /// Ordered boundary points (x, y).
    pub points: Vec<(u32, u32)>,
    /// `true` if this is an outer boundary, `false` for a hole boundary.
    pub is_outer: bool,
}

/// Find all contours in a binary image.
///
/// Input pixel values > 0.5 are treated as foreground. Returns a list of
/// contours ordered by discovery (top-to-bottom, left-to-right scan).
///
/// # Example
///
/// ```
/// use scry_cv::prelude::*;
/// use scry_cv::components::contour::find_contours;
///
/// let mut data = vec![0.0f32; 16 * 16];
/// for y in 3..7 { for x in 3..7 { data[y * 16 + x] = 1.0; } }
/// let img = GrayImageF::from_vec(data, 16, 16).unwrap();
/// let contours = find_contours(&img).unwrap();
/// assert_eq!(contours.len(), 1);
/// assert!(contours[0].is_outer);
/// ```
pub fn find_contours(img: &ImageBuf<f32, Gray>) -> Result<Vec<Contour>> {
    let w = img.width() as usize;
    let h = img.height() as usize;
    let data = img.as_slice();

    let fg: Vec<bool> = data.iter().map(|&v| v > 0.5).collect();
    let mut visited = vec![false; w * h];
    let mut contours = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !fg[idx] || visited[idx] {
                continue;
            }
            if !is_border(&fg, x, y, w, h) {
                continue;
            }

            let is_outer = x == 0 || !fg[y * w + x - 1];
            let pts = trace_contour(&fg, &mut visited, x, y, w, h);
            if !pts.is_empty() {
                contours.push(Contour {
                    points: pts,
                    is_outer,
                });
            }
        }
    }

    Ok(contours)
}

/// Check if a foreground pixel has at least one background 8-neighbor.
fn is_border(fg: &[bool], x: usize, y: usize, w: usize, h: usize) -> bool {
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx < 0 || nx >= w as i32 || ny < 0 || ny >= h as i32 {
                return true;
            }
            if !fg[ny as usize * w + nx as usize] {
                return true;
            }
        }
    }
    false
}

/// Moore neighborhood tracing starting from (`start_x`, `start_y`).
///
/// Uses the standard Moore-neighbor boundary tracing algorithm: at each
/// boundary pixel, search clockwise from the backtrack pixel (the last
/// background pixel) to find the next boundary pixel.
fn trace_contour(
    fg: &[bool],
    visited: &mut [bool],
    start_x: usize,
    start_y: usize,
    w: usize,
    h: usize,
) -> Vec<(u32, u32)> {
    // Moore neighborhood: 8 directions clockwise from right.
    const DIRS: [(i32, i32); 8] = [
        (1, 0),   // 0: right
        (1, 1),   // 1: down-right
        (0, 1),   // 2: down
        (-1, 1),  // 3: down-left
        (-1, 0),  // 4: left
        (-1, -1), // 5: up-left
        (0, -1),  // 6: up
        (1, -1),  // 7: up-right
    ];

    // When we find a foreground pixel at direction d from the current pixel,
    // the backtrack direction at the NEW pixel (direction from new pixel to
    // the last background pixel checked) is given by this lookup table.
    const BACKTRACK: [usize; 8] = [6, 6, 0, 0, 2, 2, 4, 4];

    let mut points = Vec::new();
    let mut cx = start_x;
    let mut cy = start_y;
    // Initial backtrack: the pixel to the left (found via left-to-right scan)
    let mut b_dir = 4usize;

    let max_steps = w * h * 2;
    for _ in 0..max_steps {
        visited[cy * w + cx] = true;
        points.push((cx as u32, cy as u32));

        let mut found = false;
        for i in 1..=8 {
            let d = (b_dir + i) % 8;
            let nx = cx as i32 + DIRS[d].0;
            let ny = cy as i32 + DIRS[d].1;

            if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                let nidx = ny as usize * w + nx as usize;
                if fg[nidx] {
                    b_dir = BACKTRACK[d];
                    cx = nx as usize;
                    cy = ny as usize;
                    found = true;
                    break;
                }
            }
        }

        if !found {
            break; // isolated pixel
        }

        if cx == start_x && cy == start_y {
            break;
        }
    }

    points
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangle_contour() {
        let (w, h) = (16u32, 16u32);
        let mut data = vec![0.0f32; (w * h) as usize];
        for y in 3..7 {
            for x in 3..7 {
                data[y * w as usize + x] = 1.0;
            }
        }
        let img = ImageBuf::<f32, Gray>::from_vec(data, w, h).unwrap();
        let contours = find_contours(&img).unwrap();

        assert_eq!(contours.len(), 1);
        assert!(contours[0].is_outer);
        // 4x4 block: perimeter = 12 border pixels
        assert!(
            contours[0].points.len() >= 12,
            "expected >= 12 border points, got {}",
            contours[0].points.len()
        );
    }

    #[test]
    fn empty_image_no_contours() {
        let img = ImageBuf::<f32, Gray>::new(16, 16).unwrap();
        let contours = find_contours(&img).unwrap();
        assert!(contours.is_empty());
    }

    #[test]
    fn single_pixel_contour() {
        let mut data = vec![0.0f32; 8 * 8];
        data[3 * 8 + 3] = 1.0;
        let img = ImageBuf::<f32, Gray>::from_vec(data, 8, 8).unwrap();
        let contours = find_contours(&img).unwrap();
        assert_eq!(contours.len(), 1);
        assert_eq!(contours[0].points.len(), 1);
        assert_eq!(contours[0].points[0], (3, 3));
    }

    #[test]
    fn two_separate_contours() {
        let (w, h) = (20u32, 20u32);
        let mut data = vec![0.0f32; (w * h) as usize];
        // Blob 1
        for y in 2..5 {
            for x in 2..5 {
                data[y * w as usize + x] = 1.0;
            }
        }
        // Blob 2
        for y in 10..13 {
            for x in 10..13 {
                data[y * w as usize + x] = 1.0;
            }
        }
        let img = ImageBuf::<f32, Gray>::from_vec(data, w, h).unwrap();
        let contours = find_contours(&img).unwrap();
        assert_eq!(contours.len(), 2);
    }
}
