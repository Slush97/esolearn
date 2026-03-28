// SPDX-License-Identifier: MIT OR Apache-2.0
//! Source image generators — create synthetic test images procedurally.

use scry_cv::prelude::GrayImageF;

pub fn solid_color(value: f32, w: u32, h: u32) -> Result<GrayImageF, String> {
    GrayImageF::from_vec(vec![value; (w * h) as usize], w, h).map_err(|e| e.to_string())
}

pub fn checkerboard(cell_size: u32, w: u32, h: u32) -> Result<GrayImageF, String> {
    let cell = cell_size.max(1);
    let data: Vec<f32> = (0..h)
        .flat_map(|y| {
            (0..w).map(move |x| {
                if ((x / cell) + (y / cell)).is_multiple_of(2) {
                    0.9
                } else {
                    0.1
                }
            })
        })
        .collect();
    GrayImageF::from_vec(data, w, h).map_err(|e| e.to_string())
}

pub fn gradient(w: u32, h: u32) -> Result<GrayImageF, String> {
    let data: Vec<f32> = (0..h)
        .flat_map(|_| (0..w).map(move |x| x as f32 / (w - 1).max(1) as f32))
        .collect();
    GrayImageF::from_vec(data, w, h).map_err(|e| e.to_string())
}

pub fn rectangle(
    w: u32,
    h: u32,
    rx: u32,
    ry: u32,
    rw: u32,
    rh: u32,
) -> Result<GrayImageF, String> {
    let mut data = vec![0.0f32; (w * h) as usize];
    for y in ry..(ry + rh).min(h) {
        for x in rx..(rx + rw).min(w) {
            data[y as usize * w as usize + x as usize] = 1.0;
        }
    }
    GrayImageF::from_vec(data, w, h).map_err(|e| e.to_string())
}

pub fn gaussian_blob(w: u32, h: u32, sigma: f32) -> Result<GrayImageF, String> {
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let s2 = 2.0 * sigma * sigma;
    let data: Vec<f32> = (0..h)
        .flat_map(|y| {
            (0..w).map(move |x| {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                (-(dx * dx + dy * dy) / s2).exp()
            })
        })
        .collect();
    GrayImageF::from_vec(data, w, h).map_err(|e| e.to_string())
}

pub fn load_file(path: &str) -> Result<GrayImageF, String> {
    let dyn_img = image::open(path).map_err(|e| e.to_string())?;
    let gray = dyn_img.to_luma32f();
    let (w, h) = gray.dimensions();
    let data: Vec<f32> = gray.into_raw();
    GrayImageF::from_vec(data, w, h).map_err(|e| e.to_string())
}
