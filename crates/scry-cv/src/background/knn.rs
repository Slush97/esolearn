// SPDX-License-Identifier: MIT OR Apache-2.0
//! KNN background subtractor.
//!
//! Per-pixel background model based on a history of recent samples. A new pixel
//! is classified by counting how many of the K nearest historical samples fall
//! within a distance threshold.
//!
//! Simpler and often more robust than MOG2 for scenes with multi-modal
//! backgrounds (e.g. swaying trees, water).

use crate::background::BackgroundSubtractor;
use crate::error::Result;
use crate::image::{Gray, ImageBuf};

/// Number of history samples per pixel.
const HISTORY_SIZE: usize = 50;

/// KNN background subtractor.
///
/// # Example
///
/// ```
/// use scry_cv::background::{KnnBackground, BackgroundSubtractor};
/// use scry_cv::prelude::*;
///
/// let mut knn = KnnBackground::new(32, 32);
/// let frame = GrayImageF::new(32, 32).unwrap();
/// let mask = knn.apply(&frame, -1.0).unwrap();
/// ```
pub struct KnnBackground {
    width: u32,
    height: u32,
    /// Per-pixel sample history: `[pixel_idx * HISTORY_SIZE + sample_idx]`.
    history: Vec<f32>,
    /// Number of valid samples per pixel (grows up to `HISTORY_SIZE`).
    n_samples: Vec<u16>,
    /// Ring-buffer write index per pixel.
    write_idx: Vec<u16>,
    /// Distance threshold squared for matching.
    dist2_threshold: f32,
    /// Minimum number of close neighbors to classify as background.
    n_matches: u32,
    /// Default learning rate (probability of updating per frame).
    default_lr: f32,
    /// Shadow detection enabled.
    detect_shadows: bool,
    /// Shadow threshold (brightness ratio).
    shadow_threshold: f32,
    /// Frame count.
    n_frames: u32,
}

impl KnnBackground {
    /// Create a new KNN subtractor for the given frame dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        let n = width as usize * height as usize;
        Self {
            width,
            height,
            history: vec![0.0; n * HISTORY_SIZE],
            n_samples: vec![0; n],
            write_idx: vec![0; n],
            dist2_threshold: 0.02,
            n_matches: 2,
            default_lr: 0.05,
            detect_shadows: true,
            shadow_threshold: 0.5,
            n_frames: 0,
        }
    }

    /// Set distance threshold (default 0.02).
    #[must_use]
    pub fn dist2_threshold(mut self, v: f32) -> Self {
        self.dist2_threshold = v;
        self
    }

    /// Set minimum matches for background classification (default 2).
    #[must_use]
    pub fn n_matches(mut self, v: u32) -> Self {
        self.n_matches = v;
        self
    }

    /// Enable or disable shadow detection (default true).
    #[must_use]
    pub fn detect_shadows(mut self, v: bool) -> Self {
        self.detect_shadows = v;
        self
    }
}

impl BackgroundSubtractor for KnnBackground {
    fn apply(
        &mut self,
        frame: &ImageBuf<f32, Gray>,
        learning_rate: f64,
    ) -> Result<ImageBuf<u8, Gray>> {
        let data = frame.as_slice();
        let n = self.width as usize * self.height as usize;

        let lr = if learning_rate < 0.0 {
            self.default_lr
        } else {
            learning_rate as f32
        };

        let mut mask = vec![0u8; n];

        for i in 0..n {
            let value = data[i];
            let base = i * HISTORY_SIZE;
            let ns = self.n_samples[i] as usize;

            // Count matches in the history
            let mut matches = 0u32;
            let mut closest_bg_mean = 0.0f32;
            let mut closest_bg_count = 0u32;

            for k in 0..ns {
                let diff = value - self.history[base + k];
                if diff * diff < self.dist2_threshold {
                    matches += 1;
                    closest_bg_mean += self.history[base + k];
                    closest_bg_count += 1;
                }
            }

            let is_bg = matches >= self.n_matches;

            if is_bg {
                // Shadow detection
                if self.detect_shadows && closest_bg_count > 0 {
                    let bg_mean = closest_bg_mean / closest_bg_count as f32;
                    if bg_mean > 1e-6 {
                        let ratio = value / bg_mean;
                        if ratio > self.shadow_threshold && ratio < 1.0 {
                            mask[i] = 127; // shadow
                        }
                        // else: background (0)
                    }
                }
                // else: background (0)
            } else {
                mask[i] = 255; // foreground
            }

            // Update history via ring buffer.
            // Every pixel is updated every frame; the learning rate controls
            // how quickly old samples are overwritten (via HISTORY_SIZE).
            if lr > 0.0 {
                let widx = self.write_idx[i] as usize;
                self.history[base + widx] = value;
                self.write_idx[i] = ((widx + 1) % HISTORY_SIZE) as u16;
                if (self.n_samples[i] as usize) < HISTORY_SIZE {
                    self.n_samples[i] += 1;
                }
            }
        }

        self.n_frames += 1;
        ImageBuf::from_vec(mask, self.width, self.height)
    }

    fn background(&self) -> Result<ImageBuf<f32, Gray>> {
        let n = self.width as usize * self.height as usize;
        let mut bg = vec![0.0f32; n];

        for i in 0..n {
            let base = i * HISTORY_SIZE;
            let ns = self.n_samples[i] as usize;
            if ns > 0 {
                let sum: f32 = self.history[base..base + ns].iter().sum();
                bg[i] = sum / ns as f32;
            }
        }

        ImageBuf::from_vec(bg, self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converges_on_static_scene() {
        let w = 16u32;
        let h = 16u32;
        let n = (w * h) as usize;
        let frame = ImageBuf::<f32, Gray>::from_vec(vec![0.4; n], w, h).unwrap();

        let mut knn = KnnBackground::new(w, h);

        let mut mask = ImageBuf::<u8, Gray>::new(w, h).unwrap();
        for _ in 0..60 {
            mask = knn.apply(&frame, -1.0).unwrap();
        }

        let fg_count = mask.as_slice().iter().filter(|&&v| v == 255).count();
        assert_eq!(
            fg_count, 0,
            "static scene should be all-background after convergence, got {fg_count} fg"
        );
    }

    #[test]
    fn detects_foreground_change() {
        let w = 16u32;
        let h = 16u32;
        let n = (w * h) as usize;

        let bg_frame = ImageBuf::<f32, Gray>::from_vec(vec![0.3; n], w, h).unwrap();
        let mut knn = KnnBackground::new(w, h);

        for _ in 0..60 {
            knn.apply(&bg_frame, -1.0).unwrap();
        }

        // Bright foreground
        let mut fg_data = vec![0.3f32; n];
        for y in 4..12 {
            for x in 4..12 {
                fg_data[y * w as usize + x] = 0.9;
            }
        }
        let fg_frame = ImageBuf::<f32, Gray>::from_vec(fg_data, w, h).unwrap();
        let mask = knn.apply(&fg_frame, 0.0).unwrap();

        let mask_data = mask.as_slice();
        let center_fg = (4..12usize)
            .flat_map(|y| (4..12usize).map(move |x| (y, x)))
            .filter(|&(y, x)| mask_data[y * w as usize + x] == 255)
            .count();

        assert!(
            center_fg > 32,
            "bright region should be foreground, got {center_fg}/64"
        );
    }

    #[test]
    fn background_model_matches_scene() {
        let w = 16u32;
        let h = 16u32;
        let bg_val = 0.7f32;
        let frame = ImageBuf::<f32, Gray>::from_vec(vec![bg_val; (w * h) as usize], w, h).unwrap();

        let mut knn = KnnBackground::new(w, h);
        for _ in 0..60 {
            knn.apply(&frame, -1.0).unwrap();
        }

        let bg = knn.background().unwrap();
        let avg: f32 = bg.as_slice().iter().sum::<f32>() / bg.as_slice().len() as f32;

        assert!(
            (avg - bg_val).abs() < 0.05,
            "background model should be ~{bg_val}, got {avg}"
        );
    }
}
