// SPDX-License-Identifier: MIT OR Apache-2.0
//! MOG2 (Mixture of Gaussians) background subtractor.
//!
//! Reference: Zivkovic, "Improved Adaptive Gaussian Mixture Model for Background
//! Subtraction" (2004), and Zivkovic & van der Heijden, "Efficient Adaptive
//! Density Estimation per Image Pixel for the Task of Background Subtraction" (2006).
//!
//! Per-pixel adaptive Gaussian mixture model with automatic component count
//! selection and optional shadow detection.

use crate::background::BackgroundSubtractor;
use crate::error::Result;
use crate::image::{Gray, ImageBuf};

/// Maximum number of Gaussian components per pixel.
const MAX_COMPONENTS: usize = 5;

/// MOG2 background subtractor.
///
/// # Example
///
/// ```
/// use scry_cv::background::{Mog2, BackgroundSubtractor};
/// use scry_cv::prelude::*;
///
/// let mut mog = Mog2::new(64, 64);
/// let frame = GrayImageF::new(64, 64).unwrap();
/// let mask = mog.apply(&frame, -1.0).unwrap();
/// assert_eq!(mask.dimensions(), (64, 64));
/// ```
pub struct Mog2 {
    width: u32,
    height: u32,
    /// Per-pixel mixture components: \[`pixel_idx` * `MAX_COMPONENTS` + k\]
    weights: Vec<f32>,
    means: Vec<f32>,
    variances: Vec<f32>,
    /// Number of active components per pixel.
    n_components: Vec<u8>,
    /// Background ratio threshold.
    bg_ratio: f32,
    /// Variance threshold for matching (Mahalanobis distance squared).
    var_threshold: f32,
    /// Default learning rate.
    default_lr: f32,
    /// Initial variance for new components.
    initial_variance: f32,
    /// Minimum variance (prevents collapse).
    min_variance: f32,
    /// Shadow detection enabled.
    detect_shadows: bool,
    /// Shadow threshold: ratio of pixel to background (brightness).
    shadow_threshold: f32,
    /// Number of frames processed.
    n_frames: u32,
}

impl Mog2 {
    /// Create a new MOG2 subtractor for the given frame dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        let n = width as usize * height as usize;
        let total = n * MAX_COMPONENTS;

        let mut weights = vec![0.0f32; total];
        let means = vec![0.0f32; total];
        let mut variances = vec![0.0f32; total];
        let mut n_components = vec![0u8; n];

        // Initialize first component for each pixel
        let initial_var = 0.02;
        for i in 0..n {
            weights[i * MAX_COMPONENTS] = 1.0;
            variances[i * MAX_COMPONENTS] = initial_var;
            n_components[i] = 1;
        }

        Self {
            width,
            height,
            weights,
            means,
            variances,
            n_components,
            bg_ratio: 0.9,
            var_threshold: 16.0,
            default_lr: 0.005,
            initial_variance: initial_var,
            min_variance: 0.001,
            detect_shadows: true,
            shadow_threshold: 0.5,
            n_frames: 0,
        }
    }

    /// Set background ratio threshold (default 0.9).
    #[must_use]
    pub fn bg_ratio(mut self, v: f32) -> Self {
        self.bg_ratio = v;
        self
    }

    /// Set variance threshold for matching (default 16.0).
    #[must_use]
    pub fn var_threshold(mut self, v: f32) -> Self {
        self.var_threshold = v;
        self
    }

    /// Enable or disable shadow detection (default true).
    #[must_use]
    pub fn detect_shadows(mut self, v: bool) -> Self {
        self.detect_shadows = v;
        self
    }

    /// Process a single pixel, returning the foreground classification.
    /// Returns: 0 = background, 127 = shadow, 255 = foreground.
    fn process_pixel(&mut self, pixel_idx: usize, value: f32, alpha: f32) -> u8 {
        let base = pixel_idx * MAX_COMPONENTS;
        let nk = self.n_components[pixel_idx] as usize;

        let mut matched = false;
        let mut match_k = 0;

        // Try to match the pixel to an existing component
        for k in 0..nk {
            let idx = base + k;
            let diff = value - self.means[idx];
            let var = self.variances[idx];

            if diff * diff < self.var_threshold * var {
                // Match found — update this component
                matched = true;
                match_k = k;

                let rho = alpha / self.weights[idx].max(1e-10);
                self.means[idx] += rho * diff;
                self.variances[idx] = ((1.0 - rho) * var + rho * diff * diff)
                    .max(self.min_variance);

                // Update weight
                self.weights[idx] += alpha * (1.0 - self.weights[idx]);
                // Decrease other weights
                for j in 0..nk {
                    if j != k {
                        self.weights[base + j] *= 1.0 - alpha;
                    }
                }
                break;
            }
        }

        if !matched {
            // No match: replace the weakest component or add new one
            if nk < MAX_COMPONENTS {
                // Add new component
                let idx = base + nk;
                self.weights[idx] = alpha;
                self.means[idx] = value;
                self.variances[idx] = self.initial_variance;
                self.n_components[pixel_idx] = (nk + 1) as u8;
            } else {
                // Replace the component with the lowest weight/variance ratio
                let mut min_k = 0;
                let mut min_score = f32::MAX;
                for k in 0..nk {
                    let score = self.weights[base + k];
                    if score < min_score {
                        min_score = score;
                        min_k = k;
                    }
                }
                let idx = base + min_k;
                self.weights[idx] = alpha;
                self.means[idx] = value;
                self.variances[idx] = self.initial_variance;
            }

            // Normalize weights
            let sum: f32 = (0..self.n_components[pixel_idx] as usize)
                .map(|k| self.weights[base + k])
                .sum();
            if sum > 1e-10 {
                let inv = 1.0 / sum;
                for k in 0..self.n_components[pixel_idx] as usize {
                    self.weights[base + k] *= inv;
                }
            }

            return 255; // foreground
        }

        // Normalize weights
        let nk = self.n_components[pixel_idx] as usize;
        let sum: f32 = (0..nk).map(|k| self.weights[base + k]).sum();
        if sum > 1e-10 {
            let inv = 1.0 / sum;
            for k in 0..nk {
                self.weights[base + k] *= inv;
            }
        }

        // Determine if the matched component is background
        // Sort components by weight/variance (descending weight) and check if
        // the cumulative weight of top components exceeds bg_ratio.
        let mut sorted: Vec<(f32, usize)> = (0..nk)
            .map(|k| (self.weights[base + k], k))
            .collect();
        sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut cum_weight = 0.0f32;
        let mut is_bg = false;
        for &(w, k) in &sorted {
            cum_weight += w;
            if k == match_k {
                is_bg = true;
                break;
            }
            if cum_weight > self.bg_ratio {
                break;
            }
        }

        if is_bg {
            // Check for shadow
            if self.detect_shadows {
                let bg_mean = self.means[base + match_k];
                if bg_mean > 1e-6 {
                    let ratio = value / bg_mean;
                    if ratio > self.shadow_threshold && ratio < 1.0 {
                        return 127; // shadow
                    }
                }
            }
            0 // background
        } else {
            255 // foreground
        }
    }
}

impl BackgroundSubtractor for Mog2 {
    fn apply(
        &mut self,
        frame: &ImageBuf<f32, Gray>,
        learning_rate: f64,
    ) -> Result<ImageBuf<u8, Gray>> {
        let data = frame.as_slice();
        let n = self.width as usize * self.height as usize;

        let alpha = if learning_rate < 0.0 {
            // Use default, decaying with frame count for initial adaptation
            if self.n_frames < 100 {
                1.0 / (self.n_frames as f32 + 1.0)
            } else {
                self.default_lr
            }
        } else {
            learning_rate as f32
        };

        let mut mask = vec![0u8; n];
        for i in 0..n {
            mask[i] = self.process_pixel(i, data[i], alpha);
        }

        self.n_frames += 1;
        ImageBuf::from_vec(mask, self.width, self.height)
    }

    fn background(&self) -> Result<ImageBuf<f32, Gray>> {
        let n = self.width as usize * self.height as usize;
        let mut bg = vec![0.0f32; n];

        for (i, bg_val) in bg.iter_mut().enumerate() {
            let base = i * MAX_COMPONENTS;
            let nk = self.n_components[i] as usize;

            // Background = weighted mean of the top components
            let mut max_w = 0.0f32;
            let mut best_mean = 0.0f32;
            for k in 0..nk {
                if self.weights[base + k] > max_w {
                    max_w = self.weights[base + k];
                    best_mean = self.means[base + k];
                }
            }
            *bg_val = best_mean;
        }

        ImageBuf::from_vec(bg, self.width, self.height)
    }
}

#[cfg(test)]
#[allow(clippy::naive_bytecount)]
mod tests {
    use super::*;

    #[test]
    fn converges_on_static_scene() {
        let w = 16u32;
        let h = 16u32;
        let bg_val = 0.5f32;
        let frame = ImageBuf::<f32, Gray>::from_vec(vec![bg_val; (w * h) as usize], w, h).unwrap();

        let mut mog = Mog2::new(w, h);

        // Feed the same frame many times
        let mut mask = ImageBuf::<u8, Gray>::new(w, h).unwrap();
        for _ in 0..50 {
            mask = mog.apply(&frame, -1.0).unwrap();
        }

        // After convergence, all pixels should be background
        let fg_count = mask.as_slice().iter().filter(|&&v| v == 255).count();
        assert_eq!(
            fg_count, 0,
            "static scene should converge to all-background, got {fg_count} foreground pixels"
        );
    }

    #[test]
    fn detects_foreground_change() {
        let w = 16u32;
        let h = 16u32;
        let n = (w * h) as usize;
        let bg_frame = ImageBuf::<f32, Gray>::from_vec(vec![0.3; n], w, h).unwrap();

        let mut mog = Mog2::new(w, h);

        // Train on background
        for _ in 0..50 {
            mog.apply(&bg_frame, -1.0).unwrap();
        }

        // Introduce a foreground object (bright spot)
        let mut fg_data = vec![0.3f32; n];
        for y in 4..12 {
            for x in 4..12 {
                fg_data[y * w as usize + x] = 0.9;
            }
        }
        let fg_frame = ImageBuf::<f32, Gray>::from_vec(fg_data, w, h).unwrap();
        let mask = mog.apply(&fg_frame, -1.0).unwrap();

        // Center pixels should be foreground
        let mask_data = mask.as_slice();
        let center_fg = (4..12usize)
            .flat_map(|y| (4..12usize).map(move |x| (y, x)))
            .filter(|&(y, x)| mask_data[y * w as usize + x] == 255)
            .count();

        assert!(
            center_fg > 32,
            "bright spot should be detected as foreground, got {center_fg}/64"
        );
    }

    #[test]
    fn background_model_matches_scene() {
        let w = 16u32;
        let h = 16u32;
        let bg_val = 0.6f32;
        let frame = ImageBuf::<f32, Gray>::from_vec(vec![bg_val; (w * h) as usize], w, h).unwrap();

        let mut mog = Mog2::new(w, h);
        for _ in 0..100 {
            mog.apply(&frame, -1.0).unwrap();
        }

        let bg = mog.background().unwrap();
        let avg: f32 =
            bg.as_slice().iter().sum::<f32>() / bg.as_slice().len() as f32;

        assert!(
            (avg - bg_val).abs() < 0.05,
            "background model should match scene value {bg_val}, got {avg}"
        );
    }

    #[test]
    fn shadow_detection() {
        let w = 16u32;
        let h = 16u32;
        let n = (w * h) as usize;

        let bg_frame = ImageBuf::<f32, Gray>::from_vec(vec![0.8; n], w, h).unwrap();
        let mut mog = Mog2::new(w, h)
            .detect_shadows(true)
            .var_threshold(36.0); // wider threshold so shadow pixels still match

        for _ in 0..100 {
            mog.apply(&bg_frame, -1.0).unwrap();
        }

        // Shadow: slightly darker version of background (ratio ~0.9)
        // Must be close enough to match the component but darker
        let shadow_frame =
            ImageBuf::<f32, Gray>::from_vec(vec![0.72; n], w, h).unwrap();
        let mask = mog.apply(&shadow_frame, 0.0).unwrap();

        let shadow_count = mask.as_slice().iter().filter(|&&v| v == 127).count();
        assert!(
            shadow_count > n / 4,
            "darkened scene should be classified as shadow, got {shadow_count}/{n}"
        );
    }
}
