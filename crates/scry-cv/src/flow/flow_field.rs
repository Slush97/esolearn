// SPDX-License-Identifier: MIT OR Apache-2.0
//! Flow field types for optical flow results.

/// Dense optical flow field: per-pixel (vx, vy) displacement vectors.
///
/// Stored as two separate row-major buffers for cache-friendly access during
/// computation.
#[derive(Clone, Debug)]
pub struct FlowField {
    /// Horizontal displacement per pixel.
    pub vx: Vec<f32>,
    /// Vertical displacement per pixel.
    pub vy: Vec<f32>,
    /// Width of the flow field.
    pub width: u32,
    /// Height of the flow field.
    pub height: u32,
}

impl FlowField {
    /// Create a zero-initialized flow field.
    pub fn zeros(width: u32, height: u32) -> Self {
        let n = width as usize * height as usize;
        Self {
            vx: vec![0.0; n],
            vy: vec![0.0; n],
            width,
            height,
        }
    }

    /// Get the flow vector at `(x, y)`.
    #[inline]
    pub fn at(&self, x: u32, y: u32) -> (f32, f32) {
        let idx = y as usize * self.width as usize + x as usize;
        (self.vx[idx], self.vy[idx])
    }

    /// Set the flow vector at `(x, y)`.
    #[inline]
    pub fn set(&mut self, x: u32, y: u32, dx: f32, dy: f32) {
        let idx = y as usize * self.width as usize + x as usize;
        self.vx[idx] = dx;
        self.vy[idx] = dy;
    }

    /// Compute the average endpoint error against a ground truth flow.
    pub fn endpoint_error(&self, gt: &FlowField) -> f32 {
        assert_eq!(self.width, gt.width);
        assert_eq!(self.height, gt.height);
        let n = self.vx.len() as f32;
        let sum: f32 = self
            .vx
            .iter()
            .zip(&self.vy)
            .zip(gt.vx.iter().zip(&gt.vy))
            .map(|((&ex, &ey), (&gx, &gy))| (ex - gx).hypot(ey - gy))
            .sum();
        sum / n
    }
}

/// Result of sparse optical flow computation.
#[derive(Clone, Debug)]
pub struct SparseFlowResult {
    /// Tracked point positions in the next frame.
    pub next_pts: Vec<(f32, f32)>,
    /// Per-point status: `true` if tracking succeeded.
    pub status: Vec<bool>,
    /// Per-point tracking error (optional, algorithm-dependent).
    pub errors: Vec<f32>,
}
