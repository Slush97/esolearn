// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2D affine transforms (3×2 matrix).

/// A 2D affine transformation matrix (3×2).
///
/// Stored as `[a, b, c, d, tx, ty]` representing:
/// ```text
/// | a  b  tx |
/// | c  d  ty |
/// | 0  0   1 |
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine2D {
    /// Scale/rotation element (0,0).
    pub a: f32,
    /// Scale/rotation element (0,1).
    pub b: f32,
    /// Scale/rotation element (1,0).
    pub c: f32,
    /// Scale/rotation element (1,1).
    pub d: f32,
    /// Translation X.
    pub tx: f32,
    /// Translation Y.
    pub ty: f32,
}

impl Affine2D {
    /// Identity transform.
    pub const IDENTITY: Self = Self {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        tx: 0.0,
        ty: 0.0,
    };

    /// Create a translation transform.
    pub fn translate(tx: f32, ty: f32) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            tx,
            ty,
        }
    }

    /// Create a scaling transform.
    pub fn scale(sx: f32, sy: f32) -> Self {
        Self {
            a: sx,
            b: 0.0,
            c: 0.0,
            d: sy,
            tx: 0.0,
            ty: 0.0,
        }
    }

    /// Create a rotation transform (angle in radians).
    pub fn rotate(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            a: c,
            b: -s,
            c: s,
            d: c,
            tx: 0.0,
            ty: 0.0,
        }
    }

    /// Compose: apply `self` then `other` (other × self).
    #[allow(clippy::suspicious_operation_groupings)]
    pub fn then(self, other: Self) -> Self {
        Self {
            a: (other.a * self.a) + (other.b * self.c),
            b: (other.a * self.b) + (other.b * self.d),
            c: (other.c * self.a) + (other.d * self.c),
            d: (other.c * self.b) + (other.d * self.d),
            tx: other.a * self.tx + other.b * self.ty + other.tx,
            ty: other.c * self.tx + other.d * self.ty + other.ty,
        }
    }

    /// Transform a point.
    pub fn apply(self, p: [f32; 2]) -> [f32; 2] {
        [
            self.a * p[0] + self.b * p[1] + self.tx,
            self.c * p[0] + self.d * p[1] + self.ty,
        ]
    }

    /// Convert to a column-major 3×3 matrix for GPU uniform upload.
    pub fn to_mat3_cols(self) -> [f32; 12] {
        // Column-major 3×3 with std140 padding (each column = vec4)
        [
            self.a, self.c, 0.0, 0.0, // col 0
            self.b, self.d, 0.0, 0.0, // col 1
            self.tx, self.ty, 1.0, 0.0, // col 2
        ]
    }
}

impl Default for Affine2D {
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity() {
        let p = Affine2D::IDENTITY.apply([3.0, 4.0]);
        assert!((p[0] - 3.0).abs() < 1e-6);
        assert!((p[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn translate() {
        let p = Affine2D::translate(10.0, 20.0).apply([1.0, 2.0]);
        assert!((p[0] - 11.0).abs() < 1e-6);
        assert!((p[1] - 22.0).abs() < 1e-6);
    }

    #[test]
    fn scale() {
        let p = Affine2D::scale(2.0, 3.0).apply([4.0, 5.0]);
        assert!((p[0] - 8.0).abs() < 1e-6);
        assert!((p[1] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn compose() {
        let t = Affine2D::translate(10.0, 0.0);
        let s = Affine2D::scale(2.0, 2.0);
        // Scale first, then translate
        let combined = s.then(t);
        let p = combined.apply([5.0, 0.0]);
        assert!((p[0] - 20.0).abs() < 1e-5);
    }

    #[test]
    fn rotate_90() {
        let r = Affine2D::rotate(std::f32::consts::FRAC_PI_2);
        let p = r.apply([1.0, 0.0]);
        assert!(p[0].abs() < 1e-5);
        assert!((p[1] - 1.0).abs() < 1e-5);
    }
}
