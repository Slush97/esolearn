// SPDX-License-Identifier: MIT OR Apache-2.0
//! Keypoint and descriptor types.

/// A detected interest point in an image.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct KeyPoint {
    /// X coordinate (sub-pixel).
    pub x: f32,
    /// Y coordinate (sub-pixel).
    pub y: f32,
    /// Scale (octave level or sigma).
    pub scale: f32,
    /// Orientation in radians.
    pub angle: f32,
    /// Detector response strength.
    pub response: f32,
    /// Octave index.
    pub octave: i32,
}

impl KeyPoint {
    /// Create a new keypoint.
    #[must_use]
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            scale: 1.0,
            angle: 0.0,
            response: 0.0,
            octave: 0,
        }
    }

    /// Set the response value.
    #[must_use]
    pub fn with_response(mut self, r: f32) -> Self {
        self.response = r;
        self
    }

    /// Set the orientation.
    #[must_use]
    pub fn with_angle(mut self, a: f32) -> Self {
        self.angle = a;
        self
    }

    /// Set the scale.
    #[must_use]
    pub fn with_scale(mut self, s: f32) -> Self {
        self.scale = s;
        self
    }

    /// Set the octave.
    #[must_use]
    pub fn with_octave(mut self, o: i32) -> Self {
        self.octave = o;
        self
    }
}

/// A binary feature descriptor (for ORB, BRISK, BRIEF).
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BinaryDescriptor {
    /// Bit-packed descriptor data.
    pub data: Vec<u8>,
}

impl BinaryDescriptor {
    /// Number of bits in the descriptor.
    #[inline]
    pub fn n_bits(&self) -> usize {
        self.data.len() * 8
    }

    /// Hamming distance to another binary descriptor.
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        self.data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| (a ^ b).count_ones())
            .sum()
    }
}

/// A floating-point feature descriptor (for SIFT-like).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FloatDescriptor {
    /// Descriptor vector.
    pub data: Vec<f32>,
}

impl FloatDescriptor {
    /// L2 (Euclidean) distance to another descriptor.
    pub fn l2_distance(&self, other: &Self) -> f32 {
        self.data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_distance_identical() {
        let d = BinaryDescriptor {
            data: vec![0xFF, 0x00, 0xAB],
        };
        assert_eq!(d.hamming_distance(&d), 0);
    }

    #[test]
    fn hamming_distance_opposite() {
        let a = BinaryDescriptor {
            data: vec![0x00, 0x00],
        };
        let b = BinaryDescriptor {
            data: vec![0xFF, 0xFF],
        };
        assert_eq!(a.hamming_distance(&b), 16);
    }

    #[test]
    fn l2_distance_identical() {
        let d = FloatDescriptor {
            data: vec![1.0, 2.0, 3.0],
        };
        assert!((d.l2_distance(&d)).abs() < 1e-6);
    }

    #[test]
    fn l2_distance_known() {
        let a = FloatDescriptor {
            data: vec![0.0, 0.0],
        };
        let b = FloatDescriptor {
            data: vec![3.0, 4.0],
        };
        assert!((a.l2_distance(&b) - 5.0).abs() < 1e-5);
    }
}
