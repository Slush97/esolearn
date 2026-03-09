// SPDX-License-Identifier: MIT OR Apache-2.0
//! Minimal xoshiro256** PRNG — avoids pulling in `fastrand` at runtime.

use std::ops::{Range, RangeInclusive};

/// Xoshiro256** PRNG with Box-Muller normal generation.
pub(crate) struct FastRng {
    s: [u64; 4],
}

impl FastRng {
    /// Seed the generator via SplitMix64.
    pub fn new(seed: u64) -> Self {
        let mut state = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *slot = z ^ (z >> 31);
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform f64 in \[0, 1).
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller transform.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.f64().max(1e-300);
        let u2 = self.f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Random `usize` in the given range (exclusive or inclusive end).
    pub fn usize(&mut self, range: impl UsizeRange) -> usize {
        let (start, len) = range.start_and_len();
        debug_assert!(len > 0, "empty range");
        start + (self.next_u64() as usize) % len
    }

    /// Shuffle a slice in-place (Fisher-Yates).
    pub fn shuffle(&mut self, indices: &mut [usize]) {
        for i in (1..indices.len()).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            indices.swap(i, j);
        }
    }
}

/// Trait that lets `usize()` accept both `Range<usize>` and `RangeInclusive<usize>`.
pub(crate) trait UsizeRange {
    fn start_and_len(self) -> (usize, usize);
}

impl UsizeRange for Range<usize> {
    fn start_and_len(self) -> (usize, usize) {
        (self.start, self.end - self.start)
    }
}

impl UsizeRange for RangeInclusive<usize> {
    fn start_and_len(self) -> (usize, usize) {
        let (start, end) = (*self.start(), *self.end());
        (start, end - start + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_in_unit_interval() {
        let mut rng = FastRng::new(42);
        for _ in 0..1000 {
            let v = rng.f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn usize_range() {
        let mut rng = FastRng::new(7);
        for _ in 0..200 {
            let v = rng.usize(5..10);
            assert!((5..10).contains(&v));
        }
    }

    #[test]
    fn usize_range_inclusive() {
        let mut rng = FastRng::new(7);
        for _ in 0..200 {
            let v = rng.usize(0..=3);
            assert!(v <= 3);
        }
    }

    #[test]
    fn normal_distribution() {
        let mut rng = FastRng::new(42);
        let n = 10_000;
        let samples: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let var: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.1, "mean should be ~0, got {mean}");
        assert!((var - 1.0).abs() < 0.1, "variance should be ~1, got {var}");
    }
}
