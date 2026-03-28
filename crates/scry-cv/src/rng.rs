// SPDX-License-Identifier: MIT OR Apache-2.0
//! Minimal xoshiro256** PRNG — deterministic sampling for RANSAC and tests.

use std::ops::{Range, RangeInclusive};

/// Xoshiro256** PRNG seeded via `SplitMix64`.
#[allow(clippy::redundant_pub_crate)]
pub(crate) struct FastRng {
    s: [u64; 4],
}

impl FastRng {
    /// Seed the generator via `SplitMix64`.
    pub(crate) fn new(seed: u64) -> Self {
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
    pub(crate) fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform f32 in \[0, 1).
    pub(crate) fn f32(&mut self) -> f32 {
        self.f64() as f32
    }

    /// Random `usize` in the given range.
    pub(crate) fn usize(&mut self, range: impl UsizeRange) -> usize {
        let (start, len) = range.start_and_len();
        debug_assert!(len > 0, "empty range");
        start + (self.next_u64() as usize) % len
    }

    /// Shuffle a slice in-place (Fisher-Yates).
    pub(crate) fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

/// Trait that lets `usize()` accept both `Range<usize>` and `RangeInclusive<usize>`.
#[allow(clippy::redundant_pub_crate)]
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
    fn deterministic_sequence() {
        let mut a = FastRng::new(42);
        let mut b = FastRng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

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
}
