// SPDX-License-Identifier: MIT OR Apache-2.0
//! sRGB ↔ linear RGB gamma encode/decode.

/// Decode a single sRGB gamma-encoded channel to linear light.
pub fn decode(s: f32) -> f32 {
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Encode a single linear-light channel to sRGB gamma.
pub fn encode(l: f32) -> f32 {
    let l = l.clamp(0.0, 1.0);
    if l <= 0.003_130_8 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        for i in 0..=255 {
            let s = f32::from(i) / 255.0;
            let l = decode(s);
            let back = (encode(l) * 255.0 + 0.5) as u8;
            assert_eq!(back, i, "roundtrip failed for {i}");
        }
    }

    #[test]
    fn black_white() {
        assert!((decode(0.0)).abs() < 1e-6);
        assert!((decode(1.0) - 1.0).abs() < 1e-6);
        assert!((encode(0.0)).abs() < 1e-6);
        assert!((encode(1.0) - 1.0).abs() < 1e-6);
    }
}
