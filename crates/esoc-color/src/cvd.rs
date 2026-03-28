// SPDX-License-Identifier: MIT OR Apache-2.0
//! Color vision deficiency (CVD) simulation.
//!
//! - Protanopia/Deuteranopia: Viénot, Brettel & Mollon 1999 (3×3 in linear RGB)
//! - Tritanopia: Brettel, Viénot & Mollon 1997

use crate::Color;

/// Type of color vision deficiency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CvdType {
    /// Red-blind (missing L cones).
    Protanopia,
    /// Green-blind (missing M cones).
    Deuteranopia,
    /// Blue-blind (missing S cones).
    Tritanopia,
}

/// Simulate how a color appears to someone with a given CVD.
///
/// Input and output are linear RGB. The simulation matrices assume
/// fully dichromatic vision (severity = 1.0).
pub fn simulate(c: Color, cvd: CvdType) -> Color {
    let (r, g, b) = match cvd {
        CvdType::Protanopia => simulate_protan(c.r, c.g, c.b),
        CvdType::Deuteranopia => simulate_deutan(c.r, c.g, c.b),
        CvdType::Tritanopia => simulate_tritan(c.r, c.g, c.b),
    };
    Color::new(r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0), c.a)
}

/// Simulate with partial severity `[0, 1]`.
pub fn simulate_partial(c: Color, cvd: CvdType, severity: f32) -> Color {
    let s = severity.clamp(0.0, 1.0);
    let simulated = simulate(c, cvd);
    c.lerp(simulated, s)
}

// Viénot 1999 protanopia matrix (linear RGB)
fn simulate_protan(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (
        0.152_286_88 * r + 1.052_583_3 * g - 0.204_868_44 * b,
        0.114_503_35 * r + 0.786_281_2 * g + 0.099_215_44 * b,
        -0.003_882_363 * r - 0.048_116_41 * g + 1.051_998_8 * b,
    )
}

// Viénot 1999 deuteranopia matrix (linear RGB)
fn simulate_deutan(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (
        0.367_322_4 * r + 0.860_977_8 * g - 0.228_300_18 * b,
        0.280_851_5 * r + 0.672_684_6 * g + 0.046_463_87 * b,
        -0.011_819_782 * r + 0.042_940_71 * g + 0.968_879_1 * b,
    )
}

// Brettel 1997 tritanopia (simplified single-plane approximation)
fn simulate_tritan(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (
        1.255_528_4 * r - 0.076_749_1 * g - 0.178_779_3 * b,
        -0.078_411_46 * r + 0.930_809_6 * g + 0.147_601_8 * b,
        0.004_733_144 * r + 0.691_367_4 * g + 0.303_899_44 * b,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_unchanged() {
        // Gray should look the same under all CVD types
        let gray = Color::GRAY;
        for cvd in [
            CvdType::Protanopia,
            CvdType::Deuteranopia,
            CvdType::Tritanopia,
        ] {
            let sim = simulate(gray, cvd);
            assert!(
                (sim.r - gray.r).abs() < 0.05,
                "{cvd:?} changed gray.r: {:.3} → {:.3}",
                gray.r,
                sim.r
            );
        }
    }

    #[test]
    fn partial_severity_zero() {
        let c = Color::RED;
        let sim = simulate_partial(c, CvdType::Protanopia, 0.0);
        assert!((sim.r - c.r).abs() < 1e-5);
    }

    #[test]
    fn output_in_range() {
        let colors = [Color::RED, Color::GREEN, Color::BLUE, Color::WHITE];
        for c in colors {
            for cvd in [
                CvdType::Protanopia,
                CvdType::Deuteranopia,
                CvdType::Tritanopia,
            ] {
                let sim = simulate(c, cvd);
                assert!(sim.r >= 0.0 && sim.r <= 1.0, "{cvd:?} r={}", sim.r);
                assert!(sim.g >= 0.0 && sim.g <= 1.0, "{cvd:?} g={}", sim.g);
                assert!(sim.b >= 0.0 && sim.b <= 1.0, "{cvd:?} b={}", sim.b);
            }
        }
    }
}
