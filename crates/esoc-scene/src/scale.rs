// SPDX-License-Identifier: MIT OR Apache-2.0
//! Scales: data → visual coordinate mapping.

/// A scale maps data values to visual coordinates.
///
/// Data stays f64 (scientific precision) until mapped through a Scale
/// to f32 visual coordinates.
#[derive(Clone, Debug)]
pub enum Scale {
    /// Linear mapping.
    Linear {
        /// Data domain `(min, max)`.
        domain: (f64, f64),
        /// Visual range `(min, max)` in pixels.
        range: (f32, f32),
    },
    /// Logarithmic mapping.
    Log {
        /// Data domain `(min, max)` — must be positive.
        domain: (f64, f64),
        /// Visual range `(min, max)`.
        range: (f32, f32),
        /// Log base (typically 10 or e).
        base: f64,
    },
    /// Band scale for categorical data.
    Band {
        /// Category labels.
        domain: Vec<String>,
        /// Visual range.
        range: (f32, f32),
        /// Padding between bands as fraction `[0, 1)`.
        padding: f32,
    },
    /// Time scale (Unix milliseconds).
    Time {
        /// Domain as Unix ms `(start, end)`.
        domain: (i64, i64),
        /// Visual range.
        range: (f32, f32),
    },
    /// Square root mapping (emphasizes smaller values).
    Sqrt {
        /// Data domain `(min, max)`.
        domain: (f64, f64),
        /// Visual range.
        range: (f32, f32),
    },
    /// Power mapping with configurable exponent.
    Power {
        /// Data domain `(min, max)`.
        domain: (f64, f64),
        /// Visual range.
        range: (f32, f32),
        /// Exponent (1 = linear, 2 = quadratic, 0.5 = sqrt).
        exponent: f64,
    },
    /// Symmetric log: handles positive, negative, and zero.
    Symlog {
        /// Data domain `(min, max)`.
        domain: (f64, f64),
        /// Visual range.
        range: (f32, f32),
        /// Linearity threshold constant.
        constant: f64,
    },
    /// Ordinal: maps discrete string values to specific positions.
    Ordinal {
        /// Ordered labels.
        domain: Vec<String>,
        /// Corresponding pixel positions.
        range: Vec<f32>,
    },
}

impl Scale {
    /// Map a continuous data value to a visual coordinate.
    pub fn map(&self, value: f64) -> f32 {
        match self {
            Self::Linear { domain, range } => {
                let t = if (domain.1 - domain.0).abs() < 1e-15 {
                    0.5
                } else {
                    (value - domain.0) / (domain.1 - domain.0)
                };
                range.0 + (range.1 - range.0) * t as f32
            }
            Self::Log {
                domain,
                range,
                base,
            } => {
                let log_val = value.max(1e-15).log(*base);
                let log_min = domain.0.max(1e-15).log(*base);
                let log_max = domain.1.max(1e-15).log(*base);
                let t = if (log_max - log_min).abs() < 1e-15 {
                    0.5
                } else {
                    (log_val - log_min) / (log_max - log_min)
                };
                range.0 + (range.1 - range.0) * t as f32
            }
            Self::Band {
                domain,
                range,
                padding,
            } => {
                // Return the center of the band
                if domain.is_empty() {
                    return (range.0 + range.1) * 0.5;
                }
                let total = range.1 - range.0;
                let n = domain.len() as f32;
                let band_width = total / (n + (n + 1.0) * padding);
                let step = band_width + band_width * padding;
                // Find index (default to 0)
                let idx = domain
                    .iter()
                    .position(|s| {
                        // Match against stringified value
                        let v_str = format!("{value}");
                        s == &v_str
                    })
                    .unwrap_or(0) as f32;
                range.0 + step * padding + idx * step + band_width * 0.5
            }
            Self::Time { domain, range } => {
                let t = if domain.1 == domain.0 {
                    0.5
                } else {
                    (value as i64 - domain.0) as f64 / (domain.1 - domain.0) as f64
                };
                range.0 + (range.1 - range.0) * t as f32
            }
            Self::Sqrt { domain, range } => {
                let sqrt_val = value.max(0.0).sqrt();
                let sqrt_min = domain.0.max(0.0).sqrt();
                let sqrt_max = domain.1.max(0.0).sqrt();
                let t = if (sqrt_max - sqrt_min).abs() < 1e-15 {
                    0.5
                } else {
                    (sqrt_val - sqrt_min) / (sqrt_max - sqrt_min)
                };
                range.0 + (range.1 - range.0) * t as f32
            }
            Self::Power { domain, range, exponent } => {
                let pow_val = value.max(0.0).powf(*exponent);
                let pow_min = domain.0.max(0.0).powf(*exponent);
                let pow_max = domain.1.max(0.0).powf(*exponent);
                let t = if (pow_max - pow_min).abs() < 1e-15 {
                    0.5
                } else {
                    (pow_val - pow_min) / (pow_max - pow_min)
                };
                range.0 + (range.1 - range.0) * t as f32
            }
            Self::Symlog { domain, range, constant } => {
                let symlog = |v: f64| v.signum() * (v.abs() / constant).ln_1p();
                let sl_val = symlog(value);
                let sl_min = symlog(domain.0);
                let sl_max = symlog(domain.1);
                let t = if (sl_max - sl_min).abs() < 1e-15 {
                    0.5
                } else {
                    (sl_val - sl_min) / (sl_max - sl_min)
                };
                range.0 + (range.1 - range.0) * t as f32
            }
            Self::Ordinal { domain, range } => {
                // Map by index lookup from stringified value
                let idx = domain
                    .iter()
                    .position(|s| {
                        let v_str = format!("{value}");
                        s == &v_str
                    })
                    .unwrap_or(0);
                range.get(idx).copied().unwrap_or(0.0)
            }
        }
    }

    /// Map a band category to its center position and width.
    pub fn map_band(&self, category: &str) -> Option<(f32, f32)> {
        match self {
            Self::Band {
                domain,
                range,
                padding,
            } => {
                let idx = domain.iter().position(|s| s == category)?;
                let total = range.1 - range.0;
                let n = domain.len() as f32;
                let band_width = total / (n + (n + 1.0) * padding);
                let step = band_width + band_width * padding;
                let center = range.0 + step * padding + idx as f32 * step + band_width * 0.5;
                Some((center, band_width))
            }
            _ => None,
        }
    }

    /// Invert: map from visual coordinate back to data value.
    pub fn invert(&self, visual: f32) -> f64 {
        match self {
            Self::Linear { domain, range } => {
                let t = if (range.1 - range.0).abs() < 1e-10 {
                    0.5
                } else {
                    (visual - range.0) / (range.1 - range.0)
                };
                domain.0 + (domain.1 - domain.0) * f64::from(t)
            }
            Self::Log {
                domain,
                range,
                base,
            } => {
                let t = if (range.1 - range.0).abs() < 1e-10 {
                    0.5
                } else {
                    (visual - range.0) / (range.1 - range.0)
                };
                let log_min = domain.0.max(1e-15).log(*base);
                let log_max = domain.1.max(1e-15).log(*base);
                let log_val = log_min + (log_max - log_min) * f64::from(t);
                base.powf(log_val)
            }
            Self::Time { domain, range } => {
                let t = if (range.1 - range.0).abs() < 1e-10 {
                    0.5
                } else {
                    (visual - range.0) / (range.1 - range.0)
                };
                domain.0 as f64 + (domain.1 - domain.0) as f64 * f64::from(t)
            }
            Self::Band { .. } | Self::Ordinal { .. } => 0.0, // Not invertible
            Self::Sqrt { domain, range } => {
                let t = if (range.1 - range.0).abs() < 1e-10 {
                    0.5
                } else {
                    (visual - range.0) / (range.1 - range.0)
                };
                let sqrt_min = domain.0.max(0.0).sqrt();
                let sqrt_max = domain.1.max(0.0).sqrt();
                let sqrt_val = sqrt_min + (sqrt_max - sqrt_min) * f64::from(t);
                sqrt_val * sqrt_val
            }
            Self::Power { domain, range, exponent } => {
                let t = if (range.1 - range.0).abs() < 1e-10 {
                    0.5
                } else {
                    (visual - range.0) / (range.1 - range.0)
                };
                let pow_min = domain.0.max(0.0).powf(*exponent);
                let pow_max = domain.1.max(0.0).powf(*exponent);
                let pow_val = pow_min + (pow_max - pow_min) * f64::from(t);
                pow_val.powf(1.0 / exponent)
            }
            Self::Symlog { domain, range, constant } => {
                let symlog = |v: f64| v.signum() * (v.abs() / constant).ln_1p();
                let t = if (range.1 - range.0).abs() < 1e-10 {
                    0.5
                } else {
                    (visual - range.0) / (range.1 - range.0)
                };
                let sl_min = symlog(domain.0);
                let sl_max = symlog(domain.1);
                let sl_val = sl_min + (sl_max - sl_min) * f64::from(t);
                // Inverse of symlog: sign(y) * c * (exp(|y|) - 1)
                sl_val.signum() * constant * (sl_val.abs()).exp_m1()
            }
        }
    }

    /// Generate nice tick positions.
    pub fn ticks(&self, target_count: usize) -> Vec<f64> {
        match self {
            Self::Linear { domain, .. } => nice_ticks_linear(domain.0, domain.1, target_count),
            Self::Log { domain, base, .. } => nice_ticks_log(domain.0, domain.1, *base),
            Self::Band { domain, .. } => (0..domain.len()).map(|i| i as f64).collect(),
            Self::Time { domain, .. } => nice_ticks_linear(domain.0 as f64, domain.1 as f64, target_count),
            Self::Sqrt { domain, .. } | Self::Power { domain, .. } | Self::Symlog { domain, .. } => {
                nice_ticks_linear(domain.0, domain.1, target_count)
            }
            Self::Ordinal { domain, .. } => (0..domain.len()).map(|i| i as f64).collect(),
        }
    }

    /// Extend the domain to nice round boundaries so ticks align with domain edges.
    ///
    /// For Linear scales, this expands the domain outward to the nearest nice
    /// tick boundaries (using the Heckbert algorithm). After nicing, `ticks()`
    /// will never produce values outside the domain.
    pub fn nice(&self, target_count: usize) -> Self {
        match self {
            Self::Linear { domain, range } => {
                let (min, max) = *domain;
                if (max - min).abs() < 1e-15 {
                    return self.clone();
                }
                let target = target_count.max(2) as f64;
                let r = nice_num(max - min, false);
                let step = nice_num(r / (target - 1.0), true);
                let nice_min = (min / step).floor() * step;
                let nice_max = (max / step).ceil() * step;
                Self::Linear {
                    domain: (nice_min, nice_max),
                    range: *range,
                }
            }
            // Other scale types: return unchanged for now
            _ => self.clone(),
        }
    }

    /// Format a tick value as a label.
    pub fn format_tick(&self, value: f64) -> String {
        match self {
            Self::Band { domain, .. } | Self::Ordinal { domain, .. } => {
                let idx = value as usize;
                domain.get(idx).cloned().unwrap_or_default()
            }
            _ => format_number(value),
        }
    }
}

/// Generate nice linear tick positions (Heckbert algorithm).
fn nice_ticks_linear(min: f64, max: f64, target_count: usize) -> Vec<f64> {
    if (max - min).abs() < 1e-15 {
        return vec![min];
    }

    let target = target_count.max(2) as f64;
    let range = nice_num(max - min, false);
    let step = nice_num(range / (target - 1.0), true);

    let graph_min = (min / step).floor() * step;
    let graph_max = (max / step).ceil() * step;

    let mut positions = Vec::new();
    let mut v = graph_min;
    let max_ticks = (target_count + 5) * 2;
    while v <= graph_max + step * 0.5 && positions.len() < max_ticks {
        positions.push(v);
        v += step;
    }
    positions
}

/// Generate log tick positions.
fn nice_ticks_log(min: f64, max: f64, base: f64) -> Vec<f64> {
    let log_min = min.max(1e-15).log(base).floor() as i32;
    let log_max = max.max(1e-15).log(base).ceil() as i32;
    (log_min..=log_max).map(|e| base.powi(e)).collect()
}

/// Compute a "nice" number approximately equal to `x`.
fn nice_num(x: f64, round: bool) -> f64 {
    let exp = x.abs().log10().floor();
    let frac = x / 10.0_f64.powf(exp);

    let nice_frac = if round {
        if frac < 1.5 {
            1.0
        } else if frac < 3.0 {
            2.0
        } else if frac < 7.0 {
            5.0
        } else {
            10.0
        }
    } else if frac <= 1.0 {
        1.0
    } else if frac <= 2.0 {
        2.0
    } else if frac <= 5.0 {
        5.0
    } else {
        10.0
    };

    nice_frac * 10.0_f64.powf(exp)
}

/// Format a number as a concise tick label.
///
/// Uses SI prefixes for large values, comma grouping for mid-range,
/// and scientific notation only for very small numbers.
pub fn format_number(value: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    let abs = value.abs();
    let sign = if value < 0.0 { "-" } else { "" };

    if abs >= 1e9 {
        let v = value / 1e9;
        return format_si(v, sign, "B");
    }
    if abs >= 1e6 {
        let v = value / 1e6;
        return format_si(v, sign, "M");
    }
    if abs >= 1e4 {
        // Comma-grouped integer
        let rounded = value.round() as i64;
        return format_with_commas(rounded);
    }
    if abs >= 1.0 {
        // Integer or one decimal
        if (value - value.round()).abs() < 1e-9 {
            return format!("{}", value as i64);
        }
        return format!("{value:.1}");
    }
    if abs >= 0.01 {
        return format!("{value:.2}");
    }
    if abs >= 1e-6 {
        // Small but not tiny — use enough decimals
        let decimals = (-abs.log10().floor() as usize) + 2;
        return format!("{value:.prec$}", prec = decimals.min(8));
    }
    // Very small: scientific notation
    format!("{value:.2e}")
}

/// Format a value with an SI suffix, trimming trailing zeros.
fn format_si(v: f64, sign: &str, suffix: &str) -> String {
    if (v.abs() - v.abs().round()).abs() < 0.05 {
        format!("{sign}{}{suffix}", v.abs().round() as i64)
    } else {
        format!("{sign}{:.1}{suffix}", v.abs())
    }
}

/// Format an integer with comma grouping (e.g. 12345 → "12,345").
fn format_with_commas(value: i64) -> String {
    let neg = value < 0;
    let s = value.unsigned_abs().to_string();
    let bytes = s.as_bytes();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    if neg {
        result.push('-');
    }
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(b as char);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_map() {
        let s = Scale::Linear {
            domain: (0.0, 100.0),
            range: (0.0, 500.0),
        };
        assert!((s.map(50.0) - 250.0).abs() < 1e-3);
        assert!((s.map(0.0)).abs() < 1e-3);
        assert!((s.map(100.0) - 500.0).abs() < 1e-3);
    }

    #[test]
    fn linear_invert() {
        let s = Scale::Linear {
            domain: (0.0, 100.0),
            range: (0.0, 500.0),
        };
        assert!((s.invert(250.0) - 50.0).abs() < 1e-3);
    }

    #[test]
    fn log_map() {
        let s = Scale::Log {
            domain: (1.0, 1000.0),
            range: (0.0, 300.0),
            base: 10.0,
        };
        assert!((s.map(1.0)).abs() < 1e-3);
        assert!((s.map(1000.0) - 300.0).abs() < 1e-3);
        // 10 should be at 1/3
        assert!((s.map(10.0) - 100.0).abs() < 1e-3);
    }

    #[test]
    fn band_map() {
        let s = Scale::Band {
            domain: vec!["A".into(), "B".into(), "C".into()],
            range: (0.0, 300.0),
            padding: 0.1,
        };
        let (center_a, width) = s.map_band("A").unwrap();
        let (center_b, _) = s.map_band("B").unwrap();
        assert!(center_a < center_b);
        assert!(width > 0.0);
    }

    #[test]
    fn sqrt_map() {
        let s = Scale::Sqrt {
            domain: (0.0, 100.0),
            range: (0.0, 500.0),
        };
        assert!((s.map(0.0)).abs() < 1e-3);
        assert!((s.map(100.0) - 500.0).abs() < 1e-3);
        // sqrt(25) = 5, sqrt(100) = 10, so t = 5/10 = 0.5 → 250
        assert!((s.map(25.0) - 250.0).abs() < 1e-3);
    }

    #[test]
    fn symlog_map() {
        let s = Scale::Symlog {
            domain: (-100.0, 100.0),
            range: (0.0, 500.0),
            constant: 1.0,
        };
        // 0 should map to midpoint
        let mid = s.map(0.0);
        assert!((mid - 250.0).abs() < 1e-3, "mid = {mid}");
        // Symmetric: map(-x) + map(x) should equal 2*mid
        let pos = s.map(50.0);
        let neg = s.map(-50.0);
        assert!((pos + neg - 500.0).abs() < 1e-2, "pos={pos}, neg={neg}");
    }

    #[test]
    fn power_map() {
        let s = Scale::Power {
            domain: (0.0, 10.0),
            range: (0.0, 100.0),
            exponent: 2.0,
        };
        assert!((s.map(0.0)).abs() < 1e-3);
        assert!((s.map(10.0) - 100.0).abs() < 1e-3);
        // Power(5, 2) = 25, Power(10, 2) = 100, t = 25/100 = 0.25 → 25
        assert!((s.map(5.0) - 25.0).abs() < 1e-3);
    }

    #[test]
    fn ordinal_map() {
        let s = Scale::Ordinal {
            domain: vec!["low".into(), "med".into(), "high".into()],
            range: vec![50.0, 150.0, 250.0],
        };
        // Ordinal maps by string lookup, these aren't numeric so we test map_band indirectly
        // The ticks function should return indices
        let ticks = s.ticks(3);
        assert_eq!(ticks, vec![0.0, 1.0, 2.0]);
        assert_eq!(s.format_tick(0.0), "low");
        assert_eq!(s.format_tick(2.0), "high");
    }

    #[test]
    fn nice_ticks() {
        let s = Scale::Linear {
            domain: (0.0, 100.0),
            range: (0.0, 500.0),
        };
        let ticks = s.ticks(5);
        assert!(!ticks.is_empty());
        assert!(ticks[0] <= 0.0);
        assert!(*ticks.last().unwrap() >= 100.0);
    }

    #[test]
    fn nice_expands_domain_to_tick_boundaries() {
        let s = Scale::Linear {
            domain: (3.7, 97.2),
            range: (0.0, 500.0),
        };
        let niced = s.nice(5);
        let Scale::Linear { domain, .. } = &niced else {
            panic!("expected Linear");
        };
        // Niced domain should be at nice round numbers that contain the original
        assert!(domain.0 <= 3.7, "niced min {} should be <= 3.7", domain.0);
        assert!(domain.1 >= 97.2, "niced max {} should be >= 97.2", domain.1);
        // All ticks from the niced scale should be within the niced domain
        let ticks = niced.ticks(5);
        for &t in &ticks {
            assert!(
                t >= domain.0 - 1e-9 && t <= domain.1 + 1e-9,
                "tick {} outside niced domain [{}, {}]",
                t, domain.0, domain.1,
            );
        }
    }
}
