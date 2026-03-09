// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tick generation: "nice numbers" algorithm for readable axis ticks.

/// Computed tick marks for an axis.
#[derive(Clone, Debug)]
pub struct Ticks {
    /// Tick positions in data space.
    pub positions: Vec<f64>,
    /// Formatted tick labels.
    pub labels: Vec<String>,
}

/// Generate "nice" tick positions for a given data range.
///
/// Based on Paul Heckbert's "Nice Numbers for Graph Labels" algorithm.
/// Produces approximately `target_count` ticks at round intervals.
pub fn nice_ticks(min: f64, max: f64, target_count: usize) -> Ticks {
    if (max - min).abs() < 1e-15 {
        let label = format_tick(min);
        return Ticks {
            positions: vec![min],
            labels: vec![label],
        };
    }

    let target = target_count.max(2) as f64;
    let range = nice_num(max - min, false);
    let step = nice_num(range / (target - 1.0), true);

    let graph_min = (min / step).floor() * step;
    let graph_max = (max / step).ceil() * step;

    let mut positions = Vec::new();
    let mut v = graph_min;
    // Safety bound to prevent infinite loops
    let max_ticks = (target_count + 5) * 2;
    while v <= graph_max + step * 0.5 && positions.len() < max_ticks {
        positions.push(v);
        v += step;
    }

    let labels = positions.iter().map(|&v| format_tick(v)).collect();

    Ticks { positions, labels }
}

/// Generate nice tick positions for logarithmic axes.
pub fn nice_ticks_log(min: f64, max: f64) -> Ticks {
    let log_min = min.max(1e-15).log10().floor() as i32;
    let log_max = max.max(1e-15).log10().ceil() as i32;

    let mut positions = Vec::new();
    for exp in log_min..=log_max {
        positions.push(10.0_f64.powi(exp));
    }

    let labels = positions.iter().map(|&v| format_tick(v)).collect();
    Ticks { positions, labels }
}

/// Compute a "nice" number that is approximately equal to `x`.
///
/// If `round` is true, rounds to the nearest nice number.
/// If false, takes the ceiling.
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

/// Format a tick value as a concise label.
pub fn format_tick(value: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }

    let abs = value.abs();
    if !(1e-3..1e6).contains(&abs) {
        // Scientific notation for very large/small
        format!("{value:.2e}")
    } else if (value - value.round()).abs() < 1e-9 {
        // Integer-looking values
        format!("{}", value as i64)
    } else if abs >= 1.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.3}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nice_ticks_basic() {
        let ticks = nice_ticks(0.0, 100.0, 5);
        assert!(!ticks.positions.is_empty());
        assert!(ticks.positions[0] <= 0.0);
        assert!(*ticks.positions.last().unwrap() >= 100.0);
        // Steps should be round numbers
        if ticks.positions.len() >= 2 {
            let step = ticks.positions[1] - ticks.positions[0];
            assert!(step > 0.0);
        }
    }

    #[test]
    fn test_nice_ticks_small_range() {
        let ticks = nice_ticks(0.0, 1.0, 5);
        assert!(ticks.positions.len() >= 2);
    }

    #[test]
    fn test_format_tick() {
        assert_eq!(format_tick(0.0), "0");
        assert_eq!(format_tick(100.0), "100");
        assert_eq!(format_tick(2.5), "2.5");
    }

    #[test]
    fn test_nice_ticks_same_value() {
        let ticks = nice_ticks(5.0, 5.0, 5);
        assert_eq!(ticks.positions.len(), 1);
    }
}
