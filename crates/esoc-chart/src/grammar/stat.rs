// SPDX-License-Identifier: MIT OR Apache-2.0
//! Statistical transforms applied to data before encoding.

/// A statistical transform.
#[derive(Clone, Debug)]
pub enum Stat {
    /// Identity (no transform).
    Identity,
    /// Bin data into histogram buckets.
    Bin {
        /// Number of bins.
        bins: usize,
    },
    /// Box plot statistics (quartiles, whiskers).
    BoxPlot,
    /// LOESS/local regression smoothing.
    Smooth {
        /// Bandwidth parameter.
        bandwidth: f64,
    },
    /// Aggregate (mean, sum, count, etc.).
    Aggregate {
        /// Aggregation function.
        func: AggregateFunc,
    },
}

/// Aggregation functions.
#[derive(Clone, Copy, Debug)]
pub enum AggregateFunc {
    /// Count of values.
    Count,
    /// Sum of values.
    Sum,
    /// Arithmetic mean.
    Mean,
    /// Median.
    Median,
    /// Minimum.
    Min,
    /// Maximum.
    Max,
}
