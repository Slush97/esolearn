// SPDX-License-Identifier: MIT OR Apache-2.0
//! Stereo disparity estimation.
//!
//! Semi-Global Block Matching (SGBM) for dense disparity maps from
//! rectified stereo image pairs.

pub mod census;
pub mod cost_volume;
pub mod sgbm;

pub use sgbm::SgbmStereo;
