// SPDX-License-Identifier: MIT OR Apache-2.0
//! Classical computer vision library for Rust.
//!
//! `scry-cv` targets verified gaps in the Rust CV ecosystem: feature
//! detection (ORB, BRISK), descriptor matching, optical flow (Farneback),
//! stereo disparity (SGBM), background subtraction (MOG2), region analysis,
//! and scientific imaging filters.
//!
//! # Quick Start
//!
//! ```rust
//! use scry_cv::prelude::*;
//!
//! // Create a grayscale f32 image
//! let img = GrayImageF::new(64, 64).unwrap();
//! assert_eq!(img.dimensions(), (64, 64));
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod components;
pub mod error;
pub mod features;
pub mod filter;
pub mod hough;
pub mod image;
pub mod integral;
pub mod matching;
pub mod math;
pub mod morphology;
pub mod prelude;
pub mod pyramid;
pub mod registration;

#[cfg(feature = "flow")]
pub mod flow;

#[cfg(feature = "background")]
pub mod background;

#[cfg(feature = "stereo")]
pub mod stereo;

pub(crate) mod rng;
