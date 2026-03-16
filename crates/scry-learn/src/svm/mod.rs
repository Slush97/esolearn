// SPDX-License-Identifier: MIT OR Apache-2.0
//! Support Vector Machine classifiers and regressors.
//!
//! Provides [`LinearSVC`] and [`LinearSVR`] using Pegasos SGD.
//!
//! Kernel-based SVMs (`KernelSVC`, `KernelSVR`) are available behind
//! `feature = "experimental"` — they use O(n^2) SMO and are impractical
//! on datasets larger than ~2000 samples.

#[cfg(feature = "experimental")]
pub mod kernel;
#[cfg(feature = "experimental")]
pub mod kernel_svr;
pub mod linear;

#[cfg(feature = "experimental")]
pub use kernel::{Gamma, Kernel, KernelSVC};
#[cfg(feature = "experimental")]
pub use kernel_svr::KernelSVR;
pub use linear::{LinearSVC, LinearSVR};
