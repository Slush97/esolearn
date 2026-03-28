// SPDX-License-Identifier: MIT OR Apache-2.0
//! Connected component labeling and contour tracing.

pub mod contour;
pub mod labeling;

pub use contour::{find_contours, Contour};
pub use labeling::{connected_components, ComponentStats, Connectivity, ConnectedComponents};
