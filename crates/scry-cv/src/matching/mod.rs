// SPDX-License-Identifier: MIT OR Apache-2.0
//! Descriptor matching: brute-force, ratio test, cross-check.

pub mod brute_force;
pub mod match_types;
pub mod ratio_test;
pub mod template;

pub use brute_force::{knn_match_binary, knn_match_float, match_binary, match_float};
pub use match_types::DMatch;
pub use ratio_test::{cross_check, ratio_test};
pub use template::{find_best_match, match_template, TemplateMatchMethod};
