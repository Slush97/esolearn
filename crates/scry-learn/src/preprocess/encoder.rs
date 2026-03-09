// SPDX-License-Identifier: MIT OR Apache-2.0
//! Label encoding for categorical variables.

use crate::error::{Result, ScryLearnError};

/// Encode string labels as integer indices.
///
/// Maintains a bidirectional mapping between labels and their numeric indices.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct LabelEncoder {
    classes: Vec<String>,
    fitted: bool,
}

impl LabelEncoder {
    /// Create a new unfitted encoder.
    pub fn new() -> Self {
        Self {
            classes: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the encoder on a set of string labels.
    pub fn fit(&mut self, labels: &[&str]) {
        let mut unique: Vec<String> = labels
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        unique.sort();
        unique.dedup();
        self.classes = unique;
        self.fitted = true;
    }

    /// Transform string labels to numeric indices.
    pub fn transform(&self, labels: &[&str]) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        labels
            .iter()
            .map(|&label| {
                self.classes
                    .iter()
                    .position(|c| c == label)
                    .map(|i| i as f64)
                    .ok_or_else(|| {
                        ScryLearnError::InvalidParameter(format!("unknown label: {label}"))
                    })
            })
            .collect()
    }

    /// Reverse-transform numeric indices back to string labels.
    pub fn inverse_transform(&self, indices: &[f64]) -> Result<Vec<String>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        indices
            .iter()
            .map(|&idx| {
                let i = idx as usize;
                self.classes.get(i).cloned().ok_or_else(|| {
                    ScryLearnError::InvalidParameter(format!("index out of range: {i}"))
                })
            })
            .collect()
    }

    /// Get the list of known classes.
    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    /// Number of known classes.
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

impl Default for LabelEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_encoder_roundtrip() {
        let mut enc = LabelEncoder::new();
        enc.fit(&["cat", "dog", "bird", "cat"]);
        assert_eq!(enc.n_classes(), 3);

        let encoded = enc.transform(&["dog", "cat", "bird"]).unwrap();
        assert_eq!(encoded, vec![2.0, 1.0, 0.0]); // sorted: bird=0, cat=1, dog=2

        let decoded = enc.inverse_transform(&encoded).unwrap();
        assert_eq!(decoded, vec!["dog", "cat", "bird"]);
    }

    #[test]
    fn test_label_encoder_unknown() {
        let mut enc = LabelEncoder::new();
        enc.fit(&["a", "b"]);
        assert!(enc.transform(&["c"]).is_err());
    }
}
