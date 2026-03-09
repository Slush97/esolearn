// SPDX-License-Identifier: MIT OR Apache-2.0
//! Text processing and feature extraction for NLP tasks.
//!
//! Provides tokenization, count-based vectorization, and TF-IDF weighting.
//! All vectorizers produce sparse CSR matrices via [`crate::sparse::CsrMatrix`].
//!
//! # Example
//!
//! ```ignore
//! use scry_learn::text::{CountVectorizer, TfidfVectorizer};
//!
//! let docs = ["the cat sat", "the dog sat", "the cat played"];
//!
//! // Count vectorizer
//! let mut cv = CountVectorizer::new();
//! let counts = cv.fit_transform(&docs);
//!
//! // TF-IDF vectorizer
//! let mut tfidf = TfidfVectorizer::new();
//! let matrix = tfidf.fit_transform(&docs);
//! ```

pub mod count;
pub mod tfidf;
pub mod tokenizer;

pub use count::CountVectorizer;
pub use tfidf::{TfidfNorm, TfidfVectorizer};

use crate::dataset::Dataset;
use crate::sparse::CsrMatrix;

/// Convert a sparse CSR matrix (from a text vectorizer) into a [`Dataset`].
///
/// The CsrMatrix is row-major (documents × features). This function
/// transposes into column-major format and attaches the provided target
/// vector and feature names.
///
/// # Example
///
/// ```ignore
/// use scry_learn::text::{CountVectorizer, sparse_to_dataset};
///
/// let docs = ["good movie", "bad movie", "good film"];
/// let target = vec![1.0, 0.0, 1.0];
///
/// let mut cv = CountVectorizer::new();
/// let matrix = cv.fit_transform(&docs);
/// let dataset = sparse_to_dataset(&matrix, target, cv.get_feature_names(), "label");
/// ```
pub fn sparse_to_dataset(
    matrix: &CsrMatrix,
    target: Vec<f64>,
    feature_names: Vec<String>,
    target_name: &str,
) -> Dataset {
    let n_rows = matrix.n_rows();
    let n_cols = matrix.n_cols();
    assert_eq!(
        target.len(),
        n_rows,
        "target length must match number of documents"
    );

    // Convert row-major sparse to column-major dense.
    let dense_rows = matrix.to_dense(); // Vec<Vec<f64>>, [n_rows][n_cols]
    let mut features_col_major = vec![vec![0.0; n_rows]; n_cols];
    for (i, row) in dense_rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            features_col_major[j][i] = val;
        }
    }

    Dataset::new(features_col_major, target, feature_names, target_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_to_dataset_roundtrip() {
        let docs = ["good movie", "bad movie", "good film", "bad film"];
        let target = vec![1.0, 0.0, 1.0, 0.0];

        let mut cv = CountVectorizer::new();
        let matrix = cv.fit_transform(&docs);
        let dataset =
            sparse_to_dataset(&matrix, target.clone(), cv.get_feature_names(), "sentiment");

        assert_eq!(dataset.n_samples(), 4);
        assert_eq!(dataset.n_features(), cv.n_features());
        assert_eq!(dataset.target, target);
    }

    #[test]
    fn sparse_to_dataset_feeds_into_multinomial_nb() {
        let docs = [
            "good great awesome",
            "good nice wonderful",
            "bad terrible awful",
            "bad horrible nasty",
            "good fantastic",
            "bad disgusting",
        ];
        let target = vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0];

        let mut cv = CountVectorizer::new();
        let matrix = cv.fit_transform(&docs);
        let dataset = sparse_to_dataset(&matrix, target, cv.get_feature_names(), "sentiment");

        let mut nb = crate::naive_bayes::MultinomialNB::new();
        nb.fit(&dataset).unwrap();

        // Predict on training data (just checking it doesn't crash).
        let rows = dataset.feature_matrix();
        let preds = nb.predict(&rows).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn tfidf_to_dataset_feeds_into_logistic() {
        let docs = [
            "good great awesome",
            "good nice wonderful",
            "bad terrible awful",
            "bad horrible nasty",
            "good fantastic nice",
            "bad disgusting terrible",
        ];
        let target = vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0];

        let mut tfidf = TfidfVectorizer::new();
        let matrix = tfidf.fit_transform(&docs);
        let dataset = sparse_to_dataset(&matrix, target, tfidf.get_feature_names(), "sentiment");

        let mut lr = crate::linear::LogisticRegression::new();
        lr.fit(&dataset).unwrap();

        let rows = dataset.feature_matrix();
        let preds = lr.predict(&rows).unwrap();
        assert_eq!(preds.len(), 6);
    }
}
