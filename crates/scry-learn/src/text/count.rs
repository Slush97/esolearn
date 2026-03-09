// SPDX-License-Identifier: MIT OR Apache-2.0
//! Count-based text vectorizer.
//!
//! Converts a collection of text documents into a sparse matrix of token
//! counts, analogous to scikit-learn's `CountVectorizer`.

use crate::sparse::CsrMatrix;
use std::collections::HashMap;

/// Converts text documents into a sparse term-count matrix.
///
/// Each document becomes a row, each unique token a column. Cell values
/// are the number of times that token appears in that document.
///
/// # Example
///
/// ```ignore
/// use scry_learn::text::CountVectorizer;
///
/// let mut cv = CountVectorizer::new();
/// let docs = ["the cat sat", "the dog sat", "the cat played"];
/// let matrix = cv.fit_transform(&docs);
///
/// assert_eq!(matrix.n_rows(), 3);
/// assert_eq!(matrix.n_cols(), cv.vocabulary().len());
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CountVectorizer {
    /// Token → column index mapping.
    vocabulary: HashMap<String, usize>,
    /// Minimum document frequency (absolute count).
    min_df: usize,
    /// Maximum document frequency as a fraction of total documents.
    max_df: f64,
    /// N-gram range `(min_n, max_n)`.
    ngram_range: (usize, usize),
    /// Maximum number of features (vocabulary size).
    max_features: Option<usize>,
    /// If true, all non-zero counts are set to 1.
    binary: bool,
    /// Whether fit() has been called.
    fitted: bool,
}

impl CountVectorizer {
    /// Create a new `CountVectorizer` with default settings.
    pub fn new() -> Self {
        Self {
            vocabulary: HashMap::new(),
            min_df: 1,
            max_df: 1.0,
            ngram_range: (1, 1),
            max_features: None,
            binary: false,
            fitted: false,
        }
    }

    /// Set minimum document frequency (absolute). Tokens appearing in
    /// fewer documents are excluded. Default: 1.
    pub fn min_df(mut self, n: usize) -> Self {
        self.min_df = n.max(1);
        self
    }

    /// Set maximum document frequency as a fraction in `(0.0, 1.0]`.
    /// Tokens appearing in more than this fraction of documents are
    /// excluded. Default: 1.0 (no filtering).
    pub fn max_df(mut self, frac: f64) -> Self {
        self.max_df = frac.clamp(0.0, 1.0);
        self
    }

    /// Set n-gram range. Default: `(1, 1)` (unigrams only).
    pub fn ngram_range(mut self, min_n: usize, max_n: usize) -> Self {
        self.ngram_range = (min_n.max(1), max_n.max(min_n.max(1)));
        self
    }

    /// Limit vocabulary to the top `n` features by total frequency.
    /// Default: no limit.
    pub fn max_features(mut self, n: usize) -> Self {
        self.max_features = Some(n);
        self
    }

    /// If true, all non-zero counts become 1 (presence/absence).
    /// Default: false.
    pub fn binary(mut self, b: bool) -> Self {
        self.binary = b;
        self
    }

    /// Learn vocabulary from documents.
    pub fn fit<S: AsRef<str>>(&mut self, documents: &[S]) {
        let n_docs = documents.len();

        // Count document frequency for each token.
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let mut total_freq: HashMap<String, usize> = HashMap::new();

        for doc in documents {
            let tokens = super::tokenizer::default_tokenize(doc.as_ref());
            let grams = super::tokenizer::ngrams(&tokens, self.ngram_range);

            // Track unique tokens per document for doc frequency.
            let mut seen = std::collections::HashSet::new();
            for gram in &grams {
                if seen.insert(gram.clone()) {
                    *doc_freq.entry(gram.clone()).or_insert(0) += 1;
                }
                *total_freq.entry(gram.clone()).or_insert(0) += 1;
            }
        }

        // Apply min_df / max_df filters.
        let max_df_abs = (self.max_df * n_docs as f64).ceil() as usize;
        let mut candidates: Vec<(String, usize)> = total_freq
            .into_iter()
            .filter(|(token, _)| {
                let df = doc_freq.get(token).copied().unwrap_or(0);
                df >= self.min_df && df <= max_df_abs
            })
            .collect();

        // Sort by frequency descending, then alphabetically for stability.
        candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        // Apply max_features cap.
        if let Some(max_f) = self.max_features {
            candidates.truncate(max_f);
        }

        // Build vocabulary map.
        // Sort alphabetically for stable column ordering.
        candidates.sort_by(|a, b| a.0.cmp(&b.0));
        self.vocabulary.clear();
        for (idx, (token, _)) in candidates.into_iter().enumerate() {
            self.vocabulary.insert(token, idx);
        }

        self.fitted = true;
    }

    /// Transform documents into a sparse CSR matrix of counts.
    ///
    /// Panics if `fit()` has not been called.
    pub fn transform<S: AsRef<str>>(&self, documents: &[S]) -> CsrMatrix {
        assert!(
            self.fitted,
            "CountVectorizer: must call fit() before transform()"
        );

        let n_rows = documents.len();
        let n_cols = self.vocabulary.len();

        if n_rows == 0 || n_cols == 0 {
            return CsrMatrix::from_dense(&[]);
        }

        let mut triplet_rows = Vec::new();
        let mut triplet_cols = Vec::new();
        let mut triplet_vals = Vec::new();

        for (row_idx, doc) in documents.iter().enumerate() {
            let tokens = super::tokenizer::default_tokenize(doc.as_ref());
            let grams = super::tokenizer::ngrams(&tokens, self.ngram_range);

            // Count occurrences.
            let mut counts: HashMap<usize, f64> = HashMap::new();
            for gram in &grams {
                if let Some(&col) = self.vocabulary.get(gram) {
                    *counts.entry(col).or_insert(0.0) += 1.0;
                }
            }

            for (col, val) in counts {
                let v = if self.binary { 1.0 } else { val };
                triplet_rows.push(row_idx);
                triplet_cols.push(col);
                triplet_vals.push(v);
            }
        }

        CsrMatrix::from_triplets(&triplet_rows, &triplet_cols, &triplet_vals, n_rows, n_cols)
            .expect("CountVectorizer: internal CSR construction error")
    }

    /// Fit the vocabulary and transform in one step.
    pub fn fit_transform<S: AsRef<str>>(&mut self, documents: &[S]) -> CsrMatrix {
        self.fit(documents);
        self.transform(documents)
    }

    /// Return the learned vocabulary (token → column index).
    pub fn vocabulary(&self) -> &HashMap<String, usize> {
        &self.vocabulary
    }

    /// Return feature names sorted by column index.
    pub fn get_feature_names(&self) -> Vec<String> {
        let mut pairs: Vec<(&String, &usize)> = self.vocabulary.iter().collect();
        pairs.sort_by_key(|&(_, &idx)| idx);
        pairs.into_iter().map(|(name, _)| name.clone()).collect()
    }

    /// Number of features in the vocabulary.
    pub fn n_features(&self) -> usize {
        self.vocabulary.len()
    }

    /// Whether fit() has been called.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Tokenize and generate n-grams for a single document (internal helper).
    pub(crate) fn tokenize_doc(&self, text: &str) -> Vec<String> {
        let tokens = super::tokenizer::default_tokenize(text);
        super::tokenizer::ngrams(&tokens, self.ngram_range)
    }
}

impl Default for CountVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn fit_transform_basic() {
        let docs = ["the cat sat", "the dog sat", "the cat played"];
        let mut cv = CountVectorizer::new();
        let matrix = cv.fit_transform(&docs);

        assert_eq!(matrix.n_rows(), 3);
        assert_eq!(matrix.n_cols(), cv.vocabulary().len());
        // "the" appears in all 3 docs
        assert!(cv.vocabulary().contains_key("the"));
        assert!(cv.vocabulary().contains_key("cat"));
        assert!(cv.vocabulary().contains_key("dog"));
        assert!(cv.vocabulary().contains_key("sat"));
        assert!(cv.vocabulary().contains_key("played"));
        assert_eq!(cv.n_features(), 5); // the, cat, dog, sat, played
    }

    #[test]
    fn vocabulary_order() {
        let docs = ["b c a", "a b"];
        let mut cv = CountVectorizer::new();
        cv.fit(&docs);

        let names = cv.get_feature_names();
        assert_eq!(names, vec!["a", "b", "c"]); // alphabetical
    }

    #[test]
    fn counts_are_correct() {
        let docs = ["a a b"];
        let mut cv = CountVectorizer::new();
        let matrix = cv.fit_transform(&docs);
        let dense = matrix.to_dense();

        let a_idx = cv.vocabulary()["a"];
        let b_idx = cv.vocabulary()["b"];
        assert_eq!(dense[0][a_idx], 2.0);
        assert_eq!(dense[0][b_idx], 1.0);
    }

    #[test]
    fn binary_mode() {
        let docs = ["a a a b"];
        let mut cv = CountVectorizer::new().binary(true);
        let matrix = cv.fit_transform(&docs);
        let dense = matrix.to_dense();

        let a_idx = cv.vocabulary()["a"];
        assert_eq!(dense[0][a_idx], 1.0); // binary: max 1
    }

    #[test]
    fn min_df_filters() {
        let docs = ["a b c", "a b", "a"];
        let mut cv = CountVectorizer::new().min_df(2);
        cv.fit(&docs);

        assert!(cv.vocabulary().contains_key("a"));
        assert!(cv.vocabulary().contains_key("b"));
        assert!(!cv.vocabulary().contains_key("c")); // only in 1 doc
    }

    #[test]
    fn max_df_filters() {
        let docs = ["a b", "a c", "a d"];
        let mut cv = CountVectorizer::new().max_df(0.5);
        cv.fit(&docs);

        // "a" is in 100% of docs (> 50%), should be filtered
        assert!(!cv.vocabulary().contains_key("a"));
        assert!(cv.vocabulary().contains_key("b"));
    }

    #[test]
    fn max_features_limits() {
        let docs = ["a a a b b c"];
        let mut cv = CountVectorizer::new().max_features(2);
        cv.fit(&docs);

        assert_eq!(cv.n_features(), 2);
    }

    #[test]
    fn bigrams() {
        let docs = ["the cat sat"];
        let mut cv = CountVectorizer::new().ngram_range(2, 2);
        let matrix = cv.fit_transform(&docs);

        assert!(cv.vocabulary().contains_key("the cat"));
        assert!(cv.vocabulary().contains_key("cat sat"));
        assert_eq!(matrix.n_cols(), 2);
    }

    #[test]
    fn unigrams_and_bigrams() {
        let docs = ["the cat sat"];
        let mut cv = CountVectorizer::new().ngram_range(1, 2);
        cv.fit(&docs);

        // 3 unigrams + 2 bigrams = 5 features
        assert_eq!(cv.n_features(), 5);
    }

    #[test]
    fn transform_unseen_terms() {
        let train = ["the cat sat"];
        let test = ["the bird flew"];

        let mut cv = CountVectorizer::new();
        cv.fit(&train);

        let matrix = cv.transform(&test);
        let dense = matrix.to_dense();

        // "bird" and "flew" are not in vocabulary, should be ignored
        let the_idx = cv.vocabulary()["the"];
        assert_eq!(dense[0][the_idx], 1.0);

        // Total non-zero should be 1 (only "the")
        let nnz: f64 = dense[0].iter().sum();
        assert_eq!(nnz, 1.0);
    }

    #[test]
    fn empty_documents() {
        let docs: [&str; 0] = [];
        let mut cv = CountVectorizer::new();
        let matrix = cv.fit_transform(&docs);

        assert_eq!(matrix.n_rows(), 0);
        assert_eq!(matrix.n_cols(), 0);
    }

    #[test]
    fn string_refs_accepted() {
        // Verify it works with Vec<String> too, not just &[&str]
        let docs: Vec<String> = vec!["hello world".into(), "hello test".into()];
        let mut cv = CountVectorizer::new();
        let matrix = cv.fit_transform(&docs);
        assert_eq!(matrix.n_rows(), 2);
    }
}
