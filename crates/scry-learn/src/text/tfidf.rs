// SPDX-License-Identifier: MIT OR Apache-2.0
//! TF-IDF text vectorizer.
//!
//! Term Frequency–Inverse Document Frequency weighting, built on top of
//! [`CountVectorizer`]. Analogous to scikit-learn's `TfidfVectorizer`.

use super::count::CountVectorizer;
use crate::sparse::CsrMatrix;

/// Normalization method for TF-IDF vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TfidfNorm {
    /// L1 normalization (sum of absolute values = 1).
    L1,
    /// L2 normalization (Euclidean length = 1). Default.
    L2,
    /// No normalization.
    None,
}

/// TF-IDF text vectorizer.
///
/// Combines count vectorization with IDF weighting and optional
/// normalization. Produces a sparse CSR matrix.
///
/// # Example
///
/// ```ignore
/// use scry_learn::text::TfidfVectorizer;
///
/// let docs = ["the cat sat", "the dog sat", "the cat played"];
/// let mut tfidf = TfidfVectorizer::new();
/// let matrix = tfidf.fit_transform(&docs);
/// ```
#[derive(Debug, Clone)]
pub struct TfidfVectorizer {
    /// Underlying count vectorizer.
    count: CountVectorizer,
    /// Learned IDF weights (one per vocabulary term).
    idf_values: Vec<f64>,
    /// Normalization method.
    norm: TfidfNorm,
    /// If true, use `1 + log(tf)` instead of raw `tf`.
    sublinear_tf: bool,
    /// If true, add 1 to document frequencies to prevent zero division.
    smooth_idf: bool,
    /// Whether fit() has been called.
    fitted: bool,
}

impl TfidfVectorizer {
    /// Create a new `TfidfVectorizer` with default settings.
    pub fn new() -> Self {
        Self {
            count: CountVectorizer::new(),
            idf_values: Vec::new(),
            norm: TfidfNorm::L2,
            sublinear_tf: false,
            smooth_idf: true,
            fitted: false,
        }
    }

    /// Set minimum document frequency.
    pub fn min_df(mut self, n: usize) -> Self {
        self.count = self.count.min_df(n);
        self
    }

    /// Set maximum document frequency fraction.
    pub fn max_df(mut self, frac: f64) -> Self {
        self.count = self.count.max_df(frac);
        self
    }

    /// Set n-gram range.
    pub fn ngram_range(mut self, min_n: usize, max_n: usize) -> Self {
        self.count = self.count.ngram_range(min_n, max_n);
        self
    }

    /// Limit vocabulary size.
    pub fn max_features(mut self, n: usize) -> Self {
        self.count = self.count.max_features(n);
        self
    }

    /// Set normalization method. Default: L2.
    pub fn norm(mut self, norm: TfidfNorm) -> Self {
        self.norm = norm;
        self
    }

    /// Enable sublinear TF scaling: `tf = 1 + log(tf)`.
    pub fn sublinear_tf(mut self, enable: bool) -> Self {
        self.sublinear_tf = enable;
        self
    }

    /// Enable smooth IDF: adds 1 to document frequencies.
    /// Default: true (matches sklearn).
    pub fn smooth_idf(mut self, enable: bool) -> Self {
        self.smooth_idf = enable;
        self
    }

    /// Learn vocabulary and IDF weights from documents.
    pub fn fit<S: AsRef<str>>(&mut self, documents: &[S]) {
        self.count.fit(documents);

        let n_docs = documents.len();
        let n_features = self.count.n_features();

        // Compute document frequency for each term.
        let mut doc_freq = vec![0usize; n_features];
        let vocab = self.count.vocabulary();

        for doc in documents {
            let grams = self.count.tokenize_doc(doc.as_ref());
            let mut seen = std::collections::HashSet::new();
            for gram in &grams {
                if let Some(&idx) = vocab.get(gram) {
                    if seen.insert(idx) {
                        doc_freq[idx] += 1;
                    }
                }
            }
        }

        // Compute IDF.
        self.idf_values = vec![0.0; n_features];
        let smooth = if self.smooth_idf { 1.0 } else { 0.0 };
        let n = n_docs as f64 + smooth;

        for (i, &df) in doc_freq.iter().enumerate() {
            let df_smooth = df as f64 + smooth;
            self.idf_values[i] = (n / df_smooth).ln() + 1.0;
        }

        self.fitted = true;
    }

    /// Transform documents into a TF-IDF weighted sparse matrix.
    pub fn transform<S: AsRef<str>>(&self, documents: &[S]) -> CsrMatrix {
        assert!(
            self.fitted,
            "TfidfVectorizer: must call fit() before transform()"
        );

        let counts = self.count.transform(documents);
        let n_rows = counts.n_rows();
        let n_cols = counts.n_cols();

        if n_rows == 0 || n_cols == 0 {
            return CsrMatrix::from_dense(&[]);
        }

        // Get dense counts, apply TF-IDF weighting, then rebuild as sparse.
        let count_dense = counts.to_dense();

        let mut triplet_rows = Vec::new();
        let mut triplet_cols = Vec::new();
        let mut triplet_vals = Vec::new();

        for (row_idx, row) in count_dense.iter().enumerate() {
            let mut row_entries: Vec<(usize, f64)> = Vec::new();

            for (col, &count) in row.iter().enumerate() {
                if count == 0.0 {
                    continue;
                }

                let tf = if self.sublinear_tf {
                    1.0 + count.ln()
                } else {
                    count
                };

                let idf = self.idf_values.get(col).copied().unwrap_or(1.0);
                let tfidf = tf * idf;
                row_entries.push((col, tfidf));
            }

            // Normalize.
            if !row_entries.is_empty() {
                match self.norm {
                    TfidfNorm::L2 => {
                        let norm: f64 = row_entries.iter().map(|(_, v)| v * v).sum::<f64>().sqrt();
                        if norm > 0.0 {
                            for entry in &mut row_entries {
                                entry.1 /= norm;
                            }
                        }
                    }
                    TfidfNorm::L1 => {
                        let norm: f64 = row_entries.iter().map(|(_, v)| v.abs()).sum();
                        if norm > 0.0 {
                            for entry in &mut row_entries {
                                entry.1 /= norm;
                            }
                        }
                    }
                    TfidfNorm::None => {}
                }
            }

            for (col, val) in row_entries {
                triplet_rows.push(row_idx);
                triplet_cols.push(col);
                triplet_vals.push(val);
            }
        }

        CsrMatrix::from_triplets(&triplet_rows, &triplet_cols, &triplet_vals, n_rows, n_cols)
            .expect("TfidfVectorizer: internal CSR construction error")
    }

    /// Fit and transform in one step.
    pub fn fit_transform<S: AsRef<str>>(&mut self, documents: &[S]) -> CsrMatrix {
        self.fit(documents);
        self.transform(documents)
    }

    /// Return the learned IDF weights.
    pub fn idf(&self) -> &[f64] {
        &self.idf_values
    }

    /// Return the underlying vocabulary.
    pub fn vocabulary(&self) -> &std::collections::HashMap<String, usize> {
        self.count.vocabulary()
    }

    /// Return feature names sorted by column index.
    pub fn get_feature_names(&self) -> Vec<String> {
        self.count.get_feature_names()
    }

    /// Number of features.
    pub fn n_features(&self) -> usize {
        self.count.n_features()
    }
}

impl Default for TfidfVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_fit_transform() {
        let docs = ["the cat sat", "the dog sat", "the cat played"];
        let mut tfidf = TfidfVectorizer::new();
        let matrix = tfidf.fit_transform(&docs);

        assert_eq!(matrix.n_rows(), 3);
        assert_eq!(matrix.n_cols(), tfidf.n_features());
        assert_eq!(tfidf.n_features(), 5); // the, cat, dog, sat, played
    }

    #[test]
    fn idf_values_are_positive() {
        let docs = ["hello world", "hello test"];
        let mut tfidf = TfidfVectorizer::new();
        tfidf.fit(&docs);

        for &idf in tfidf.idf() {
            assert!(idf > 0.0, "IDF should be positive, got {idf}");
        }
    }

    #[test]
    fn l2_normalization() {
        let docs = ["a b c", "a b b"];
        let mut tfidf = TfidfVectorizer::new().norm(TfidfNorm::L2);
        let matrix = tfidf.fit_transform(&docs);
        let dense = matrix.to_dense();

        for row in &dense {
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                assert!(
                    (norm - 1.0).abs() < 1e-10,
                    "L2 norm should be 1.0, got {norm}"
                );
            }
        }
    }

    #[test]
    fn l1_normalization() {
        let docs = ["a b c"];
        let mut tfidf = TfidfVectorizer::new().norm(TfidfNorm::L1);
        let matrix = tfidf.fit_transform(&docs);
        let dense = matrix.to_dense();

        let norm: f64 = dense[0].iter().map(|v| v.abs()).sum();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "L1 norm should be 1.0, got {norm}"
        );
    }

    #[test]
    fn no_normalization() {
        let docs = ["a a"];
        let mut tfidf = TfidfVectorizer::new().norm(TfidfNorm::None);
        let matrix = tfidf.fit_transform(&docs);
        let dense = matrix.to_dense();

        // tf=2, idf=ln(1+1/1+1)+1 = ln(1)+1 = 1.0 with smooth_idf
        // So tfidf = 2 * 1.0 = 2.0
        assert!(
            dense[0].iter().any(|&v| v > 1.0),
            "Expected unnormalized values"
        );
    }

    #[test]
    fn smooth_idf_default() {
        let docs = ["a", "b"];
        let mut tfidf = TfidfVectorizer::new();
        tfidf.fit(&docs);

        // With smooth_idf: idf = ln((n+1)/(df+1)) + 1
        // For "a": idf = ln(3/2) + 1 ≈ 1.405
        for &idf in tfidf.idf() {
            assert!(idf > 1.0, "Smooth IDF should be > 1.0, got {idf}");
        }
    }

    #[test]
    fn sublinear_tf() {
        let docs = ["a a a a a"];
        let mut tfidf = TfidfVectorizer::new()
            .sublinear_tf(true)
            .norm(TfidfNorm::None);
        let matrix = tfidf.fit_transform(&docs);
        let dense = matrix.to_dense();

        // With sublinear_tf: tf = 1 + ln(5) ≈ 2.609
        // Without: tf = 5
        // So the value should be less than 5 * idf
        let val = dense[0].iter().find(|&&v| v > 0.0).unwrap();
        // idf = ln(2/2) + 1 = 1.0 with smooth
        assert!(*val < 5.0, "Sublinear TF should reduce high counts");
    }

    #[test]
    fn unseen_terms_ignored() {
        let train = ["cat dog"];
        let test = ["cat bird"]; // "bird" not in vocabulary

        let mut tfidf = TfidfVectorizer::new();
        tfidf.fit(&train);

        let matrix = tfidf.transform(&test);
        let dense = matrix.to_dense();

        let nnz: usize = dense[0].iter().filter(|&&v| v > 0.0).count();
        assert_eq!(nnz, 1, "Only 'cat' should have a non-zero value");
    }

    #[test]
    fn bigrams_tfidf() {
        let docs = ["the cat sat"];
        let mut tfidf = TfidfVectorizer::new().ngram_range(2, 2);
        let matrix = tfidf.fit_transform(&docs);

        assert_eq!(matrix.n_cols(), 2); // "the cat", "cat sat"
    }

    #[test]
    fn empty_documents() {
        let docs: [&str; 0] = [];
        let mut tfidf = TfidfVectorizer::new();
        let matrix = tfidf.fit_transform(&docs);

        assert_eq!(matrix.n_rows(), 0);
    }
}
