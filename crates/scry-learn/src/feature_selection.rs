// SPDX-License-Identifier: MIT OR Apache-2.0
//! Feature selection transformers.
//!
//! Remove low-information features before training to reduce
//! overfitting and speed up downstream models.
//!
//! # Examples
//!
//! ```ignore
//! use scry_learn::prelude::*;
//!
//! let mut vt = VarianceThreshold::new().threshold(0.1);
//! vt.fit(&data)?;
//! vt.transform(&mut data)?; // drops constant / near-constant columns
//! ```

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::preprocess::Transformer;

// ---------------------------------------------------------------------------
// VarianceThreshold
// ---------------------------------------------------------------------------

/// Remove features whose variance falls below a threshold.
///
/// By default, removes only constant features (threshold = 0.0).
/// Useful as a lightweight first step before expensive feature selectors.
///
/// # Examples
///
/// ```
/// use scry_learn::dataset::Dataset;
/// use scry_learn::preprocess::Transformer;
/// use scry_learn::feature_selection::VarianceThreshold;
///
/// let mut data = Dataset::new(
///     vec![
///         vec![1.0, 2.0, 3.0],  // variable
///         vec![5.0, 5.0, 5.0],  // constant — will be removed
///     ],
///     vec![0.0, 1.0, 0.0],
///     vec!["a".into(), "b".into()],
///     "target",
/// );
///
/// let mut vt = VarianceThreshold::new();
/// vt.fit_transform(&mut data).unwrap();
/// assert_eq!(data.n_features(), 1); // only "a" remains
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct VarianceThreshold {
    threshold: f64,
    variances_: Vec<f64>,
    mask_: Vec<bool>,
    fitted: bool,
}

impl VarianceThreshold {
    /// Create a new selector with threshold 0.0 (remove only constants).
    pub fn new() -> Self {
        Self {
            threshold: 0.0,
            variances_: Vec::new(),
            mask_: Vec::new(),
            fitted: false,
        }
    }

    /// Set the variance threshold.
    ///
    /// Features with variance ≤ this value are removed.
    pub fn threshold(mut self, t: f64) -> Self {
        self.threshold = t;
        self
    }

    /// Per-feature variances computed during fit.
    ///
    /// # Panics
    ///
    /// Panics if called before [`VarianceThreshold::fit`].
    pub fn variances(&self) -> &[f64] {
        &self.variances_
    }

    /// Boolean mask of selected features.
    ///
    /// `true` at index `j` means feature `j` was kept.
    pub fn get_support(&self) -> &[bool] {
        &self.mask_
    }
}

impl Default for VarianceThreshold {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for VarianceThreshold {
    fn fit(&mut self, data: &Dataset) -> Result<()> {
        let n = data.n_samples();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        let nf = n as f64;

        self.variances_ = Vec::with_capacity(data.n_features());
        self.mask_ = Vec::with_capacity(data.n_features());

        for col in &data.features {
            let mean = col.iter().sum::<f64>() / nf;
            let var = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / nf;
            self.variances_.push(var);
            self.mask_.push(var > self.threshold);
        }

        self.fitted = true;
        Ok(())
    }

    fn transform(&self, data: &mut Dataset) -> Result<()> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        filter_features(data, &self.mask_);
        Ok(())
    }

    fn inverse_transform(&self, _data: &mut Dataset) -> Result<()> {
        Err(ScryLearnError::InvalidParameter(
            "VarianceThreshold is not invertible — dropped columns cannot be restored".into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// SelectKBest
// ---------------------------------------------------------------------------

/// Scoring function for feature selection.
///
/// Determines how each feature is scored relative to the target.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ScoreFn {
    /// ANOVA F-value: ratio of between-group variance to within-group variance.
    ///
    /// Best suited for classification tasks where higher F-values indicate
    /// features that separate classes well.
    FClassif,
}

/// Select the top-k highest-scoring features.
///
/// Uses a scoring function (e.g. ANOVA F-value) to rank features by their
/// discriminative power, then keeps only the `k` best.
///
/// # Examples
///
/// ```ignore
/// use scry_learn::prelude::*;
///
/// let mut sel = SelectKBest::new(ScoreFn::FClassif).k(2);
/// sel.fit(&data)?;
/// sel.transform(&mut data)?;
/// assert_eq!(data.n_features(), 2);
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct SelectKBest {
    k: usize,
    score_fn: ScoreFn,
    scores_: Vec<f64>,
    mask_: Vec<bool>,
    fitted: bool,
}

impl SelectKBest {
    /// Create a new selector with the given scoring function.
    ///
    /// Default: keep top 10 features.
    pub fn new(score_fn: ScoreFn) -> Self {
        Self {
            k: 10,
            score_fn,
            scores_: Vec::new(),
            mask_: Vec::new(),
            fitted: false,
        }
    }

    /// Set the number of top features to keep.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Per-feature scores computed during fit.
    ///
    /// Higher values indicate more discriminative features.
    pub fn scores(&self) -> &[f64] {
        &self.scores_
    }

    /// Boolean mask of selected features.
    ///
    /// `true` at index `j` means feature `j` was kept.
    pub fn get_support(&self) -> &[bool] {
        &self.mask_
    }
}

impl Transformer for SelectKBest {
    fn fit(&mut self, data: &Dataset) -> Result<()> {
        let n = data.n_samples();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        self.scores_ = match self.score_fn {
            ScoreFn::FClassif => f_classif(data),
        };

        let k = self.k.min(data.n_features());

        // Find the k-th highest score to determine the cutoff.
        let mut sorted_scores: Vec<f64> = self.scores_.clone();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let cutoff = if k > 0 && k <= sorted_scores.len() {
            sorted_scores[k - 1]
        } else {
            f64::NEG_INFINITY
        };

        // Build mask: keep features with score >= cutoff, but cap at k.
        self.mask_ = vec![false; self.scores_.len()];
        let mut kept = 0;
        // First pass: mark features with score > cutoff.
        for (i, &score) in self.scores_.iter().enumerate() {
            if score > cutoff && kept < k {
                self.mask_[i] = true;
                kept += 1;
            }
        }
        // Second pass: fill remaining slots with features exactly at cutoff.
        for (i, &score) in self.scores_.iter().enumerate() {
            if kept >= k {
                break;
            }
            if !self.mask_[i] && (score - cutoff).abs() < 1e-12 {
                self.mask_[i] = true;
                kept += 1;
            }
        }

        self.fitted = true;
        Ok(())
    }

    fn transform(&self, data: &mut Dataset) -> Result<()> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        filter_features(data, &self.mask_);
        Ok(())
    }

    fn inverse_transform(&self, _data: &mut Dataset) -> Result<()> {
        Err(ScryLearnError::InvalidParameter(
            "SelectKBest is not invertible — dropped columns cannot be restored".into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// ANOVA F-value (f_classif)
// ---------------------------------------------------------------------------

/// Compute ANOVA F-value for each feature vs. the target.
///
/// The F-value is the ratio of between-group variance to within-group
/// variance. Higher F-values indicate features that separate classes well.
///
/// # Examples
///
/// ```ignore
/// let scores = f_classif(&data);
/// // scores[j] is the F-value for feature j
/// ```
pub fn f_classif(data: &Dataset) -> Vec<f64> {
    let n = data.n_samples();
    let n_features = data.n_features();

    // Identify unique classes.
    let mut class_set: Vec<i64> = data.target.iter().map(|&v| v as i64).collect();
    class_set.sort_unstable();
    class_set.dedup();
    let n_classes = class_set.len();

    if n_classes < 2 {
        return vec![0.0; n_features];
    }

    // Build class membership lookup.
    let class_indices: Vec<Vec<usize>> = class_set
        .iter()
        .map(|&c| (0..n).filter(|&i| data.target[i] as i64 == c).collect())
        .collect();

    let mut f_values = Vec::with_capacity(n_features);

    for j in 0..n_features {
        let col = &data.features[j];
        let grand_mean = col.iter().sum::<f64>() / n as f64;

        // Between-group sum of squares.
        let mut ss_between = 0.0;
        // Within-group sum of squares.
        let mut ss_within = 0.0;

        for group in &class_indices {
            let n_g = group.len() as f64;
            if n_g == 0.0 {
                continue;
            }
            let group_mean = group.iter().map(|&i| col[i]).sum::<f64>() / n_g;
            ss_between += n_g * (group_mean - grand_mean).powi(2);

            for &i in group {
                ss_within += (col[i] - group_mean).powi(2);
            }
        }

        let df_between = (n_classes - 1) as f64;
        let df_within = (n - n_classes) as f64;

        let f_val = if df_within > 0.0 && ss_within > 1e-15 {
            (ss_between / df_between) / (ss_within / df_within)
        } else if ss_between > 1e-15 {
            // Perfect separation: zero within-group variance.
            f64::MAX
        } else {
            0.0
        };

        f_values.push(f_val);
    }

    f_values
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Filter a dataset's features and feature_names using a boolean mask.
fn filter_features(data: &mut Dataset, mask: &[bool]) {
    let mut new_features = Vec::new();
    let mut new_names = Vec::new();

    for (j, &keep) in mask.iter().enumerate() {
        if keep {
            new_features.push(data.features[j].clone());
            new_names.push(data.feature_names[j].clone());
        }
    }

    data.features = new_features;
    data.feature_names = new_names;
    data.sync_matrix();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Pipeline;
    use crate::preprocess::StandardScaler;
    use crate::tree::DecisionTreeClassifier;

    /// Iris-like dataset where petal features (f2, f3) are much more
    /// discriminative than sepal features (f0, f1).
    fn iris_like() -> Dataset {
        let n_per_class = 30;
        let n = n_per_class * 3;
        let mut f0 = Vec::with_capacity(n);
        let mut f1 = Vec::with_capacity(n);
        let mut f2 = Vec::with_capacity(n);
        let mut f3 = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);

        let mut rng = crate::rng::FastRng::new(123);

        for _ in 0..n_per_class {
            // Class 0
            f0.push(5.0 + rng.f64() * 0.5); // overlapping sepal
            f1.push(3.4 + rng.f64() * 0.4); // overlapping sepal
            f2.push(1.0 + rng.f64() * 0.5); // small petal — discriminative
            f3.push(0.1 + rng.f64() * 0.2); // small petal — discriminative
            target.push(0.0);
        }
        for _ in 0..n_per_class {
            // Class 1
            f0.push(5.5 + rng.f64() * 0.8); // overlapping sepal
            f1.push(2.5 + rng.f64() * 0.5); // overlapping sepal
            f2.push(4.0 + rng.f64() * 0.5); // medium petal
            f3.push(1.2 + rng.f64() * 0.3); // medium petal
            target.push(1.0);
        }
        for _ in 0..n_per_class {
            // Class 2
            f0.push(6.0 + rng.f64() * 1.0); // overlapping sepal
            f1.push(2.8 + rng.f64() * 0.5); // overlapping sepal
            f2.push(5.5 + rng.f64() * 0.5); // large petal
            f3.push(2.0 + rng.f64() * 0.3); // large petal
            target.push(2.0);
        }

        Dataset::new(
            vec![f0, f1, f2, f3],
            target,
            vec![
                "sepal_len".into(),
                "sepal_wid".into(),
                "petal_len".into(),
                "petal_wid".into(),
            ],
            "species",
        )
    }

    #[test]
    fn test_variance_threshold_removes_constant() {
        let mut data = Dataset::new(
            vec![
                vec![1.0, 2.0, 3.0, 4.0], // variable
                vec![5.0, 5.0, 5.0, 5.0], // constant → removed
                vec![0.0, 1.0, 0.0, 1.0], // variable
            ],
            vec![0.0, 1.0, 0.0, 1.0],
            vec!["a".into(), "b".into(), "c".into()],
            "t",
        );

        let mut vt = VarianceThreshold::new();
        vt.fit_transform(&mut data).unwrap();

        assert_eq!(data.n_features(), 2);
        assert_eq!(data.feature_names, vec!["a", "c"]);
    }

    #[test]
    fn test_variance_threshold_custom() {
        let mut data = Dataset::new(
            vec![
                vec![1.0, 1.0, 1.0, 1.1],   // variance ≈ 0.0019
                vec![0.0, 10.0, 0.0, 10.0], // variance = 25
            ],
            vec![0.0; 4],
            vec!["low_var".into(), "high_var".into()],
            "t",
        );

        let mut vt = VarianceThreshold::new().threshold(0.01);
        vt.fit_transform(&mut data).unwrap();

        assert_eq!(data.n_features(), 1);
        assert_eq!(data.feature_names, vec!["high_var"]);
    }

    #[test]
    fn test_select_k_best_petal_features_rank_highest() {
        let data = iris_like();

        let mut sel = SelectKBest::new(ScoreFn::FClassif).k(2);
        sel.fit(&data).unwrap();

        let scores = sel.scores();
        // Petal features (indices 2, 3) should have higher F-values
        // than sepal features (indices 0, 1).
        assert!(
            scores[2] > scores[0],
            "petal_len ({:.1}) should rank higher than sepal_len ({:.1})",
            scores[2],
            scores[0]
        );
        assert!(
            scores[3] > scores[1],
            "petal_wid ({:.1}) should rank higher than sepal_wid ({:.1})",
            scores[3],
            scores[1]
        );

        // After transform, only 2 features remain.
        let mut data_copy = data.clone();
        sel.transform(&mut data_copy).unwrap();
        assert_eq!(data_copy.n_features(), 2);

        // The kept features should be petal_len and petal_wid.
        let support = sel.get_support();
        assert!(!support[0], "sepal_len should be dropped");
        assert!(!support[1], "sepal_wid should be dropped");
        assert!(support[2], "petal_len should be kept");
        assert!(support[3], "petal_wid should be kept");
    }

    #[test]
    fn test_select_k_best_not_fitted() {
        let sel = SelectKBest::new(ScoreFn::FClassif);
        let mut data = Dataset::new(vec![vec![1.0]], vec![0.0], vec!["x".into()], "t");
        assert!(sel.transform(&mut data).is_err());
    }

    #[test]
    fn test_f_classif_basic() {
        // One perfectly discriminative feature, one random.
        let data = Dataset::new(
            vec![
                vec![1.0, 1.0, 1.0, 10.0, 10.0, 10.0], // perfect separator
                vec![3.0, 7.0, 2.0, 5.0, 8.0, 1.0],    // noise
            ],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec!["good".into(), "noise".into()],
            "class",
        );

        let scores = f_classif(&data);
        assert!(
            scores[0] > scores[1],
            "good feature ({:.1}) should have higher F-value than noise ({:.1})",
            scores[0],
            scores[1]
        );
    }

    #[test]
    fn test_pipeline_vt_scaler_dt() {
        // End-to-end: VarianceThreshold → StandardScaler → DecisionTree.
        let features = vec![
            vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0], // discriminative
            vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0],    // constant → removed
            vec![0.0, 0.5, 1.0, 5.0, 5.5, 6.0],    // discriminative
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let data = Dataset::new(
            features,
            target,
            vec!["a".into(), "b".into(), "c".into()],
            "class",
        );

        let mut pipeline = Pipeline::new()
            .add_transformer(VarianceThreshold::new())
            .add_transformer(StandardScaler::new())
            .set_model(DecisionTreeClassifier::new());

        pipeline.fit(&data).unwrap();
        let preds = pipeline.predict(&data).unwrap();
        assert_eq!(preds.len(), 6);
    }
}
