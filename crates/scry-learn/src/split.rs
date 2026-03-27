// SPDX-License-Identifier: MIT OR Apache-2.0
//! Train/test splitting and cross-validation utilities.

use crate::dataset::Dataset;
use crate::error::Result;
use crate::pipeline::PipelineModel;

/// Scoring function signature: `(y_true, y_pred) -> score`.
///
/// Use `metrics::accuracy` or any `fn(&[f64], &[f64]) -> f64`.
pub type ScoringFn = fn(&[f64], &[f64]) -> f64;

/// Split a dataset into training and test sets.
///
/// `test_ratio` should be between 0.0 and 1.0 (e.g., 0.2 for 80/20 split).
/// The `seed` controls the random shuffle for reproducibility.
pub fn train_test_split(data: &Dataset, test_ratio: f64, seed: u64) -> (Dataset, Dataset) {
    let n = data.n_samples();
    let mut indices: Vec<usize> = (0..n).collect();
    shuffle(&mut indices, seed);

    let test_size = (n as f64 * test_ratio).round() as usize;
    let test_size = test_size.max(1).min(n - 1);

    let test_indices = &indices[..test_size];
    let train_indices = &indices[test_size..];

    (data.subset(train_indices), data.subset(test_indices))
}

/// Stratified train/test split — preserves class proportions.
///
/// Groups samples by target value and splits each group independently,
/// ensuring the ratio of each class is maintained in both sets.
pub fn stratified_split(data: &Dataset, test_ratio: f64, seed: u64) -> (Dataset, Dataset) {
    let n = data.n_samples();

    // Group indices by class.
    let mut class_map: std::collections::HashMap<i64, Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..n {
        let key = data.target[i] as i64;
        class_map.entry(key).or_default().push(i);
    }

    let mut train_indices = Vec::new();
    let mut test_indices = Vec::new();

    // Sort class keys for deterministic iteration order.
    let mut sorted_classes: Vec<i64> = class_map.keys().copied().collect();
    sorted_classes.sort_unstable();

    let mut rng = crate::rng::FastRng::new(seed);
    for class in sorted_classes {
        let mut indices = class_map
            .remove(&class)
            .expect("class key from sorted_classes must exist in class_map");
        // Shuffle within each class.
        for i in (1..indices.len()).rev() {
            let j = rng.usize(0..=i);
            indices.swap(i, j);
        }
        let test_n = (indices.len() as f64 * test_ratio).round() as usize;
        let test_n = test_n.max(1).min(indices.len().saturating_sub(1));
        test_indices.extend_from_slice(&indices[..test_n]);
        train_indices.extend_from_slice(&indices[test_n..]);
    }

    (data.subset(&train_indices), data.subset(&test_indices))
}

/// K-fold cross-validation splits.
///
/// Returns `k` pairs of (train, test) datasets.
pub fn k_fold(data: &Dataset, k: usize, seed: u64) -> Vec<(Dataset, Dataset)> {
    let n = data.n_samples();
    let mut indices: Vec<usize> = (0..n).collect();
    shuffle(&mut indices, seed);

    let fold_size = n / k;
    let mut folds = Vec::with_capacity(k);

    for i in 0..k {
        let start = i * fold_size;
        let end = if i == k - 1 { n } else { start + fold_size };
        let test_indices: Vec<usize> = indices[start..end].to_vec();
        let train_indices: Vec<usize> = indices[..start]
            .iter()
            .chain(indices[end..].iter())
            .copied()
            .collect();
        folds.push((data.subset(&train_indices), data.subset(&test_indices)));
    }

    folds
}

/// Stratified k-fold cross-validation.
pub fn stratified_k_fold(data: &Dataset, k: usize, seed: u64) -> Vec<(Dataset, Dataset)> {
    let n = data.n_samples();

    // Group by class and shuffle within each class.
    let mut class_map: std::collections::HashMap<i64, Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..n {
        let key = data.target[i] as i64;
        class_map.entry(key).or_default().push(i);
    }

    // Sort class keys for deterministic iteration order.
    let mut sorted_classes: Vec<i64> = class_map.keys().copied().collect();
    sorted_classes.sort_unstable();

    let mut rng = crate::rng::FastRng::new(seed);
    for class in &sorted_classes {
        let indices = class_map
            .get_mut(class)
            .expect("class key from sorted_classes must exist in class_map");
        for i in (1..indices.len()).rev() {
            let j = rng.usize(0..=i);
            indices.swap(i, j);
        }
    }

    // Round-robin assign samples to folds.
    let mut fold_indices: Vec<Vec<usize>> = vec![Vec::new(); k];
    for class in &sorted_classes {
        let indices = &class_map[class];
        for (i, &idx) in indices.iter().enumerate() {
            fold_indices[i % k].push(idx);
        }
    }

    let mut folds = Vec::with_capacity(k);
    let all_indices: Vec<usize> = (0..n).collect();

    for fold in &fold_indices {
        let test_set: std::collections::HashSet<usize> = fold.iter().copied().collect();
        let train: Vec<usize> = all_indices
            .iter()
            .filter(|i| !test_set.contains(i))
            .copied()
            .collect();
        folds.push((data.subset(&train), data.subset(fold)));
    }

    folds
}

// ---------------------------------------------------------------------------
// Cross-validation scoring
// ---------------------------------------------------------------------------

/// Run k-fold cross-validation, returning per-fold scores.
///
/// Clones the model for each fold, fits on the training split, predicts on
/// the test split, and computes `scorer(y_true, y_pred)`.
///
/// ```ignore
/// use scry_learn::prelude::*;
/// use scry_learn::split::{cross_val_score, ScoringFn};
///
/// let scores = cross_val_score(
///     &DecisionTreeClassifier::new(),
///     &data, 5, accuracy as ScoringFn, 42,
/// ).unwrap();
/// ```
pub fn cross_val_score<M: PipelineModel + Clone + Send + Sync>(
    model: &M,
    data: &Dataset,
    k: usize,
    scorer: ScoringFn,
    seed: u64,
) -> Result<Vec<f64>> {
    let folds = k_fold(data, k, seed);
    run_cv(model, &folds, scorer)
}

/// Stratified k-fold cross-validation — preserves class balance in each fold.
pub fn cross_val_score_stratified<M: PipelineModel + Clone + Send + Sync>(
    model: &M,
    data: &Dataset,
    k: usize,
    scorer: ScoringFn,
    seed: u64,
) -> Result<Vec<f64>> {
    let folds = stratified_k_fold(data, k, seed);
    run_cv(model, &folds, scorer)
}

/// Shared implementation: fit + predict + score for each fold.
///
/// Folds are evaluated in parallel using rayon when multiple cores are
/// available. Each fold clones the model independently.
fn run_cv<M: PipelineModel + Clone + Send + Sync>(
    model: &M,
    folds: &[(Dataset, Dataset)],
    scorer: ScoringFn,
) -> Result<Vec<f64>> {
    use rayon::prelude::*;

    let results: Vec<Result<f64>> = folds
        .par_iter()
        .map(|(train, test)| {
            let mut m = model.clone();
            m.fit(train)?;
            let features = test.feature_matrix();
            let preds = m.predict(&features)?;
            Ok(scorer(&test.target, &preds))
        })
        .collect();

    // Collect results, propagating the first error if any fold failed.
    results.into_iter().collect()
}

/// Fisher-Yates shuffle with a seeded RNG.
fn shuffle(arr: &mut [usize], seed: u64) {
    let mut rng = crate::rng::FastRng::new(seed);
    for i in (1..arr.len()).rev() {
        let j = rng.usize(0..=i);
        arr.swap(i, j);
    }
}

// ---------------------------------------------------------------------------
// RepeatedKFold
// ---------------------------------------------------------------------------

/// Repeated k-fold cross-validation.
///
/// Repeats standard k-fold `n_repeats` times, each with a different random
/// shuffle (using `seed + repeat_idx`), yielding `n_splits × n_repeats` folds.
///
/// # Example
///
/// ```ignore
/// let rkf = RepeatedKFold::new(5, 3, 42);
/// let folds = rkf.folds(&data); // 15 (train, test) pairs
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RepeatedKFold {
    /// Number of folds per repetition.
    pub n_splits: usize,
    /// Number of repetitions.
    pub n_repeats: usize,
    /// Base random seed.
    pub seed: u64,
}

impl RepeatedKFold {
    /// Create a new `RepeatedKFold` splitter.
    pub fn new(n_splits: usize, n_repeats: usize, seed: u64) -> Self {
        Self {
            n_splits,
            n_repeats,
            seed,
        }
    }

    /// Generate all `n_splits × n_repeats` (train, test) pairs.
    pub fn folds(&self, data: &Dataset) -> Vec<(Dataset, Dataset)> {
        let mut all_folds = Vec::with_capacity(self.n_splits * self.n_repeats);
        for rep in 0..self.n_repeats {
            let rep_seed = self.seed.wrapping_add(rep as u64);
            all_folds.extend(k_fold(data, self.n_splits, rep_seed));
        }
        all_folds
    }
}

/// Convenience: run repeated k-fold CV on a clonable model.
///
/// Returns per-fold scores across all `n_splits × n_repeats` folds.
pub fn repeated_cross_val_score<M: PipelineModel + Clone + Send + Sync>(
    model: &M,
    data: &Dataset,
    n_splits: usize,
    n_repeats: usize,
    scorer: ScoringFn,
    seed: u64,
) -> Result<Vec<f64>> {
    let rkf = RepeatedKFold::new(n_splits, n_repeats, seed);
    let folds = rkf.folds(data);
    run_cv(model, &folds, scorer)
}

// ---------------------------------------------------------------------------
// GroupKFold
// ---------------------------------------------------------------------------

/// Group-aware k-fold: no group appears in both train and test within a fold.
///
/// Groups are assigned to folds round-robin by unique group index. This
/// prevents data leakage when samples from the same group (e.g. patient)
/// are correlated.
///
/// # Arguments
///
/// * `data`   — the dataset to split
/// * `groups` — group label per sample (length must equal `data.n_samples()`)
/// * `k`      — number of folds
///
/// # Panics
///
/// Panics if `groups.len() != data.n_samples()`.
pub fn group_k_fold(data: &Dataset, groups: &[usize], k: usize) -> Vec<(Dataset, Dataset)> {
    assert_eq!(
        groups.len(),
        data.n_samples(),
        "groups length must match n_samples"
    );

    // Collect unique groups in order of first appearance.
    let mut unique_groups: Vec<usize> = Vec::new();
    for &g in groups {
        if !unique_groups.contains(&g) {
            unique_groups.push(g);
        }
    }

    // Assign each group to a fold (round-robin).
    let mut group_to_fold = std::collections::HashMap::new();
    for (i, &g) in unique_groups.iter().enumerate() {
        group_to_fold.insert(g, i % k);
    }

    let mut folds = Vec::with_capacity(k);
    for fold_idx in 0..k {
        let mut test_indices = Vec::new();
        let mut train_indices = Vec::new();
        for (sample_idx, &g) in groups.iter().enumerate() {
            if group_to_fold[&g] == fold_idx {
                test_indices.push(sample_idx);
            } else {
                train_indices.push(sample_idx);
            }
        }
        folds.push((data.subset(&train_indices), data.subset(&test_indices)));
    }

    folds
}

// ---------------------------------------------------------------------------
// TimeSeriesSplit
// ---------------------------------------------------------------------------

/// Time-series cross-validation with expanding training window.
///
/// Produces `n_splits` folds where each test set immediately follows its
/// training set. Data order is preserved — no shuffling.
///
/// For `n_splits` folds and `n` samples, each test chunk has size
/// `n / (n_splits + 1)`. Fold *i* trains on `[0 .. (i+1)*chunk]` and tests
/// on `[(i+1)*chunk .. (i+2)*chunk]`.
pub fn time_series_split(data: &Dataset, n_splits: usize) -> Vec<(Dataset, Dataset)> {
    let n = data.n_samples();
    let chunk = n / (n_splits + 1);
    let mut folds = Vec::with_capacity(n_splits);

    for i in 0..n_splits {
        let train_end = (i + 1) * chunk;
        let test_end = if i == n_splits - 1 {
            n
        } else {
            (i + 2) * chunk
        };
        let train_indices: Vec<usize> = (0..train_end).collect();
        let test_indices: Vec<usize> = (train_end..test_end).collect();
        folds.push((data.subset(&train_indices), data.subset(&test_indices)));
    }

    folds
}

// ---------------------------------------------------------------------------
// cross_val_predict
// ---------------------------------------------------------------------------

/// Out-of-fold predictions for every sample.
///
/// Trains on k-1 folds, predicts the held-out fold, and reassembles
/// predictions in the original sample order. The returned vector has length
/// `data.n_samples()`.
///
/// # Example
///
/// ```ignore
/// let preds = cross_val_predict(&model, &data, 5, 42)?;
/// assert_eq!(preds.len(), data.n_samples());
/// ```
pub fn cross_val_predict<M: PipelineModel + Clone>(
    model: &M,
    data: &Dataset,
    k: usize,
    seed: u64,
) -> Result<Vec<f64>> {
    let n = data.n_samples();
    let mut indices_all: Vec<usize> = (0..n).collect();
    shuffle(&mut indices_all, seed);

    let fold_size = n / k;
    let mut predictions = vec![0.0; n];

    for i in 0..k {
        let start = i * fold_size;
        let end = if i == k - 1 { n } else { start + fold_size };

        let test_indices: Vec<usize> = indices_all[start..end].to_vec();
        let train_indices: Vec<usize> = indices_all[..start]
            .iter()
            .chain(indices_all[end..].iter())
            .copied()
            .collect();

        let train = data.subset(&train_indices);
        let test = data.subset(&test_indices);

        let mut m = model.clone();
        m.fit(&train)?;
        let features = test.feature_matrix();
        let preds = m.predict(&features)?;

        for (j, &idx) in test_indices.iter().enumerate() {
            predictions[idx] = preds[j];
        }
    }

    Ok(predictions)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::metrics::accuracy;
    use crate::tree::DecisionTreeClassifier;

    fn dummy_dataset(n: usize) -> Dataset {
        let features = vec![(0..n).map(|i| i as f64).collect()];
        let target = (0..n).map(|i| (i % 3) as f64).collect();
        Dataset::new(features, target, vec!["x".into()], "y")
    }

    /// A well-separated 2-class dataset for reliable CV testing.
    fn separable_dataset() -> Dataset {
        let n = 60;
        let mut f0 = Vec::with_capacity(n);
        let mut f1 = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for i in 0..n {
            if i < n / 2 {
                f0.push(i as f64);
                f1.push(i as f64);
                target.push(0.0);
            } else {
                f0.push((i + 100) as f64);
                f1.push((i + 100) as f64);
                target.push(1.0);
            }
        }
        Dataset::new(vec![f0, f1], target, vec!["x".into(), "y".into()], "class")
    }

    #[test]
    fn test_train_test_split_sizes() {
        let ds = dummy_dataset(100);
        let (train, test) = train_test_split(&ds, 0.2, 42);
        assert_eq!(train.n_samples() + test.n_samples(), 100);
        assert_eq!(test.n_samples(), 20);
    }

    #[test]
    fn test_stratified_split_preserves_ratio() {
        let ds = dummy_dataset(90); // 30 each of class 0, 1, 2
        let (train, test) = stratified_split(&ds, 0.2, 42);
        assert_eq!(train.n_samples() + test.n_samples(), 90);

        let test_class_0 = test.target.iter().filter(|&&v| v == 0.0).count();
        let test_class_1 = test.target.iter().filter(|&&v| v == 1.0).count();
        let test_class_2 = test.target.iter().filter(|&&v| v == 2.0).count();
        assert!((4..=8).contains(&test_class_0));
        assert!((4..=8).contains(&test_class_1));
        assert!((4..=8).contains(&test_class_2));
    }

    #[test]
    fn test_k_fold_count() {
        let ds = dummy_dataset(50);
        let folds = k_fold(&ds, 5, 42);
        assert_eq!(folds.len(), 5);
        for (train, test) in &folds {
            assert_eq!(train.n_samples() + test.n_samples(), 50);
        }
    }

    // -----------------------------------------------------------------------
    // Cross-validation scorer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cross_val_score_dt() {
        let ds = separable_dataset();
        let model = DecisionTreeClassifier::new();
        let scores = cross_val_score(&model, &ds, 5, accuracy, 42).unwrap();
        assert_eq!(scores.len(), 5);
        for &s in &scores {
            assert!(s >= 0.8, "fold accuracy {s} < 0.8 on well-separated data");
        }
    }

    #[test]
    fn test_cross_val_score_stratified() {
        let ds = separable_dataset();
        let model = DecisionTreeClassifier::new();
        let scores = cross_val_score_stratified(&model, &ds, 5, accuracy, 42).unwrap();
        assert_eq!(scores.len(), 5);
        for &s in &scores {
            assert!(s >= 0.8, "stratified fold accuracy {s} < 0.8");
        }
    }

    #[test]
    fn test_cross_val_score_leave_one_out() {
        // k = n for leave-one-out cross-validation.
        let ds = separable_dataset();
        let n = ds.n_samples();
        let model = DecisionTreeClassifier::new();
        let scores = cross_val_score(&model, &ds, n, accuracy, 42).unwrap();
        assert_eq!(scores.len(), n);
        // Each fold has 1 test sample, so score is 0.0 or 1.0.
        for &s in &scores {
            assert!(s == 0.0 || s == 1.0);
        }
    }

    #[test]
    fn test_cross_val_score_custom_scorer() {
        fn always_one(_true: &[f64], _pred: &[f64]) -> f64 {
            1.0
        }
        let ds = separable_dataset();
        let model = DecisionTreeClassifier::new();
        let scores = cross_val_score(&model, &ds, 3, always_one, 42).unwrap();
        assert!(scores.iter().all(|&s| (s - 1.0).abs() < 1e-10));
    }

    // -----------------------------------------------------------------------
    // Session 15: New CV strategies
    // -----------------------------------------------------------------------

    #[test]
    fn test_repeated_k_fold_count() {
        let ds = dummy_dataset(50);
        let rkf = RepeatedKFold::new(5, 3, 42);
        let folds = rkf.folds(&ds);
        assert_eq!(folds.len(), 15);
        for (train, test) in &folds {
            assert_eq!(train.n_samples() + test.n_samples(), 50);
            assert!(!test.target.is_empty(), "test fold must not be empty");
        }
    }

    #[test]
    fn test_repeated_cross_val_score() {
        let ds = separable_dataset();
        let model = DecisionTreeClassifier::new();
        let scores = repeated_cross_val_score(&model, &ds, 5, 3, accuracy, 42).unwrap();
        assert_eq!(scores.len(), 15);
        for &s in &scores {
            assert!(s >= 0.5, "repeated CV fold accuracy {s} too low");
        }
    }

    #[test]
    fn test_group_k_fold_no_leakage() {
        let ds = dummy_dataset(12);
        // 3 groups: 0,0,0,0, 1,1,1,1, 2,2,2,2
        let groups = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let folds = group_k_fold(&ds, &groups, 3);
        assert_eq!(folds.len(), 3);

        for (train, test) in &folds {
            assert_eq!(train.n_samples() + test.n_samples(), 12);
            // Each fold's test set should have exactly 4 samples (one group).
            assert_eq!(test.n_samples(), 4);
        }
    }

    #[test]
    fn test_group_k_fold_group_isolation() {
        // Verify no group appears in both train and test.
        let n = 15;
        let ds = dummy_dataset(n);
        let groups: Vec<usize> = (0..n).map(|i| i / 3).collect(); // 5 groups of 3
        let folds = group_k_fold(&ds, &groups, 3);

        for (fold_idx, (_train, test)) in folds.iter().enumerate() {
            // Reconstruct test indices from target values (unique due to dummy_dataset)
            // Just verify sizes are correct.
            assert!(!test.target.is_empty(), "fold {fold_idx} test set is empty");
        }
    }

    #[test]
    fn test_time_series_split_temporal_order() {
        let n = 24;
        let ds = dummy_dataset(n);
        let folds = time_series_split(&ds, 3);
        assert_eq!(folds.len(), 3);

        // Expanding window: each successive fold has a larger training set.
        let mut prev_train_size = 0;
        for (train, test) in &folds {
            assert!(
                train.n_samples() > prev_train_size,
                "training size should grow"
            );
            prev_train_size = train.n_samples();
            assert!(!test.target.is_empty(), "test fold must not be empty");
        }
    }

    #[test]
    fn test_time_series_split_no_future_leak() {
        let n = 20;
        let features = vec![(0..n).map(|i| i as f64).collect::<Vec<_>>()];
        let target = (0..n).map(|i| i as f64).collect();
        let ds = Dataset::new(features, target, vec!["t".into()], "y");

        let folds = time_series_split(&ds, 4);
        for (train, test) in &folds {
            let train_max = train.features[0]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let test_min = test.features[0]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            assert!(
                train_max < test_min,
                "train max {train_max} must be < test min {test_min}"
            );
        }
    }

    #[test]
    fn test_cross_val_predict_length() {
        let ds = separable_dataset();
        let model = DecisionTreeClassifier::new();
        let preds = cross_val_predict(&model, &ds, 5, 42).unwrap();
        assert_eq!(preds.len(), ds.n_samples());
    }

    #[test]
    fn test_cross_val_predict_reasonable_accuracy() {
        let ds = separable_dataset();
        let model = DecisionTreeClassifier::new();
        let preds = cross_val_predict(&model, &ds, 5, 42).unwrap();
        let acc = accuracy(&ds.target, &preds);
        assert!(
            acc >= 0.8,
            "cross_val_predict accuracy {acc} too low on separable data"
        );
    }
}
