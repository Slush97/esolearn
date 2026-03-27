// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tabular dataset container for ML workflows.
//!
//! [`Dataset`] provides a lightweight, column-major representation of
//! features + target, with CSV loading and basic column access.

use std::sync::OnceLock;

use crate::error::{Result, ScryLearnError};

use crate::matrix::DenseMatrix;
use crate::sparse::CscMatrix;

/// Internal feature storage format.
#[derive(Clone, Debug, Default)]
pub(crate) enum Storage {
    /// Dense column-major features (current default).
    #[default]
    Dense,
    /// Sparse CSC matrix (column-oriented for fit).
    Sparse(CscMatrix),
}

/// Descriptive statistics for a single column.
#[derive(Clone, Debug)]
pub struct ColumnStats {
    /// Column name.
    pub name: String,
    /// Number of finite (non-NaN) values.
    pub count: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// Sample standard deviation (ddof=1).
    pub std: f64,
    /// Minimum value.
    pub min: f64,
    /// 25th percentile.
    pub q25: f64,
    /// Median (50th percentile).
    pub median: f64,
    /// 75th percentile.
    pub q75: f64,
    /// Maximum value.
    pub max: f64,
}

/// A tabular dataset with features and a target column.
///
/// Features are stored column-major (`features[feature_idx][sample_idx]`)
/// for cache-friendly access during tree split evaluation.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Dataset {
    /// Feature columns: `features[feature_idx][sample_idx]`.
    pub features: Vec<Vec<f64>>,
    /// Target values: `target[sample_idx]`.
    pub target: Vec<f64>,
    /// Feature column names.
    pub feature_names: Vec<String>,
    /// Target column name.
    pub target_name: String,
    /// Class label mapping (index → label string) for classification tasks.
    pub class_labels: Option<Vec<String>>,
    /// Lazily-computed contiguous column-major feature matrix.
    ///
    /// Built on first access from `features` via [`OnceCell::get_or_init`],
    /// avoiding the upfront clone in [`Dataset::new`].
    #[cfg_attr(feature = "serde", serde(skip))]
    matrix: OnceLock<DenseMatrix>,
    /// Lazily-computed contiguous row-major feature buffer.
    ///
    /// Layout: `[sample_0_feat_0, sample_0_feat_1, ..., sample_n_feat_m]`.
    /// Populated on first call to [`flat_feature_matrix`].
    #[cfg_attr(feature = "serde", serde(skip))]
    row_major_cache: Option<Vec<f64>>,
    /// Storage format — dense (default) or sparse CSC.
    #[cfg_attr(feature = "serde", serde(skip))]
    storage: Storage,
}

impl Dataset {
    /// Create a dataset from pre-computed features and target.
    ///
    /// # Panics
    ///
    /// Panics if feature columns have mismatched lengths, or if
    /// `feature_names.len() != features.len()`.
    pub fn new(
        features: Vec<Vec<f64>>,
        target: Vec<f64>,
        feature_names: Vec<String>,
        target_name: impl Into<String>,
    ) -> Self {
        assert!(
            feature_names.len() == features.len(),
            "feature_names.len()={} but features.len()={}",
            feature_names.len(),
            features.len(),
        );
        if let Some(first) = features.first() {
            for (i, col) in features.iter().enumerate().skip(1) {
                assert!(
                    col.len() == first.len(),
                    "feature column {i} has {} rows but column 0 has {}",
                    col.len(),
                    first.len(),
                );
            }
        }
        Self {
            features,
            target,
            feature_names,
            target_name: target_name.into(),
            class_labels: None,
            matrix: OnceLock::new(),
            row_major_cache: None,
            storage: Storage::Dense,
        }
    }

    /// Create a dataset from a [`DenseMatrix`], target, and column names.
    ///
    /// The `features` field is populated from the matrix for backward compat.
    pub fn from_matrix(
        matrix: DenseMatrix,
        target: Vec<f64>,
        feature_names: Vec<String>,
        target_name: impl Into<String>,
    ) -> Self {
        let features = matrix.to_col_vecs();
        let cell = OnceLock::new();
        let _ = cell.set(matrix);
        Self {
            features,
            target,
            feature_names,
            target_name: target_name.into(),
            class_labels: None,
            matrix: cell,
            row_major_cache: None,
            storage: Storage::Dense,
        }
    }

    /// The contiguous column-major feature matrix.
    ///
    /// Lazily built from `features` on first access. Subsequent calls
    /// return the cached matrix without recomputation.
    #[inline]
    pub fn matrix(&self) -> &DenseMatrix {
        self.matrix.get_or_init(|| {
            DenseMatrix::from_col_major_ref(&self.features)
                .expect("DenseMatrix build from features failed")
        })
    }

    /// Load a dataset from a CSV file.
    ///
    /// The `target_column` is extracted as the target; all other numeric
    /// columns become features. String columns are label-encoded automatically.
    ///
    /// Requires the `csv` feature.
    #[cfg(feature = "csv")]
    pub fn from_csv(path: &str, target_column: &str) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(ScryLearnError::Io)?;
        Self::from_csv_reader(file, target_column)
    }

    /// Load a dataset from any reader producing CSV data.
    ///
    /// Requires the `csv` feature.
    #[cfg(feature = "csv")]
    pub fn from_csv_reader(rdr: impl std::io::Read, target_column: &str) -> Result<Self> {
        let mut csv_rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_reader(rdr);

        let headers: Vec<String> = csv_rdr
            .headers()
            .map_err(|e| ScryLearnError::Csv(e.to_string()))?
            .iter()
            .map(std::string::ToString::to_string)
            .collect();

        let target_idx = headers
            .iter()
            .position(|h| h.eq_ignore_ascii_case(target_column))
            .ok_or_else(|| ScryLearnError::InvalidColumn(target_column.to_string()))?;

        // Collect all rows as string records.
        let mut rows: Vec<Vec<String>> = Vec::new();
        for result in csv_rdr.records() {
            let record = result.map_err(|e| ScryLearnError::Csv(e.to_string()))?;
            rows.push(
                record
                    .iter()
                    .map(std::string::ToString::to_string)
                    .collect(),
            );
        }

        if rows.is_empty() {
            return Err(ScryLearnError::EmptyDataset);
        }

        // Determine which columns are features (all except target).
        let feature_indices: Vec<usize> = (0..headers.len()).filter(|&i| i != target_idx).collect();

        let n_samples = rows.len();
        let n_features = feature_indices.len();

        // Parse target — try numeric first, fall back to label encoding.
        let (target, class_labels) = parse_target_column(&rows, target_idx);

        // Parse feature columns — try numeric, label-encode strings.
        let mut features = vec![vec![0.0; n_samples]; n_features];
        let mut feature_names = Vec::with_capacity(n_features);

        for (feat_col, &col_idx) in feature_indices.iter().enumerate() {
            feature_names.push(headers[col_idx].clone());
            for (row_idx, row) in rows.iter().enumerate() {
                let val = row.get(col_idx).map_or("", std::string::String::as_str);
                features[feat_col][row_idx] = val.parse::<f64>().unwrap_or(f64::NAN);
            }
        }

        Ok(Self {
            features,
            target,
            feature_names,
            target_name: headers[target_idx].clone(),
            class_labels,
            matrix: OnceLock::new(),
            row_major_cache: None,
            storage: Storage::Dense,
        })
    }

    /// Number of samples (rows).
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.target.len()
    }

    /// Number of features (columns).
    #[inline]
    pub fn n_features(&self) -> usize {
        match &self.storage {
            Storage::Sparse(csc) => csc.n_cols(),
            Storage::Dense => self.features.len(),
        }
    }

    /// Number of unique classes in the target (for classification).
    pub fn n_classes(&self) -> usize {
        self.class_labels.as_ref().map_or_else(
            || {
                let mut vals: Vec<i64> = self.target.iter().map(|&v| v as i64).collect();
                vals.sort_unstable();
                vals.dedup();
                vals.len()
            },
            Vec::len,
        )
    }

    /// Get a single feature column by index.
    pub fn feature(&self, idx: usize) -> &[f64] {
        &self.features[idx]
    }

    /// Get a single sample (row) as a vector of feature values.
    pub fn sample(&self, idx: usize) -> Vec<f64> {
        self.features.iter().map(|col| col[idx]).collect()
    }

    /// Get the feature matrix as row-major `[n_samples][n_features]`.
    pub fn feature_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.n_samples();
        let m = self.n_features();
        let mut matrix = vec![vec![0.0; m]; n];
        for (j, feat_col) in self.features.iter().enumerate() {
            for (i, &val) in feat_col.iter().enumerate() {
                matrix[i][j] = val;
            }
        }
        matrix
    }

    /// Get a contiguous row-major feature buffer, computing on first call.
    ///
    /// Layout: `[sample_0_feat_0, sample_0_feat_1, ..., sample_n_feat_m]`.
    /// Subsequent calls return the cached slice without recomputation.
    pub fn flat_feature_matrix(&mut self) -> &[f64] {
        if self.row_major_cache.is_none() {
            let n = self.n_samples();
            let m = self.n_features();
            let mut buf = vec![0.0; n * m];
            if let Some(mat) = self.matrix.get() {
                let src = mat.as_slice();
                for j in 0..m {
                    let col_off = j * n;
                    for i in 0..n {
                        buf[i * m + j] = src[col_off + i];
                    }
                }
            } else {
                for j in 0..m {
                    for i in 0..n {
                        buf[i * m + j] = self.features[j][i];
                    }
                }
            }
            self.row_major_cache = Some(buf);
        }
        self.row_major_cache
            .as_ref()
            .expect("row_major_cache populated above")
    }

    /// Get a zero-copy row slice from a pre-computed flat feature buffer.
    ///
    /// `cache` should be the result of [`Dataset::flat_feature_matrix`].
    #[inline]
    pub fn sample_row<'a>(&self, cache: &'a [f64], idx: usize) -> &'a [f64] {
        let m = self.n_features();
        &cache[idx * m..(idx + 1) * m]
    }

    /// Create a subset of this dataset with the given sample indices.
    pub fn subset(&self, indices: &[usize]) -> Self {
        let target: Vec<f64> = indices.iter().map(|&i| self.target[i]).collect();

        if let Storage::Sparse(csc) = &self.storage {
            let new_csc = subset_csc(csc, indices);
            return Self {
                features: Vec::new(),
                target,
                feature_names: self.feature_names.clone(),
                target_name: self.target_name.clone(),
                class_labels: self.class_labels.clone(),
                matrix: OnceLock::new(),
                row_major_cache: None,
                storage: Storage::Sparse(new_csc),
            };
        }

        let features: Vec<Vec<f64>> = self
            .features
            .iter()
            .map(|col| indices.iter().map(|&i| col[i]).collect())
            .collect();
        Self {
            features,
            target,
            feature_names: self.feature_names.clone(),
            target_name: self.target_name.clone(),
            class_labels: self.class_labels.clone(),
            matrix: OnceLock::new(),
            row_major_cache: None,
            storage: Storage::Dense,
        }
    }

    /// Clear the cached matrix so it will be lazily rebuilt from `features`
    /// on the next call to [`matrix()`](Self::matrix).
    ///
    /// Call this after mutating `features` in place (e.g. after a
    /// transformer's `transform()` step).
    pub fn sync_matrix(&mut self) {
        self.matrix = OnceLock::new();
        self.row_major_cache = None;
    }

    /// Mark the matrix cache as stale after in-place feature mutations.
    ///
    /// The matrix will be lazily rebuilt from `features` on next access.
    #[inline]
    pub fn invalidate_matrix(&mut self) {
        self.matrix = OnceLock::new();
        self.row_major_cache = None;
    }

    /// Returns `Err(InvalidData)` if any feature or target value is NaN or ±Inf.
    pub fn validate_finite(&self) -> Result<()> {
        // Check sparse storage values if present.
        if let Storage::Sparse(csc) = &self.storage {
            for j in 0..csc.n_cols() {
                for (i, v) in csc.col(j).iter() {
                    if !v.is_finite() {
                        let name = self
                            .feature_names
                            .get(j)
                            .map_or_else(|| format!("feature[{j}]"), std::clone::Clone::clone);
                        return Err(ScryLearnError::InvalidData(format!(
                            "non-finite value ({v}) in {name} at sample {i}"
                        )));
                    }
                }
            }
        } else {
            for (j, col) in self.features.iter().enumerate() {
                for (i, &v) in col.iter().enumerate() {
                    if !v.is_finite() {
                        let name = self
                            .feature_names
                            .get(j)
                            .map_or_else(|| format!("feature[{j}]"), std::clone::Clone::clone);
                        return Err(ScryLearnError::InvalidData(format!(
                            "non-finite value ({v}) in {name} at sample {i}"
                        )));
                    }
                }
            }
        }
        for (i, &v) in self.target.iter().enumerate() {
            if !v.is_finite() {
                return Err(ScryLearnError::InvalidData(format!(
                    "non-finite value ({v}) in target at sample {i}"
                )));
            }
        }
        Ok(())
    }

    /// Returns `Err(InvalidData)` if any feature or target value is ±Inf.
    ///
    /// Unlike [`validate_finite`](Self::validate_finite), this allows NaN
    /// values (useful for imputers that intentionally handle NaN).
    pub fn validate_no_inf(&self) -> Result<()> {
        if let Storage::Sparse(csc) = &self.storage {
            for j in 0..csc.n_cols() {
                for (i, v) in csc.col(j).iter() {
                    if v.is_infinite() {
                        let name = self
                            .feature_names
                            .get(j)
                            .map_or_else(|| format!("feature[{j}]"), std::clone::Clone::clone);
                        return Err(ScryLearnError::InvalidData(format!(
                            "infinite value ({v}) in {name} at sample {i}"
                        )));
                    }
                }
            }
        } else {
            for (j, col) in self.features.iter().enumerate() {
                for (i, &v) in col.iter().enumerate() {
                    if v.is_infinite() {
                        let name = self
                            .feature_names
                            .get(j)
                            .map_or_else(|| format!("feature[{j}]"), std::clone::Clone::clone);
                        return Err(ScryLearnError::InvalidData(format!(
                            "infinite value ({v}) in {name} at sample {i}"
                        )));
                    }
                }
            }
        }
        for (i, &v) in self.target.iter().enumerate() {
            if v.is_infinite() {
                return Err(ScryLearnError::InvalidData(format!(
                    "infinite value ({v}) in target at sample {i}"
                )));
            }
        }
        Ok(())
    }

    /// Attach class labels for classification.
    pub fn with_class_labels(mut self, labels: Vec<String>) -> Self {
        self.class_labels = Some(labels);
        self
    }

    /// Create a dataset from a sparse CSC matrix.
    ///
    /// The `features` field is left empty. Call [`ensure_dense`](Self::ensure_dense)
    /// before accessing `features` directly on a sparse dataset.
    pub fn from_sparse(
        csc: CscMatrix,
        target: Vec<f64>,
        feature_names: Vec<String>,
        target_name: impl Into<String>,
    ) -> Self {
        Self {
            features: Vec::new(),
            target,
            feature_names,
            target_name: target_name.into(),
            class_labels: None,
            matrix: OnceLock::new(),
            row_major_cache: None,
            storage: Storage::Sparse(csc),
        }
    }

    /// Whether this dataset uses sparse storage.
    #[inline]
    pub fn is_sparse(&self) -> bool {
        matches!(self.storage, Storage::Sparse(_))
    }

    /// Get the sparse CSC matrix if available.
    pub fn sparse_csc(&self) -> Option<&CscMatrix> {
        match &self.storage {
            Storage::Sparse(m) => Some(m),
            Storage::Dense => None,
        }
    }

    /// Get the sparse CSR matrix (converted from CSC on demand).
    pub fn sparse_csr(&self) -> Option<crate::sparse::CsrMatrix> {
        self.sparse_csc().map(CscMatrix::to_csr)
    }

    /// Compute descriptive statistics for every feature column and the target.
    ///
    /// Returns one [`ColumnStats`] per feature (in order) followed by one for
    /// the target column. NaN values are excluded from all computations.
    /// Standard deviation uses `ddof=1` (sample std) to match pandas.
    pub fn summary(&self) -> Vec<ColumnStats> {
        let n_feat = self.n_features();
        let mut stats = Vec::with_capacity(n_feat + 1);

        for j in 0..n_feat {
            let name = self
                .feature_names
                .get(j)
                .cloned()
                .unwrap_or_else(|| format!("feature[{j}]"));

            let col: Vec<f64> = if let Some(csc) = self.sparse_csc() {
                // Reconstruct the full column (zeros + stored values).
                let n_rows = csc.n_rows();
                let mut dense = vec![0.0_f64; n_rows];
                for (i, v) in csc.col(j).iter() {
                    dense[i] = v;
                }
                dense
            } else {
                self.features[j].clone()
            };

            stats.push(compute_column_stats(&name, &col));
        }

        stats.push(compute_column_stats(&self.target_name, &self.target));
        stats
    }

    /// Print a pandas-style descriptive statistics table to stdout.
    ///
    /// Internally calls [`summary()`](Self::summary).
    pub fn describe(&self) {
        let stats = self.summary();
        if stats.is_empty() {
            return;
        }

        let labels = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"];
        let label_width = labels.iter().map(|l| l.len()).max().unwrap_or(0);

        let col_widths: Vec<usize> = stats
            .iter()
            .map(|s| s.name.len().max(12))
            .collect();

        // Header row.
        print!("{:>width$}", "", width = label_width);
        for (i, s) in stats.iter().enumerate() {
            print!("  {:>width$}", s.name, width = col_widths[i]);
        }
        println!();

        // Data rows.
        for (row_idx, label) in labels.iter().enumerate() {
            print!("{:>width$}", label, width = label_width);
            for (i, s) in stats.iter().enumerate() {
                let val = match row_idx {
                    0 => s.count as f64,
                    1 => s.mean,
                    2 => s.std,
                    3 => s.min,
                    4 => s.q25,
                    5 => s.median,
                    6 => s.q75,
                    7 => s.max,
                    _ => unreachable!(),
                };
                print!("  {:>width$.6}", val, width = col_widths[i]);
            }
            println!();
        }
    }

    /// Populate the `features` field from sparse storage.
    ///
    /// No-op if the dataset is already dense. After calling this,
    /// `features[j][i]` is available as usual.
    pub fn ensure_dense(&mut self) {
        if let Storage::Sparse(csc) = &self.storage {
            let n_cols = csc.n_cols();
            let n_rows = csc.n_rows();
            let mut features = vec![vec![0.0; n_rows]; n_cols];
            for (j, feat_col) in features.iter_mut().enumerate() {
                for (i, v) in csc.col(j).iter() {
                    feat_col[i] = v;
                }
            }
            self.features = features;
            self.matrix = OnceLock::new();
        }
    }
}

/// Subset a CSC matrix by selecting specific row indices.
///
/// Returns a new CSC matrix with `indices.len()` rows, where row `k` in the
/// output corresponds to row `indices[k]` in the input.
///
/// Uses `CscMatrix::from_dense` (column-major) to avoid a pre-existing
/// dedup bug in `CscMatrix::from_triplets`.
fn subset_csc(csc: &CscMatrix, indices: &[usize]) -> CscMatrix {
    let n_new_rows = indices.len();
    let n_cols = csc.n_cols();

    // Build old→new row mapping.
    let mut row_map = std::collections::HashMap::with_capacity(n_new_rows);
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        row_map.insert(old_idx, new_idx);
    }

    // Build column-major dense vectors for the subset.
    let mut cols: Vec<Vec<f64>> = vec![vec![0.0; n_new_rows]; n_cols];
    for (j, col) in cols.iter_mut().enumerate() {
        for (old_row, val) in csc.col(j).iter() {
            if let Some(&new_row) = row_map.get(&old_row) {
                col[new_row] = val;
            }
        }
    }

    CscMatrix::from_dense(&cols)
}

/// Compute descriptive statistics for a single column, filtering NaN values.
fn compute_column_stats(name: &str, values: &[f64]) -> ColumnStats {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_unstable_by(|a, b| a.total_cmp(b));

    let count = sorted.len();
    if count == 0 {
        return ColumnStats {
            name: name.to_string(),
            count: 0,
            mean: f64::NAN,
            std: f64::NAN,
            min: f64::NAN,
            q25: f64::NAN,
            median: f64::NAN,
            q75: f64::NAN,
            max: f64::NAN,
        };
    }

    let sum: f64 = sorted.iter().sum();
    let mean = sum / count as f64;

    let std = if count <= 1 {
        0.0
    } else {
        let var = sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1) as f64;
        var.sqrt()
    };

    let min = sorted[0];
    let max = sorted[count - 1];
    let q25 = percentile(&sorted, 0.25);
    let median = percentile(&sorted, 0.50);
    let q75 = percentile(&sorted, 0.75);

    ColumnStats {
        name: name.to_string(),
        count,
        mean,
        std,
        min,
        q25,
        median,
        q75,
        max,
    }
}

/// Linear-interpolation percentile on a pre-sorted slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = p * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = lo + 1;
    let frac = idx - lo as f64;
    if hi >= n {
        sorted[lo]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

#[cfg(feature = "csv")]
/// Parse a target column: try numeric, fall back to label encoding.
///
/// Returns `(encoded_values, Option<class_labels>)`.
fn parse_target_column(rows: &[Vec<String>], col_idx: usize) -> (Vec<f64>, Option<Vec<String>>) {
    // Try parsing all as numeric first.
    let numeric: Vec<Option<f64>> = rows
        .iter()
        .map(|row| row.get(col_idx).and_then(|s| s.parse::<f64>().ok()))
        .collect();

    let all_numeric = numeric.iter().all(std::option::Option::is_some);
    if all_numeric {
        return (numeric.into_iter().flatten().collect(), None);
    }

    // Label-encode string values.
    let mut labels: Vec<String> = Vec::new();
    let mut encoded = Vec::with_capacity(rows.len());

    for row in rows {
        let val = row.get(col_idx).map_or("", std::string::String::as_str);
        let idx = labels.iter().position(|l| l == val).unwrap_or_else(|| {
            labels.push(val.to_string());
            labels.len() - 1
        });
        encoded.push(idx as f64);
    }

    (encoded, Some(labels))
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_new() {
        let features = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let target = vec![0.0, 1.0, 0.0];
        let ds = Dataset::new(features, target, vec!["f1".into(), "f2".into()], "label");
        assert_eq!(ds.n_samples(), 3);
        assert_eq!(ds.n_features(), 2);
        assert_eq!(ds.feature(0), &[1.0, 2.0, 3.0]);
        assert_eq!(ds.sample(1), vec![2.0, 5.0]);
    }

    #[cfg(feature = "csv")]
    #[test]
    fn test_dataset_from_csv_reader() {
        let csv = "f1,f2,target\n1.0,4.0,a\n2.0,5.0,b\n3.0,6.0,a\n";
        let ds = Dataset::from_csv_reader(csv.as_bytes(), "target").unwrap();
        assert_eq!(ds.n_samples(), 3);
        assert_eq!(ds.n_features(), 2);
        assert_eq!(ds.target, vec![0.0, 1.0, 0.0]);
        assert_eq!(
            ds.class_labels,
            Some(vec!["a".to_string(), "b".to_string()])
        );
    }

    #[test]
    fn test_dataset_subset() {
        let features = vec![vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]];
        let target = vec![0.0, 1.0, 0.0, 1.0];
        let ds = Dataset::new(features, target, vec!["a".into(), "b".into()], "t");
        let sub = ds.subset(&[0, 2]);
        assert_eq!(sub.n_samples(), 2);
        assert_eq!(sub.feature(0), &[1.0, 3.0]);
        assert_eq!(sub.target, vec![0.0, 0.0]);
    }

    #[cfg(feature = "csv")]
    #[test]
    fn test_empty_csv() {
        let csv = "f1,target\n";
        let err = Dataset::from_csv_reader(csv.as_bytes(), "target");
        assert!(err.is_err());
    }

    #[test]
    fn test_n_classes() {
        let ds = Dataset::new(
            vec![vec![1.0, 2.0, 3.0]],
            vec![0.0, 1.0, 2.0],
            vec!["f".into()],
            "t",
        );
        assert_eq!(ds.n_classes(), 3);
    }

    #[test]
    fn test_matrix_accessor() {
        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let ds = Dataset::new(features, vec![0.0, 1.0], vec!["a".into(), "b".into()], "t");
        let mat = ds.matrix();
        assert_eq!(mat.n_rows(), 2);
        assert_eq!(mat.n_cols(), 2);
        assert_eq!(mat.col(0), &[1.0, 2.0]);
        assert_eq!(mat.col(1), &[3.0, 4.0]);
    }

    #[test]
    fn test_from_matrix() {
        let mat = DenseMatrix::from_col_major(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let ds = Dataset::from_matrix(mat, vec![0.0, 1.0], vec!["a".into(), "b".into()], "t");
        assert_eq!(ds.n_samples(), 2);
        assert_eq!(ds.n_features(), 2);
        assert_eq!(ds.feature(0), &[1.0, 2.0]);
        assert_eq!(ds.matrix().col(1), &[3.0, 4.0]);
    }

    // -------------------------------------------------------------------
    // Sparse dataset tests
    // -------------------------------------------------------------------

    fn sample_csc() -> CscMatrix {
        // 3 samples × 2 features (column-major):
        //   col 0: [1.0, 0.0, 3.0]
        //   col 1: [0.0, 2.0, 0.0]
        CscMatrix::from_dense(&[vec![1.0, 0.0, 3.0], vec![0.0, 2.0, 0.0]])
    }

    #[test]
    fn test_from_sparse_basic() {
        let csc = sample_csc();
        let ds = Dataset::from_sparse(csc, vec![0.0, 1.0, 0.0], vec!["a".into(), "b".into()], "t");
        assert!(ds.is_sparse());
        assert_eq!(ds.n_samples(), 3);
        assert_eq!(ds.n_features(), 2);
    }

    #[test]
    fn test_sparse_csc_accessor() {
        let csc = sample_csc();
        let ds = Dataset::from_sparse(csc, vec![0.0, 1.0, 0.0], vec!["a".into(), "b".into()], "t");
        let csc_ref = ds.sparse_csc().expect("should have CSC");
        assert_eq!(csc_ref.n_rows(), 3);
        assert_eq!(csc_ref.n_cols(), 2);
        assert_eq!(csc_ref.get(0, 0), 1.0);
        assert_eq!(csc_ref.get(1, 1), 2.0);
        assert_eq!(csc_ref.get(1, 0), 0.0);
    }

    #[test]
    fn test_sparse_csr_conversion() {
        let csc = sample_csc();
        let ds = Dataset::from_sparse(csc, vec![0.0, 1.0, 0.0], vec!["a".into(), "b".into()], "t");
        let csr = ds.sparse_csr().expect("should convert to CSR");
        assert_eq!(csr.n_rows(), 3);
        assert_eq!(csr.n_cols(), 2);
        assert_eq!(csr.get(0, 0), 1.0);
        assert_eq!(csr.get(2, 0), 3.0);
        assert_eq!(csr.get(1, 1), 2.0);
    }

    #[test]
    fn test_sparse_subset() {
        let csc = sample_csc();
        let ds = Dataset::from_sparse(csc, vec![0.0, 1.0, 2.0], vec!["a".into(), "b".into()], "t");
        let sub = ds.subset(&[0, 2]);
        assert!(sub.is_sparse());
        assert_eq!(sub.n_samples(), 2);
        assert_eq!(sub.n_features(), 2);
        assert_eq!(sub.target, vec![0.0, 2.0]);
        let csc_ref = sub.sparse_csc().unwrap();
        assert_eq!(csc_ref.get(0, 0), 1.0); // row 0 of subset = original row 0
        assert_eq!(csc_ref.get(1, 0), 3.0); // row 1 of subset = original row 2
    }

    #[test]
    fn test_sparse_with_class_labels() {
        let csc = sample_csc();
        let ds = Dataset::from_sparse(csc, vec![0.0, 1.0, 0.0], vec!["a".into(), "b".into()], "t")
            .with_class_labels(vec!["cat".into(), "dog".into()]);
        assert!(ds.is_sparse());
        assert_eq!(
            ds.class_labels,
            Some(vec!["cat".to_string(), "dog".to_string()])
        );
    }

    #[test]
    fn test_n_features_consistency() {
        // Dense and sparse datasets with same data should report same n_features.
        let dense_ds = Dataset::new(
            vec![vec![1.0, 0.0, 3.0], vec![0.0, 2.0, 0.0]],
            vec![0.0, 1.0, 0.0],
            vec!["a".into(), "b".into()],
            "t",
        );
        let csc = sample_csc();
        let sparse_ds =
            Dataset::from_sparse(csc, vec![0.0, 1.0, 0.0], vec!["a".into(), "b".into()], "t");
        assert_eq!(dense_ds.n_features(), sparse_ds.n_features());
    }

    #[test]
    fn test_ensure_dense() {
        let csc = sample_csc();
        let mut ds =
            Dataset::from_sparse(csc, vec![0.0, 1.0, 0.0], vec!["a".into(), "b".into()], "t");
        assert!(ds.features.is_empty());
        ds.ensure_dense();
        assert_eq!(ds.features.len(), 2);
        assert_eq!(ds.features[0], vec![1.0, 0.0, 3.0]);
        assert_eq!(ds.features[1], vec![0.0, 2.0, 0.0]);
    }

    #[test]
    fn test_dense_not_sparse() {
        let ds = Dataset::new(vec![vec![1.0, 2.0]], vec![0.0, 1.0], vec!["x".into()], "y");
        assert!(!ds.is_sparse());
        assert!(ds.sparse_csc().is_none());
        assert!(ds.sparse_csr().is_none());
    }

    #[test]
    fn test_matrix_lazy_rebuild_after_invalidate() {
        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let mut ds = Dataset::new(features, vec![0.0, 1.0], vec!["a".into(), "b".into()], "t");

        // Matrix is available after construction.
        assert_eq!(ds.matrix().col(0), &[1.0, 2.0]);

        // Invalidate.
        ds.invalidate_matrix();

        // matrix() should lazily rebuild — no panic.
        assert_eq!(ds.matrix().col(0), &[1.0, 2.0]);
        assert_eq!(ds.matrix().col(1), &[3.0, 4.0]);
    }

    #[test]
    fn test_describe_summary() {
        let features = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![10.0, 20.0, 30.0, 40.0],
        ];
        let target = vec![0.0, 1.0, 0.0, 1.0];
        let ds = Dataset::new(
            features,
            target,
            vec!["a".into(), "b".into()],
            "t",
        );

        let stats = ds.summary();
        assert_eq!(stats.len(), 3); // 2 features + 1 target

        // Feature "a": [1, 2, 3, 4]
        assert_eq!(stats[0].name, "a");
        assert_eq!(stats[0].count, 4);
        assert!((stats[0].mean - 2.5).abs() < 1e-10);
        assert!((stats[0].min - 1.0).abs() < 1e-10);
        assert!((stats[0].max - 4.0).abs() < 1e-10);

        // Feature "b": [10, 20, 30, 40]
        assert_eq!(stats[1].name, "b");
        assert_eq!(stats[1].count, 4);
        assert!((stats[1].mean - 25.0).abs() < 1e-10);
        assert!((stats[1].min - 10.0).abs() < 1e-10);
        assert!((stats[1].max - 40.0).abs() < 1e-10);

        // Target "t": [0, 1, 0, 1]
        assert_eq!(stats[2].name, "t");
        assert_eq!(stats[2].count, 4);
        assert!((stats[2].mean - 0.5).abs() < 1e-10);

        // describe() should not panic.
        ds.describe();
    }

    #[test]
    fn test_matrix_lazy_rebuild_reflects_feature_mutation() {
        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let mut ds = Dataset::new(features, vec![0.0, 1.0], vec!["a".into(), "b".into()], "t");

        // Mutate features and invalidate.
        ds.features[0][0] = 99.0;
        ds.invalidate_matrix();

        // Lazy rebuild should reflect the mutation.
        assert_eq!(ds.matrix().col(0), &[99.0, 2.0]);
    }
}
