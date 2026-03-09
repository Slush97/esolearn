// SPDX-License-Identifier: MIT OR Apache-2.0
//! Polars DataFrame ↔ Dataset interop.
//!
//! Requires the `polars` feature flag.

use polars::prelude::*;

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};

impl Dataset {
    /// Create a Dataset from a Polars DataFrame.
    ///
    /// The `target_col` column becomes the target. All other numeric
    /// (f64/f32/i64/i32/bool) columns become features. Non-numeric columns
    /// are silently skipped.
    ///
    /// # Errors
    ///
    /// Returns [`ScryLearnError::InvalidColumn`] if `target_col` doesn't
    /// exist or is not numeric. Returns [`ScryLearnError::InvalidParameter`]
    /// if no numeric feature columns remain, or if any numeric column
    /// contains null values (use `SimpleImputer` first).
    pub fn from_dataframe(df: &DataFrame, target_col: &str) -> Result<Self> {
        // Extract target column.
        let target_series = df
            .column(target_col)
            .map_err(|_| ScryLearnError::InvalidColumn(target_col.to_string()))?;

        let target_f64 = cast_column_to_f64(target_series).ok_or_else(|| {
            ScryLearnError::InvalidColumn(format!(
                "target column '{target_col}' is not numeric (type: {})",
                target_series.dtype()
            ))
        })?;

        let target = extract_f64_vec(&target_f64, target_col)?;

        // Extract feature columns (all numeric columns except the target).
        let mut features: Vec<Vec<f64>> = Vec::new();
        let mut feature_names: Vec<String> = Vec::new();

        for col in df.get_columns() {
            let name = col.name().as_str();
            if name == target_col {
                continue;
            }
            if let Some(cast) = cast_column_to_f64(col) {
                let vals = extract_f64_vec(&cast, name)?;
                features.push(vals);
                feature_names.push(name.to_string());
            }
            // Non-numeric columns are silently skipped.
        }

        if features.is_empty() {
            return Err(ScryLearnError::InvalidParameter(
                "no numeric feature columns found in DataFrame".to_string(),
            ));
        }

        Ok(Dataset::new(features, target, feature_names, target_col))
    }

    /// Convert this Dataset to a Polars DataFrame.
    ///
    /// Each feature column becomes a named `Float64` Series, plus the target.
    pub fn to_dataframe(&self) -> Result<DataFrame> {
        let mut columns: Vec<Column> = Vec::with_capacity(self.feature_names.len() + 1);

        for (i, name) in self.feature_names.iter().enumerate() {
            let s = Column::new(name.as_str().into(), &self.features[i]);
            columns.push(s);
        }

        let target_col = Column::new(self.target_name.as_str().into(), &self.target);
        columns.push(target_col);

        DataFrame::new(columns).map_err(|e| {
            ScryLearnError::InvalidParameter(format!("failed to create DataFrame: {e}"))
        })
    }
}

/// Try to cast a polars Column to f64. Returns `None` if the dtype is not
/// numeric/boolean.
fn cast_column_to_f64(col: &Column) -> Option<Column> {
    match col.dtype() {
        DataType::Float64 => Some(col.clone()),
        DataType::Float32
        | DataType::Int64
        | DataType::Int32
        | DataType::Int16
        | DataType::Int8
        | DataType::UInt64
        | DataType::UInt32
        | DataType::UInt16
        | DataType::UInt8
        | DataType::Boolean => col.cast(&DataType::Float64).ok(),
        _ => None,
    }
}

/// Extract a `Vec<f64>` from a column known to be Float64.
/// Errors on null values.
fn extract_f64_vec(col: &Column, col_name: &str) -> Result<Vec<f64>> {
    let ca = col
        .f64()
        .map_err(|e| ScryLearnError::InvalidParameter(format!("column '{col_name}': {e}")))?;

    let mut out = Vec::with_capacity(ca.len());
    for opt_val in ca {
        match opt_val {
            Some(v) => out.push(v),
            None => {
                return Err(ScryLearnError::InvalidParameter(format!(
                    "column '{col_name}' contains null values — \
                     use SimpleImputer or DataFrame.fill_null() first"
                )));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_dataframe_basic() {
        let df = df!(
            "f1" => [1.0, 2.0, 3.0],
            "f2" => [4.0, 5.0, 6.0],
            "target" => [0.0, 1.0, 0.0]
        )
        .unwrap();

        let ds = Dataset::from_dataframe(&df, "target").unwrap();
        assert_eq!(ds.n_samples(), 3);
        assert_eq!(ds.n_features(), 2);
        assert_eq!(ds.target, vec![0.0, 1.0, 0.0]);
        assert_eq!(ds.feature(0), &[1.0, 2.0, 3.0]);
        assert_eq!(ds.feature(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_to_dataframe_basic() {
        let ds = Dataset::new(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![0.0, 1.0],
            vec!["a".into(), "b".into()],
            "t",
        );

        let df = ds.to_dataframe().unwrap();
        assert_eq!(df.shape(), (2, 3)); // 2 rows, 3 columns (a, b, t)
        assert_eq!(
            df.column("a").unwrap().f64().unwrap().to_vec(),
            vec![Some(1.0), Some(2.0)]
        );
        assert_eq!(
            df.column("t").unwrap().f64().unwrap().to_vec(),
            vec![Some(0.0), Some(1.0)]
        );
    }

    #[test]
    fn test_round_trip() {
        let ds = Dataset::new(
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            vec![10.0, 20.0, 30.0],
            vec!["x".into(), "y".into()],
            "target",
        );

        let df = ds.to_dataframe().unwrap();
        let ds2 = Dataset::from_dataframe(&df, "target").unwrap();

        assert_eq!(ds2.n_samples(), ds.n_samples());
        assert_eq!(ds2.n_features(), ds.n_features());
        assert_eq!(ds2.target, ds.target);
        assert_eq!(ds2.feature_names, ds.feature_names);
        for i in 0..ds.n_features() {
            assert_eq!(ds2.feature(i), ds.feature(i));
        }
    }

    #[test]
    fn test_mixed_types() {
        // i64 and bool columns should be cast; string should be skipped.
        let df = DataFrame::new(vec![
            Column::new("float_col".into(), &[1.0_f64, 2.0, 3.0]),
            Column::new("int_col".into(), &[10_i64, 20, 30]),
            Column::new("bool_col".into(), &[true, false, true]),
            Column::new("str_col".into(), &["a", "b", "c"]),
            Column::new("target".into(), &[0.0_f64, 1.0, 0.0]),
        ])
        .unwrap();

        let ds = Dataset::from_dataframe(&df, "target").unwrap();
        assert_eq!(ds.n_features(), 3); // float, int, bool — string skipped
        assert_eq!(ds.feature_names, vec!["float_col", "int_col", "bool_col"]);
        assert_eq!(ds.feature(1), &[10.0, 20.0, 30.0]); // i64 → f64
        assert_eq!(ds.feature(2), &[1.0, 0.0, 1.0]); // bool → f64
    }

    #[test]
    fn test_null_handling_errors() {
        let s1 = Column::new("f1".into(), &[1.0_f64, 2.0, 3.0]);
        let s2: Column = {
            let ca = Float64Chunked::new("f2".into(), &[Some(1.0), None, Some(3.0)]);
            ca.into_column()
        };
        let s3 = Column::new("target".into(), &[0.0_f64, 1.0, 0.0]);

        let df = DataFrame::new(vec![s1, s2, s3]).unwrap();
        let err = Dataset::from_dataframe(&df, "target");
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("null"), "expected null error, got: {msg}");
    }

    #[test]
    fn test_missing_target_errors() {
        let df = df!(
            "f1" => [1.0, 2.0],
            "f2" => [3.0, 4.0]
        )
        .unwrap();

        let err = Dataset::from_dataframe(&df, "nonexistent");
        assert!(err.is_err());
    }

    #[test]
    fn test_single_feature() {
        let df = df!(
            "feat" => [1.0, 2.0, 3.0],
            "target" => [0.0, 1.0, 0.0]
        )
        .unwrap();

        let ds = Dataset::from_dataframe(&df, "target").unwrap();
        assert_eq!(ds.n_features(), 1);
        assert_eq!(ds.feature(0), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_empty_dataframe() {
        let df = df!(
            "f1" => Vec::<f64>::new(),
            "target" => Vec::<f64>::new()
        )
        .unwrap();

        let ds = Dataset::from_dataframe(&df, "target").unwrap();
        assert_eq!(ds.n_samples(), 0);
        assert_eq!(ds.n_features(), 1);
    }

    #[test]
    fn test_large_dataframe() {
        let n = 10_000;
        let mut rng = fastrand::Rng::with_seed(42);
        let f1: Vec<f64> = (0..n).map(|_| rng.f64()).collect();
        let f2: Vec<f64> = (0..n).map(|_| rng.f64()).collect();
        let target: Vec<f64> = (0..n).map(|_| (rng.u32(0..2)) as f64).collect();

        let df = df!(
            "f1" => f1.clone(),
            "f2" => f2.clone(),
            "target" => target.clone()
        )
        .unwrap();

        let ds = Dataset::from_dataframe(&df, "target").unwrap();
        assert_eq!(ds.n_samples(), n);
        assert_eq!(ds.n_features(), 2);
        assert_eq!(ds.target, target);
        assert_eq!(ds.feature(0), f1.as_slice());
        assert_eq!(ds.feature(1), f2.as_slice());
    }
}
