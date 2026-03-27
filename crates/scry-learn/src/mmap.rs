// SPDX-License-Identifier: MIT OR Apache-2.0
//! Memory-mapped dataset loading via the `.scry` binary format.
//!
//! [`MmapDataset`] maps a `.scry` file into virtual memory so the OS pages in
//! columns on demand — a 1 GB file does not require 1 GB of RAM.
//!
//! # File Format
//!
//! ```text
//! [Magic: 4 bytes "SCRY"]
//! [Version: u32 LE]           // currently 1
//! [n_rows: u64 LE]
//! [n_cols: u64 LE]            // n_features + 1 (target is last column)
//! [n_feature_names: u64 LE]
//! [target_name_len: u16 LE]
//! [target_name: UTF-8 bytes]
//! [feature_name_lens: n_feature_names × u16 LE]
//! [feature_names: concatenated UTF-8 bytes]
//! [padding: 0–7 bytes to reach 8-byte alignment]
//! [data: n_rows × n_cols × f64 LE, column-major]
//! ```
//!
//! # Example
//!
//! ```ignore
//! use scry_learn::mmap::MmapDataset;
//!
//! let mmap = MmapDataset::open("big_data.scry")?;
//! println!("{} samples × {} features", mmap.n_samples(), mmap.n_features());
//!
//! // Stream through in batches with partial_fit
//! for start in (0..mmap.n_samples()).step_by(10_000) {
//!     let end = (start + 10_000).min(mmap.n_samples());
//!     let batch = mmap.batch(start, end);
//!     model.partial_fit(&batch)?;
//! }
//! ```

use std::io::Write;
use std::path::Path;

use memmap2::Mmap;

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};

const MAGIC: &[u8; 4] = b"SCRY";
const VERSION: u32 = 1;

/// Read exactly `N` little-endian bytes from `buf` at `pos`, or return an error
/// if the buffer is too short.
fn read_le_bytes<const N: usize>(buf: &[u8], pos: usize) -> Result<[u8; N]> {
    buf.get(pos..pos + N)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| {
            ScryLearnError::InvalidParameter(format!(
                ".scry file truncated at offset {pos} (need {N} bytes)"
            ))
        })
}

/// A memory-mapped dataset backed by a `.scry` binary file.
///
/// Data is not loaded into memory — the OS pages in columns on demand.
/// This enables working with datasets larger than available RAM.
#[non_exhaustive]
pub struct MmapDataset {
    mmap: Mmap,
    n_rows: usize,
    n_cols: usize,
    feature_names: Vec<String>,
    target_name: String,
    data_offset: usize,
}

impl MmapDataset {
    /// Open a memory-mapped dataset from a `.scry` file.
    #[allow(unsafe_code)]
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())?;
        // SAFETY: `Mmap::map` requires that the underlying file is not
        // concurrently modified (truncated, overwritten) while the mapping
        // is live — otherwise the OS may deliver SIGBUS or expose
        // uninitialised pages. We uphold this by:
        //   1. Opening the file read-only (`File::open`).
        //   2. Not exposing the `Mmap` handle — it is stored in the
        //      private `mmap` field and never handed out.
        //   3. Documenting that callers must not modify the file while
        //      the `MmapDataset` is alive.
        let mmap = unsafe { Mmap::map(&file) }.map_err(ScryLearnError::Io)?;
        Self::from_mmap(mmap)
    }

    fn from_mmap(mmap: Mmap) -> Result<Self> {
        let buf = &mmap[..];
        if buf.len() < 4 + 4 + 8 + 8 + 8 + 2 {
            return Err(ScryLearnError::InvalidParameter(
                "file too small for .scry header".into(),
            ));
        }

        // Magic
        if &buf[0..4] != MAGIC {
            return Err(ScryLearnError::InvalidParameter(
                "not a .scry file (bad magic)".into(),
            ));
        }

        let mut pos = 4;

        // Version
        let version = u32::from_le_bytes(read_le_bytes::<4>(buf, pos)?);
        pos += 4;
        if version != VERSION {
            return Err(ScryLearnError::InvalidParameter(format!(
                "unsupported .scry version: {version}"
            )));
        }

        // n_rows, n_cols
        let n_rows = u64::from_le_bytes(read_le_bytes::<8>(buf, pos)?) as usize;
        pos += 8;
        let n_cols = u64::from_le_bytes(read_le_bytes::<8>(buf, pos)?) as usize;
        pos += 8;

        if n_cols == 0 {
            return Err(ScryLearnError::InvalidParameter(
                ".scry file has 0 columns".into(),
            ));
        }

        // n_feature_names
        let n_feature_names = u64::from_le_bytes(read_le_bytes::<8>(buf, pos)?) as usize;
        pos += 8;

        // target_name
        let target_name_len = u16::from_le_bytes(read_le_bytes::<2>(buf, pos)?) as usize;
        pos += 2;
        if pos + target_name_len > buf.len() {
            return Err(ScryLearnError::InvalidParameter(
                "file truncated reading target name".into(),
            ));
        }
        let target_name = std::str::from_utf8(&buf[pos..pos + target_name_len])
            .map_err(|e| {
                ScryLearnError::InvalidParameter(format!("bad UTF-8 in target name: {e}"))
            })?
            .to_string();
        pos += target_name_len;

        // feature_names
        let mut feature_names = Vec::with_capacity(n_feature_names.min(10_000));
        let mut name_lens = Vec::with_capacity(n_feature_names.min(10_000));
        for _ in 0..n_feature_names {
            let len = u16::from_le_bytes(read_le_bytes::<2>(buf, pos)?) as usize;
            pos += 2;
            name_lens.push(len);
        }
        for len in name_lens {
            if pos + len > buf.len() {
                return Err(ScryLearnError::InvalidParameter(
                    "file truncated reading feature name".into(),
                ));
            }
            let name = std::str::from_utf8(&buf[pos..pos + len])
                .map_err(|e| {
                    ScryLearnError::InvalidParameter(format!("bad UTF-8 in feature name: {e}"))
                })?
                .to_string();
            pos += len;
            feature_names.push(name);
        }

        // Align to 8 bytes for f64 data
        let remainder = pos % 8;
        if remainder != 0 {
            pos += 8 - remainder;
        }

        let data_offset = pos;
        let expected_size = data_offset + n_rows * n_cols * 8;
        if buf.len() < expected_size {
            return Err(ScryLearnError::InvalidParameter(format!(
                "file truncated: expected {expected_size} bytes, got {}",
                buf.len()
            )));
        }

        Ok(Self {
            mmap,
            n_rows,
            n_cols,
            feature_names,
            target_name,
            data_offset,
        })
    }

    /// Number of samples (rows).
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_rows
    }

    /// Number of features (excluding target).
    #[inline]
    pub fn n_features(&self) -> usize {
        self.n_cols.saturating_sub(1)
    }

    /// Feature column names.
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Target column name.
    pub fn target_name(&self) -> &str {
        &self.target_name
    }

    /// Get a feature column as a zero-copy slice via `bytemuck`.
    ///
    /// Column `j` spans bytes `[data_offset + j * n_rows * 8 .. +n_rows*8]`
    /// in the memory-mapped file. On little-endian platforms this is a direct
    /// reinterpret cast with no allocation.
    ///
    /// # Panics
    ///
    /// Panics if `j >= n_cols`.
    pub fn col(&self, j: usize) -> &[f64] {
        assert!(
            j < self.n_cols,
            "column index {j} out of bounds (n_cols={})",
            self.n_cols
        );
        let offset = self.data_offset + j * self.n_rows * 8;
        let bytes = &self.mmap[offset..offset + self.n_rows * 8];
        bytemuck::cast_slice(bytes)
    }

    /// Get the target column as a zero-copy slice.
    ///
    /// The target is stored as the last column in the `.scry` file.
    ///
    /// # Panics
    ///
    /// Panics if `n_cols == 0` (no columns in the file).
    pub fn target(&self) -> &[f64] {
        assert!(
            self.n_cols > 0,
            "MmapDataset has 0 columns — cannot read target"
        );
        self.col(self.n_cols - 1)
    }

    /// Extract a batch of rows as a [`Dataset`] (copies into memory).
    ///
    /// Use this with [`PartialFit`](crate::partial_fit::PartialFit) for
    /// streaming training over datasets larger than RAM:
    ///
    /// ```ignore
    /// let mmap = MmapDataset::open("big_data.scry")?;
    /// let mut model = LogisticRegression::new();
    /// for start in (0..mmap.n_samples()).step_by(10_000) {
    ///     let end = (start + 10_000).min(mmap.n_samples());
    ///     let batch = mmap.batch(start, end);
    ///     model.partial_fit(&batch)?;
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `start > end` or `end > n_samples`.
    pub fn batch(&self, start: usize, end: usize) -> Dataset {
        assert!(start <= end, "start ({start}) > end ({end})");
        assert!(
            end <= self.n_rows,
            "end ({end}) > n_samples ({})",
            self.n_rows
        );
        let n_features = self.n_features();

        let mut features = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let col = self.col(j);
            features.push(col[start..end].to_vec());
        }

        let target_col = self.col(self.n_cols - 1);
        let target = target_col[start..end].to_vec();

        Dataset::new(
            features,
            target,
            self.feature_names.clone(),
            &self.target_name,
        )
    }

    /// Convert the entire mmap to a regular [`Dataset`] (loads all into memory).
    pub fn to_dataset(&self) -> Dataset {
        self.batch(0, self.n_rows)
    }

    /// Convert a CSV file to `.scry` format.
    ///
    /// Reads the CSV once and writes the binary format. Subsequent loads via
    /// [`open()`](Self::open) are near-instant.
    ///
    /// Requires the `csv` feature.
    #[cfg(feature = "csv")]
    pub fn from_csv(
        csv_path: impl AsRef<Path>,
        target_col: &str,
        output_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let path_str = csv_path.as_ref().to_str().ok_or_else(|| {
            ScryLearnError::InvalidParameter("CSV path contains invalid UTF-8".into())
        })?;
        let dataset = Dataset::from_csv(path_str, target_col)?;
        save_scry(&dataset, &output_path)?;
        Self::open(output_path)
    }
}

/// Save a dataset to a `.scry` binary file for fast memory-mapped loading.
///
/// This is a free function rather than a method on `Dataset` to avoid
/// modifying `dataset.rs` (which other agents may be editing).
pub fn save_scry(dataset: &Dataset, path: impl AsRef<Path>) -> Result<()> {
    let mut file = std::fs::File::create(path.as_ref())?;

    let n_rows = dataset.n_samples();
    let n_features = dataset.n_features();
    let n_cols = n_features + 1; // features + target

    // Magic + version
    file.write_all(MAGIC)?;
    file.write_all(&VERSION.to_le_bytes())?;

    // Dimensions
    file.write_all(&(n_rows as u64).to_le_bytes())?;
    file.write_all(&(n_cols as u64).to_le_bytes())?;
    file.write_all(&(n_features as u64).to_le_bytes())?;

    // Target name
    let target_bytes = dataset.target_name.as_bytes();
    file.write_all(&(target_bytes.len() as u16).to_le_bytes())?;
    file.write_all(target_bytes)?;

    // Feature name lengths
    for name in &dataset.feature_names {
        file.write_all(&(name.len() as u16).to_le_bytes())?;
    }
    // Feature name bytes
    for name in &dataset.feature_names {
        file.write_all(name.as_bytes())?;
    }

    // Calculate current position for alignment padding
    let mut pos = 4 + 4 + 8 + 8 + 8 + 2 + target_bytes.len();
    for name in &dataset.feature_names {
        pos += 2 + name.len(); // u16 len + bytes
    }
    let remainder = pos % 8;
    if remainder != 0 {
        let padding = 8 - remainder;
        file.write_all(&vec![0u8; padding])?;
    }

    // Write column-major f64 data: features first, then target
    for j in 0..n_features {
        let col = &dataset.features[j];
        for &val in col {
            file.write_all(&val.to_le_bytes())?;
        }
    }

    // Target column (last)
    for &val in &dataset.target {
        file.write_all(&val.to_le_bytes())?;
    }

    file.flush()?;
    Ok(())
}

impl std::fmt::Debug for MmapDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapDataset")
            .field("n_rows", &self.n_rows)
            .field("n_cols", &self.n_cols)
            .field("feature_names", &self.feature_names)
            .field("target_name", &self.target_name)
            .field("data_offset", &self.data_offset)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("scry_mmap_test_{name}_{}.scry", std::process::id()))
    }

    fn sample_dataset(n_rows: usize, n_cols: usize) -> Dataset {
        let mut rng = fastrand::Rng::with_seed(42);
        let features: Vec<Vec<f64>> = (0..n_cols)
            .map(|_| (0..n_rows).map(|_| rng.f64() * 10.0 - 5.0).collect())
            .collect();
        let target: Vec<f64> = (0..n_rows).map(|_| (rng.f64() * 3.0).floor()).collect();
        let names: Vec<String> = (0..n_cols).map(|i| format!("f{i}")).collect();
        Dataset::new(features, target, names, "target")
    }

    // Test 1: Write + read round-trip
    #[test]
    fn test_roundtrip() {
        let path = temp_path("roundtrip");
        let ds = sample_dataset(100, 5);
        save_scry(&ds, &path).unwrap();
        let mmap = MmapDataset::open(&path).unwrap();

        assert_eq!(mmap.n_samples(), 100);
        assert_eq!(mmap.n_features(), 5);

        for j in 0..5 {
            let col = mmap.col(j);
            assert_eq!(col.len(), 100);
            for (i, &val) in col.iter().enumerate() {
                assert!(
                    (val - ds.features[j][i]).abs() < f64::EPSILON,
                    "mismatch at col {j}, row {i}"
                );
            }
        }

        let target = mmap.target();
        for (i, &val) in target.iter().enumerate() {
            assert!(
                (val - ds.target[i]).abs() < f64::EPSILON,
                "target mismatch at row {i}"
            );
        }

        std::fs::remove_file(&path).ok();
    }

    // Test 2: Header parsing
    #[test]
    fn test_header_parsing() {
        let path = temp_path("header");
        let ds = sample_dataset(50, 3);
        save_scry(&ds, &path).unwrap();
        let mmap = MmapDataset::open(&path).unwrap();

        assert_eq!(mmap.n_samples(), 50);
        assert_eq!(mmap.n_features(), 3);
        assert_eq!(mmap.feature_names(), &["f0", "f1", "f2"]);
        assert_eq!(mmap.target_name(), "target");

        std::fs::remove_file(&path).ok();
    }

    // Test 3: col() zero-copy values
    #[test]
    fn test_col_zero_copy() {
        let path = temp_path("col_zero_copy");
        let features = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let target = vec![0.0, 1.0, 0.0];
        let ds = Dataset::new(features, target, vec!["a".into(), "b".into()], "t");
        save_scry(&ds, &path).unwrap();
        let mmap = MmapDataset::open(&path).unwrap();

        assert_eq!(mmap.col(0), &[1.0, 2.0, 3.0]);
        assert_eq!(mmap.col(1), &[4.0, 5.0, 6.0]);

        std::fs::remove_file(&path).ok();
    }

    // Test 4: target() accessor
    #[test]
    fn test_target_accessor() {
        let path = temp_path("target");
        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let target = vec![10.0, 20.0];
        let ds = Dataset::new(features, target, vec!["a".into(), "b".into()], "y");
        save_scry(&ds, &path).unwrap();
        let mmap = MmapDataset::open(&path).unwrap();

        assert_eq!(mmap.target(), &[10.0, 20.0]);

        std::fs::remove_file(&path).ok();
    }

    // Test 5: batch() extraction
    #[test]
    fn test_batch() {
        let path = temp_path("batch");
        let ds = sample_dataset(100, 4);
        save_scry(&ds, &path).unwrap();
        let mmap = MmapDataset::open(&path).unwrap();

        let batch = mmap.batch(10, 20);
        assert_eq!(batch.n_samples(), 10);
        assert_eq!(batch.n_features(), 4);

        for j in 0..4 {
            for i in 0..10 {
                assert!(
                    (batch.features[j][i] - ds.features[j][i + 10]).abs() < f64::EPSILON,
                    "batch mismatch at col {j}, row {i}"
                );
            }
        }
        for i in 0..10 {
            assert!(
                (batch.target[i] - ds.target[i + 10]).abs() < f64::EPSILON,
                "batch target mismatch at row {i}"
            );
        }

        std::fs::remove_file(&path).ok();
    }

    // Test 6: to_dataset() full materialization
    #[test]
    fn test_to_dataset() {
        let path = temp_path("to_dataset");
        let ds = sample_dataset(50, 3);
        save_scry(&ds, &path).unwrap();
        let mmap = MmapDataset::open(&path).unwrap();

        let materialized = mmap.to_dataset();
        assert_eq!(materialized.n_samples(), 50);
        assert_eq!(materialized.n_features(), 3);
        assert_eq!(materialized.target_name, "target");
        assert_eq!(materialized.feature_names, ds.feature_names);

        for j in 0..3 {
            assert_eq!(materialized.features[j], ds.features[j]);
        }
        assert_eq!(materialized.target, ds.target);

        std::fs::remove_file(&path).ok();
    }

    // Test 7: Batch iteration covers all rows
    #[test]
    fn test_batch_iteration() {
        let path = temp_path("batch_iter");
        let ds = sample_dataset(1000, 5);
        save_scry(&ds, &path).unwrap();
        let mmap = MmapDataset::open(&path).unwrap();

        let batch_size = 100;
        let mut total_rows = 0;
        for start in (0..mmap.n_samples()).step_by(batch_size) {
            let end = (start + batch_size).min(mmap.n_samples());
            let batch = mmap.batch(start, end);
            total_rows += batch.n_samples();

            // Verify first row of each batch
            for j in 0..5 {
                assert!(
                    (batch.features[j][0] - ds.features[j][start]).abs() < f64::EPSILON,
                    "batch start mismatch at batch starting {start}"
                );
            }
        }
        assert_eq!(total_rows, 1000);

        std::fs::remove_file(&path).ok();
    }

    // Test 8: Empty dataset
    #[test]
    fn test_empty_dataset() {
        let path = temp_path("empty");
        let ds = Dataset::new(
            vec![Vec::new(), Vec::new()],
            Vec::new(),
            vec!["a".into(), "b".into()],
            "t",
        );
        save_scry(&ds, &path).unwrap();
        let mmap = MmapDataset::open(&path).unwrap();

        assert_eq!(mmap.n_samples(), 0);
        assert_eq!(mmap.n_features(), 2);
        assert_eq!(mmap.col(0), &[] as &[f64]);
        assert_eq!(mmap.target(), &[] as &[f64]);

        let batch = mmap.batch(0, 0);
        assert_eq!(batch.n_samples(), 0);

        std::fs::remove_file(&path).ok();
    }

    // Test 9: Large dataset — verify file size
    #[test]
    fn test_large_dataset_file_size() {
        let path = temp_path("large");
        let n_rows = 100_000;
        let n_cols = 50;
        let ds = sample_dataset(n_rows, n_cols);
        save_scry(&ds, &path).unwrap();

        let metadata = std::fs::metadata(&path).unwrap();
        let file_size = metadata.len() as usize;

        // Data portion: n_rows × (n_cols + 1) × 8 bytes (features + target)
        let data_size = n_rows * (n_cols + 1) * 8;
        // Header is small relative to data — just check total is at least data_size
        assert!(
            file_size >= data_size,
            "file too small: {file_size} < {data_size}"
        );
        // And not unreasonably large (header + padding < 10KB for 50 features)
        assert!(
            file_size < data_size + 10_000,
            "file unexpectedly large: {file_size}"
        );

        // Verify we can open and read it
        let mmap = MmapDataset::open(&path).unwrap();
        assert_eq!(mmap.n_samples(), n_rows);
        assert_eq!(mmap.n_features(), n_cols);

        // Spot-check a value
        assert!(
            (mmap.col(0)[0] - ds.features[0][0]).abs() < f64::EPSILON,
            "value mismatch in large dataset"
        );

        std::fs::remove_file(&path).ok();
    }

    // Test 10: File not found
    #[test]
    fn test_file_not_found() {
        let result = MmapDataset::open("/tmp/nonexistent_scry_test_file_12345.scry");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ScryLearnError::Io(_)),
            "expected Io error, got: {err:?}"
        );
    }

    // Test 11: Miri-compatible write→open→batch round-trip exercising the unsafe mmap path.
    //
    // Miri does not support mmap syscalls, so this test is skipped under miri.
    // Under normal execution it validates the unsafe `Mmap::map()` path with
    // various dataset shapes.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mmap_unsafe_roundtrip() {
        // Single row, single feature.
        let path = temp_path("miri_1x1");
        let ds = Dataset::new(vec![vec![3.14]], vec![1.0], vec!["x".into()], "y");
        save_scry(&ds, &path).unwrap();
        let mmap = MmapDataset::open(&path).unwrap();
        assert_eq!(mmap.n_samples(), 1);
        assert_eq!(mmap.n_features(), 1);
        assert_eq!(mmap.col(0), &[3.14]);
        assert_eq!(mmap.target(), &[1.0]);
        let batch = mmap.batch(0, 1);
        assert_eq!(batch.n_samples(), 1);
        assert_eq!(batch.features[0][0], 3.14);
        std::fs::remove_file(&path).ok();

        // Exact boundary: batch(0, n) == to_dataset().
        let path = temp_path("miri_boundary");
        let ds = sample_dataset(10, 3);
        save_scry(&ds, &path).unwrap();
        let mmap = MmapDataset::open(&path).unwrap();
        let full = mmap.to_dataset();
        let batch_full = mmap.batch(0, 10);
        for j in 0..3 {
            assert_eq!(full.features[j], batch_full.features[j]);
        }
        assert_eq!(full.target, batch_full.target);
        std::fs::remove_file(&path).ok();

        // Edge: batch(5, 5) — zero-length batch.
        let path = temp_path("miri_zero_batch");
        let ds = sample_dataset(10, 2);
        save_scry(&ds, &path).unwrap();
        let mmap = MmapDataset::open(&path).unwrap();
        let batch = mmap.batch(5, 5);
        assert_eq!(batch.n_samples(), 0);
        std::fs::remove_file(&path).ok();
    }

    // Test 12: Header rejection — malformed files must return Err, not panic/UB.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_malformed_headers() {
        // Too-short file.
        let path = temp_path("miri_short");
        std::fs::write(&path, &[0u8; 10]).unwrap();
        assert!(MmapDataset::open(&path).is_err());
        std::fs::remove_file(&path).ok();

        // Bad magic.
        let path = temp_path("miri_bad_magic");
        let mut buf = vec![0u8; 128];
        buf[0..4].copy_from_slice(b"NOPE");
        std::fs::write(&path, &buf).unwrap();
        assert!(MmapDataset::open(&path).is_err());
        std::fs::remove_file(&path).ok();

        // Wrong version.
        let path = temp_path("miri_bad_version");
        let mut buf = vec![0u8; 128];
        buf[0..4].copy_from_slice(b"SCRY");
        buf[4..8].copy_from_slice(&99u32.to_le_bytes()); // version 99
        std::fs::write(&path, &buf).unwrap();
        assert!(MmapDataset::open(&path).is_err());
        std::fs::remove_file(&path).ok();

        // Truncated data section (valid header, but data section is too small).
        let path = temp_path("miri_truncated");
        let ds = sample_dataset(100, 5);
        save_scry(&ds, &path).unwrap();
        // Truncate the file to just the header.
        let metadata = std::fs::metadata(&path).unwrap();
        let truncated_len = (metadata.len() / 2) as usize;
        let full = std::fs::read(&path).unwrap();
        std::fs::write(&path, &full[..truncated_len]).unwrap();
        assert!(MmapDataset::open(&path).is_err());
        std::fs::remove_file(&path).ok();
    }
}
