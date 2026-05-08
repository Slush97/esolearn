# Changelog

## 0.2.0 — 2026-05-07

This release addresses an external review (codex, 2026-05-07) and closes a
soundness hole in the tree prediction path. Most of 0.1.0's API surface is
preserved; the breaking change is encapsulation of `Dataset` data fields.

### Fixed

- **Soundness — predict-time row-width validation.** `FlatTree`'s hot
  prediction path uses `unsafe { sample.get_unchecked(feature_idx) }`, where
  `feature_idx` is bounded against the *training* `n_features` but not against
  the *predict-time* row length. A short row was therefore reachable UB from
  safe code. Public predict APIs (`DecisionTree*::predict` /
  `predict_proba`, `RandomForest*::predict` / `predict_proba`) now validate
  every row via `ensure_row_widths`, returning `ScryLearnError::ShapeMismatch`
  on the first mismatch.
- **RandomForest training-error tracking.** Fit-time errors on individual
  trees were silently swallowed via `.ok()`, but `predict_proba` and
  `feature_importances_` divided by the planned `n_estimators`, biasing
  averages toward zero. Failed trees are now dropped after parallel build,
  and `n_trees()` / `n_failed_trees()` getters are exposed on both classifier
  and regressor. If every tree fails, `fit()` returns
  `InvalidParameter` instead of producing a silently-empty model.
- **CSV string-feature label encoding.** `Dataset::from_csv` documented
  string features as label-encoded but actually parsed every cell as `f64`,
  silently turning unparseable strings into `NaN`. Feature columns are now
  detected and label-encoded the same way the target column is, with the
  mapping recorded in the new `Dataset::feature_label_maps()` accessor.
- **Metric and split shape guards.** `accuracy`, `log_loss`,
  `confusion_matrix`, `precision`, `recall`, `f1_score`, `balanced_accuracy`,
  `cohen_kappa_score`, `classification_report`, all regression metrics, ROC /
  PR curves, ARI / Calinski-Harabasz / Davies-Bouldin, and `k_fold` /
  `stratified_k_fold` / `train_test_split` now panic with a clear message on
  mismatched input lengths or `k == 0` rather than producing silently-wrong
  scores or unrelated index-out-of-bounds panics.

### Changed (Breaking)

- **Dataset fields are now `pub(crate)`.** External code accesses data via
  new accessors: `features()`, `target()`, `feature_names()`, `target_name()`,
  `class_labels()`, `feature_label_maps()`. Direct mutation is no longer
  possible from outside the crate; rebuild the dataset via `Dataset::new`
  with adjusted columns. `Dataset::new` itself now also panics if
  `target.len()` disagrees with the feature row count.
- Added `Dataset::validate()` for code that ingests an already-built
  `Dataset` from an untrusted source (deserialization, FFI).

### Documentation

- README quickstart now mentions the `csv` feature and uses the new accessor
  signatures (`test.feature_matrix()`, `test.target()`).
- README `#![deny(unsafe_code)]` line clarifies that the crate-root deny is
  locally re-enabled in the `FlatTree` DFS loop, with the public predict APIs
  enforcing the contract upstream.
