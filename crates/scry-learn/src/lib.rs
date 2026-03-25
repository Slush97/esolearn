// SPDX-License-Identifier: MIT OR Apache-2.0
//! # scry-learn
//!
//! Machine learning toolkit in pure Rust.
//!
//! ## Quick Start
//!
//! ```ignore
//! use scry_learn::prelude::*;
//!
//! let data = Dataset::from_csv("iris.csv", "species")?;
//! let (train, test) = train_test_split(&data, 0.2, 42);
//!
//! let mut model = RandomForestClassifier::new()
//!     .n_estimators(100)
//!     .max_depth(10);
//! model.fit(&train)?;
//!
//! let preds = model.predict(&test)?;
//! let report = classification_report(&test.target, &preds);
//! println!("{report}");
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_fields_in_debug)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::redundant_pub_crate)]
#![allow(clippy::use_self)]
#![allow(clippy::suspicious_operation_groupings)]
#![allow(clippy::used_underscore_binding)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::type_complexity)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::explicit_counter_loop)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::redundant_clone)]
#![allow(clippy::significant_drop_tightening)]
#![allow(clippy::or_fun_call)]
#![allow(clippy::redundant_closure_for_method_calls)]

pub(crate) mod accel;
pub mod anomaly;
pub mod calibration;
pub(crate) mod constants;
pub mod cluster;
pub mod dataset;
pub mod distance;
pub mod ensemble;
pub mod error;
pub mod explain;
pub mod feature_selection;
pub mod linear;
pub(crate) mod matrix;
pub mod metrics;
pub mod naive_bayes;
pub mod neighbors;
pub mod neural;
pub mod partial_fit;
pub mod pipeline;
pub mod preprocess;
pub(crate) mod rng;
pub mod search;
pub mod sparse;
pub mod split;
pub mod svm;
pub mod text;
pub mod tree;
pub(crate) mod version;
pub mod weights;

#[cfg(feature = "experimental")]
pub mod onnx;
#[cfg(feature = "mmap")]
pub mod mmap;
#[cfg(feature = "polars")]
pub mod polars_interop;

/// Convenience re-exports for common usage.
pub mod prelude {
    pub use crate::anomaly::IsolationForest;
    pub use crate::calibration::{
        CalibratedClassifierCV, CalibrationMethod, IsotonicRegression, PlattScaling,
    };
    pub use crate::cluster::{
        silhouette_score, AgglomerativeClustering, Dbscan, Hdbscan, KMeans, Linkage,
        MiniBatchKMeans,
    };
    pub use crate::dataset::{ColumnStats, Dataset};
    pub use crate::ensemble::{StackingClassifier, Voting, VotingClassifier};
    pub use crate::error::ScryLearnError;
    pub use crate::explain::{ensemble_tree_shap, permutation_importance, tree_shap};
    pub use crate::feature_selection::{f_classif, ScoreFn, SelectKBest, VarianceThreshold};
    pub use crate::linear::{
        ElasticNet, LassoRegression, LinearRegression, LogisticRegression, Penalty, Ridge, Solver,
    };
    pub use crate::matrix::DenseMatrix;
    pub use crate::metrics::{
        accuracy, adjusted_rand_index, balanced_accuracy, calinski_harabasz_score,
        classification_report, cohen_kappa_score, confusion_matrix, davies_bouldin_score,
        explained_variance_score, f1_score, log_loss, mean_absolute_percentage_error,
        mean_squared_error, pr_curve, precision, r2_score, recall, roc_auc_score, roc_curve,
        ClassMetrics, ClassificationReport, ConfusionMatrix, PrCurve, RocCurve,
    };
    #[cfg(feature = "mmap")]
    pub use crate::mmap::{save_scry, MmapDataset};
    pub use crate::naive_bayes::{BernoulliNB, GaussianNb, MultinomialNB};
    pub use crate::neighbors::{
        Algorithm, DistanceMetric, KdTree, KnnClassifier, KnnRegressor, WeightFunction,
    };
    #[cfg(feature = "experimental")]
    pub use crate::neural::{Conv2D, Flatten, MaxPool2D};
    pub use crate::neural::{
        Activation, BackwardOutput, CallbackAction, Layer, MLPClassifier, MLPRegressor,
        OptimizerKind, TrainingCallback,
    };
    pub use crate::partial_fit::PartialFit;
    pub use crate::pipeline::Pipeline;
    pub use crate::preprocess::{
        ColumnTransformer, DropStrategy, LabelEncoder, MinMaxScaler, Norm, Normalizer,
        OneHotEncoder, Pca, PolynomialFeatures, RobustScaler, SimpleImputer, StandardScaler,
        Strategy, Transformer, UnknownStrategy,
    };
    pub use crate::search::{
        BayesSearchCV, CvResult, GridSearchCV, ParamDistribution, ParamGrid, ParamSpace,
        ParamValue, RandomizedSearchCV, Tunable,
    };
    pub use crate::sparse::{CscMatrix, CsrMatrix};
    pub use crate::split::{
        cross_val_predict, cross_val_score, cross_val_score_stratified, group_k_fold,
        repeated_cross_val_score, stratified_split, time_series_split, train_test_split,
        RepeatedKFold, ScoringFn,
    };
    #[cfg(feature = "experimental")]
    pub use crate::svm::{Gamma, Kernel, KernelSVC, KernelSVR};
    pub use crate::svm::{LinearSVC, LinearSVR};
    pub use crate::text::sparse_to_dataset;
    pub use crate::tree::{
        DecisionTreeClassifier, DecisionTreeRegressor, GradientBoostingClassifier,
        GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
        RandomForestClassifier, RandomForestRegressor, RegressionLoss, SplitCriterion,
    };
    pub use crate::weights::ClassWeight;
}
