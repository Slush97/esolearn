// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Tunable` trait and implementations for all model types.
//!
//! Models that implement `Tunable` can participate in [`GridSearchCV`] and
//! [`RandomizedSearchCV`] hyperparameter search.

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};

use super::ParamValue;

/// A model whose hyperparameters can be set dynamically by name.
///
/// Implement this trait on any model that should participate in
/// [`GridSearchCV`](super::GridSearchCV) or [`RandomizedSearchCV`](super::RandomizedSearchCV).
///
/// # Examples
///
/// ```ignore
/// use scry_learn::search::{Tunable, ParamValue};
///
/// let mut dt = DecisionTreeClassifier::new();
/// dt.set_param("max_depth", ParamValue::Int(5)).unwrap();
/// ```
pub trait Tunable {
    /// Apply a named hyperparameter.
    ///
    /// Returns [`ScryLearnError::InvalidParameter`] if the parameter name
    /// is unrecognised or the value type is wrong.
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()>;

    /// Clone this model into a boxed trait object.
    fn clone_box(&self) -> Box<dyn Tunable>;

    /// Train on a dataset.
    fn fit(&mut self, data: &Dataset) -> Result<()>;

    /// Predict on row-major features.
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>>;
}

// ---------------------------------------------------------------------------
// impl_tunable! macro — generates the boilerplate for the common case:
//   - clone + builder for each parameter
//   - clone_box via self.clone()
//   - fit delegates to self.fit(data)
//   - predict delegates to self.predict(features)
//
// For models that need custom fit/predict (KMeans, IsolationForest),
// keep a manual impl below the macro invocation.
// ---------------------------------------------------------------------------

macro_rules! impl_tunable {
    (
        $(
            $(#[$meta:meta])*
            $Model:ty {
                $( $param:ident : $kind:ident ),* $(,)?
            }
        );* $(;)?
    ) => {
        $(
            $(#[$meta])*
            impl Tunable for $Model {
                fn set_param(&mut self, name: &str, _value: ParamValue) -> Result<()> {
                    match name {
                        $(
                            stringify!($param) => {
                                impl_tunable!(@extract _value, $kind, $param, self)
                            }
                        )*
                        _ => Err(ScryLearnError::InvalidParameter(format!(
                            "unknown parameter: {name}"
                        ))),
                    }
                }

                fn clone_box(&self) -> Box<dyn Tunable> {
                    Box::new(self.clone())
                }

                fn fit(&mut self, data: &Dataset) -> Result<()> {
                    self.fit(data)
                }

                fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
                    self.predict(features)
                }
            }
        )*
    };

    // Internal: extract Int parameter.
    (@extract $value:ident, Int, $param:ident, $self:ident) => {
        if let ParamValue::Int(v) = $value {
            *$self = $self.clone().$param(v);
            Ok(())
        } else {
            Err(ScryLearnError::InvalidParameter(format!(
                concat!(stringify!($param), " expects Int, got {}"), $value
            )))
        }
    };

    // Internal: extract Float parameter.
    (@extract $value:ident, Float, $param:ident, $self:ident) => {
        if let ParamValue::Float(v) = $value {
            *$self = $self.clone().$param(v);
            Ok(())
        } else {
            Err(ScryLearnError::InvalidParameter(format!(
                concat!(stringify!($param), " expects Float, got {}"), $value
            )))
        }
    };
}

// ---------------------------------------------------------------------------
// Standard impls via macro
// ---------------------------------------------------------------------------

impl_tunable! {
    crate::tree::DecisionTreeClassifier {
        max_depth: Int,
        min_samples_split: Int,
        min_samples_leaf: Int,
    };
    crate::tree::DecisionTreeRegressor {
        max_depth: Int,
        min_samples_split: Int,
        min_samples_leaf: Int,
    };
    crate::tree::RandomForestClassifier {
        n_estimators: Int,
        max_depth: Int,
    };
    crate::linear::LogisticRegression {
        learning_rate: Float,
        max_iter: Int,
        alpha: Float,
        tolerance: Float,
    };
    crate::neighbors::KnnClassifier {
        k: Int,
    };
    crate::neighbors::KnnRegressor {
        k: Int,
    };
    crate::tree::GradientBoostingRegressor {
        n_estimators: Int,
        learning_rate: Float,
        max_depth: Int,
        min_samples_split: Int,
        min_samples_leaf: Int,
    };
    crate::tree::GradientBoostingClassifier {
        n_estimators: Int,
        learning_rate: Float,
        max_depth: Int,
        min_samples_split: Int,
        min_samples_leaf: Int,
    };
    crate::svm::LinearSVC {
        c: Float,
        max_iter: Int,
        tol: Float,
    };
    crate::svm::LinearSVR {
        c: Float,
        epsilon: Float,
        max_iter: Int,
        tol: Float,
    };
    #[cfg(feature = "experimental")]
    crate::svm::KernelSVC {
        c: Float,
        tol: Float,
        max_iter: Int,
    };
    #[cfg(feature = "experimental")]
    crate::svm::KernelSVR {
        c: Float,
        epsilon: Float,
        tol: Float,
        max_iter: Int,
    };
    crate::naive_bayes::GaussianNb {};
    crate::naive_bayes::BernoulliNB {
        alpha: Float,
    };
    crate::naive_bayes::MultinomialNB {
        alpha: Float,
    };
    crate::linear::LassoRegression {
        alpha: Float,
        max_iter: Int,
        tol: Float,
    };
    crate::linear::ElasticNet {
        alpha: Float,
        l1_ratio: Float,
        max_iter: Int,
        tol: Float,
    };
    crate::tree::HistGradientBoostingRegressor {
        n_estimators: Int,
        learning_rate: Float,
        max_leaf_nodes: Int,
        max_depth: Int,
        min_samples_leaf: Int,
    };
    crate::tree::HistGradientBoostingClassifier {
        n_estimators: Int,
        learning_rate: Float,
        max_leaf_nodes: Int,
        max_depth: Int,
        min_samples_leaf: Int,
    };
    crate::neural::MLPClassifier {
        learning_rate: Float,
        alpha: Float,
        max_iter: Int,
        batch_size: Int,
    };
    crate::neural::MLPRegressor {
        learning_rate: Float,
        alpha: Float,
        max_iter: Int,
        batch_size: Int,
    };
}

// ---------------------------------------------------------------------------
// Manual impls for models with custom fit/predict
// ---------------------------------------------------------------------------

impl Tunable for crate::cluster::KMeans {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "max_iter" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().max_iter(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "max_iter expects Int, got {value}"
                    )))
                }
            }
            "tolerance" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().tolerance(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "tolerance expects Float, got {value}"
                    )))
                }
            }
            "n_init" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().n_init(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "n_init expects Int, got {value}"
                    )))
                }
            }
            _ => Err(ScryLearnError::InvalidParameter(format!(
                "unknown parameter: {name}"
            ))),
        }
    }
    fn clone_box(&self) -> Box<dyn Tunable> {
        Box::new(self.clone())
    }
    fn fit(&mut self, data: &Dataset) -> Result<()> {
        self.fit(data)
    }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        let labels = crate::cluster::KMeans::predict(self, features)?;
        Ok(labels.into_iter().map(|l| l as f64).collect())
    }
}

impl Tunable for crate::anomaly::IsolationForest {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "n_estimators" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().n_estimators(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "n_estimators expects Int, got {value}"
                    )))
                }
            }
            "max_samples" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().max_samples(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "max_samples expects Int, got {value}"
                    )))
                }
            }
            "contamination" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().contamination(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "contamination expects Float, got {value}"
                    )))
                }
            }
            _ => Err(ScryLearnError::InvalidParameter(format!(
                "unknown parameter: {name}"
            ))),
        }
    }
    fn clone_box(&self) -> Box<dyn Tunable> {
        Box::new(self.clone())
    }
    fn fit(&mut self, data: &Dataset) -> Result<()> {
        let features = data.feature_matrix();
        self.fit(&features)
    }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        Ok(self.predict(features))
    }
}
