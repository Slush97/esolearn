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
// Tunable impls for existing models
// ---------------------------------------------------------------------------

impl Tunable for crate::tree::DecisionTreeClassifier {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "max_depth" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().max_depth(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "max_depth expects Int, got {value}"
                    )))
                }
            }
            "min_samples_split" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().min_samples_split(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "min_samples_split expects Int, got {value}"
                    )))
                }
            }
            "min_samples_leaf" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().min_samples_leaf(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "min_samples_leaf expects Int, got {value}"
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
        self.predict(features)
    }
}

impl Tunable for crate::tree::RandomForestClassifier {
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
            "max_depth" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().max_depth(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "max_depth expects Int, got {value}"
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
        self.predict(features)
    }
}

impl Tunable for crate::linear::LogisticRegression {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "learning_rate" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().learning_rate(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "learning_rate expects Float, got {value}"
                    )))
                }
            }
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
            "alpha" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().alpha(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "alpha expects Float, got {value}"
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

impl Tunable for crate::neighbors::KnnClassifier {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "k" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().k(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "k expects Int, got {value}"
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
        self.predict(features)
    }
}

impl Tunable for crate::neighbors::KnnRegressor {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "k" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().k(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "k expects Int, got {value}"
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
        self.predict(features)
    }
}

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

impl Tunable for crate::tree::GradientBoostingRegressor {
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
            "learning_rate" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().learning_rate(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "learning_rate expects Float, got {value}"
                    )))
                }
            }
            "max_depth" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().max_depth(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "max_depth expects Int, got {value}"
                    )))
                }
            }
            "min_samples_split" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().min_samples_split(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "min_samples_split expects Int, got {value}"
                    )))
                }
            }
            "min_samples_leaf" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().min_samples_leaf(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "min_samples_leaf expects Int, got {value}"
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
        self.predict(features)
    }
}

impl Tunable for crate::tree::GradientBoostingClassifier {
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
            "learning_rate" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().learning_rate(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "learning_rate expects Float, got {value}"
                    )))
                }
            }
            "max_depth" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().max_depth(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "max_depth expects Int, got {value}"
                    )))
                }
            }
            "min_samples_split" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().min_samples_split(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "min_samples_split expects Int, got {value}"
                    )))
                }
            }
            "min_samples_leaf" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().min_samples_leaf(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "min_samples_leaf expects Int, got {value}"
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
        self.predict(features)
    }
}

impl Tunable for crate::svm::LinearSVC {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "c" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().c(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "c expects Float, got {value}"
                    )))
                }
            }
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
            "tol" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().tol(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "tol expects Float, got {value}"
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
        self.predict(features)
    }
}

impl Tunable for crate::svm::LinearSVR {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "c" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().c(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "c expects Float, got {value}"
                    )))
                }
            }
            "epsilon" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().epsilon(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "epsilon expects Float, got {value}"
                    )))
                }
            }
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
            "tol" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().tol(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "tol expects Float, got {value}"
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
        self.predict(features)
    }
}

#[cfg(feature = "experimental")]
impl Tunable for crate::svm::KernelSVC {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "c" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().c(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "c expects Float, got {value}"
                    )))
                }
            }
            "tol" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().tol(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "tol expects Float, got {value}"
                    )))
                }
            }
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

#[cfg(feature = "experimental")]
impl Tunable for crate::svm::KernelSVR {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "c" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().c(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "c expects Float, got {value}"
                    )))
                }
            }
            "epsilon" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().epsilon(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "epsilon expects Float, got {value}"
                    )))
                }
            }
            "tol" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().tol(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "tol expects Float, got {value}"
                    )))
                }
            }
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

impl Tunable for crate::naive_bayes::GaussianNb {
    fn set_param(&mut self, name: &str, _value: ParamValue) -> Result<()> {
        Err(ScryLearnError::InvalidParameter(format!(
            "unknown parameter: {name}"
        )))
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

impl Tunable for crate::naive_bayes::BernoulliNB {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "alpha" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().alpha(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "alpha expects Float, got {value}"
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
        self.predict(features)
    }
}

impl Tunable for crate::naive_bayes::MultinomialNB {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "alpha" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().alpha(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "alpha expects Float, got {value}"
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
        self.predict(features)
    }
}

impl Tunable for crate::linear::LassoRegression {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "alpha" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().alpha(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "alpha expects Float, got {value}"
                    )))
                }
            }
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
            "tol" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().tol(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "tol expects Float, got {value}"
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
        self.predict(features)
    }
}

impl Tunable for crate::linear::ElasticNet {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "alpha" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().alpha(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "alpha expects Float, got {value}"
                    )))
                }
            }
            "l1_ratio" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().l1_ratio(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "l1_ratio expects Float, got {value}"
                    )))
                }
            }
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
            "tol" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().tol(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "tol expects Float, got {value}"
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
        self.predict(features)
    }
}

impl Tunable for crate::tree::HistGradientBoostingRegressor {
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
            "learning_rate" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().learning_rate(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "learning_rate expects Float, got {value}"
                    )))
                }
            }
            "max_leaf_nodes" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().max_leaf_nodes(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "max_leaf_nodes expects Int, got {value}"
                    )))
                }
            }
            "max_depth" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().max_depth(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "max_depth expects Int, got {value}"
                    )))
                }
            }
            "min_samples_leaf" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().min_samples_leaf(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "min_samples_leaf expects Int, got {value}"
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
        self.predict(features)
    }
}

impl Tunable for crate::tree::HistGradientBoostingClassifier {
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
            "learning_rate" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().learning_rate(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "learning_rate expects Float, got {value}"
                    )))
                }
            }
            "max_leaf_nodes" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().max_leaf_nodes(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "max_leaf_nodes expects Int, got {value}"
                    )))
                }
            }
            "max_depth" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().max_depth(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "max_depth expects Int, got {value}"
                    )))
                }
            }
            "min_samples_leaf" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().min_samples_leaf(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "min_samples_leaf expects Int, got {value}"
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
        self.predict(features)
    }
}

impl Tunable for crate::tree::DecisionTreeRegressor {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "max_depth" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().max_depth(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "max_depth expects Int, got {value}"
                    )))
                }
            }
            "min_samples_split" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().min_samples_split(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "min_samples_split expects Int, got {value}"
                    )))
                }
            }
            "min_samples_leaf" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().min_samples_leaf(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "min_samples_leaf expects Int, got {value}"
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
        self.predict(features)
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

impl Tunable for crate::neural::MLPClassifier {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "learning_rate" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().learning_rate(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "learning_rate expects Float, got {value}"
                    )))
                }
            }
            "alpha" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().alpha(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "alpha expects Float, got {value}"
                    )))
                }
            }
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
            "batch_size" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().batch_size(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "batch_size expects Int, got {value}"
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
        self.predict(features)
    }
}

impl Tunable for crate::neural::MLPRegressor {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "learning_rate" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().learning_rate(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "learning_rate expects Float, got {value}"
                    )))
                }
            }
            "alpha" => {
                if let ParamValue::Float(v) = value {
                    *self = self.clone().alpha(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "alpha expects Float, got {value}"
                    )))
                }
            }
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
            "batch_size" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().batch_size(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(format!(
                        "batch_size expects Int, got {value}"
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
        self.predict(features)
    }
}
