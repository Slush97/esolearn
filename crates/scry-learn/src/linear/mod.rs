// SPDX-License-Identifier: MIT OR Apache-2.0
//! Linear models: OLS, Ridge, Logistic, Lasso, and ElasticNet.
//!
//! # Regularization naming convention
//!
//! scry-learn uses **`alpha`** as the regularization strength parameter across
//! all linear models — matching scikit-learn's `Ridge`, `Lasso`, and `ElasticNet`:
//!
//! | Model | Parameter | Meaning |
//! |-------|-----------|---------|
//! | [`LinearRegression`] | `alpha` | L2 penalty strength (0 = OLS) |
//! | [`Ridge`] | `alpha` | L2 penalty strength (constructor arg) |
//! | [`LassoRegression`] | `alpha` | L1 penalty strength |
//! | [`ElasticNet`] | `alpha` | Total penalty strength |
//! | [`LogisticRegression`] | `alpha` | Penalty strength (type set by [`Penalty`]) |
//!
//! ## sklearn migration note
//!
//! scikit-learn's `LogisticRegression` and `SVC` use **`C = 1/alpha`** (inverse
//! regularization strength). When porting sklearn code, convert via `alpha = 1.0 / C`.
//! All other sklearn linear models (`Ridge`, `Lasso`, `ElasticNet`) already use
//! `alpha`, so those translate directly.

mod elastic_net;
mod lasso;
mod lbfgs;
mod logistic;
pub(crate) mod qr;
mod regression;
pub(crate) mod svd;

pub use elastic_net::ElasticNet;
pub use lasso::LassoRegression;
pub use logistic::{LogisticRegression, Penalty, Solver};
pub use regression::LinearRegression;

use crate::dataset::Dataset;
use crate::error::Result;

/// Ridge regression — [`LinearRegression`] with L2 regularization.
///
/// This is a thin wrapper around [`LinearRegression`] that provides a more
/// discoverable API for users coming from scikit-learn's `Ridge` class.
///
/// # Example
///
/// ```
/// use scry_learn::linear::Ridge;
/// # use scry_learn::dataset::Dataset;
///
/// let data = Dataset::new(
///     vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]],
///     vec![2.0, 4.0, 6.0, 8.0, 10.0],
///     vec!["x".into()],
///     "y",
/// );
///
/// let mut model = Ridge::new(1.0);
/// model.fit(&data).unwrap();
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Ridge {
    inner: LinearRegression,
}

impl Ridge {
    /// Create a new Ridge regression model with the given L2 regularization strength.
    ///
    /// Equivalent to `LinearRegression::new().alpha(alpha)`.
    pub fn new(alpha: f64) -> Self {
        Self {
            inner: LinearRegression::new().alpha(alpha),
        }
    }

    /// Train the model on the given dataset.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        self.inner.fit(data)
    }

    /// Predict target values for the given feature matrix.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        self.inner.predict(features)
    }

    /// Get the learned coefficients.
    pub fn coefficients(&self) -> &[f64] {
        self.inner.coefficients()
    }

    /// Get the learned intercept.
    pub fn intercept(&self) -> f64 {
        self.inner.intercept()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ridge_alias() {
        let features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        // Ridge(1.0) should produce the same result as LinearRegression::new().alpha(1.0).
        let mut ridge = Ridge::new(1.0);
        ridge.fit(&data).unwrap();

        let mut lr = LinearRegression::new().alpha(1.0);
        lr.fit(&data).unwrap();

        assert!(
            (ridge.coefficients()[0] - lr.coefficients()[0]).abs() < 1e-10,
            "Ridge and LinearRegression(alpha=1.0) should produce identical coefficients"
        );
        assert!(
            (ridge.intercept() - lr.intercept()).abs() < 1e-10,
            "Ridge and LinearRegression(alpha=1.0) should produce identical intercepts"
        );

        // Sanity: coefficient should be shrunk below 2.0 (the OLS solution).
        assert!(ridge.coefficients()[0] < 2.0);
        assert!(ridge.coefficients()[0] > 1.0);
    }
}
