// SPDX-License-Identifier: MIT OR Apache-2.0
//! Naive Bayes classifiers: Gaussian, Bernoulli, and Multinomial.

mod bernoulli;
mod gaussian;
mod multinomial;

pub use bernoulli::BernoulliNB;
pub use gaussian::GaussianNb;
pub use multinomial::MultinomialNB;
