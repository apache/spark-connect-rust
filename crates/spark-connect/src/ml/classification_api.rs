// SPDX-License-Identifier: Apache-2.0
//! The typed accessors, summaries and persistence of the classification models, over the server-side
//! objects of [`super::object`].
//!
//! [`ClassificationModel`] and [`ProbabilisticClassificationModel`] hold the members that every
//! classification model has, and [`ClassificationSummary`], [`BinaryClassificationSummary`] and
//! [`TrainingSummary`] the members of the summaries, as the base classes of PySpark's
//! `pyspark.ml.classification` do. Each family has its own file of its model's members and its
//! summary types.

mod model;
pub use model::*;
mod summary;
pub use summary::*;

mod fm;
#[allow(unused_imports)]
pub use fm::*;
mod linear_svc;
#[allow(unused_imports)]
pub use linear_svc::*;
mod logistic;
#[allow(unused_imports)]
pub use logistic::*;
mod mlp;
#[allow(unused_imports)]
pub use mlp::*;
mod naive_bayes;
#[allow(unused_imports)]
pub use naive_bayes::*;
mod one_vs_rest;
#[allow(unused_imports)]
pub use one_vs_rest::*;
mod tree;
#[allow(unused_imports)]
pub use tree::*;
