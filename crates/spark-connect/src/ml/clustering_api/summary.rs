// SPDX-License-Identifier: Apache-2.0
//! The summaries of the clustering models, as PySpark's `ClusteringSummary`, `KMeansSummary`,
//! `BisectingKMeansSummary` and `GaussianMixtureSummary`: each the training summary of a model,
//! `"<model>.summary"`, which carries the training dataset for the server to recreate a lost
//! summary.

use super::as_i64s;
use crate::dataframe::DataFrame;
use crate::ml::object::{attribute_relation, MlTarget};
use crate::ml::tree_api::{as_f64, as_i32, as_string, fetch_value};
use crate::ml::MlSummary;
use spark_connect_core::error::Result;

/// The members of a clustering summary (`ClusteringSummary`): each a fetch of the summary's target,
/// and an attribute relation of a member that PySpark returns as a DataFrame.
pub trait ClusteringSummary: MlSummary {
    /// The name of the prediction column (`predictionCol`).
    fn prediction_col(&self) -> Result<String> {
        let value = fetch_value(self.summary_target(), "predictionCol", vec![])?;
        as_string("predictionCol", value)
    }

    /// The DataFrame of the predictions of the model (`predictions`).
    fn predictions(&self) -> Result<DataFrame> {
        attribute_relation(self.summary_target(), "predictions", vec![])
    }

    /// The name of the features column (`featuresCol`).
    fn features_col(&self) -> Result<String> {
        let value = fetch_value(self.summary_target(), "featuresCol", vec![])?;
        as_string("featuresCol", value)
    }

    /// The number of clusters of the model (`k`).
    fn k(&self) -> Result<i32> {
        as_i32("k", fetch_value(self.summary_target(), "k", vec![])?)
    }

    /// The DataFrame of the cluster of each training instance (`cluster`).
    fn cluster(&self) -> Result<DataFrame> {
        attribute_relation(self.summary_target(), "cluster", vec![])
    }

    /// The size of each cluster (`clusterSizes`).
    fn cluster_sizes(&self) -> Result<Vec<i64>> {
        let value = fetch_value(self.summary_target(), "clusterSizes", vec![])?;
        as_i64s("clusterSizes", value)
    }

    /// The number of iterations of the fit (`numIter`).
    fn num_iter(&self) -> Result<i32> {
        as_i32(
            "numIter",
            fetch_value(self.summary_target(), "numIter", vec![])?,
        )
    }
}

/// The summary of a k-means model of its training dataset (`KMeansSummary`), of
/// [`crate::ml::KMeansModel::summary`]: the members of [`ClusteringSummary`] and its cost.
#[derive(Debug, Clone)]
pub struct KMeansSummary {
    pub(crate) target: MlTarget,
}

impl MlSummary for KMeansSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl ClusteringSummary for KMeansSummary {}

impl KMeansSummary {
    /// The k-means cost, the sum of the squared distances of the points to their nearest center
    /// (`trainingCost`).
    pub fn training_cost(&self) -> Result<f64> {
        let value = fetch_value(&self.target, "trainingCost", vec![])?;
        as_f64("trainingCost", value)
    }
}

/// The summary of a bisecting k-means model of its training dataset (`BisectingKMeansSummary`), of
/// [`crate::ml::BisectingKMeansModel::summary`]: the members of [`ClusteringSummary`] and its
/// cost.
#[derive(Debug, Clone)]
pub struct BisectingKMeansSummary {
    pub(crate) target: MlTarget,
}

impl MlSummary for BisectingKMeansSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl ClusteringSummary for BisectingKMeansSummary {}

impl BisectingKMeansSummary {
    /// The sum of the squared distances of the points to their nearest center (`trainingCost`).
    pub fn training_cost(&self) -> Result<f64> {
        let value = fetch_value(&self.target, "trainingCost", vec![])?;
        as_f64("trainingCost", value)
    }
}

/// The summary of a Gaussian mixture model of its training dataset (`GaussianMixtureSummary`), of
/// [`crate::ml::GaussianMixtureModel::summary`]: the members of [`ClusteringSummary`] and those of
/// the probabilities.
#[derive(Debug, Clone)]
pub struct GaussianMixtureSummary {
    pub(crate) target: MlTarget,
}

impl MlSummary for GaussianMixtureSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl ClusteringSummary for GaussianMixtureSummary {}

impl GaussianMixtureSummary {
    /// The name of the column of the probability of each cluster (`probabilityCol`).
    pub fn probability_col(&self) -> Result<String> {
        let value = fetch_value(&self.target, "probabilityCol", vec![])?;
        as_string("probabilityCol", value)
    }

    /// The DataFrame of the probability of each cluster of each training instance (`probability`).
    pub fn probability(&self) -> Result<DataFrame> {
        attribute_relation(&self.target, "probability", vec![])
    }

    /// The total log-likelihood of the model of its training dataset (`logLikelihood`).
    pub fn log_likelihood(&self) -> Result<f64> {
        let value = fetch_value(&self.target, "logLikelihood", vec![])?;
        as_f64("logLikelihood", value)
    }
}
