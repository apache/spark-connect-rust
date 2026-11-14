// SPDX-License-Identifier: Apache-2.0
//! The bisecting k-means model, as PySpark's `BisectingKMeansModel`. Its prediction is a cluster
//! index, so it is not a [`crate::ml::PredictionModel`].

use super::{matrix_rows, transformed_summary_dataset, BisectingKMeansSummary};
use crate::dataframe::DataFrame;
use crate::ml::object::{FetchArg, MlValue};
use crate::ml::regression_api::{as_matrix, summary_or_error, training_summary};
use crate::ml::tree_api::{as_f64, as_i32, fetch_value};
use crate::ml::{
    fit_via_command, read_model, read_operator, BisectingKMeans, BisectingKMeansModel, Estimator,
    MlVector, Model, OperatorType,
};
use crate::session::SparkSession;
use spark_connect_core::error::Result;

const BISECTING_KMEANS: &str = "org.apache.spark.ml.clustering.BisectingKMeans";
const BISECTING_KMEANS_MODEL: &str = "org.apache.spark.ml.clustering.BisectingKMeansModel";

impl BisectingKMeans {
    /// Fit the model of `df`, with its training summary, as PySpark's `fit` returns a
    /// `BisectingKMeansModel`.
    pub fn fit(&mut self, df: &DataFrame) -> Result<BisectingKMeansModel> {
        self.ensure_operator();
        let info = fit_via_command(&self.operator, &self.params, df)?;
        let mut model = self.model_from_info(info, &df.session)?;
        let summary_dataset = transformed_summary_dataset(&model.cached, df)?;
        model.training_summary = Some(training_summary(&model.cached, &summary_dataset));
        Ok(model)
    }

    /// The estimator saved at `path`, as `BisectingKMeans.load` of PySpark.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (operator, params) =
            read_operator(session, BISECTING_KMEANS, OperatorType::Estimator, path)?;
        Ok(Self { operator, params })
    }
}

impl BisectingKMeansModel {
    /// The centers of the clusters, each the array of its coordinates (`clusterCenters`), as
    /// PySpark reads the rows of the matrix of the centers (`clusterCenterMatrix`).
    pub fn cluster_centers(&self) -> Result<Vec<Vec<f64>>> {
        let value = fetch_value(&self.model_target()?, "clusterCenterMatrix", vec![])?;
        Ok(matrix_rows(&as_matrix("clusterCenterMatrix", value)?))
    }

    /// The sum of the squared distances of the points of `dataset` to their nearest center
    /// (`computeCost`).
    pub fn compute_cost(&self, dataset: &DataFrame) -> Result<f64> {
        let args = vec![FetchArg::Input(dataset.clone())];
        as_f64(
            "computeCost",
            fetch_value(&self.model_target()?, "computeCost", args)?,
        )
    }

    /// The number of features that the model was trained on (`numFeatures`).
    pub fn num_features(&self) -> Result<i32> {
        let value = fetch_value(&self.model_target()?, "numFeatures", vec![])?;
        as_i32("numFeatures", value)
    }

    /// The index of the cluster of `features` (`predict`).
    pub fn predict(&self, features: &MlVector) -> Result<i32> {
        let args = vec![FetchArg::Value(MlValue::Vector(features.clone()))];
        as_i32(
            "predict",
            fetch_value(&self.model_target()?, "predict", args)?,
        )
    }

    /// Whether the model has a training summary (`hasSummary`): a model that this client fitted
    /// has one, and a loaded model has none, as PySpark tells of a remote model.
    pub fn has_summary(&self) -> bool {
        self.training_summary.is_some()
    }

    /// The summary of the model of its training dataset (`summary`), or an error of a model that
    /// has none.
    pub fn summary(&self) -> Result<BisectingKMeansSummary> {
        let target = summary_or_error(&self.training_summary, "BisectingKMeansModel")?;
        Ok(BisectingKMeansSummary { target })
    }

    /// The model saved at `path`, as `BisectingKMeansModel.load` of PySpark. A loaded model has
    /// no training summary.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (cached, operator, params) = read_model(session, BISECTING_KMEANS_MODEL, path)?;
        Ok(Self {
            operator,
            params,
            cached,
            training_summary: None,
        })
    }
}
