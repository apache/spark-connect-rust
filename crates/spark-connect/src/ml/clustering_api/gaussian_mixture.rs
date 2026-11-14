// SPDX-License-Identifier: Apache-2.0
//! The Gaussian mixture model, as PySpark's `GaussianMixtureModel`. Its prediction is a cluster
//! index, so it is not a [`crate::ml::PredictionModel`].

use super::{matrix_of_value, vector_of_value, GaussianMixtureSummary};
use crate::dataframe::DataFrame;
use crate::ml::object::{attribute_relation, summary_transform, FetchArg, MlValue};
use crate::ml::regression_api::{summary_or_error, training_summary};
use crate::ml::tree_api::{as_f64s, as_i32, as_vector, fetch_value};
use crate::ml::{
    fit_via_command, read_model, read_operator, Estimator, GaussianMixture, GaussianMixtureModel,
    MlVector, Model, MultivariateGaussian, OperatorType,
};
use crate::session::SparkSession;
use spark_connect_core::error::{Result, SparkError};

const GAUSSIAN_MIXTURE: &str = "org.apache.spark.ml.clustering.GaussianMixture";
const GAUSSIAN_MIXTURE_MODEL: &str = "org.apache.spark.ml.clustering.GaussianMixtureModel";

impl GaussianMixture {
    /// Fit the model of `df`, with its training summary, as PySpark's `fit` returns a
    /// `GaussianMixtureModel`.
    pub fn fit(&mut self, df: &DataFrame) -> Result<GaussianMixtureModel> {
        self.ensure_operator();
        let info = fit_via_command(&self.operator, &self.params, df)?;
        let mut model = self.model_from_info(info, &df.session)?;
        let summary_dataset = summary_transform(&model.cached, df)?;
        model.training_summary = Some(training_summary(&model.cached, &summary_dataset));
        Ok(model)
    }

    /// The estimator saved at `path`, as `GaussianMixture.load` of PySpark.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (operator, params) =
            read_operator(session, GAUSSIAN_MIXTURE, OperatorType::Estimator, path)?;
        Ok(Self { operator, params })
    }
}

impl GaussianMixtureModel {
    /// The weight of each component of the mixture, which sum to 1 (`weights`).
    pub fn weights(&self) -> Result<Vec<f64>> {
        let value = fetch_value(&self.model_target()?, "weights", vec![])?;
        as_f64s("weights", value)
    }

    /// The DataFrame of the components of the mixture, a row of the `mean` vector and the `cov`
    /// matrix of each (`gaussiansDF`).
    pub fn gaussians_df(&self) -> Result<DataFrame> {
        attribute_relation(&self.model_target()?, "gaussiansDF", vec![])
    }

    /// The components of the mixture (`gaussians`), as PySpark builds them of the collected rows
    /// of [`Self::gaussians_df`].
    pub fn gaussians(&self) -> Result<Vec<MultivariateGaussian>> {
        self.gaussians_df()?
            .collect()?
            .iter()
            .map(|row| {
                let mean = row.get_by_name("mean").and_then(vector_of_value);
                let cov = row.get_by_name("cov").and_then(matrix_of_value);
                match (mean, cov) {
                    (Some(mean), Some(cov)) => Ok(MultivariateGaussian { mean, cov }),
                    _ => Err(SparkError::value_msg(format!(
                        "expected a mean vector and a cov matrix of `gaussiansDF`, got {row:?}"
                    ))),
                }
            })
            .collect()
    }

    /// The number of features that the model was trained on (`numFeatures`).
    pub fn num_features(&self) -> Result<i32> {
        let value = fetch_value(&self.model_target()?, "numFeatures", vec![])?;
        as_i32("numFeatures", value)
    }

    /// The index of the most likely component of `features` (`predict`).
    pub fn predict(&self, features: &MlVector) -> Result<i32> {
        let args = vec![FetchArg::Value(MlValue::Vector(features.clone()))];
        as_i32(
            "predict",
            fetch_value(&self.model_target()?, "predict", args)?,
        )
    }

    /// The probability of each component of `features` (`predictProbability`).
    pub fn predict_probability(&self, features: &MlVector) -> Result<MlVector> {
        let args = vec![FetchArg::Value(MlValue::Vector(features.clone()))];
        let value = fetch_value(&self.model_target()?, "predictProbability", args)?;
        as_vector("predictProbability", value)
    }

    /// Whether the model has a training summary (`hasSummary`): a model that this client fitted
    /// has one, and a loaded model has none, as PySpark tells of a remote model.
    pub fn has_summary(&self) -> bool {
        self.training_summary.is_some()
    }

    /// The summary of the model of its training dataset (`summary`), or an error of a model that
    /// has none.
    pub fn summary(&self) -> Result<GaussianMixtureSummary> {
        let target = summary_or_error(&self.training_summary, "GaussianMixtureModel")?;
        Ok(GaussianMixtureSummary { target })
    }

    /// The model saved at `path`, as `GaussianMixtureModel.load` of PySpark. A loaded model has
    /// no training summary.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (cached, operator, params) = read_model(session, GAUSSIAN_MIXTURE_MODEL, path)?;
        Ok(Self {
            operator,
            params,
            cached,
            training_summary: None,
        })
    }
}
