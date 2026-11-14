// SPDX-License-Identifier: Apache-2.0
//! The factorization machines regression model, as PySpark's `FMRegressionModel`.

use super::as_matrix;
use crate::dataframe::DataFrame;
use crate::ml::tree_api::{as_f64, as_vector, fetch_value};
use crate::ml::{
    fit_via_command, read_model, read_operator, Estimator, FMRegressionModel, FMRegressor,
    MlMatrix, MlVector, Model, OperatorType, PredictionModel,
};
use crate::session::SparkSession;
use spark_connect_core::error::Result;

const FM_REGRESSOR: &str = "org.apache.spark.ml.regression.FMRegressor";
const FM_REGRESSION_MODEL: &str = "org.apache.spark.ml.regression.FMRegressionModel";

impl FMRegressor {
    /// Fit the model of `df`, as PySpark's `fit` returns an `FMRegressionModel`.
    pub fn fit(&mut self, df: &DataFrame) -> Result<FMRegressionModel> {
        self.ensure_operator();
        let info = fit_via_command(&self.operator, &self.params, df)?;
        self.model_from_info(info, &df.session)
    }

    /// The estimator saved at `path`, as `FMRegressor.load` of PySpark.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (operator, params) =
            read_operator(session, FM_REGRESSOR, OperatorType::Estimator, path)?;
        Ok(Self { operator, params })
    }
}

impl FMRegressionModel {
    /// The intercept (`intercept`).
    pub fn intercept(&self) -> Result<f64> {
        as_f64(
            "intercept",
            fetch_value(&self.model_target()?, "intercept", vec![])?,
        )
    }

    /// The coefficients of the linear term (`linear`).
    pub fn linear(&self) -> Result<MlVector> {
        as_vector(
            "linear",
            fetch_value(&self.model_target()?, "linear", vec![])?,
        )
    }

    /// The factors of the pairwise interactions, a matrix of a row per feature (`factors`).
    pub fn factors(&self) -> Result<MlMatrix> {
        as_matrix(
            "factors",
            fetch_value(&self.model_target()?, "factors", vec![])?,
        )
    }

    /// The model saved at `path`, as `FMRegressionModel.load` of PySpark.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (cached, operator, params) = read_model(session, FM_REGRESSION_MODEL, path)?;
        Ok(Self {
            operator,
            params,
            cached,
        })
    }
}

impl PredictionModel for FMRegressionModel {}
