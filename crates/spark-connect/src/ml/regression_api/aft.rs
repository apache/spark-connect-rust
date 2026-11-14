// SPDX-License-Identifier: Apache-2.0
//! The accelerated failure time survival regression model, as PySpark's
//! `AFTSurvivalRegressionModel`.

use crate::dataframe::DataFrame;
use crate::ml::object::{FetchArg, MlValue};
use crate::ml::tree_api::{as_f64, as_vector, fetch_value};
use crate::ml::{
    fit_via_command, read_model, read_operator, AFTSurvivalRegression, AFTSurvivalRegressionModel,
    Estimator, MlVector, Model, OperatorType, PredictionModel,
};
use crate::session::SparkSession;
use spark_connect_core::error::Result;

const AFT_SURVIVAL_REGRESSION: &str = "org.apache.spark.ml.regression.AFTSurvivalRegression";
const AFT_SURVIVAL_REGRESSION_MODEL: &str =
    "org.apache.spark.ml.regression.AFTSurvivalRegressionModel";

impl AFTSurvivalRegression {
    /// Fit the model of `df`, as PySpark's `fit` returns an `AFTSurvivalRegressionModel`.
    pub fn fit(&mut self, df: &DataFrame) -> Result<AFTSurvivalRegressionModel> {
        self.ensure_operator();
        let info = fit_via_command(&self.operator, &self.params, df)?;
        self.model_from_info(info, &df.session)
    }

    /// The estimator saved at `path`, as `AFTSurvivalRegression.load` of PySpark.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (operator, params) = read_operator(
            session,
            AFT_SURVIVAL_REGRESSION,
            OperatorType::Estimator,
            path,
        )?;
        Ok(Self { operator, params })
    }
}

impl AFTSurvivalRegressionModel {
    /// The coefficients of the features (`coefficients`).
    pub fn coefficients(&self) -> Result<MlVector> {
        let value = fetch_value(&self.model_target()?, "coefficients", vec![])?;
        as_vector("coefficients", value)
    }

    /// The intercept (`intercept`).
    pub fn intercept(&self) -> Result<f64> {
        as_f64(
            "intercept",
            fetch_value(&self.model_target()?, "intercept", vec![])?,
        )
    }

    /// The scale of the model (`scale`).
    pub fn scale(&self) -> Result<f64> {
        as_f64(
            "scale",
            fetch_value(&self.model_target()?, "scale", vec![])?,
        )
    }

    /// The quantiles of the survival time of `features`, at the quantile probabilities of the
    /// model (`predictQuantiles`).
    pub fn predict_quantiles(&self, features: &MlVector) -> Result<MlVector> {
        let args = vec![FetchArg::Value(MlValue::Vector(features.clone()))];
        let value = fetch_value(&self.model_target()?, "predictQuantiles", args)?;
        as_vector("predictQuantiles", value)
    }

    /// The model saved at `path`, as `AFTSurvivalRegressionModel.load` of PySpark.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (cached, operator, params) = read_model(session, AFT_SURVIVAL_REGRESSION_MODEL, path)?;
        Ok(Self {
            operator,
            params,
            cached,
        })
    }
}

impl PredictionModel for AFTSurvivalRegressionModel {}
