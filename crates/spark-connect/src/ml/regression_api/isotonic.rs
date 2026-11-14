// SPDX-License-Identifier: Apache-2.0
//! The isotonic regression model, as PySpark's `IsotonicRegressionModel`. Its prediction is of a
//! single feature value rather than of a vector, so it is not a [`crate::ml::PredictionModel`].

use crate::dataframe::DataFrame;
use crate::ml::object::{FetchArg, MlValue};
use crate::ml::tree_api::{as_f64, as_i32, as_vector, fetch_value};
use crate::ml::{
    fit_via_command, read_model, read_operator, Estimator, IsotonicRegression,
    IsotonicRegressionModel, MlVector, Model, OperatorType,
};
use crate::session::SparkSession;
use spark_connect_core::error::Result;

const ISOTONIC_REGRESSION: &str = "org.apache.spark.ml.regression.IsotonicRegression";
const ISOTONIC_REGRESSION_MODEL: &str = "org.apache.spark.ml.regression.IsotonicRegressionModel";

impl IsotonicRegression {
    /// Fit the model of `df`, as PySpark's `fit` returns an `IsotonicRegressionModel`.
    pub fn fit(&mut self, df: &DataFrame) -> Result<IsotonicRegressionModel> {
        self.ensure_operator();
        let info = fit_via_command(&self.operator, &self.params, df)?;
        self.model_from_info(info, &df.session)
    }

    /// The estimator saved at `path`, as `IsotonicRegression.load` of PySpark.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (operator, params) =
            read_operator(session, ISOTONIC_REGRESSION, OperatorType::Estimator, path)?;
        Ok(Self { operator, params })
    }
}

impl IsotonicRegressionModel {
    /// The boundaries of the feature, in increasing order, of the intervals of the piecewise linear
    /// function (`boundaries`).
    pub fn boundaries(&self) -> Result<MlVector> {
        let value = fetch_value(&self.model_target()?, "boundaries", vec![])?;
        as_vector("boundaries", value)
    }

    /// The prediction at each boundary (`predictions`).
    pub fn predictions(&self) -> Result<MlVector> {
        let value = fetch_value(&self.model_target()?, "predictions", vec![])?;
        as_vector("predictions", value)
    }

    /// The number of features that the model was trained on (`numFeatures`).
    pub fn num_features(&self) -> Result<i32> {
        let value = fetch_value(&self.model_target()?, "numFeatures", vec![])?;
        as_i32("numFeatures", value)
    }

    /// The prediction of the feature value `value` (`predict`).
    pub fn predict(&self, value: f64) -> Result<f64> {
        let args = vec![FetchArg::Value(MlValue::Double(value))];
        as_f64(
            "predict",
            fetch_value(&self.model_target()?, "predict", args)?,
        )
    }

    /// The model saved at `path`, as `IsotonicRegressionModel.load` of PySpark.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (cached, operator, params) = read_model(session, ISOTONIC_REGRESSION_MODEL, path)?;
        Ok(Self {
            operator,
            params,
            cached,
        })
    }
}
