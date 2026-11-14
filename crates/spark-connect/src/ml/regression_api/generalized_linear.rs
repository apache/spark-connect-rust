// SPDX-License-Identifier: Apache-2.0
//! The generalized linear regression model and its summaries, as PySpark's
//! `GeneralizedLinearRegressionModel`, `GeneralizedLinearRegressionSummary` and
//! `GeneralizedLinearRegressionTrainingSummary`.

use super::{as_i64, evaluated_summary, summary_or_error, training_summary};
use crate::dataframe::DataFrame;
use crate::ml::object::{attribute_relation, fetch, FetchArg, MlTarget, MlValue};
use crate::ml::tree_api::{as_f64, as_f64s, as_i32, as_string, as_vector, fetch_value};
use crate::ml::{
    fit_via_command, read_model, read_operator, Estimator, GeneralizedLinearRegression,
    GeneralizedLinearRegressionModel, MlSummary, MlVector, Model, OperatorType, PredictionModel,
};
use crate::session::SparkSession;
use spark_connect_core::error::Result;

const GENERALIZED_LINEAR_REGRESSION: &str =
    "org.apache.spark.ml.regression.GeneralizedLinearRegression";
const GENERALIZED_LINEAR_REGRESSION_MODEL: &str =
    "org.apache.spark.ml.regression.GeneralizedLinearRegressionModel";

impl GeneralizedLinearRegression {
    /// Fit the model of `df`, with its training summary, as PySpark's `fit` returns a
    /// `GeneralizedLinearRegressionModel`.
    pub fn fit(&mut self, df: &DataFrame) -> Result<GeneralizedLinearRegressionModel> {
        self.ensure_operator();
        let info = fit_via_command(&self.operator, &self.params, df)?;
        let mut model = self.model_from_info(info, &df.session)?;
        // GeneralizedLinearRegressionModel overrides PySpark's `_summary_dataset` with the training
        // dataset itself.
        model.training_summary = Some(training_summary(&model.cached, df));
        Ok(model)
    }

    /// The estimator saved at `path`, as `GeneralizedLinearRegression.load` of PySpark.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (operator, params) = read_operator(
            session,
            GENERALIZED_LINEAR_REGRESSION,
            OperatorType::Estimator,
            path,
        )?;
        Ok(Self { operator, params })
    }
}

impl GeneralizedLinearRegressionModel {
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

    /// Whether the model has a training summary (`hasSummary`): a model that this client fitted
    /// has one, and a loaded model has none, as PySpark tells of a remote model.
    pub fn has_summary(&self) -> bool {
        self.training_summary.is_some()
    }

    /// The summary of the model of its training dataset (`summary`), or an error of a model that
    /// has none.
    pub fn summary(&self) -> Result<GeneralizedLinearRegressionTrainingSummary> {
        let target = summary_or_error(&self.training_summary, "GeneralizedLinearRegressionModel")?;
        Ok(GeneralizedLinearRegressionTrainingSummary { target })
    }

    /// The summary of the model of `dataset` (`evaluate`).
    pub fn evaluate(&self, dataset: &DataFrame) -> Result<GeneralizedLinearRegressionSummary> {
        let target = self.model_target()?;
        let output = fetch(&target, "evaluate", vec![FetchArg::Input(dataset.clone())])?;
        Ok(GeneralizedLinearRegressionSummary {
            target: evaluated_summary(target.session(), output)?,
        })
    }

    /// The model saved at `path`, as `GeneralizedLinearRegressionModel.load` of PySpark. A loaded
    /// model has no training summary.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (cached, operator, params) =
            read_model(session, GENERALIZED_LINEAR_REGRESSION_MODEL, path)?;
        Ok(Self {
            operator,
            params,
            cached,
            training_summary: None,
        })
    }
}

impl PredictionModel for GeneralizedLinearRegressionModel {}

/// The members of a generalized linear regression summary (`GeneralizedLinearRegressionSummary`),
/// which the summary of an evaluation and the training summary share: each a fetch of the
/// summary's target.
pub trait GeneralizedLinearRegressionSummaryApi: MlSummary {
    /// The DataFrame of the predictions of the model (`predictions`).
    fn predictions(&self) -> Result<DataFrame> {
        attribute_relation(self.summary_target(), "predictions", vec![])
    }

    /// The name of the prediction column (`predictionCol`).
    fn prediction_col(&self) -> Result<String> {
        let value = fetch_value(self.summary_target(), "predictionCol", vec![])?;
        as_string("predictionCol", value)
    }

    /// The number of instances (`numInstances`).
    fn num_instances(&self) -> Result<i64> {
        long_of(self, "numInstances")
    }

    /// The numeric rank of the fitted linear model (`rank`).
    fn rank(&self) -> Result<i64> {
        long_of(self, "rank")
    }

    /// The degrees of freedom (`degreesOfFreedom`).
    fn degrees_of_freedom(&self) -> Result<i64> {
        long_of(self, "degreesOfFreedom")
    }

    /// The residual degrees of freedom (`residualDegreeOfFreedom`).
    fn residual_degree_of_freedom(&self) -> Result<i64> {
        long_of(self, "residualDegreeOfFreedom")
    }

    /// The residual degrees of freedom of the null model (`residualDegreeOfFreedomNull`).
    fn residual_degree_of_freedom_null(&self) -> Result<i64> {
        long_of(self, "residualDegreeOfFreedomNull")
    }

    /// The DataFrame of the residuals of the type `residuals_type`: "deviance" (PySpark's
    /// default), "pearson", "working" or "response" (`residuals`).
    fn residuals(&self, residuals_type: &str) -> Result<DataFrame> {
        let args = vec![FetchArg::Value(MlValue::String(residuals_type.to_string()))];
        attribute_relation(self.summary_target(), "residuals", args)
    }

    /// The deviance of the null model (`nullDeviance`).
    fn null_deviance(&self) -> Result<f64> {
        double_of(self, "nullDeviance")
    }

    /// The deviance of the fitted model (`deviance`).
    fn deviance(&self) -> Result<f64> {
        double_of(self, "deviance")
    }

    /// The dispersion of the fitted model (`dispersion`).
    fn dispersion(&self) -> Result<f64> {
        double_of(self, "dispersion")
    }

    /// The Akaike information criterion of the fitted model (`aic`).
    fn aic(&self) -> Result<f64> {
        double_of(self, "aic")
    }
}

fn long_of<S: MlSummary + ?Sized>(summary: &S, method: &str) -> Result<i64> {
    as_i64(
        method,
        fetch_value(summary.summary_target(), method, vec![])?,
    )
}

fn double_of<S: MlSummary + ?Sized>(summary: &S, method: &str) -> Result<f64> {
    as_f64(
        method,
        fetch_value(summary.summary_target(), method, vec![])?,
    )
}

fn doubles_of<S: MlSummary + ?Sized>(summary: &S, method: &str) -> Result<Vec<f64>> {
    as_f64s(
        method,
        fetch_value(summary.summary_target(), method, vec![])?,
    )
}

/// The summary of a generalized linear regression model of a dataset
/// (`GeneralizedLinearRegressionSummary`), of [`GeneralizedLinearRegressionModel::evaluate`]. Its
/// members are those of [`GeneralizedLinearRegressionSummaryApi`].
#[derive(Debug, Clone)]
pub struct GeneralizedLinearRegressionSummary {
    target: MlTarget,
}

impl MlSummary for GeneralizedLinearRegressionSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl GeneralizedLinearRegressionSummaryApi for GeneralizedLinearRegressionSummary {}

/// The summary of a generalized linear regression model of its training dataset
/// (`GeneralizedLinearRegressionTrainingSummary`), of
/// [`GeneralizedLinearRegressionModel::summary`]: the members of
/// [`GeneralizedLinearRegressionSummaryApi`] and those of the fit.
#[derive(Debug, Clone)]
pub struct GeneralizedLinearRegressionTrainingSummary {
    target: MlTarget,
}

impl MlSummary for GeneralizedLinearRegressionTrainingSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl GeneralizedLinearRegressionSummaryApi for GeneralizedLinearRegressionTrainingSummary {}

impl GeneralizedLinearRegressionTrainingSummary {
    /// The number of iterations of the fit (`numIterations`).
    pub fn num_iterations(&self) -> Result<i32> {
        let value = fetch_value(&self.target, "numIterations", vec![])?;
        as_i32("numIterations", value)
    }

    /// The numeric solver of the fit (`solver`).
    pub fn solver(&self) -> Result<String> {
        as_string("solver", fetch_value(&self.target, "solver", vec![])?)
    }

    /// The standard errors of the coefficients and the intercept (`coefficientStandardErrors`).
    pub fn coefficient_standard_errors(&self) -> Result<Vec<f64>> {
        doubles_of(self, "coefficientStandardErrors")
    }

    /// The t-statistics of the coefficients and the intercept (`tValues`).
    pub fn t_values(&self) -> Result<Vec<f64>> {
        doubles_of(self, "tValues")
    }

    /// The two-sided p-values of the coefficients and the intercept (`pValues`).
    pub fn p_values(&self) -> Result<Vec<f64>> {
        doubles_of(self, "pValues")
    }

    /// The server's table of the coefficients and the fit, as PySpark's `__repr__` of the summary
    /// reads it (its `toString`).
    pub fn describe(&self) -> Result<String> {
        fetch(&self.target, "toString", vec![])?.into_string()
    }
}
