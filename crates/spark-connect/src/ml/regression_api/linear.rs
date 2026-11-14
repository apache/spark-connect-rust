// SPDX-License-Identifier: Apache-2.0
//! The linear regression model and its summaries, as PySpark's `LinearRegressionModel`,
//! `LinearRegressionSummary` and `LinearRegressionTrainingSummary`.

use super::{as_i64, evaluated_summary, summary_or_error, training_summary};
use crate::dataframe::DataFrame;
use crate::ml::object::{attribute_relation, fetch, FetchArg, MlTarget};
use crate::ml::tree_api::{as_f64, as_f64s, as_string, as_vector, fetch_value};
use crate::ml::{
    fit_via_command, read_model, read_operator, Estimator, LinearRegression, LinearRegressionModel,
    MlSummary, MlVector, Model, OperatorType, PredictionModel, TrainingSummary,
};
use crate::session::SparkSession;
use spark_connect_core::error::Result;

const LINEAR_REGRESSION: &str = "org.apache.spark.ml.regression.LinearRegression";
const LINEAR_REGRESSION_MODEL: &str = "org.apache.spark.ml.regression.LinearRegressionModel";

impl LinearRegression {
    /// Fit the model of `df`, with its training summary, as PySpark's `fit` returns a
    /// `LinearRegressionModel`.
    pub fn fit(&mut self, df: &DataFrame) -> Result<LinearRegressionModel> {
        self.ensure_operator();
        let info = fit_via_command(&self.operator, &self.params, df)?;
        let mut model = self.model_from_info(info, &df.session)?;
        // LinearRegressionModel overrides PySpark's `_summary_dataset` with the training dataset
        // itself.
        model.training_summary = Some(training_summary(&model.cached, df));
        Ok(model)
    }

    /// The estimator saved at `path`, as `LinearRegression.load` of PySpark.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (operator, params) =
            read_operator(session, LINEAR_REGRESSION, OperatorType::Estimator, path)?;
        Ok(Self { operator, params })
    }
}

impl LinearRegressionModel {
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

    /// The scale of the Huber loss, 1.0 of the squared error loss (`scale`).
    pub fn scale(&self) -> Result<f64> {
        as_f64(
            "scale",
            fetch_value(&self.model_target()?, "scale", vec![])?,
        )
    }

    /// Whether the model has a training summary (`hasSummary`): a model that this client fitted
    /// has one, and a loaded model has none, as PySpark tells of a remote model.
    pub fn has_summary(&self) -> bool {
        self.training_summary.is_some()
    }

    /// The summary of the model of its training dataset (`summary`), or an error of a model that
    /// has none.
    pub fn summary(&self) -> Result<LinearRegressionTrainingSummary> {
        let target = summary_or_error(&self.training_summary, "LinearRegressionModel")?;
        Ok(LinearRegressionTrainingSummary { target })
    }

    /// The summary of the model of `dataset` (`evaluate`).
    pub fn evaluate(&self, dataset: &DataFrame) -> Result<LinearRegressionSummary> {
        let target = self.model_target()?;
        let output = fetch(&target, "evaluate", vec![FetchArg::Input(dataset.clone())])?;
        Ok(LinearRegressionSummary {
            target: evaluated_summary(target.session(), output)?,
        })
    }

    /// The model saved at `path`, as `LinearRegressionModel.load` of PySpark. A loaded model has no
    /// training summary.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (cached, operator, params) = read_model(session, LINEAR_REGRESSION_MODEL, path)?;
        Ok(Self {
            operator,
            params,
            cached,
            training_summary: None,
        })
    }
}

impl PredictionModel for LinearRegressionModel {}

/// The members of a linear regression summary (`LinearRegressionSummary`), which the summary of an
/// evaluation and the training summary share: each a fetch of the summary's target.
pub trait LinearRegressionSummaryApi: MlSummary {
    /// The DataFrame of the predictions of the model (`predictions`).
    fn predictions(&self) -> Result<DataFrame> {
        attribute_relation(self.summary_target(), "predictions", vec![])
    }

    /// The name of the prediction column (`predictionCol`).
    fn prediction_col(&self) -> Result<String> {
        string_of(self, "predictionCol")
    }

    /// The name of the label column (`labelCol`).
    fn label_col(&self) -> Result<String> {
        string_of(self, "labelCol")
    }

    /// The name of the features column (`featuresCol`).
    fn features_col(&self) -> Result<String> {
        string_of(self, "featuresCol")
    }

    /// The explained variance regression score (`explainedVariance`).
    fn explained_variance(&self) -> Result<f64> {
        double_of(self, "explainedVariance")
    }

    /// The mean absolute error (`meanAbsoluteError`).
    fn mean_absolute_error(&self) -> Result<f64> {
        double_of(self, "meanAbsoluteError")
    }

    /// The mean squared error (`meanSquaredError`).
    fn mean_squared_error(&self) -> Result<f64> {
        double_of(self, "meanSquaredError")
    }

    /// The root mean squared error (`rootMeanSquaredError`).
    fn root_mean_squared_error(&self) -> Result<f64> {
        double_of(self, "rootMeanSquaredError")
    }

    /// The coefficient of determination (`r2`).
    fn r2(&self) -> Result<f64> {
        double_of(self, "r2")
    }

    /// The coefficient of determination adjusted for the number of features (`r2adj`).
    fn r2adj(&self) -> Result<f64> {
        double_of(self, "r2adj")
    }

    /// The DataFrame of the residuals, the labels minus the predictions (`residuals`).
    fn residuals(&self) -> Result<DataFrame> {
        attribute_relation(self.summary_target(), "residuals", vec![])
    }

    /// The number of instances (`numInstances`).
    fn num_instances(&self) -> Result<i64> {
        let value = fetch_value(self.summary_target(), "numInstances", vec![])?;
        as_i64("numInstances", value)
    }

    /// The degrees of freedom (`degreesOfFreedom`).
    fn degrees_of_freedom(&self) -> Result<i64> {
        let value = fetch_value(self.summary_target(), "degreesOfFreedom", vec![])?;
        as_i64("degreesOfFreedom", value)
    }

    /// The weighted residuals of the deviance (`devianceResiduals`).
    fn deviance_residuals(&self) -> Result<Vec<f64>> {
        doubles_of(self, "devianceResiduals")
    }

    /// The standard errors of the coefficients and the intercept, of the normal solver
    /// (`coefficientStandardErrors`).
    fn coefficient_standard_errors(&self) -> Result<Vec<f64>> {
        doubles_of(self, "coefficientStandardErrors")
    }

    /// The t-statistics of the coefficients and the intercept, of the normal solver (`tValues`).
    fn t_values(&self) -> Result<Vec<f64>> {
        doubles_of(self, "tValues")
    }

    /// The two-sided p-values of the coefficients and the intercept, of the normal solver
    /// (`pValues`).
    fn p_values(&self) -> Result<Vec<f64>> {
        doubles_of(self, "pValues")
    }
}

fn string_of<S: MlSummary + ?Sized>(summary: &S, method: &str) -> Result<String> {
    as_string(
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

/// The summary of a linear regression model of a dataset (`LinearRegressionSummary`), of
/// [`LinearRegressionModel::evaluate`]. Its members are those of [`LinearRegressionSummaryApi`].
#[derive(Debug, Clone)]
pub struct LinearRegressionSummary {
    target: MlTarget,
}

impl MlSummary for LinearRegressionSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl LinearRegressionSummaryApi for LinearRegressionSummary {}

/// The summary of a linear regression model of its training dataset
/// (`LinearRegressionTrainingSummary`), of [`LinearRegressionModel::summary`]. Its members are
/// those of [`LinearRegressionSummaryApi`] and of [`TrainingSummary`].
#[derive(Debug, Clone)]
pub struct LinearRegressionTrainingSummary {
    target: MlTarget,
}

impl MlSummary for LinearRegressionTrainingSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl LinearRegressionSummaryApi for LinearRegressionTrainingSummary {}

impl TrainingSummary for LinearRegressionTrainingSummary {}
