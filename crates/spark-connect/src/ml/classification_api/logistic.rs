// SPDX-License-Identifier: Apache-2.0
//! The typed accessors, summaries and persistence of LogisticRegression and
//! LogisticRegressionModel, as `pyspark.ml.classification` of PySpark 4.3.0 has them.
//!
//! A model that a fit returns holds the target of its training summary (`_summary`), so that
//! `has_summary` is client side. Its summary is the binary one when `numClasses` is at most 2, as
//! PySpark chooses it at the fit (`_summaryCls`). The threshold getters are the Params logic of the
//! estimator, which this file does not repeat.

use crate::dataframe::DataFrame;
use crate::ml::tree_api::{as_f64, as_string, as_vector, fetch_value};
use crate::ml::{
    fetch, read_model, read_operator, BinaryClassificationSummary, ClassificationModel,
    ClassificationSummary, Estimator, FetchArg, LogisticRegression, LogisticRegressionModel,
    MlMatrix, MlOperator, MlRef, MlSummary, MlTarget, MlValue, MlVector, Model, OperatorType,
    Params, PredictionModel, ProbabilisticClassificationModel, TrainingSummary,
};
use crate::session::SparkSession;
use spark_connect_core::error::{Result, SparkError};

/// The JVM class of the estimator, as PySpark saves and reads it.
const LOGISTIC_REGRESSION: &str = "org.apache.spark.ml.classification.LogisticRegression";
/// The JVM class of the model.
const LOGISTIC_REGRESSION_MODEL: &str =
    "org.apache.spark.ml.classification.LogisticRegressionModel";

/// The members that the four LogisticRegression summaries share, as `_LogisticRegressionSummary`
/// of PySpark: the names of the features and probability columns.
pub trait LogisticRegressionSummaryApi: MlSummary {
    /// The name of the features column (`featuresCol`).
    fn features_col(&self) -> Result<String> {
        as_string(
            "featuresCol",
            fetch_value(self.summary_target(), "featuresCol", vec![])?,
        )
    }

    /// The name of the probability column (`probabilityCol`).
    fn probability_col(&self) -> Result<String> {
        as_string(
            "probabilityCol",
            fetch_value(self.summary_target(), "probabilityCol", vec![])?,
        )
    }
}

/// The summary of a multiclass LogisticRegressionModel (`LogisticRegressionSummary`), which has
/// no ROC curve.
#[derive(Debug, Clone)]
pub struct LogisticRegressionSummary {
    target: MlTarget,
}

impl LogisticRegressionSummary {
    /// The summary of the target `target`, of a summary that the server holds.
    pub(crate) fn from_target(target: MlTarget) -> Self {
        Self { target }
    }
}

impl MlSummary for LogisticRegressionSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl ClassificationSummary for LogisticRegressionSummary {}
impl LogisticRegressionSummaryApi for LogisticRegressionSummary {}

/// The training summary of a multiclass LogisticRegressionModel
/// (`LogisticRegressionTrainingSummary`).
#[derive(Debug, Clone)]
pub struct LogisticRegressionTrainingSummary {
    target: MlTarget,
}

impl LogisticRegressionTrainingSummary {
    /// The summary of the target `target`, of a summary that the server holds.
    pub(crate) fn from_target(target: MlTarget) -> Self {
        Self { target }
    }
}

impl MlSummary for LogisticRegressionTrainingSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl ClassificationSummary for LogisticRegressionTrainingSummary {}
impl LogisticRegressionSummaryApi for LogisticRegressionTrainingSummary {}
impl TrainingSummary for LogisticRegressionTrainingSummary {}

/// The summary of a binary LogisticRegressionModel (`BinaryLogisticRegressionSummary`).
#[derive(Debug, Clone)]
pub struct BinaryLogisticRegressionSummary {
    target: MlTarget,
}

impl BinaryLogisticRegressionSummary {
    /// The summary of the target `target`, of a summary that the server holds.
    pub(crate) fn from_target(target: MlTarget) -> Self {
        Self { target }
    }
}

impl MlSummary for BinaryLogisticRegressionSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl ClassificationSummary for BinaryLogisticRegressionSummary {}
impl BinaryClassificationSummary for BinaryLogisticRegressionSummary {}
impl LogisticRegressionSummaryApi for BinaryLogisticRegressionSummary {}

/// The training summary of a binary LogisticRegressionModel
/// (`BinaryLogisticRegressionTrainingSummary`).
#[derive(Debug, Clone)]
pub struct BinaryLogisticRegressionTrainingSummary {
    target: MlTarget,
}

impl BinaryLogisticRegressionTrainingSummary {
    /// The summary of the target `target`, of a summary that the server holds.
    pub(crate) fn from_target(target: MlTarget) -> Self {
        Self { target }
    }
}

impl MlSummary for BinaryLogisticRegressionTrainingSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl ClassificationSummary for BinaryLogisticRegressionTrainingSummary {}
impl BinaryClassificationSummary for BinaryLogisticRegressionTrainingSummary {}
impl LogisticRegressionSummaryApi for BinaryLogisticRegressionTrainingSummary {}
impl TrainingSummary for BinaryLogisticRegressionTrainingSummary {}

/// The training summary of a LogisticRegressionModel that a fit returns: binary when the model
/// has at most two classes, as `LogisticRegressionModel.summary` of PySpark returns it.
#[derive(Debug, Clone)]
pub enum LogisticRegressionModelSummary {
    Binary(BinaryLogisticRegressionTrainingSummary),
    Multiclass(LogisticRegressionTrainingSummary),
}

/// The summary of a LogisticRegressionModel's `evaluate`: binary when the model has at most two
/// classes, as PySpark's `evaluate` chooses it.
#[derive(Debug, Clone)]
pub enum LogisticRegressionEvaluation {
    Binary(BinaryLogisticRegressionSummary),
    Multiclass(LogisticRegressionSummary),
}

impl LogisticRegression {
    /// Fit the typed model, which holds the target of its training summary (`_summary` of
    /// PySpark's `_fit`). The summary's dataset is the training dataset itself, as
    /// `_summary_dataset` of PySpark returns it.
    pub fn fit(&mut self, df: &DataFrame) -> Result<LogisticRegressionModel> {
        self.ensure_operator();
        let info = crate::ml::fit_via_command(&self.operator, &self.params, df)?;
        let mut model = self.model_from_info(info, &df.session)?;
        let binary = model.num_classes()? <= 2;
        model.summary = Some(
            model
                .model_target()?
                .child("summary")
                .with_summary_dataset(df),
        );
        model.binary_summary = binary;
        Ok(model)
    }

    /// Fit one model per param map, in order: each a copy of this estimator with the map's params
    /// set over its own, as `Estimator.fitMultiple` of PySpark does with `copy` and `fit`.
    pub fn fit_multiple(
        &self,
        df: &DataFrame,
        param_maps: &[Params],
    ) -> Result<Vec<LogisticRegressionModel>> {
        param_maps
            .iter()
            .map(|map| {
                let mut estimator = self.clone();
                estimator.params.params.extend(map.params.clone());
                estimator.fit(df)
            })
            .collect()
    }

    /// The estimator that `write().save(path)` saved, as `LogisticRegression.load` of PySpark
    /// reads it.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (operator, params) =
            read_operator(session, LOGISTIC_REGRESSION, OperatorType::Estimator, path)?;
        Ok(Self { operator, params })
    }
}

impl LogisticRegressionModel {
    /// The model of the server-side object `ml_ref` with the params `params`, of no training
    /// summary, as a read that returns a model gives it.
    pub(crate) fn from_ml_ref(ml_ref: MlRef, params: Params) -> Self {
        Self {
            operator: MlOperator::new(LOGISTIC_REGRESSION_MODEL, OperatorType::Model),
            params,
            cached: ml_ref,
            summary: None,
            binary_summary: false,
        }
    }

    /// The model that `write().save(path)` saved, as `LogisticRegressionModel.load` of PySpark
    /// reads it. A loaded model has no training summary.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (object, operator, params) = read_model(session, LOGISTIC_REGRESSION_MODEL, path)?;
        let mut model = Self::from_ml_ref(object, params);
        model.operator = operator;
        Ok(model)
    }

    /// The coefficients of a binary model (`coefficients`). A multinomial model has a matrix of
    /// coefficients instead, and the server's error says so.
    pub fn coefficients(&self) -> Result<MlVector> {
        as_vector(
            "coefficients",
            fetch_value(&self.model_target()?, "coefficients", vec![])?,
        )
    }

    /// The coefficients, as a matrix with one row per class (`coefficientMatrix`).
    pub fn coefficient_matrix(&self) -> Result<MlMatrix> {
        as_matrix(
            "coefficientMatrix",
            fetch_value(&self.model_target()?, "coefficientMatrix", vec![])?,
        )
    }

    /// The intercept of a binary model (`intercept`). A multinomial model has a vector of
    /// intercepts instead, and the server's error says so.
    pub fn intercept(&self) -> Result<f64> {
        as_f64(
            "intercept",
            fetch_value(&self.model_target()?, "intercept", vec![])?,
        )
    }

    /// The intercepts, one per class (`interceptVector`).
    pub fn intercept_vector(&self) -> Result<MlVector> {
        as_vector(
            "interceptVector",
            fetch_value(&self.model_target()?, "interceptVector", vec![])?,
        )
    }

    /// Whether the model has a training summary, which a fit gives and a loaded model lacks
    /// (`hasSummary`). It is client side, as PySpark's is under Spark Connect.
    pub fn has_summary(&self) -> bool {
        self.summary.is_some()
    }

    /// The training summary of the model (`summary`): binary when the model has at most two
    /// classes. An error when the model has no training summary.
    pub fn summary(&self) -> Result<LogisticRegressionModelSummary> {
        let target = self.summary.clone().ok_or_else(|| {
            SparkError::value_msg("No training summary available for this LogisticRegressionModel")
        })?;
        Ok(if self.binary_summary {
            LogisticRegressionModelSummary::Binary(
                BinaryLogisticRegressionTrainingSummary::from_target(target),
            )
        } else {
            LogisticRegressionModelSummary::Multiclass(
                LogisticRegressionTrainingSummary::from_target(target),
            )
        })
    }

    /// The summary of the model on `df` (`evaluate`), binary when the model has at most two
    /// classes. The summary is the server's own object, which this client does not release, as
    /// PySpark does not release it.
    pub fn evaluate(&self, df: &DataFrame) -> Result<LogisticRegressionEvaluation> {
        let target = self.model_target()?;
        let id = fetch(&target, "evaluate", vec![FetchArg::Input(df.clone())])?.into_string()?;
        let summary = MlTarget::unowned(target.session(), &id);
        Ok(if self.num_classes()? <= 2 {
            LogisticRegressionEvaluation::Binary(BinaryLogisticRegressionSummary::from_target(
                summary,
            ))
        } else {
            LogisticRegressionEvaluation::Multiclass(LogisticRegressionSummary::from_target(
                summary,
            ))
        })
    }

    /// Set the features column of the model's transform (`setFeaturesCol`).
    pub fn set_features_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("featuresCol", value);
        self
    }

    /// Set the prediction column of the model's transform (`setPredictionCol`).
    pub fn set_prediction_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("predictionCol", value);
        self
    }

    /// Set the probability column of the model's transform (`setProbabilityCol`).
    pub fn set_probability_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("probabilityCol", value);
        self
    }

    /// Set the raw prediction column of the model's transform (`setRawPredictionCol`).
    pub fn set_raw_prediction_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("rawPredictionCol", value);
        self
    }

    /// Set the decision threshold of the model (`setThreshold`).
    pub fn set_threshold(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("threshold", value);
        self
    }

    /// Set the per-class thresholds of the model (`setThresholds`).
    pub fn set_thresholds(mut self, values: impl IntoIterator<Item = f64>) -> Self {
        self.params = self.params.set_param_double_array("thresholds", values);
        self
    }
}

impl PredictionModel for LogisticRegressionModel {}
impl ClassificationModel for LogisticRegressionModel {}
impl ProbabilisticClassificationModel for LogisticRegressionModel {}

/// The matrix of a `Matrix` value of `method`.
fn as_matrix(method: &str, value: MlValue) -> Result<MlMatrix> {
    match value {
        MlValue::Matrix(matrix) => Ok(matrix),
        other => Err(SparkError::value_msg(format!(
            "expected a matrix of `{method}`, got {other:?}"
        ))),
    }
}
