// SPDX-License-Identifier: Apache-2.0
//! The typed accessors, summaries and persistence of MultilayerPerceptronClassificationModel and
//! its summaries.
//!
//! The model has a training summary when a fit made it (`hasSummary`, client side, as PySpark's
//! `HasTrainingSummary`). Its evaluation summary is the summary that `evaluate` returns.

use super::{
    ClassificationModel, ClassificationSummary, MlSummary, ProbabilisticClassificationModel,
    TrainingSummary,
};
use crate::dataframe::DataFrame;
use crate::ml::object::{fetch, read_model, FetchArg, MlTarget};
use crate::ml::tree_api::{as_vector, fetch_value};
use crate::ml::{
    java_string_hash, MlOperator, MlRef, MlVector, Model, MultilayerPerceptronClassificationModel,
    MultilayerPerceptronClassifier, OperatorType, Params, PredictionModel,
};
use crate::session::SparkSession;
use spark_connect_core::error::{Result, SparkError};

impl MultilayerPerceptronClassificationModel {
    /// The weights of the layers, as one vector (`weights`).
    pub fn weights(&self) -> Result<MlVector> {
        as_vector(
            "weights",
            fetch_value(&self.model_target()?, "weights", vec![])?,
        )
    }

    /// Whether a training summary exists for the model (`hasSummary`), which a fit gives it.
    pub fn has_summary(&self) -> bool {
        self.summary_target.is_some()
    }

    /// The training summary of the model (`summary()`), which is an error of a model without one.
    pub fn summary(&self) -> Result<MultilayerPerceptronClassificationTrainingSummary> {
        match &self.summary_target {
            Some(target) => Ok(MultilayerPerceptronClassificationTrainingSummary {
                target: target.clone(),
            }),
            None => Err(SparkError::value_msg(
                "No training summary available for this MultilayerPerceptronClassificationModel",
            )),
        }
    }

    /// The summary of the model on `df` (`evaluate`), whose server-side summary this client does
    /// not release, as PySpark does not release it.
    pub fn evaluate(&self, df: &DataFrame) -> Result<MultilayerPerceptronClassificationSummary> {
        let target = self.model_target()?;
        let id = fetch(&target, "evaluate", vec![FetchArg::Input(df.clone())])?.into_string()?;
        Ok(MultilayerPerceptronClassificationSummary {
            target: MlTarget::unowned(target.session(), &id),
        })
    }

    /// The model that `path` saves (`MultilayerPerceptronClassificationModel.load`), of the
    /// session `session`. A loaded model has no training summary.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (object, operator, params) = read_model(
            session,
            "org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel",
            path,
        )?;
        let mut model = Self::from_ml_ref(object, params);
        model.operator = operator;
        Ok(model)
    }

    /// The model of the server-side object `ml_ref`, with the params `params`.
    pub(crate) fn from_ml_ref(ml_ref: MlRef, params: Params) -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel",
                OperatorType::Model,
            ),
            params,
            cached: ml_ref,
            summary_target: None,
        }
    }
}

impl PredictionModel for MultilayerPerceptronClassificationModel {}

impl ClassificationModel for MultilayerPerceptronClassificationModel {}

impl ProbabilisticClassificationModel for MultilayerPerceptronClassificationModel {}

/// The summary of a multilayer perceptron model on a dataset (`evaluate`), as PySpark's
/// `MultilayerPerceptronClassificationSummary`: a multiclass classification summary.
#[derive(Debug, Clone)]
pub struct MultilayerPerceptronClassificationSummary {
    target: MlTarget,
}

impl MlSummary for MultilayerPerceptronClassificationSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl ClassificationSummary for MultilayerPerceptronClassificationSummary {}

/// The training summary of a multilayer perceptron model, as PySpark's
/// `MultilayerPerceptronClassificationTrainingSummary`.
#[derive(Debug, Clone)]
pub struct MultilayerPerceptronClassificationTrainingSummary {
    target: MlTarget,
}

impl MlSummary for MultilayerPerceptronClassificationTrainingSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl ClassificationSummary for MultilayerPerceptronClassificationTrainingSummary {}

impl TrainingSummary for MultilayerPerceptronClassificationTrainingSummary {}

ml_params! {
    MultilayerPerceptronClassifier {
        set_block_size(value: i32) => "blockSize", set_param_int32;
        block_size -> i32 = |this| this.params.int_or("blockSize", 128);
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        set_initial_weights(value: impl Into<MlVector>) => "initialWeights", set_param_vector;
        initial_weights -> Option<MlVector> =
            |this| this.params.literal("initialWeights").and_then(MlVector::from_literal);
        set_label_col(value: &str) => "labelCol", set_param_string;
        label_col -> &str = |this| this.params.string_or("labelCol", "label");
        set_layers(value: impl IntoIterator<Item = i32>) => "layers", set_param_int32_array;
        layers -> Option<Vec<i32>> = |this| this.params.int_array_opt("layers");
        set_max_iter(value: i32) => "maxIter", set_param_int32;
        max_iter -> i32 = |this| this.params.int_or("maxIter", 100);
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_probability_col(value: &str) => "probabilityCol", set_param_string;
        probability_col -> &str = |this| this.params.string_or("probabilityCol", "probability");
        set_raw_prediction_col(value: &str) => "rawPredictionCol", set_param_string;
        raw_prediction_col -> &str =
            |this| this.params.string_or("rawPredictionCol", "rawPrediction");
        set_seed(value: i64) => "seed", set_param_int_narrow;
        seed -> i64 = |this| {
            this.params.long_opt("seed").unwrap_or(java_string_hash(
                "org.apache.spark.ml.classification.MultilayerPerceptronClassifier",
            ))
        };
        set_solver(value: &str) => "solver", set_param_string;
        solver -> &str = |this| this.params.string_or("solver", "l-bfgs");
        set_step_size(value: f64) => "stepSize", set_param_double;
        step_size -> f64 = |this| this.params.double_or("stepSize", 0.03);
        set_thresholds(value: impl IntoIterator<Item = f64>) =>
            "thresholds", set_param_double_array;
        thresholds -> Option<Vec<f64>> = |this| this.params.double_array_opt("thresholds");
        set_tol(value: f64) => "tol", set_param_double;
        tol -> f64 = |this| this.params.double_or("tol", 1e-06);
    }
}

ml_params! {
    MultilayerPerceptronClassificationModel {
        block_size -> i32 = |this| this.params.int_or("blockSize", 128) => "blockSize";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        initial_weights -> Option<MlVector> =
            |this| this.params.literal(
                "initialWeights").and_then(MlVector::from_literal) => "initialWeights";
        label_col -> &str = |this| this.params.string_or("labelCol", "label") => "labelCol";
        layers -> Option<Vec<i32>> = |this| this.params.int_array_opt("layers") => "layers";
        max_iter -> i32 = |this| this.params.int_or("maxIter", 100) => "maxIter";
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_probability_col(value: &str) => "probabilityCol", set_param_string;
        probability_col -> &str = |this| this.params.string_or("probabilityCol", "probability");
        set_raw_prediction_col(value: &str) => "rawPredictionCol", set_param_string;
        raw_prediction_col -> &str =
            |this| this.params.string_or("rawPredictionCol", "rawPrediction");
        seed -> i64 = |this| {
            this.params.long_opt("seed").unwrap_or(java_string_hash(
                "org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel",
            ))
        } => "seed";
        solver -> &str = |this| this.params.string_or("solver", "l-bfgs") => "solver";
        step_size -> f64 = |this| this.params.double_or("stepSize", 0.03) => "stepSize";
        set_thresholds(value: impl IntoIterator<Item = f64>) =>
            "thresholds", set_param_double_array;
        thresholds -> Option<Vec<f64>> = |this| this.params.double_array_opt("thresholds");
        tol -> f64 = |this| this.params.double_or("tol", 1e-06) => "tol";
    }
}
