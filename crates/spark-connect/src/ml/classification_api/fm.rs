// SPDX-License-Identifier: Apache-2.0
//! The typed accessors, summaries and persistence of FMClassificationModel and its summaries.
//!
//! The model has a training summary when a fit made it (`hasSummary`, client side, as PySpark's
//! `HasTrainingSummary`). Both summaries are binary classification summaries, as PySpark's.

use super::naive_bayes::as_matrix;
use super::{
    BinaryClassificationSummary, ClassificationModel, ClassificationSummary, MlSummary,
    ProbabilisticClassificationModel, TrainingSummary,
};
use crate::dataframe::DataFrame;
use crate::ml::object::{fetch, read_model, FetchArg, MlTarget};
use crate::ml::tree_api::{as_f64, as_vector, fetch_value};
use crate::ml::{
    java_string_hash, FMClassificationModel, FMClassifier, MlMatrix, MlOperator, MlRef, MlVector,
    Model, OperatorType, Params, PredictionModel,
};
use crate::session::SparkSession;
use spark_connect_core::error::{Result, SparkError};

impl FMClassificationModel {
    /// The factor matrix of the model, one row per feature (`factors`).
    pub fn factors(&self) -> Result<MlMatrix> {
        as_matrix(
            "factors",
            fetch_value(&self.model_target()?, "factors", vec![])?,
        )
    }

    /// The linear coefficients of the model (`linear`).
    pub fn linear(&self) -> Result<MlVector> {
        as_vector(
            "linear",
            fetch_value(&self.model_target()?, "linear", vec![])?,
        )
    }

    /// The intercept of the model (`intercept`).
    pub fn intercept(&self) -> Result<f64> {
        as_f64(
            "intercept",
            fetch_value(&self.model_target()?, "intercept", vec![])?,
        )
    }

    /// Whether a training summary exists for the model (`hasSummary`), which a fit gives it.
    pub fn has_summary(&self) -> bool {
        self.summary_target.is_some()
    }

    /// The training summary of the model (`summary()`), which is an error of a model without one.
    pub fn summary(&self) -> Result<FMClassificationTrainingSummary> {
        match &self.summary_target {
            Some(target) => Ok(FMClassificationTrainingSummary {
                target: target.clone(),
            }),
            None => Err(SparkError::value_msg(
                "No training summary available for this FMClassificationModel",
            )),
        }
    }

    /// The summary of the model on `df` (`evaluate`), whose server-side summary this client does
    /// not release, as PySpark does not release it.
    pub fn evaluate(&self, df: &DataFrame) -> Result<FMClassificationSummary> {
        let target = self.model_target()?;
        let id = fetch(&target, "evaluate", vec![FetchArg::Input(df.clone())])?.into_string()?;
        Ok(FMClassificationSummary {
            target: MlTarget::unowned(target.session(), &id),
        })
    }

    /// The model that `path` saves (`FMClassificationModel.load`), of the session `session`. A
    /// loaded model has no training summary.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (object, operator, params) = read_model(
            session,
            "org.apache.spark.ml.classification.FMClassificationModel",
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
                "org.apache.spark.ml.classification.FMClassificationModel",
                OperatorType::Model,
            ),
            params,
            cached: ml_ref,
            summary_target: None,
        }
    }
}

impl PredictionModel for FMClassificationModel {}

impl ClassificationModel for FMClassificationModel {}

impl ProbabilisticClassificationModel for FMClassificationModel {}

/// The summary of a factorization machines model on a dataset (`evaluate`), as PySpark's
/// `FMClassificationSummary`: a binary classification summary.
#[derive(Debug, Clone)]
pub struct FMClassificationSummary {
    target: MlTarget,
}

impl MlSummary for FMClassificationSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl ClassificationSummary for FMClassificationSummary {}

impl BinaryClassificationSummary for FMClassificationSummary {}

/// The training summary of a factorization machines model, as PySpark's
/// `FMClassificationTrainingSummary`.
#[derive(Debug, Clone)]
pub struct FMClassificationTrainingSummary {
    target: MlTarget,
}

impl MlSummary for FMClassificationTrainingSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl ClassificationSummary for FMClassificationTrainingSummary {}

impl BinaryClassificationSummary for FMClassificationTrainingSummary {}

impl TrainingSummary for FMClassificationTrainingSummary {}

ml_params! {
    FMClassifier {
        set_factor_size(value: i32) => "factorSize", set_param_int32;
        factor_size -> i32 = |this| this.params.int_or("factorSize", 8);
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        set_fit_intercept(value: bool) => "fitIntercept", set_param_bool;
        fit_intercept -> bool = |this| this.params.bool_or("fitIntercept", true);
        set_fit_linear(value: bool) => "fitLinear", set_param_bool;
        fit_linear -> bool = |this| this.params.bool_or("fitLinear", true);
        set_init_std(value: f64) => "initStd", set_param_double;
        init_std -> f64 = |this| this.params.double_or("initStd", 0.01);
        set_label_col(value: &str) => "labelCol", set_param_string;
        label_col -> &str = |this| this.params.string_or("labelCol", "label");
        set_max_iter(value: i32) => "maxIter", set_param_int32;
        max_iter -> i32 = |this| this.params.int_or("maxIter", 100);
        set_mini_batch_fraction(value: f64) => "miniBatchFraction", set_param_double;
        mini_batch_fraction -> f64 = |this| this.params.double_or("miniBatchFraction", 1.0);
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_probability_col(value: &str) => "probabilityCol", set_param_string;
        probability_col -> &str = |this| this.params.string_or("probabilityCol", "probability");
        set_raw_prediction_col(value: &str) => "rawPredictionCol", set_param_string;
        raw_prediction_col -> &str =
            |this| this.params.string_or("rawPredictionCol", "rawPrediction");
        set_reg_param(value: f64) => "regParam", set_param_double;
        reg_param -> f64 = |this| this.params.double_or("regParam", 0.0);
        set_seed(value: i64) => "seed", set_param_int_narrow;
        seed -> i64 = |this| {
            this.params.long_opt("seed").unwrap_or(java_string_hash(
                "org.apache.spark.ml.classification.FMClassifier",
            ))
        };
        set_solver(value: &str) => "solver", set_param_string;
        solver -> &str = |this| this.params.string_or("solver", "adamW");
        set_step_size(value: f64) => "stepSize", set_param_double;
        step_size -> f64 = |this| this.params.double_or("stepSize", 1.0);
        set_thresholds(value: impl IntoIterator<Item = f64>) =>
            "thresholds", set_param_double_array;
        thresholds -> Option<Vec<f64>> = |this| this.params.double_array_opt("thresholds");
        set_tol(value: f64) => "tol", set_param_double;
        tol -> f64 = |this| this.params.double_or("tol", 1e-06);
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol") => "weightCol";
    }
}

ml_params! {
    FMClassificationModel {
        factor_size -> i32 = |this| this.params.int_or("factorSize", 8) => "factorSize";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        fit_intercept -> bool = |this| this.params.bool_or("fitIntercept", true) => "fitIntercept";
        fit_linear -> bool = |this| this.params.bool_or("fitLinear", true) => "fitLinear";
        init_std -> f64 = |this| this.params.double_or("initStd", 0.01) => "initStd";
        label_col -> &str = |this| this.params.string_or("labelCol", "label") => "labelCol";
        max_iter -> i32 = |this| this.params.int_or("maxIter", 100) => "maxIter";
        mini_batch_fraction -> f64 =
            |this| this.params.double_or("miniBatchFraction", 1.0) => "miniBatchFraction";
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_probability_col(value: &str) => "probabilityCol", set_param_string;
        probability_col -> &str = |this| this.params.string_or("probabilityCol", "probability");
        set_raw_prediction_col(value: &str) => "rawPredictionCol", set_param_string;
        raw_prediction_col -> &str =
            |this| this.params.string_or("rawPredictionCol", "rawPrediction");
        reg_param -> f64 = |this| this.params.double_or("regParam", 0.0) => "regParam";
        seed -> i64 = |this| {
            this.params.long_opt("seed").unwrap_or(java_string_hash(
                "org.apache.spark.ml.classification.FMClassificationModel",
            ))
        } => "seed";
        solver -> &str = |this| this.params.string_or("solver", "adamW") => "solver";
        step_size -> f64 = |this| this.params.double_or("stepSize", 1.0) => "stepSize";
        set_thresholds(value: impl IntoIterator<Item = f64>) =>
            "thresholds", set_param_double_array;
        thresholds -> Option<Vec<f64>> = |this| this.params.double_array_opt("thresholds");
        tol -> f64 = |this| this.params.double_or("tol", 1e-06) => "tol";
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol") => "weightCol";
    }
}
