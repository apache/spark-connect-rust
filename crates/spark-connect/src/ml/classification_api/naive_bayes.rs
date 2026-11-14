// SPDX-License-Identifier: Apache-2.0
//! The typed accessors, persistence and classification members of NaiveBayesModel. PySpark 4.3.0
//! gives the model no training summary, so its own members are the class parameters `pi`,
//! `theta` and `sigma`, and the prediction members of the classification models.

use super::{ClassificationModel, ProbabilisticClassificationModel};
use crate::ml::object::{read_model, MlValue};
use crate::ml::tree_api::{as_vector, fetch_value};
use crate::ml::{
    MlMatrix, MlOperator, MlRef, MlVector, Model, NaiveBayes, NaiveBayesModel, OperatorType,
};
use crate::ml::{Params, PredictionModel};
use crate::session::SparkSession;
use spark_connect_core::error::{Result, SparkError};

impl NaiveBayesModel {
    /// The log of the class priors (`pi`), one value per class.
    pub fn pi(&self) -> Result<MlVector> {
        as_vector("pi", fetch_value(&self.model_target()?, "pi", vec![])?)
    }

    /// The log of the conditional probabilities of the features, one row per class (`theta`).
    pub fn theta(&self) -> Result<MlMatrix> {
        as_matrix(
            "theta",
            fetch_value(&self.model_target()?, "theta", vec![])?,
        )
    }

    /// The variance of each feature of each class (`sigma`): a 0 by 0 matrix for a multinomial,
    /// complement or Bernoulli model.
    pub fn sigma(&self) -> Result<MlMatrix> {
        as_matrix(
            "sigma",
            fetch_value(&self.model_target()?, "sigma", vec![])?,
        )
    }

    /// The model that `path` saves (`NaiveBayesModel.load`), of the session `session`.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (object, operator, params) = read_model(
            session,
            "org.apache.spark.ml.classification.NaiveBayesModel",
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
                "org.apache.spark.ml.classification.NaiveBayesModel",
                OperatorType::Model,
            ),
            params,
            cached: ml_ref,
        }
    }
}

impl PredictionModel for NaiveBayesModel {}

impl ClassificationModel for NaiveBayesModel {}

impl ProbabilisticClassificationModel for NaiveBayesModel {}

/// The matrix `value` of `method`, as PySpark reads a `Matrix` of `deserialize_param`.
pub(super) fn as_matrix(method: &str, value: MlValue) -> Result<MlMatrix> {
    value.as_matrix().cloned().ok_or_else(|| {
        SparkError::value_msg(format!("expected a matrix of `{method}`, got {value:?}"))
    })
}

ml_params! {
    NaiveBayes {
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        set_label_col(value: &str) => "labelCol", set_param_string;
        label_col -> &str = |this| this.params.string_or("labelCol", "label");
        set_model_type(value: &str) => "modelType", set_param_string;
        model_type -> &str = |this| this.params.string_or("modelType", "multinomial");
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_probability_col(value: &str) => "probabilityCol", set_param_string;
        probability_col -> &str = |this| this.params.string_or("probabilityCol", "probability");
        set_raw_prediction_col(value: &str) => "rawPredictionCol", set_param_string;
        raw_prediction_col -> &str =
            |this| this.params.string_or("rawPredictionCol", "rawPrediction");
        set_smoothing(value: f64) => "smoothing", set_param_double;
        smoothing -> f64 = |this| this.params.double_or("smoothing", 1.0);
        set_thresholds(value: impl IntoIterator<Item = f64>) =>
            "thresholds", set_param_double_array;
        thresholds -> Option<Vec<f64>> = |this| this.params.double_array_opt("thresholds");
        set_weight_col(value: &str) => "weightCol", set_param_string;
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol");
    }
}

ml_params! {
    NaiveBayesModel {
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        label_col -> &str = |this| this.params.string_or("labelCol", "label") => "labelCol";
        model_type -> &str =
            |this| this.params.string_or("modelType", "multinomial") => "modelType";
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_probability_col(value: &str) => "probabilityCol", set_param_string;
        probability_col -> &str = |this| this.params.string_or("probabilityCol", "probability");
        set_raw_prediction_col(value: &str) => "rawPredictionCol", set_param_string;
        raw_prediction_col -> &str =
            |this| this.params.string_or("rawPredictionCol", "rawPrediction");
        smoothing -> f64 = |this| this.params.double_or("smoothing", 1.0) => "smoothing";
        set_thresholds(value: impl IntoIterator<Item = f64>) =>
            "thresholds", set_param_double_array;
        thresholds -> Option<Vec<f64>> = |this| this.params.double_array_opt("thresholds");
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol") => "weightCol";
    }
}
