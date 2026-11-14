// SPDX-License-Identifier: Apache-2.0
//! The typed accessors, persistence and classification members of NaiveBayesModel. PySpark 4.3.0
//! gives the model no training summary, so its own members are the class parameters `pi`,
//! `theta` and `sigma`, and the prediction members of the classification models.

use super::{ClassificationModel, ProbabilisticClassificationModel};
use crate::ml::object::{read_model, MlValue};
use crate::ml::tree_api::{as_vector, fetch_value};
use crate::ml::{MlMatrix, MlOperator, MlRef, MlVector, Model, NaiveBayesModel, OperatorType};
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
