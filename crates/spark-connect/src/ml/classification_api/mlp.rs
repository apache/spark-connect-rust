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
    MlOperator, MlRef, MlVector, Model, MultilayerPerceptronClassificationModel, OperatorType,
    Params, PredictionModel,
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
