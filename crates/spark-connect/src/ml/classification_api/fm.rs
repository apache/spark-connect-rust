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
    FMClassificationModel, MlMatrix, MlOperator, MlRef, MlVector, Model, OperatorType, Params,
    PredictionModel,
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
