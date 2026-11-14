// SPDX-License-Identifier: Apache-2.0
//! The typed accessors, summaries and persistence of LinearSVC and LinearSVCModel, as
//! `pyspark.ml.classification` of PySpark 4.3.0 has them.
//!
//! A model that a fit returns holds the target of its training summary (`_summary`), so that
//! `has_summary` is client side. Its summary is always the training summary, and its dataset is
//! the training dataset itself, as `_summary_dataset` of PySpark returns it.

use crate::dataframe::DataFrame;
use crate::ml::tree_api::{as_f64, as_vector, fetch_value};
use crate::ml::{
    fetch, read_model, read_operator, BinaryClassificationSummary, ClassificationModel,
    ClassificationSummary, Estimator, FetchArg, LinearSVC, LinearSVCModel, MlOperator, MlRef,
    MlSummary, MlTarget, MlVector, Model, OperatorType, Params, PredictionModel, TrainingSummary,
};
use crate::session::SparkSession;
use spark_connect_core::error::{Result, SparkError};

/// The JVM class of the estimator, as PySpark saves and reads it.
const LINEAR_SVC: &str = "org.apache.spark.ml.classification.LinearSVC";
/// The JVM class of the model.
const LINEAR_SVC_MODEL: &str = "org.apache.spark.ml.classification.LinearSVCModel";

/// The summary of a LinearSVCModel (`LinearSVCSummary`), which is binary.
#[derive(Debug, Clone)]
pub struct LinearSVCSummary {
    target: MlTarget,
}

impl LinearSVCSummary {
    /// The summary of the target `target`, of a summary that the server holds.
    pub(crate) fn from_target(target: MlTarget) -> Self {
        Self { target }
    }
}

impl MlSummary for LinearSVCSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl ClassificationSummary for LinearSVCSummary {}
impl BinaryClassificationSummary for LinearSVCSummary {}

/// The training summary of a LinearSVCModel (`LinearSVCTrainingSummary`).
#[derive(Debug, Clone)]
pub struct LinearSVCTrainingSummary {
    target: MlTarget,
}

impl LinearSVCTrainingSummary {
    /// The summary of the target `target`, of a summary that the server holds.
    pub(crate) fn from_target(target: MlTarget) -> Self {
        Self { target }
    }
}

impl MlSummary for LinearSVCTrainingSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl ClassificationSummary for LinearSVCTrainingSummary {}
impl BinaryClassificationSummary for LinearSVCTrainingSummary {}
impl TrainingSummary for LinearSVCTrainingSummary {}

impl LinearSVC {
    /// Fit the typed model, which holds the target of its training summary (`_summary` of
    /// PySpark's `_fit`). The summary's dataset is the training dataset itself.
    pub fn fit(&mut self, df: &DataFrame) -> Result<LinearSVCModel> {
        self.ensure_operator();
        let info = crate::ml::fit_via_command(&self.operator, &self.params, df)?;
        let mut model = self.model_from_info(info, &df.session)?;
        model.summary = Some(
            model
                .model_target()?
                .child("summary")
                .with_summary_dataset(df),
        );
        Ok(model)
    }

    /// Fit one model per param map, in order: each a copy of this estimator with the map's params
    /// set over its own, as `Estimator.fitMultiple` of PySpark does with `copy` and `fit`.
    pub fn fit_multiple(
        &self,
        df: &DataFrame,
        param_maps: &[Params],
    ) -> Result<Vec<LinearSVCModel>> {
        param_maps
            .iter()
            .map(|map| {
                let mut estimator = self.clone();
                estimator.params.params.extend(map.params.clone());
                estimator.fit(df)
            })
            .collect()
    }

    /// The estimator that `write().save(path)` saved, as `LinearSVC.load` of PySpark reads it.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (operator, params) = read_operator(session, LINEAR_SVC, OperatorType::Estimator, path)?;
        Ok(Self { operator, params })
    }
}

impl LinearSVCModel {
    /// The model of the server-side object `ml_ref` with the params `params`, of no training
    /// summary, as a read that returns a model gives it.
    pub(crate) fn from_ml_ref(ml_ref: MlRef, params: Params) -> Self {
        Self {
            operator: MlOperator::new(LINEAR_SVC_MODEL, OperatorType::Model),
            params,
            cached: ml_ref,
            summary: None,
        }
    }

    /// The model that `write().save(path)` saved, as `LinearSVCModel.load` of PySpark reads it.
    /// A loaded model has no training summary.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (object, operator, params) = read_model(session, LINEAR_SVC_MODEL, path)?;
        let mut model = Self::from_ml_ref(object, params);
        model.operator = operator;
        Ok(model)
    }

    /// The coefficients of the model (`coefficients`).
    pub fn coefficients(&self) -> Result<MlVector> {
        as_vector(
            "coefficients",
            fetch_value(&self.model_target()?, "coefficients", vec![])?,
        )
    }

    /// The intercept of the model (`intercept`).
    pub fn intercept(&self) -> Result<f64> {
        as_f64(
            "intercept",
            fetch_value(&self.model_target()?, "intercept", vec![])?,
        )
    }

    /// Whether the model has a training summary, which a fit gives and a loaded model lacks
    /// (`hasSummary`). It is client side, as PySpark's is under Spark Connect.
    pub fn has_summary(&self) -> bool {
        self.summary.is_some()
    }

    /// The training summary of the model (`summary()`, a method in PySpark 4.3.0). An error when
    /// the model has no training summary.
    pub fn summary(&self) -> Result<LinearSVCTrainingSummary> {
        let target = self.summary.clone().ok_or_else(|| {
            SparkError::value_msg("No training summary available for this LinearSVCModel")
        })?;
        Ok(LinearSVCTrainingSummary::from_target(target))
    }

    /// The summary of the model on `df` (`evaluate`). The summary is the server's own object,
    /// which this client does not release, as PySpark does not release it.
    pub fn evaluate(&self, df: &DataFrame) -> Result<LinearSVCSummary> {
        let target = self.model_target()?;
        let id = fetch(&target, "evaluate", vec![FetchArg::Input(df.clone())])?.into_string()?;
        Ok(LinearSVCSummary::from_target(MlTarget::unowned(
            target.session(),
            &id,
        )))
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
}

impl PredictionModel for LinearSVCModel {}
impl ClassificationModel for LinearSVCModel {}
