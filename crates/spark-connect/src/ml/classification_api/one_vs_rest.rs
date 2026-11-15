// SPDX-License-Identifier: Apache-2.0
//! The typed accessors of OneVsRestModel. Its fit and transform run in the reduction of
//! [`crate::ml::OneVsRest`] and [`crate::ml::OneVsRestModel`]. Their persistence is PySpark's
//! Python format (`OneVsRestWriter`, `OneVsRestModelWriter` and their readers, see the
//! persistence section at the end of this file): what each of PySpark and this client saves, the
//! other loads.

use crate::ml::meta::{
    json_to_ml_value, load_python_metadata, metadata_item, metadata_str, ml_value_json,
    ONE_VS_REST_CLASS, ONE_VS_REST_MODEL_CLASS,
};
use crate::ml::{
    join_path, json_object, load_stage, save_metadata, Classifier, DecisionTreeClassificationModel,
    DecisionTreeClassifier, FMClassificationModel, FMClassifier, GBTClassificationModel,
    GBTClassifier, LinearSVC, LinearSVCModel, LogisticRegression, LogisticRegressionModel,
    MetaSave, Metadata, MlOperator, MlValue, MlWriter, Model,
    MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier, NaiveBayes,
    NaiveBayesModel, OneVsRest, OneVsRestModel, OperatorType, Params, PipelineStage,
    RandomForestClassificationModel, RandomForestClassifier,
};
use crate::session::SparkSession;
use spark_connect_core::error::{Result, SparkError};
use std::collections::HashMap;
use std::sync::Arc;

/// A binary submodel of a OneVsRestModel (`models`): its typed model, as the classifier that
/// fitted it, or the boxed model of a classifier without a typed model.
pub enum SubModel {
    LogisticRegression(LogisticRegressionModel),
    LinearSVC(LinearSVCModel),
    DecisionTreeClassification(DecisionTreeClassificationModel),
    RandomForestClassification(RandomForestClassificationModel),
    GBTClassification(GBTClassificationModel),
    NaiveBayes(NaiveBayesModel),
    FMClassification(FMClassificationModel),
    MultilayerPerceptronClassification(MultilayerPerceptronClassificationModel),
    Other(Box<dyn Model>),
}

impl SubModel {
    /// The typed submodel of `model`, which the classifier of a OneVsRest fit returned.
    fn of(model: Box<dyn Model>) -> Self {
        if let Some(m) = model.downcast_ref::<LogisticRegressionModel>() {
            return Self::LogisticRegression(m.clone());
        }
        if let Some(m) = model.downcast_ref::<LinearSVCModel>() {
            return Self::LinearSVC(m.clone());
        }
        if let Some(m) = model.downcast_ref::<DecisionTreeClassificationModel>() {
            return Self::DecisionTreeClassification(m.clone());
        }
        if let Some(m) = model.downcast_ref::<RandomForestClassificationModel>() {
            return Self::RandomForestClassification(m.clone());
        }
        if let Some(m) = model.downcast_ref::<GBTClassificationModel>() {
            return Self::GBTClassification(m.clone());
        }
        if let Some(m) = model.downcast_ref::<NaiveBayesModel>() {
            return Self::NaiveBayes(m.clone());
        }
        if let Some(m) = model.downcast_ref::<FMClassificationModel>() {
            return Self::FMClassification(m.clone());
        }
        if let Some(m) = model.downcast_ref::<MultilayerPerceptronClassificationModel>() {
            return Self::MultilayerPerceptronClassification(m.clone());
        }
        Self::Other(model)
    }
}

impl OneVsRestModel {
    /// The binary submodels, one per class, in the order of the class index (`models`).
    pub fn models(&self) -> Vec<SubModel> {
        self.models
            .iter()
            .map(|model| SubModel::of(model.clone_box()))
            .collect()
    }
}

ml_params! {
    OneVsRest {
        classifier -> Option<&Classifier> = |this| this.classifier.as_ref() => "classifier";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        set_label_col(value: &str) => "labelCol", set_param_string;
        label_col -> &str = |this| this.params.string_or("labelCol", "label");
        set_parallelism(value: i32) => "parallelism", set_param_int32;
        parallelism -> i32 = |this| this.params.int_or("parallelism", 1);
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_raw_prediction_col(value: &str) => "rawPredictionCol", set_param_string;
        raw_prediction_col -> &str =
            |this| this.params.string_or("rawPredictionCol", "rawPrediction");
        set_weight_col(value: &str) => "weightCol", set_param_string;
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol");
    }
}

ml_params! {
    OneVsRestModel {
        classifier -> Option<&Classifier> = |this| this.classifier.as_ref() => "classifier";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> Option<&str> = |this| this.params.string_opt("featuresCol");
        label_col -> Option<&str> = |this| this.params.string_opt("labelCol") => "labelCol";
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> Option<&str> = |this| this.params.string_opt("predictionCol");
        set_raw_prediction_col(value: &str) => "rawPredictionCol", set_param_string;
        raw_prediction_col -> Option<&str> = |this| this.params.string_opt("rawPredictionCol");
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol") => "weightCol";
    }
}

// ---------------------------------------------------------------------------------------------
// Persistence, as `OneVsRestWriter`, `OneVsRestModelWriter` and their readers of PySpark.
// ---------------------------------------------------------------------------------------------

/// The defaults of the params of a OneVsRest, as JSON texts, in the order of PySpark's
/// `_defaultParamMap` of the estimator.
const ESTIMATOR_DEFAULTS: &[(&str, &str)] = &[
    ("parallelism", "1"),
    ("predictionCol", "\"prediction\""),
    ("featuresCol", "\"features\""),
    ("labelCol", "\"label\""),
    ("rawPredictionCol", "\"rawPrediction\""),
];

/// The defaults of the params of a OneVsRestModel, as JSON texts, in the order of PySpark's
/// `_defaultParamMap` of the model: the estimator's, but for `parallelism`, which a model has not.
const MODEL_DEFAULTS: &[(&str, &str)] = &[
    ("predictionCol", "\"prediction\""),
    ("featuresCol", "\"features\""),
    ("labelCol", "\"label\""),
    ("rawPredictionCol", "\"rawPrediction\""),
];

/// The JSON text of the literal of a param, as `json.dumps` writes its value.
fn expression_json(expr: &spark_connect_proto::Expression) -> Result<String> {
    use spark_connect_proto::expression::ExprType;
    match &expr.expr_type {
        Some(ExprType::Literal(literal)) => ml_value_json(&MlValue::from_literal(literal)?),
        _ => Err(SparkError::value_msg(
            "a param of a OneVsRest is no literal value",
        )),
    }
}

/// The JSON text of the `paramMap` of a OneVsRest or its model, as `extractJsonParams` writes it
/// (without the classifier): of each of `defaults`, in its order, the set value or the default,
/// and then the set `weightCol`, which has no default.
fn param_map_json(params: &Params, defaults: &[(&str, &str)]) -> Result<String> {
    let mut fields = Vec::with_capacity(defaults.len() + 1);
    for (name, default) in defaults {
        let value = match params.get_param(name) {
            Some(expr) => expression_json(expr)?,
            None => default.to_string(),
        };
        fields.push((*name, value));
    }
    if let Some(expr) = params.get_param("weightCol") {
        fields.push(("weightCol", expression_json(expr)?));
    }
    Ok(json_object(&fields))
}

/// The JSON text of the `defaultParamMap` of a OneVsRest or its model: of `defaults`.
fn default_param_map_json(defaults: &[(&str, &str)]) -> String {
    let fields: Vec<(&str, String)> = defaults
        .iter()
        .map(|(name, value)| (*name, value.to_string()))
        .collect();
    json_object(&fields)
}

/// The save of a OneVsRest (its `models` are `None`) or of its model (of `models`), as
/// `_OneVsRestSharedReadWrite.saveImpl`: the metadata of its params, without the classifier, which
/// the classifier's own writer saves at `classifier`; a model also writes `numClasses` in its
/// metadata, and each of its models at `model_<index>`.
struct OneVsRestSave {
    class: &'static str,
    uid: String,
    params: Params,
    defaults: &'static [(&'static str, &'static str)],
    classifier: Option<Classifier>,
    models: Option<Vec<Box<dyn Model>>>,
}

impl MetaSave for OneVsRestSave {
    fn name(&self) -> String {
        self.uid.clone()
    }

    fn save_meta(
        &self,
        session: &SparkSession,
        path: &str,
        _options: &HashMap<String, String>,
    ) -> Result<()> {
        // As `validateParams`, before anything is written: the classifier and each model write.
        let classifier = self
            .classifier
            .as_ref()
            .ok_or_else(|| SparkError::value_msg("OneVsRest requires a native classifier"))?
            .estimator()
            .write();
        let model_writers = self
            .models
            .as_ref()
            .map(|models| {
                models
                    .iter()
                    .map(|model| model.write())
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let mut metadata = Metadata::new(
            self.class,
            &self.uid,
            &session.version()?,
            param_map_json(&self.params, self.defaults)?,
            default_param_map_json(self.defaults),
        );
        if let Some(models) = &self.models {
            metadata
                .extra
                .push(("numClasses".to_string(), models.len().to_string()));
        }
        save_metadata(session, path, &metadata.to_json())?;
        classifier
            .session(session)
            .save(&join_path(path, "classifier"))?;
        for (index, writer) in model_writers.into_iter().flatten().enumerate() {
            writer
                .session(session)
                .save(&join_path(path, &format!("model_{index}")))?;
        }
        Ok(())
    }
}

/// The typed classifier of the operator `operator` of the params `params`, which a load of the
/// classifier's own format reads: one of the native binary classifiers that OneVsRest fits.
fn typed_classifier(operator: MlOperator, params: Params) -> Result<Classifier> {
    let name = operator.name.clone();
    let classifier: Classifier = match name.as_str() {
        "org.apache.spark.ml.classification.LogisticRegression" => {
            LogisticRegression { operator, params }.into()
        }
        "org.apache.spark.ml.classification.LinearSVC" => LinearSVC { operator, params }.into(),
        "org.apache.spark.ml.classification.DecisionTreeClassifier" => {
            DecisionTreeClassifier { operator, params }.into()
        }
        "org.apache.spark.ml.classification.RandomForestClassifier" => {
            RandomForestClassifier { operator, params }.into()
        }
        "org.apache.spark.ml.classification.GBTClassifier" => {
            GBTClassifier { operator, params }.into()
        }
        "org.apache.spark.ml.classification.NaiveBayes" => NaiveBayes { operator, params }.into(),
        "org.apache.spark.ml.classification.FMClassifier" => {
            FMClassifier { operator, params }.into()
        }
        "org.apache.spark.ml.classification.MultilayerPerceptronClassifier" => {
            MultilayerPerceptronClassifier { operator, params }.into()
        }
        other => {
            return Err(SparkError::value_msg(format!(
                "the classifier {other} is no native binary classifier that OneVsRest fits"
            )))
        }
    };
    Ok(classifier)
}

/// The classifier saved at `<path>/classifier`, of its own format, as
/// `_OneVsRestSharedReadWrite.loadClassifier` loads it.
fn load_classifier(session: &SparkSession, path: &str) -> Result<Classifier> {
    match load_stage(session, &join_path(path, "classifier"))? {
        PipelineStage::Estimator(estimator) => {
            typed_classifier(estimator.operator().clone(), estimator.params().clone())
        }
        other => Err(SparkError::value_msg(format!(
            "the classifier of {path} is no estimator: {other:?}"
        ))),
    }
}

/// Whether the metadata `metadata` is of the class `expected`, as `_parseMetaData` checks it.
fn check_class(metadata: &serde_json::Value, expected: &str) -> Result<()> {
    let class = metadata_str(metadata, "class")?;
    if class != expected {
        return Err(SparkError::value_msg(format!(
            "Error loading metadata: Expected class name {expected} but found class name {class}"
        )));
    }
    Ok(())
}

/// The params of the `paramMap` of the metadata `metadata`, set as `getAndSetParams` sets them,
/// without the classifier.
fn read_params(metadata: &serde_json::Value) -> Result<Params> {
    let map = metadata_item(metadata, "paramMap")?
        .as_object()
        .ok_or_else(|| SparkError::value_msg("the paramMap of the metadata is no object"))?;
    let mut params = Params::new();
    for (name, value) in map {
        if name != "classifier" {
            params = params.set_param_literal(name, json_to_ml_value(value).to_literal()?);
        }
    }
    Ok(params)
}

impl OneVsRest {
    /// The writer of the estimator in PySpark's Python format, as `OneVsRestWriter`.
    pub(crate) fn writer(&self) -> MlWriter {
        MlWriter::meta(Arc::new(OneVsRestSave {
            class: ONE_VS_REST_CLASS,
            uid: self.operator.uid.clone(),
            params: self.params.clone(),
            defaults: ESTIMATOR_DEFAULTS,
            classifier: self.classifier.clone(),
            models: None,
        }))
    }

    /// Load the estimator saved at `path` in PySpark's Python format, as `OneVsRestReader.load`:
    /// its uid, its params and its classifier.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let metadata = load_python_metadata(session, path)?;
        check_class(&metadata, ONE_VS_REST_CLASS)?;
        Ok(Self {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.classification.OneVsRest",
                metadata_str(&metadata, "uid")?,
                OperatorType::Estimator,
            ),
            params: read_params(&metadata)?,
            classifier: Some(load_classifier(session, path)?),
        })
    }
}

impl OneVsRestModel {
    /// The writer of the model in PySpark's Python format, as `OneVsRestModelWriter`.
    pub(crate) fn writer(&self) -> MlWriter {
        MlWriter::meta(Arc::new(OneVsRestSave {
            class: ONE_VS_REST_MODEL_CLASS,
            uid: self.operator.uid.clone(),
            params: self.params.clone(),
            defaults: MODEL_DEFAULTS,
            classifier: self.classifier.clone(),
            models: Some(self.models.iter().map(|model| model.clone_box()).collect()),
        }))
    }

    /// The models of the classes, as the stages that they are (`models` of PySpark's model), for
    /// the uids of a validator's nested stages.
    pub(crate) fn stage_models(&self) -> &[Box<dyn Model>] {
        &self.models
    }

    /// Load the model saved at `path` in PySpark's Python format, as `OneVsRestModelReader.load`:
    /// its uid, its params, its classifier and its `numClasses` models.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let metadata = load_python_metadata(session, path)?;
        check_class(&metadata, ONE_VS_REST_MODEL_CLASS)?;
        let classifier = load_classifier(session, path)?;
        let count = metadata_item(&metadata, "numClasses")?
            .as_u64()
            .ok_or_else(|| SparkError::value_msg("numClasses of the metadata is no int"))?;
        let mut models = Vec::with_capacity(count as usize);
        for index in 0..count {
            match load_stage(session, &join_path(path, &format!("model_{index}")))? {
                PipelineStage::Model(model) => models.push(model),
                other => {
                    return Err(SparkError::value_msg(format!(
                        "the model_{index} of {path} is no model: {other:?}"
                    )))
                }
            }
        }
        Ok(Self {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.classification.OneVsRestModel",
                metadata_str(&metadata, "uid")?,
                OperatorType::Model,
            ),
            params: read_params(&metadata)?,
            models,
            session_id: session.session_id().to_string(),
            classifier: Some(classifier),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::{Estimator, ParamMap};

    #[test]
    fn the_param_maps_are_pysparks_defaults_in_their_order_then_the_set_weight() {
        assert_eq!(
            param_map_json(&Params::new(), ESTIMATOR_DEFAULTS).unwrap(),
            "{\"parallelism\":1,\"predictionCol\":\"prediction\",\"featuresCol\":\"features\",\
             \"labelCol\":\"label\",\"rawPredictionCol\":\"rawPrediction\"}"
        );
        assert_eq!(
            default_param_map_json(ESTIMATOR_DEFAULTS),
            param_map_json(&Params::new(), ESTIMATOR_DEFAULTS).unwrap()
        );
        // A set param replaces its default in its place, and a set weight comes after them all.
        let params = Params::new()
            .set_param_int32("parallelism", 3)
            .set_param_string("weightCol", "w");
        let json = param_map_json(&params, ESTIMATOR_DEFAULTS).unwrap();
        assert!(
            json.starts_with("{\"parallelism\":3,\"predictionCol\""),
            "{json}"
        );
        assert!(json.ends_with(",\"weightCol\":\"w\"}"), "{json}");
        // A null weight is written as null, and a model has no parallelism.
        let params = Params::new()
            .set_param_null("weightCol")
            .set_param_int32("parallelism", 3);
        let json = param_map_json(&params, MODEL_DEFAULTS).unwrap();
        assert_eq!(
            json,
            "{\"predictionCol\":\"prediction\",\"featuresCol\":\"features\",\
             \"labelCol\":\"label\",\"rawPredictionCol\":\"rawPrediction\",\
             \"weightCol\":null}"
        );
    }

    #[test]
    fn the_classifier_of_a_metadata_is_one_of_the_native_binary_classifiers() {
        let operator = |name: &str| {
            MlOperator::with_uid(
                name,
                "LogisticRegression_0123456789ab",
                OperatorType::Estimator,
            )
        };
        let classifier = typed_classifier(
            operator("org.apache.spark.ml.classification.LogisticRegression"),
            Params::new(),
        )
        .unwrap();
        assert!(matches!(classifier, Classifier::LogisticRegression(_)));
        assert_eq!(
            classifier.estimator().operator().uid,
            "LogisticRegression_0123456789ab"
        );
        let error = typed_classifier(
            operator("org.apache.spark.ml.regression.GBTRegressor"),
            Params::new(),
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("no native binary classifier"),
            "{error}"
        );
    }

    #[test]
    fn a_param_map_of_a_one_vs_rest_sets_the_params_of_its_classifier() {
        // As `OneVsRest.copy(extra)`, which copies the classifier with `extra`.
        let classifier = LinearSVC::new();
        let uid = Estimator::operator(&classifier).uid.clone();
        let mut estimator = OneVsRest::new().set_classifier(classifier);
        let map = ParamMap::new().put(&uid, "maxIter", MlValue::Int(7));
        estimator.apply_param_map(&map).unwrap();
        let Some(classifier) = estimator.classifier() else {
            panic!("a classifier");
        };
        let value = classifier.estimator().params().get_param("maxIter");
        assert!(format!("{value:?}").contains("Integer(7)"), "{value:?}");
    }
}
