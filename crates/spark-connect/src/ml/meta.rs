// SPDX-License-Identifier: Apache-2.0
//! PySpark's meta-algorithms of `pyspark.ml` in the Rust API, as PySpark 4.3.0 runs them over
//! Spark Connect, on the client: [`Pipeline`] and [`PipelineModel`] of `pyspark.ml.pipeline`, and
//! the validators [`CrossValidator`] and [`TrainValidationSplit`] of `pyspark.ml.tuning` and their
//! models, of any estimators, transformers and evaluators of this crate (or of the user's).
//! - **Param maps** ([`ParamMap`], [`ParamGridBuilder`]) are of the uid of a stage and the name
//!   of its param, as PySpark's `Param.parent` and `Param.name`; [`Estimator::apply_param_map`]
//!   sets them, as a copy of PySpark of its `extra` (a meta-algorithm in each of its stages).
//! - **Fits** are PySpark's `_fit`: a pipeline fits and transforms its stages in order; a
//!   validator splits its dataset ([`k_fold`], [`train_validation_split`]), fits a copy of its
//!   estimator of each param map on each training set in `parallelism` threads, evaluates each on
//!   its validation set, and fits the best map on the whole dataset.
//! - **Persistence** is of PySpark's Python format, so that PySpark loads what this API saves and
//!   this API loads what PySpark saves: the metadata of `DefaultParamsWriter` and each stage by its
//!   own writer ([`Pipeline::load`], [`load_stage`]).
use super::stage_kinds::STAGE_KINDS;
use super::*;
use spark_connect_core::error::Result;
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------------------------
// Param maps.
// ---------------------------------------------------------------------------------------------

/// The value of a param of a stage in a [`ParamMap`]: the uid of the stage (PySpark's
/// `Param.parent`) and the name of the param of the server's class.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamPair {
    pub parent: String,
    pub name: String,
    pub value: MlValue,
}

/// A param map, as PySpark's (a dict of Params): the values of the params of stages, in the order
/// of their insertion, each param of each stage once.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParamMap {
    pairs: Vec<ParamPair>,
}

impl ParamMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// The map with the param `name` of the stage of the uid `parent` of `value`, in place of a
    /// value that it has, as a dict sets a key.
    pub fn put(mut self, parent: &str, name: &str, value: MlValue) -> Self {
        self.insert(parent, name, value);
        self
    }

    /// Set the param `name` of the stage of the uid `parent` to `value` (see [`Self::put`]).
    pub fn insert(&mut self, parent: &str, name: &str, value: MlValue) {
        match self
            .pairs
            .iter_mut()
            .find(|p| p.parent == parent && p.name == name)
        {
            Some(pair) => pair.value = value,
            None => self.pairs.push(ParamPair {
                parent: parent.to_string(),
                name: name.to_string(),
                value,
            }),
        }
    }

    /// The value of the param `name` of the stage of the uid `parent`.
    pub fn get(&self, parent: &str, name: &str) -> Option<&MlValue> {
        self.pairs
            .iter()
            .find(|p| p.parent == parent && p.name == name)
            .map(|p| &p.value)
    }

    /// The values, in the order of their insertion.
    pub fn pairs(&self) -> &[ParamPair] {
        &self.pairs
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// The values of the params of the stage of the uid `uid`.
    pub fn of_stage<'a>(&'a self, uid: &'a str) -> impl Iterator<Item = &'a ParamPair> + 'a {
        self.pairs.iter().filter(move |p| p.parent == uid)
    }

    /// Whether the map has a param of the stage of the uid `uid`.
    pub fn has_stage(&self, uid: &str) -> bool {
        self.of_stage(uid).next().is_some()
    }
}

/// PySpark's `ParamGridBuilder`: a grid of the values of params, whose param maps
/// ([`Self::build`]) are each combination of them, as `itertools.product` of the values of the
/// params in their order (the last param varies fastest).
#[derive(Debug, Clone, Default)]
pub struct ParamGridBuilder {
    grid: Vec<(String, String, Vec<MlValue>)>,
}

impl ParamGridBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// The values of the param `name` of the stage of the uid `parent`, as `addGrid`: in place of
    /// the values that it has, as a dict sets a key.
    pub fn add_grid(
        mut self,
        parent: &str,
        name: &str,
        values: impl IntoIterator<Item = MlValue>,
    ) -> Self {
        let values: Vec<MlValue> = values.into_iter().collect();
        match self
            .grid
            .iter_mut()
            .find(|(p, n, _)| p == parent && n == name)
        {
            Some((_, _, old)) => *old = values,
            None => self
                .grid
                .push((parent.to_string(), name.to_string(), values)),
        }
        self
    }

    /// Each value of `map` as the only value of its param, as `baseOn`.
    pub fn base_on(mut self, map: &ParamMap) -> Self {
        for pair in map.pairs() {
            self = self.add_grid(&pair.parent, &pair.name, [pair.value.clone()]);
        }
        self
    }

    /// Every combination of the values of the params, as `build`: one empty map of no params.
    pub fn build(&self) -> Vec<ParamMap> {
        let mut maps = vec![ParamMap::new()];
        for (parent, name, values) in &self.grid {
            maps = maps
                .iter()
                .flat_map(|map| {
                    values
                        .iter()
                        .map(move |value| map.clone().put(parent, name, value.clone()))
                })
                .collect();
        }
        maps
    }
}

/// Set the values of `map` of the params of the stage of the uid `uid` in `params`.
pub(super) fn apply_pairs(params: &mut Params, uid: &str, map: &ParamMap) -> Result<()> {
    for pair in map.of_stage(uid) {
        let literal = pair.value.to_literal()?;
        let taken = std::mem::replace(params, Params::new());
        *params = taken.set_param_literal(&pair.name, literal);
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Stages.
// ---------------------------------------------------------------------------------------------

/// An estimator of a meta-algorithm: an [`Estimator`] that clones into a boxed stage, as a
/// meta-algorithm copies its estimator.
pub trait EstimatorStage: Estimator {
    fn clone_stage(&self) -> Box<dyn EstimatorStage>;

    /// The estimator as its concrete type, of which a meta-algorithm reads its nested stages.
    fn as_any_stage(&self) -> Option<&dyn std::any::Any>;
}

impl<T: Estimator + Clone + 'static> EstimatorStage for T {
    fn clone_stage(&self) -> Box<dyn EstimatorStage> {
        Box::new(self.clone())
    }

    fn as_any_stage(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

impl Clone for Box<dyn Model> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl std::fmt::Debug for dyn Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Model({}, {})",
            self.operator().name,
            self.operator().uid
        )
    }
}

/// A transformer of a pipeline that is no model (see [`EstimatorStage`]).
pub trait TransformerStage: Transformer {
    fn clone_stage(&self) -> Box<dyn TransformerStage>;
}

impl<T: Transformer + Clone + 'static> TransformerStage for T {
    fn clone_stage(&self) -> Box<dyn TransformerStage> {
        Box::new(self.clone())
    }
}

/// The evaluator of a validator (see [`EstimatorStage`]).
pub trait EvaluatorStage: Evaluator {
    fn clone_stage(&self) -> Box<dyn EvaluatorStage>;
}

impl<T: Evaluator + Clone + 'static> EvaluatorStage for T {
    fn clone_stage(&self) -> Box<dyn EvaluatorStage> {
        Box::new(self.clone())
    }
}

/// A stage of a pipeline: an estimator, which a fit fits, a transformer, or a fitted model.
pub enum PipelineStage {
    Estimator(Box<dyn EstimatorStage>),
    Transformer(Box<dyn TransformerStage>),
    Model(Box<dyn Model>),
}

impl Clone for PipelineStage {
    fn clone(&self) -> Self {
        match self {
            Self::Estimator(e) => Self::Estimator(e.clone_stage()),
            Self::Transformer(t) => Self::Transformer(t.clone_stage()),
            Self::Model(m) => Self::Model(m.clone_box()),
        }
    }
}

impl std::fmt::Debug for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (kind, operator) = match self {
            Self::Estimator(e) => ("Estimator", e.operator()),
            Self::Transformer(t) => ("Transformer", t.operator()),
            Self::Model(m) => ("Model", m.operator()),
        };
        write!(f, "{kind}({}, {})", operator.name, operator.uid)
    }
}

impl PipelineStage {
    /// The stage of the estimator `estimator`.
    pub fn estimator(estimator: impl EstimatorStage + 'static) -> Self {
        Self::Estimator(Box::new(estimator))
    }

    /// The stage of the transformer `transformer`, which is no model.
    pub fn transformer(transformer: impl TransformerStage + 'static) -> Self {
        Self::Transformer(Box::new(transformer))
    }

    /// The stage of the fitted model `model`.
    pub fn model(model: Box<dyn Model>) -> Self {
        Self::Model(model)
    }

    /// The operator of the stage (its class and uid).
    pub fn operator(&self) -> &MlOperator {
        match self {
            Self::Estimator(e) => e.operator(),
            Self::Transformer(t) => t.operator(),
            Self::Model(m) => m.operator(),
        }
    }

    /// The uid of the stage, as PySpark's `stage.uid`.
    pub fn uid(&self) -> &str {
        &self.operator().uid
    }

    /// Whether the stage is an estimator, which a pipeline fits.
    pub fn is_estimator(&self) -> bool {
        matches!(self, Self::Estimator(_))
    }

    /// Set the values of `map` of the params of the stage, as `stage.copy(extra)` of PySpark.
    pub fn apply_param_map(&mut self, map: &ParamMap) -> Result<()> {
        match self {
            Self::Estimator(e) => e.apply_param_map(map),
            Self::Transformer(t) => t.apply_param_map(map),
            Self::Model(m) => m.apply_param_map(map),
        }
    }

    /// The transform of `df` by the stage, a transformer or a model.
    pub fn transform(&mut self, df: &DataFrame) -> Result<DataFrame> {
        match self {
            Self::Transformer(t) => t.transform(df),
            Self::Model(m) => m.transform(df),
            Self::Estimator(e) => Err(SparkError::value_msg(format!(
                "{} is an estimator, which does not transform",
                e.operator().uid
            ))),
        }
    }

    /// The writer of the stage, of its own format: of the server's class, or of PySpark's Python
    /// format of a meta-algorithm.
    pub fn write(&self) -> Result<MlWriter> {
        match self {
            Self::Estimator(e) => Ok(e.write()),
            Self::Transformer(t) => Ok(MlWriter::operator(t.operator(), t.params())),
            Self::Model(m) => m.write(),
        }
    }
}

/// An estimator of the server's class `operator.name` that a load reads, of no class of this
/// crate that it knows: it fits as every estimator does, by the server's `Fit` of its operator and
/// params.
#[derive(Debug, Clone)]
pub struct OperatorEstimator {
    operator: MlOperator,
    params: Params,
}

impl OperatorEstimator {
    /// The estimator of the operator `operator` (its class and uid) and the params `params`.
    pub fn new(operator: MlOperator, params: Params) -> Self {
        Self { operator, params }
    }
}

impl Estimator for OperatorEstimator {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }

    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }

    fn params(&self) -> &Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> Result<Box<dyn Model>> {
        let info = fit_via_command(&self.operator, &self.params, df)?;
        let object = CachedModel::from_info(&info, &df.session, &self.operator.name)?;
        // The model of the uid of the estimator, as the server makes it, of the class of the
        // estimator's name of a model ("<class>Model").
        let operator = MlOperator::with_uid(
            &format!("{}Model", self.operator.name),
            info.uid.as_deref().unwrap_or(&self.operator.uid),
            OperatorType::Model,
        );
        Ok(Box::new(ObjectModel::new(
            object,
            operator,
            self.params.clone(),
        )))
    }
}

/// A transformer of the server's class `operator.name` that a load reads (see
/// [`OperatorEstimator`]): it transforms by the server's transform of its operator and params.
#[derive(Debug, Clone)]
pub struct OperatorTransformer {
    operator: MlOperator,
    params: Params,
}

impl OperatorTransformer {
    /// The transformer of the operator `operator` and the params `params`.
    pub fn new(operator: MlOperator, params: Params) -> Self {
        Self { operator, params }
    }
}

impl Transformer for OperatorTransformer {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }

    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }

    fn params(&self) -> &Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn transform_impl(&mut self, df: &DataFrame) -> Result<DataFrame> {
        let ml_relation = proto::Relation {
            rel_type: Some(proto::relation::RelType::MlRelation(Box::new(
                self.build_ml_relation(df)?,
            ))),
            ..Default::default()
        };
        Ok(DataFrame::new(
            df.session.clone(),
            LogicalPlan::MlTransform {
                ml_relation,
                input: df.plan.clone(),
                owner: None,
            },
        ))
    }
}

/// The JVM class kind of the class `class` of a stage of pyspark.ml (see `stage_kinds`).
fn stage_kind(class: &str) -> Option<OperatorType> {
    STAGE_KINDS
        .iter()
        .find(|(name, _)| *name == class)
        .map(|(_, kind)| *kind)
}

// ---------------------------------------------------------------------------------------------
// Metadata of PySpark's Python format.
// ---------------------------------------------------------------------------------------------

/// The JSON text `text` of a metadata of Python's `json.dumps`, parsed: of its floats `NaN`,
/// `Infinity` and `-Infinity` (which JSON has not), as the strings of [`python_float`].
pub(super) fn parse_python_json(text: &str) -> Result<serde_json::Value> {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.char_indices().peekable();
    let mut in_string = false;
    while let Some((i, c)) = chars.next() {
        if in_string {
            out.push(c);
            if c == '\\' {
                if let Some((_, escaped)) = chars.next() {
                    out.push(escaped);
                }
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if c == '"' {
            in_string = true;
            out.push(c);
            continue;
        }
        let rest = &text[i..];
        let token = ["-Infinity", "Infinity", "NaN"]
            .into_iter()
            .find(|t| rest.starts_with(t));
        if let Some(token) = token {
            // The marker, of the JSON escape of its control character.
            out.push_str(&format!("\"\\u0000float:{token}\""));
            for _ in 1..token.len() {
                chars.next();
            }
            continue;
        }
        out.push(c);
    }
    serde_json::from_str(&out).map_err(|e| SparkError::value_msg(format!("the metadata: {e}")))
}

/// The prefix of the strings of the non-finite floats of [`parse_python_json`].
const NON_FINITE: &str = "\u{0}float:";

/// The float of a JSON value of a metadata of [`parse_python_json`]: a number, or a non-finite
/// float.
pub(super) fn python_float(value: &serde_json::Value) -> Option<f64> {
    if let Some(number) = value.as_f64() {
        return Some(number);
    }
    match value.as_str()?.strip_prefix(NON_FINITE)? {
        "NaN" => Some(f64::NAN),
        "Infinity" => Some(f64::INFINITY),
        "-Infinity" => Some(f64::NEG_INFINITY),
        _ => None,
    }
}

/// The JSON text of the float `value`, as Python's `json.dumps` writes it (`NaN`, `Infinity`).
pub(super) fn python_float_json(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value == f64::INFINITY {
        "Infinity".to_string()
    } else if value == f64::NEG_INFINITY {
        "-Infinity".to_string()
    } else {
        // As Python's `repr` of a float, the shortest that reads back the same.
        let text = format!("{value:?}");
        text
    }
}

/// The metadata at `<path>/metadata`, of [`parse_python_json`].
pub(super) fn load_python_metadata(
    session: &SparkSession,
    path: &str,
) -> Result<serde_json::Value> {
    parse_python_json(&load_metadata_text(session, path)?)
}

/// `metadata[key]`, an error of a metadata without it, as Python's dict raises a KeyError.
pub(super) fn metadata_item<'a>(
    metadata: &'a serde_json::Value,
    key: &str,
) -> Result<&'a serde_json::Value> {
    metadata
        .get(key)
        .ok_or_else(|| SparkError::value_msg(format!("the metadata has no '{key}'")))
}

/// The string `metadata[key]`.
pub(super) fn metadata_str<'a>(metadata: &'a serde_json::Value, key: &str) -> Result<&'a str> {
    metadata_item(metadata, key)?
        .as_str()
        .ok_or_else(|| SparkError::value_msg(format!("the '{key}' of the metadata is no string")))
}

/// The MlValue of a JSON value of a metadata: a bool, an int (of a long), a float, a string, a
/// null, or a list of them.
pub(super) fn json_to_ml_value(value: &serde_json::Value) -> MlValue {
    match value {
        serde_json::Value::Null => MlValue::Null,
        serde_json::Value::Bool(b) => MlValue::Bool(*b),
        serde_json::Value::Number(n) => match n.as_i64() {
            Some(i) if i32::try_from(i).is_ok() => MlValue::Int(i as i32),
            Some(i) => MlValue::Long(i),
            None => MlValue::Double(n.as_f64().unwrap_or(f64::NAN)),
        },
        serde_json::Value::String(s) => match python_float(value) {
            Some(f) if s.starts_with(NON_FINITE) => MlValue::Double(f),
            _ => MlValue::String(s.clone()),
        },
        serde_json::Value::Array(items) => {
            MlValue::Array(items.iter().map(json_to_ml_value).collect())
        }
        serde_json::Value::Object(_) => MlValue::String(value.to_string()),
    }
}

/// The JSON text of an MlValue, as Python's `json.dumps` writes the value of a param.
pub(super) fn ml_value_json(value: &MlValue) -> Result<String> {
    Ok(match value {
        MlValue::Null => "null".to_string(),
        MlValue::Bool(b) => b.to_string(),
        MlValue::Byte(i) | MlValue::Short(i) | MlValue::Int(i) => i.to_string(),
        MlValue::Long(i) => i.to_string(),
        MlValue::Float(f) => python_float_json(f64::from(*f)),
        MlValue::Double(f) => python_float_json(*f),
        MlValue::String(s) => serde_json::json!(s).to_string(),
        MlValue::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(ml_value_json)
                .collect::<Result<Vec<_>>>()?
                .join(", ")
        ),
        other => {
            return Err(SparkError::value_msg(format!(
                "a value of a param of no JSON: {other:?}"
            )))
        }
    })
}

/// The stage saved at `path`, as `DefaultParamsReader.loadParamsInstance` reads it of the class of
/// its metadata: a meta-algorithm of PySpark's Python format, or an estimator, a transformer or a
/// model of the server's class (an [`OperatorEstimator`], an [`OperatorTransformer`] or an
/// [`ObjectModel`]).
pub fn load_stage(session: &SparkSession, path: &str) -> Result<PipelineStage> {
    let metadata = load_python_metadata(session, path)?;
    let class = metadata_str(&metadata, "class")?;
    Ok(match class {
        PIPELINE_CLASS => PipelineStage::estimator(Pipeline::load(session, path)?),
        PIPELINE_MODEL_CLASS => PipelineStage::Model(Box::new(PipelineModel::load(session, path)?)),
        CROSS_VALIDATOR_CLASS => PipelineStage::estimator(CrossValidator::load(session, path)?),
        CROSS_VALIDATOR_MODEL_CLASS => {
            PipelineStage::Model(Box::new(CrossValidatorModel::load(session, path)?))
        }
        TRAIN_VALIDATION_SPLIT_CLASS => {
            PipelineStage::estimator(TrainValidationSplit::load(session, path)?)
        }
        TRAIN_VALIDATION_SPLIT_MODEL_CLASS => {
            PipelineStage::Model(Box::new(TrainValidationSplitModel::load(session, path)?))
        }
        jvm => match stage_kind(jvm) {
            Some(OperatorType::Model) => {
                let (object, operator, params) = read_model(session, jvm, path)?;
                PipelineStage::Model(Box::new(ObjectModel::new(object, operator, params)))
            }
            Some(OperatorType::Estimator) => {
                let (operator, params) =
                    read_operator(session, jvm, OperatorType::Estimator, path)?;
                PipelineStage::estimator(OperatorEstimator::new(operator, params))
            }
            Some(OperatorType::Transformer) => {
                let (operator, params) =
                    read_operator(session, jvm, OperatorType::Transformer, path)?;
                PipelineStage::transformer(OperatorTransformer::new(operator, params))
            }
            _ => {
                return Err(SparkError::value_msg(format!(
                    "the stage at {path} is of the class {jvm}, which is no stage that this \
                     client loads"
                )))
            }
        },
    })
}

/// The evaluator saved at `path`, of the server's class of its metadata, as the typed evaluator of
/// this crate of the class.
pub(super) fn load_evaluator(
    session: &SparkSession,
    path: &str,
) -> Result<Box<dyn EvaluatorStage>> {
    let metadata = load_python_metadata(session, path)?;
    let class = metadata_str(&metadata, "class")?;
    let name = class.rsplit('.').next().unwrap_or(class);
    Ok(
        match (class.starts_with("org.apache.spark.ml.evaluation."), name) {
            (true, "BinaryClassificationEvaluator") => {
                Box::new(BinaryClassificationEvaluator::load(session, path)?)
            }
            (true, "MulticlassClassificationEvaluator") => {
                Box::new(MulticlassClassificationEvaluator::load(session, path)?)
            }
            (true, "MultilabelClassificationEvaluator") => {
                Box::new(MultilabelClassificationEvaluator::load(session, path)?)
            }
            (true, "RegressionEvaluator") => Box::new(RegressionEvaluator::load(session, path)?),
            (true, "ClusteringEvaluator") => Box::new(ClusteringEvaluator::load(session, path)?),
            (true, "RankingEvaluator") => Box::new(RankingEvaluator::load(session, path)?),
            _ => {
                return Err(SparkError::value_msg(format!(
                    "the evaluator at {path} is of the class {class}, which is no evaluator that \
                 this client loads"
                )))
            }
        },
    )
}

// ---------------------------------------------------------------------------------------------
// Pipeline and PipelineModel.
// ---------------------------------------------------------------------------------------------

pub(super) const PIPELINE_CLASS: &str = "pyspark.ml.pipeline.Pipeline";
pub(super) const PIPELINE_MODEL_CLASS: &str = "pyspark.ml.pipeline.PipelineModel";
pub(super) const CROSS_VALIDATOR_CLASS: &str = "pyspark.ml.tuning.CrossValidator";
pub(super) const CROSS_VALIDATOR_MODEL_CLASS: &str = "pyspark.ml.tuning.CrossValidatorModel";
pub(super) const TRAIN_VALIDATION_SPLIT_CLASS: &str = "pyspark.ml.tuning.TrainValidationSplit";
pub(super) const TRAIN_VALIDATION_SPLIT_MODEL_CLASS: &str =
    "pyspark.ml.tuning.TrainValidationSplitModel";

/// PySpark's `Pipeline`: a sequence of stages, of estimators and transformers, which a fit fits in
/// order (`Pipeline._fit`) into a [`PipelineModel`].
#[derive(Debug, Clone)]
pub struct Pipeline {
    operator: MlOperator,
    params: Params,
    stages: Vec<PipelineStage>,
}

impl Pipeline {
    /// The names of PySpark's params of the class.
    pub const PARAM_NAMES: &'static [&'static str] = &["stages"];

    /// A pipeline of no stages, of a new uid (`Pipeline_<12 hex digits>`).
    pub fn new() -> Self {
        Pipeline {
            operator: MlOperator::new("org.apache.spark.ml.Pipeline", OperatorType::Estimator),
            params: Params::new(),
            stages: Vec::new(),
        }
    }

    /// The pipeline of the stages `stages`, as `setStages`.
    pub fn set_stages(mut self, stages: impl IntoIterator<Item = PipelineStage>) -> Self {
        self.stages = stages.into_iter().collect();
        self
    }

    /// The stages, as `getStages`.
    pub fn stages(&self) -> &[PipelineStage] {
        &self.stages
    }

    /// The uid, as PySpark's `uid`.
    pub fn uid(&self) -> &str {
        &self.operator.uid
    }

    /// A copy of the pipeline, as `Pipeline.copy(extra)`: of the same uid, of copies of its
    /// stages, each with the values of `extra` of its params.
    pub fn copy(&self, extra: &ParamMap) -> Result<Self> {
        let mut that = self.clone();
        that.apply_param_map(extra)?;
        Ok(that)
    }

    /// Load the pipeline saved at `path` of PySpark's Python format, as `PipelineReader.load`.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (uid, stages) = load_with_stages(session, path)?;
        Ok(Pipeline {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.Pipeline",
                &uid,
                OperatorType::Estimator,
            ),
            params: Params::new(),
            stages,
        })
    }

    /// The fit of `df`, as `Pipeline._fit`, of the typed model.
    pub fn fit_pipeline(&mut self, df: &DataFrame) -> Result<PipelineModel> {
        let last_estimator = self.stages.iter().rposition(PipelineStage::is_estimator);
        let mut dataset = df.clone();
        let mut transformers = Vec::with_capacity(self.stages.len());
        for (i, stage) in self.stages.iter_mut().enumerate() {
            match (last_estimator, stage) {
                (Some(last), PipelineStage::Estimator(estimator)) if i <= last => {
                    let mut model = estimator.fit(&dataset)?;
                    if i < last {
                        dataset = model.transform(&dataset)?;
                    }
                    transformers.push(PipelineStage::Model(model));
                }
                (Some(last), stage) if i <= last => {
                    dataset = stage.transform(&dataset)?;
                    transformers.push(stage.clone());
                }
                (_, stage) => transformers.push(stage.clone()),
            }
        }
        Ok(PipelineModel::of_stages(transformers))
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for Pipeline {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }

    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }

    fn params(&self) -> &Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> Result<Box<dyn Model>> {
        Ok(Box::new(self.fit_pipeline(df)?))
    }

    fn apply_param_map(&mut self, map: &ParamMap) -> Result<()> {
        for stage in &mut self.stages {
            stage.apply_param_map(map)?;
        }
        Ok(())
    }

    fn write(&self) -> MlWriter {
        MlWriter::meta(Arc::new(StagesSave {
            class: PIPELINE_CLASS,
            uid: self.operator.uid.clone(),
            stages: self.stages.clone(),
        }))
    }
}

/// PySpark's `PipelineModel`: the transformers and models of a pipeline's fit, which transform in
/// order (`PipelineModel._transform`).
#[derive(Debug, Clone)]
pub struct PipelineModel {
    operator: MlOperator,
    params: Params,
    stages: Vec<PipelineStage>,
}

impl PipelineModel {
    /// The names of PySpark's params of the class (none).
    pub const PARAM_NAMES: &'static [&'static str] = &[];

    /// The model of the stages `stages`, of a new uid, as `PipelineModel(stages)`.
    pub fn of_stages(stages: Vec<PipelineStage>) -> Self {
        Self {
            operator: MlOperator::new("org.apache.spark.ml.PipelineModel", OperatorType::Model),
            params: Params::new(),
            stages,
        }
    }

    /// The stages, as `stages`.
    pub fn stages(&self) -> &[PipelineStage] {
        &self.stages
    }

    /// The uid, as PySpark's `uid`.
    pub fn uid(&self) -> &str {
        &self.operator.uid
    }

    /// A copy of the model, as `PipelineModel.copy(extra)`: of a new uid, of copies of its stages,
    /// each with the values of `extra` of its params.
    pub fn copy(&self, extra: &ParamMap) -> Result<Self> {
        let mut stages = self.stages.clone();
        for stage in &mut stages {
            stage.apply_param_map(extra)?;
        }
        Ok(Self::of_stages(stages))
    }

    /// Load the model saved at `path` of PySpark's Python format, as `PipelineModelReader.load`.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (uid, stages) = load_with_stages(session, path)?;
        Ok(Self {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.PipelineModel",
                &uid,
                OperatorType::Model,
            ),
            params: Params::new(),
            stages,
        })
    }
}

impl Transformer for PipelineModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }

    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }

    fn params(&self) -> &Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn transform_impl(&mut self, df: &DataFrame) -> Result<DataFrame> {
        let mut dataset = df.clone();
        for stage in &mut self.stages {
            dataset = stage.transform(&dataset)?;
        }
        Ok(dataset)
    }

    fn apply_param_map(&mut self, map: &ParamMap) -> Result<()> {
        for stage in &mut self.stages {
            stage.apply_param_map(map)?;
        }
        Ok(())
    }
}

impl Model for PipelineModel {
    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }

    fn write(&self) -> Result<MlWriter> {
        Ok(MlWriter::meta(Arc::new(StagesSave {
            class: PIPELINE_MODEL_CLASS,
            uid: self.operator.uid.clone(),
            stages: self.stages.clone(),
        })))
    }
}

/// The save of a pipeline or a pipeline model, as `PipelineSharedReadWrite.saveImpl`: the metadata
/// of the uids of the stages (`stageUids`, of the language "Python"), and each stage, of its own
/// writer and the session of the save, at `stages/<index>_<uid>`.
struct StagesSave {
    class: &'static str,
    uid: String,
    stages: Vec<PipelineStage>,
}

impl MetaSave for StagesSave {
    fn name(&self) -> String {
        self.uid.clone()
    }

    fn save_meta(
        &self,
        session: &SparkSession,
        path: &str,
        _options: &HashMap<String, String>,
    ) -> Result<()> {
        // As `validateStages`, before anything is written: each stage writes.
        let writers = self
            .stages
            .iter()
            .map(PipelineStage::write)
            .collect::<Result<Vec<_>>>()?;
        let uids: Vec<&str> = self.stages.iter().map(PipelineStage::uid).collect();
        let param_map = json_object(&[
            ("stageUids", serde_json::json!(uids).to_string()),
            ("language", serde_json::json!("Python").to_string()),
        ]);
        let metadata = Metadata::new(
            self.class,
            &self.uid,
            &session.version()?,
            param_map,
            "{}".to_string(),
        );
        save_metadata(session, path, &metadata.to_json())?;
        let stages_dir = join_path(path, "stages");
        for (index, (writer, uid)) in writers.into_iter().zip(&uids).enumerate() {
            writer
                .session(session)
                .save(&stage_path(uid, index, uids.len(), &stages_dir))?;
        }
        Ok(())
    }
}

/// The uid and the stages of the pipeline or model saved at `path`, as
/// `PipelineSharedReadWrite.load`.
fn load_with_stages(session: &SparkSession, path: &str) -> Result<(String, Vec<PipelineStage>)> {
    let metadata = load_python_metadata(session, path)?;
    let stage_uids = metadata_item(metadata_item(&metadata, "paramMap")?, "stageUids")?
        .as_array()
        .ok_or_else(|| SparkError::value_msg("the stageUids of the metadata are no list"))?;
    let stages_dir = join_path(path, "stages");
    let mut stages = Vec::with_capacity(stage_uids.len());
    for (index, uid) in stage_uids.iter().enumerate() {
        let uid = uid
            .as_str()
            .ok_or_else(|| SparkError::value_msg("a stage uid of the metadata is no string"))?;
        stages.push(load_stage(
            session,
            &stage_path(uid, index, stage_uids.len(), &stages_dir),
        )?);
    }
    Ok((metadata_str(&metadata, "uid")?.to_string(), stages))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_maps_set_a_param_of_a_stage_once_in_its_place() {
        let map = ParamMap::new()
            .put("A_1", "maxIter", MlValue::Int(3))
            .put("B_1", "maxIter", MlValue::Int(4))
            .put("A_1", "maxIter", MlValue::Int(5));
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("A_1", "maxIter"), Some(&MlValue::Int(5)));
        assert_eq!(map.pairs()[0].parent, "A_1");
        assert_eq!(map.of_stage("B_1").count(), 1);
        assert!(map.has_stage("A_1") && !map.has_stage("C_1"));
    }

    #[test]
    fn param_grids_are_the_product_of_their_values_of_the_last_param_fastest() {
        // As `itertools.product` of PySpark's `build`.
        let grid = ParamGridBuilder::new()
            .add_grid(
                "A_1",
                "regParam",
                [MlValue::Double(0.1), MlValue::Double(1.0)],
            )
            .add_grid("A_1", "maxIter", [MlValue::Int(1), MlValue::Int(5)])
            .base_on(&ParamMap::new().put("A_1", "tol", MlValue::Double(0.5)))
            .build();
        let values: Vec<Vec<&MlValue>> = grid
            .iter()
            .map(|m| m.pairs().iter().map(|p| &p.value).collect())
            .collect();
        assert_eq!(
            values,
            vec![
                vec![
                    &MlValue::Double(0.1),
                    &MlValue::Int(1),
                    &MlValue::Double(0.5)
                ],
                vec![
                    &MlValue::Double(0.1),
                    &MlValue::Int(5),
                    &MlValue::Double(0.5)
                ],
                vec![
                    &MlValue::Double(1.0),
                    &MlValue::Int(1),
                    &MlValue::Double(0.5)
                ],
                vec![
                    &MlValue::Double(1.0),
                    &MlValue::Int(5),
                    &MlValue::Double(0.5)
                ],
            ]
        );
        // A grid of no params is one map of none, and a param of no values none.
        assert_eq!(ParamGridBuilder::new().build(), vec![ParamMap::new()]);
        assert!(ParamGridBuilder::new()
            .add_grid("A_1", "maxIter", [])
            .build()
            .is_empty());
    }

    #[test]
    fn param_maps_set_the_params_of_their_stage_of_its_uid() {
        let mut svc = LinearSVC::new().set_max_iter(5);
        let uid = svc.operator().uid.clone();
        let map = ParamMap::new().put(&uid, "maxIter", MlValue::Int(7)).put(
            "Other_1",
            "maxIter",
            MlValue::Int(9),
        );
        svc.apply_param_map(&map).unwrap();
        assert_eq!(svc.max_iter(), 7);
        // A pipeline sets them in each of its stages.
        let mut pipeline = Pipeline::new().set_stages(vec![PipelineStage::estimator(
            LinearSVC::new().set_max_iter(5),
        )]);
        let stage_uid = pipeline.stages()[0].uid().to_string();
        let copy = pipeline
            .copy(&ParamMap::new().put(&stage_uid, "maxIter", MlValue::Int(2)))
            .unwrap();
        assert_eq!(copy.uid(), pipeline.uid());
        let PipelineStage::Estimator(stage) = &copy.stages()[0] else {
            panic!("an estimator stage");
        };
        let literal = stage.params().get_param("maxIter").cloned();
        assert!(format!("{literal:?}").contains("Integer(2)"), "{literal:?}");
        pipeline.apply_param_map(&ParamMap::new()).unwrap();
    }

    #[test]
    fn python_json_of_non_finite_floats_parses() {
        let value = parse_python_json(
            r#"{"avgMetrics":[0.5,NaN,Infinity,-Infinity],"s":"NaN in a string","n":-1}"#,
        )
        .unwrap();
        let metrics: Vec<f64> = value["avgMetrics"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| python_float(v).unwrap())
            .collect();
        assert_eq!(metrics[0], 0.5);
        assert!(metrics[1].is_nan());
        assert_eq!(metrics[2], f64::INFINITY);
        assert_eq!(metrics[3], f64::NEG_INFINITY);
        assert_eq!(value["s"], "NaN in a string");
        assert_eq!(value["n"], -1);
        assert_eq!(
            [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.1, 1.0]
                .map(python_float_json)
                .join(","),
            "NaN,Infinity,-Infinity,0.1,1.0"
        );
        assert!(parse_python_json(r#"{"a": "\"NaN"}"#).is_ok());
    }

    #[test]
    fn values_of_params_are_their_json_as_python_dumps_them() {
        assert_eq!(ml_value_json(&MlValue::Double(0.1)).unwrap(), "0.1");
        assert_eq!(ml_value_json(&MlValue::Double(1.0)).unwrap(), "1.0");
        assert_eq!(ml_value_json(&MlValue::Int(3)).unwrap(), "3");
        assert_eq!(ml_value_json(&MlValue::Bool(true)).unwrap(), "true");
        assert_eq!(
            ml_value_json(&MlValue::String("a\"b".into())).unwrap(),
            "\"a\\\"b\""
        );
        assert_eq!(
            ml_value_json(&MlValue::Array(vec![MlValue::Int(1), MlValue::Int(2)])).unwrap(),
            "[1, 2]"
        );
        assert_eq!(
            json_to_ml_value(&serde_json::json!([1, 2.5, "x", true, null])),
            MlValue::Array(vec![
                MlValue::Int(1),
                MlValue::Double(2.5),
                MlValue::String("x".into()),
                MlValue::Bool(true),
                MlValue::Null
            ])
        );
        assert_eq!(
            json_to_ml_value(&serde_json::json!(1i64 << 40)),
            MlValue::Long(1 << 40)
        );
    }

    #[test]
    fn stage_kinds_are_of_the_jvm_classes_of_pyspark_ml() {
        assert_eq!(
            stage_kind("org.apache.spark.ml.classification.LinearSVC"),
            Some(OperatorType::Estimator)
        );
        assert_eq!(
            stage_kind("org.apache.spark.ml.classification.LinearSVCModel"),
            Some(OperatorType::Model)
        );
        assert_eq!(
            stage_kind("org.apache.spark.ml.feature.VectorAssembler"),
            Some(OperatorType::Transformer)
        );
        assert_eq!(
            stage_kind("org.apache.spark.ml.evaluation.RegressionEvaluator"),
            Some(OperatorType::Evaluator)
        );
        assert_eq!(stage_kind("org.apache.spark.ml.Pipeline"), None);
    }
}
