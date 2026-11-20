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
use super::param_specs::{ParamConverter, MODEL_CLASSES, PARAM_CONVERTERS, PARAM_DOCS};
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

/// Set the values of `map` of the params of the stage of the uid `uid` and the JVM class `class`
/// in `params`, as PySpark's `copy(extra)` does: each value converted as its param's type
/// converter converts it, and none of the params that the class does not have (PySpark's
/// `_copyValues` sets only the params of the copied object).
pub(super) fn apply_pairs(
    params: &mut Params,
    class: &str,
    uid: &str,
    map: &ParamMap,
) -> Result<()> {
    for pair in map.of_stage(uid) {
        if class_known(class) && param_converter(class, &pair.name).is_none() {
            continue;
        }
        let literal = converted(class, &pair.name, &pair.value)?.to_literal()?;
        let taken = std::mem::replace(params, Params::new());
        *params = taken.set_param_literal(&pair.name, literal);
    }
    Ok(())
}

/// The simple name of a JVM class (of its package stripped), which the generated table names.
fn simple_name(class: &str) -> &str {
    class.rsplit('.').next().unwrap_or(class)
}

/// Whether the generated table of the params of pyspark.ml has the class `class`.
fn class_known(class: &str) -> bool {
    PARAM_CONVERTERS
        .iter()
        .any(|(name, _)| *name == simple_name(class))
}

/// The doc of the param `name` of the class of the simple name `class`, as PySpark's `Param.doc`
/// of the class, if the table has the param.
pub(super) fn param_doc(class: &str, name: &str) -> Option<&'static str> {
    let (_, params) = PARAM_DOCS.iter().find(|(simple, _)| *simple == class)?;
    params
        .iter()
        .find(|(param, _)| *param == name)
        .map(|(_, doc)| *doc)
}

/// The type converter of the param `name` of the class `class`, if the table has the param.
pub(super) fn param_converter(class: &str, name: &str) -> Option<ParamConverter> {
    let (_, params) = PARAM_CONVERTERS
        .iter()
        .find(|(simple, _)| *simple == simple_name(class))?;
    params
        .iter()
        .find(|(param, _)| *param == name)
        .map(|(_, converter)| *converter)
}

/// The value `value` of the param `name` of the class `class`, converted as PySpark's
/// `Param.typeConverter` converts it, or as it is for a param that the table does not have.
pub(super) fn converted(class: &str, name: &str, value: &MlValue) -> Result<MlValue> {
    match param_converter(class, name) {
        Some(converter) => convert_param(converter, value),
        None => Ok(value.clone()),
    }
}

/// The JVM class of the model that an estimator of the JVM class `estimator` fits (PySpark's
/// `_create_model`): the class of the estimator's name with `Model` appended, unless the table
/// names another (the models of the tree, ensemble, FM and perceptron classifiers and regressors,
/// and QuantileDiscretizer's `Bucketizer`).
pub(super) fn model_class(estimator: &str) -> String {
    match MODEL_CLASSES
        .iter()
        .find(|(simple, _)| *simple == simple_name(estimator))
    {
        Some((_, model)) => model.to_string(),
        None => format!("{estimator}Model"),
    }
}

/// The params of `params` that the JVM class `class` has, as PySpark's `_copyValues` copies them
/// to an object of the class. A class that the table does not have keeps all of them.
pub(super) fn params_of_class(params: &Params, class: &str) -> Params {
    if !class_known(class) {
        return params.clone();
    }
    params.retain_names(|name| param_converter(class, name).is_some())
}

/// The value of `value` as PySpark's type converter of `converter` converts it, or PySpark's
/// `TypeError` (of its message) when it cannot.
pub(super) fn convert_param(converter: ParamConverter, value: &MlValue) -> Result<MlValue> {
    let could_not = |target: &str| {
        SparkError::value_msg(format!("Could not convert {} to {target}", py_str(value)))
    };
    match converter {
        ParamConverter::Identity => Ok(value.clone()),
        ParamConverter::Boolean => match value {
            MlValue::Bool(b) => Ok(MlValue::Bool(*b)),
            other => Err(SparkError::value_msg(format!(
                "Boolean Param requires value of type bool. Found {}.",
                py_type(other)
            ))),
        },
        ParamConverter::Float => to_float(value)
            .map(MlValue::Double)
            .ok_or_else(|| could_not("float")),
        ParamConverter::Int => to_int(value).unwrap_or_else(|| Err(could_not("int"))),
        ParamConverter::String => match value {
            MlValue::String(s) => Ok(MlValue::String(s.clone())),
            other => Err(SparkError::value_msg(format!(
                "Could not convert {} to string type",
                py_type(other)
            ))),
        },
        ParamConverter::ListFloat => list_of(value, "list of floats", |v| {
            to_float(v).map(MlValue::Double)
        }),
        ParamConverter::ListInt => int_list_of(value),
        ParamConverter::ListString => list_of(value, "list of strings", |v| match v {
            MlValue::String(s) => Some(MlValue::String(s.clone())),
            _ => None,
        }),
        ParamConverter::ListListFloat => match value {
            MlValue::Array(items) => Ok(MlValue::Array(
                items
                    .iter()
                    .map(|item| convert_param(ParamConverter::ListFloat, item))
                    .collect::<Result<Vec<_>>>()?,
            )),
            _ => Err(could_not("list of list of floats")),
        },
    }
}

/// The list of the converted items of `value`, as PySpark's list converters convert it, or
/// PySpark's `TypeError` of a value that is no list of such items.
fn list_of(
    value: &MlValue,
    target: &str,
    item: impl Fn(&MlValue) -> Option<MlValue>,
) -> Result<MlValue> {
    let fail = || SparkError::value_msg(format!("Could not convert {} to {target}", py_str(value)));
    match value {
        MlValue::Array(items) => items
            .iter()
            .map(|v| item(v).ok_or_else(fail))
            .collect::<Result<Vec<_>>>()
            .map(MlValue::Array),
        _ => Err(fail()),
    }
}

/// The float of a numeric value (a Python `int` or `float`, not a `bool`), as `TypeConverters`.
fn to_float(value: &MlValue) -> Option<f64> {
    match value {
        MlValue::Byte(v) | MlValue::Short(v) | MlValue::Int(v) => Some(f64::from(*v)),
        MlValue::Long(v) => Some(*v as f64),
        MlValue::Float(v) => Some(f64::from(*v)),
        MlValue::Double(v) => Some(*v),
        _ => None,
    }
}

/// The whole number of a numeric value that is integral, as `TypeConverters._is_integer` takes it
/// and `int` converts it (an int, or a float that is whole: not a `bool`, NaN or an infinity), or
/// None of another value. Python's `int` keeps a whole float beyond the range of a long, which
/// PySpark's `LiteralExpression` cannot send: it is the `VALUE_NOT_BETWEEN` of `_infer_type`.
fn whole_long(value: &MlValue) -> Option<Result<i64>> {
    // 2^63, the least whole float beyond a long (`i64::MAX as f64` rounds to it).
    const LONG_BOUND: f64 = 9_223_372_036_854_775_808.0;
    let whole = match value {
        MlValue::Byte(v) | MlValue::Short(v) | MlValue::Int(v) => return Some(Ok(i64::from(*v))),
        MlValue::Long(v) => return Some(Ok(*v)),
        MlValue::Float(v) => f64::from(*v),
        MlValue::Double(v) => *v,
        _ => return None,
    };
    if !whole.is_finite() || whole.fract() != 0.0 {
        return None;
    }
    Some(if (-LONG_BOUND..LONG_BOUND).contains(&whole) {
        Ok(whole as i64)
    } else {
        Err(value_not_between())
    })
}

/// PySpark's error of a Python int beyond the range of a long, which its `LiteralExpression`
/// cannot send: `VALUE_NOT_BETWEEN`, whose maximum is `JVM_SHORT_MAX`, as PySpark writes it.
fn value_not_between() -> SparkError {
    SparkError::classed(
        SparkErrorKind::ValueError,
        "VALUE_NOT_BETWEEN",
        &[
            ("arg_name", "value"),
            ("min", "-9223372036854775808"),
            ("max", "32767"),
        ],
    )
}

/// The int of a numeric value that is integral, as `TypeConverters.toInt` converts it, of the
/// literal that PySpark sends of it (`_infer_type`): an `Int` when it fits 32 bits, else a `Long`,
/// and `VALUE_NOT_BETWEEN` beyond a long (see [`whole_long`]). None of another value.
fn to_int(value: &MlValue) -> Option<Result<MlValue>> {
    Some(whole_long(value)?.map(|whole| match i32::try_from(whole) {
        Ok(v) => MlValue::Int(v),
        Err(_) => MlValue::Long(whole),
    }))
}

/// The list of ints of `value`, as `TypeConverters.toListInt` converts it, typed as PySpark's
/// `LiteralExpression` sends a list: of the type of its first element (`_infer_type`), an `Int`
/// when it fits 32 bits, else a `Long`, and `VALUE_NOT_BETWEEN` of a first element beyond a long.
/// Each element is a literal of that type, whose `assert` fails of an element out of its range,
/// an `AssertionError`. A list of no element stays one, whose literal is
/// `CANNOT_INFER_ARRAY_ELEMENT_TYPE` (see [`MlValue::to_literal`]). A value that is no list of
/// integral numbers is PySpark's `TypeError`.
fn int_list_of(value: &MlValue) -> Result<MlValue> {
    let fail = || {
        SparkError::value_msg(format!(
            "Could not convert {} to list of ints",
            py_str(value)
        ))
    };
    let MlValue::Array(items) = value else {
        return Err(fail());
    };
    let wholes = items
        .iter()
        .map(|item| whole_long(item).ok_or_else(fail))
        .collect::<Result<Vec<_>>>()?;
    let Some(first) = wholes.first() else {
        return Ok(MlValue::Array(Vec::new()));
    };
    let narrow = i32::try_from(first.clone()?).is_ok();
    let (type_name, bits) = if narrow {
        ("an integer", "32")
    } else {
        ("a long", "64")
    };
    wholes
        .into_iter()
        .zip(items)
        .map(|(whole, item)| match whole {
            Ok(v) if !narrow => Ok(MlValue::Long(v)),
            Ok(v) if i32::try_from(v).is_ok() => Ok(MlValue::Int(v as i32)),
            _ => Err(assertion_error(format!(
                "the element {} of the list of ints {} is out of the range of {type_name} \
                 ({bits} bits), the type of its first element, as PySpark's LiteralExpression \
                 asserts",
                py_str(item),
                py_str(value)
            ))),
        })
        .collect::<Result<Vec<_>>>()
        .map(MlValue::Array)
}

/// The Python type of a value, as PySpark's messages name `type(value)`.
fn py_type(value: &MlValue) -> &'static str {
    match value {
        MlValue::Null => "<class 'NoneType'>",
        MlValue::Bool(_) => "<class 'bool'>",
        MlValue::Byte(_) | MlValue::Short(_) | MlValue::Int(_) | MlValue::Long(_) => {
            "<class 'int'>"
        }
        MlValue::Float(_) | MlValue::Double(_) => "<class 'float'>",
        MlValue::String(_) => "<class 'str'>",
        MlValue::Array(_) => "<class 'list'>",
        _ => "<class 'object'>",
    }
}

/// The Python `repr` of the str `text`: in single quotes, or in double quotes when it has a single
/// quote and no double one, with the backslashes, the quote and the control characters escaped.
pub(super) fn python_str_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// The Python `str` of a value, as PySpark's messages show it (a list as its `repr`).
fn py_str(value: &MlValue) -> String {
    match value {
        MlValue::Null => "None".to_string(),
        MlValue::Bool(true) => "True".to_string(),
        MlValue::Bool(false) => "False".to_string(),
        MlValue::Byte(v) | MlValue::Short(v) | MlValue::Int(v) => v.to_string(),
        MlValue::Long(v) => v.to_string(),
        MlValue::Float(v) => python_float_repr(f64::from(*v)),
        MlValue::Double(v) => python_float_repr(*v),
        MlValue::String(s) => s.clone(),
        MlValue::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(|item| match item {
                    MlValue::String(s) => python_str_repr(s),
                    other => py_str(other),
                })
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => format!("{other:?}"),
    }
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

    /// The stage of the transformer `transformer`, which is no model: a fitted model is a stage of
    /// [`Self::model`], which a save writes by the model's server-side object. A fitted model given
    /// here fails at the save, as the stage has no object to write it by.
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
            Self::Transformer(t) => {
                check_not_model(t.operator())?;
                Ok(MlWriter::operator(t.operator(), t.params()))
            }
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
        // The model of the uid of the estimator, as the server makes it, of the JVM class that
        // PySpark's `_create_model` names, with the params of that class only: PySpark's model
        // copies the params of the estimator that the model has.
        let class = model_class(&self.operator.name);
        let operator = MlOperator::with_uid(
            &class,
            info.uid.as_deref().unwrap_or(&self.operator.uid),
            OperatorType::Model,
        );
        let params = params_of_class(&self.params, &class);
        Ok(Box::new(ObjectModel::new(object, operator, params)))
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

/// The error of a fitted model that is held as a transformer stage (see
/// [`PipelineStage::transformer`]): a meta-algorithm saves a model by its server-side object, which
/// a [`PipelineStage::Model`] holds and a transformer does not.
pub(super) fn check_not_model(operator: &MlOperator) -> Result<()> {
    if operator.op_type == OperatorType::Model {
        return Err(SparkError::value_msg(format!(
            "{} is a fitted model, which a transformer stage cannot save: give it to \
             PipelineStage::model",
            operator.uid
        )));
    }
    Ok(())
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
        python_float_repr(value)
    }
}

/// The text of the finite or non-finite float `value` as Python's `repr` (and `str`) writes it:
/// the shortest digits that read back the same, of an even last digit of a tie, in the decimal form
/// for a decimal exponent from -4 to 15, else in the exponent form with a sign and at least two
/// digits (`1e-06`, `1e+16`), as [`crate::expression::python_float_repr`] writes it.
pub(super) fn python_float_repr(value: f64) -> String {
    crate::expression::python_float_repr(value)
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
        MlValue::String(s) => json_string(s),
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

/// The Python class of the class `class` of a stage's metadata, as `DefaultParamsReader` names it:
/// a class of PySpark's Python format as it is, else the JVM class with `pyspark` for
/// `org.apache.spark`. The pipelines of the JVM writer are `pyspark.ml.Pipeline` and
/// `pyspark.ml.PipelineModel` there, which are the classes of `pyspark.ml.pipeline`.
pub(super) fn python_class_of(class: &str) -> String {
    if class.starts_with("pyspark.ml.") {
        return class.to_string();
    }
    match class.replace("org.apache.spark", "pyspark").as_str() {
        "pyspark.ml.Pipeline" => PIPELINE_CLASS.to_string(),
        "pyspark.ml.PipelineModel" => PIPELINE_MODEL_CLASS.to_string(),
        python => python.to_string(),
    }
}

/// The stage saved at `path`, as `DefaultParamsReader.loadParamsInstance` reads it of the class of
/// its metadata: a meta-algorithm of PySpark's Python format (of the JVM writer's class names too,
/// see `python_class_of`), or an estimator, a transformer or a model of the server's class (an
/// [`OperatorEstimator`], an [`OperatorTransformer`] or an [`ObjectModel`]).
pub fn load_stage(session: &SparkSession, path: &str) -> Result<PipelineStage> {
    let metadata = load_python_metadata(session, path)?;
    let class = metadata_str(&metadata, "class")?;
    Ok(match python_class_of(class).as_str() {
        PIPELINE_CLASS => PipelineStage::estimator(Pipeline::load(session, path)?),
        PIPELINE_MODEL_CLASS => PipelineStage::Model(Box::new(PipelineModel::load(session, path)?)),
        ONE_VS_REST_CLASS => PipelineStage::estimator(OneVsRest::load(session, path)?),
        ONE_VS_REST_MODEL_CLASS => {
            PipelineStage::Model(Box::new(OneVsRestModel::load(session, path)?))
        }
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
        // The typed discretizer, not an `OperatorEstimator`, so that its fit gives the Bucketizer.
        QUANTILE_DISCRETIZER_CLASS => {
            PipelineStage::estimator(QuantileDiscretizer::load(session, path)?)
        }
        _ => match stage_kind(class) {
            Some(OperatorType::Model) => {
                let (object, operator, params) = read_model(session, class, path)?;
                PipelineStage::Model(Box::new(ObjectModel::new(object, operator, params)))
            }
            Some(OperatorType::Estimator) => {
                let (operator, params) =
                    read_operator(session, class, OperatorType::Estimator, path)?;
                PipelineStage::estimator(OperatorEstimator::new(operator, params))
            }
            Some(OperatorType::Transformer) => {
                let (operator, params) =
                    read_operator(session, class, OperatorType::Transformer, path)?;
                PipelineStage::transformer(OperatorTransformer::new(operator, params))
            }
            _ => {
                return Err(SparkError::value_msg(format!(
                    "the stage at {path} is of the class {class}, which is no stage that this \
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
pub(super) const ONE_VS_REST_CLASS: &str = "pyspark.ml.classification.OneVsRest";
pub(super) const ONE_VS_REST_MODEL_CLASS: &str = "pyspark.ml.classification.OneVsRestModel";
pub(super) const TRAIN_VALIDATION_SPLIT_MODEL_CLASS: &str =
    "pyspark.ml.tuning.TrainValidationSplitModel";
pub(super) const QUANTILE_DISCRETIZER_CLASS: &str = "pyspark.ml.feature.QuantileDiscretizer";

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
                    let mut fitted = fit_stage(estimator.as_mut(), &dataset)?;
                    if i < last {
                        dataset = fitted.transform(&dataset)?;
                    }
                    transformers.push(fitted);
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

/// The stage that a pipeline's fit gives for the estimator `estimator`, as PySpark's `fit` does: a
/// QuantileDiscretizer gives a Bucketizer of its own new uid (see
/// [`QuantileDiscretizer::fit_bucketizer`]), and any other estimator its model.
fn fit_stage(estimator: &mut dyn EstimatorStage, df: &DataFrame) -> Result<PipelineStage> {
    let discretizer = estimator
        .as_any_stage()
        .and_then(|any| any.downcast_ref::<QuantileDiscretizer>())
        .cloned();
    match discretizer {
        Some(mut discretizer) => Ok(PipelineStage::transformer(discretizer.fit_bucketizer(df)?)),
        None => Ok(PipelineStage::Model(estimator.fit(df)?)),
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
        let stage_uids: Vec<String> = uids.iter().map(|uid| json_string(uid)).collect();
        let param_map = json_object(&[
            ("stageUids", format!("[{}]", stage_uids.join(","))),
            ("language", json_string("Python")),
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
    fn python_floats_are_pythons_repr_and_json_dumps() {
        // The expected texts are Python 3.11's `repr` and `json.dumps` of the same floats.
        let cases: [(f64, &str, &str); 11] = [
            (1e20, "1e+20", "1e+20"),
            (1e-6, "1e-06", "1e-06"),
            (1e16, "1e+16", "1e+16"),
            (1e-5, "1e-05", "1e-05"),
            (0.1, "0.1", "0.1"),
            (1e22, "1e+22", "1e+22"),
            (5e-324, "5e-324", "5e-324"),
            (
                123456789012345678.0,
                "1.2345678901234568e+17",
                "1.2345678901234568e+17",
            ),
            (f64::NAN, "nan", "NaN"),
            (f64::INFINITY, "inf", "Infinity"),
            (f64::NEG_INFINITY, "-inf", "-Infinity"),
        ];
        for (value, repr, json) in cases {
            assert_eq!(python_float_repr(value), repr, "repr of {value:?}");
            assert_eq!(python_float_json(value), json, "json of {value:?}");
        }
        // The decimal form holds from 1e-4 up to 1e16 exclusive, and the digits are the shortest.
        for (value, text) in [
            (0.0001, "0.0001"),
            (1e15, "1000000000000000.0"),
            (9999999999999998.0, "9999999999999998.0"),
            (1.5e16, "1.5e+16"),
            (-0.0, "-0.0"),
            (1.2345678e-6, "1.2345678e-06"),
            (f64::MAX, "1.7976931348623157e+308"),
        ] {
            assert_eq!(python_float_repr(value), text);
        }
        // A value of `str` is its repr, and a value of a list is the repr of each item.
        assert_eq!(py_str(&MlValue::Double(1e-6)), "1e-06");
        assert_eq!(py_str(&MlValue::Double(f64::NAN)), "nan");
        assert_eq!(py_str(&MlValue::Double(f64::NEG_INFINITY)), "-inf");
        assert_eq!(
            py_str(&MlValue::Array(vec![
                MlValue::String("it's".into()),
                MlValue::Double(1e20),
            ])),
            "[\"it's\", 1e+20]"
        );
    }

    #[test]
    fn python_strs_are_reprs_as_python_writes_them() {
        assert_eq!(
            python_str_repr("regularization parameter (>= 0)."),
            "'regularization parameter (>= 0).'"
        );
        assert_eq!(python_str_repr("it's"), "\"it's\"");
        assert_eq!(python_str_repr("say \"hi\""), "'say \"hi\"'");
        assert_eq!(python_str_repr("a\nb\\c\u{1}"), "'a\\nb\\\\c\\x01'");
    }

    #[test]
    fn string_values_are_json_with_ascii_escapes() {
        // As `json.dumps` of the same strings, of its default `ensure_ascii`.
        assert_eq!(
            ml_value_json(&MlValue::String("\u{e9}\u{7f}\n".into())).unwrap(),
            "\"\\u00e9\\u007f\\n\""
        );
    }

    #[test]
    fn a_jvm_class_of_a_meta_algorithm_is_read_as_its_python_class() {
        // As `DefaultParamsReader.loadParamsInstance`: `org.apache.spark` is `pyspark`, and the
        // pipelines of the JVM writer are of `pyspark.ml`, which exports them.
        for (jvm, python) in [
            ("org.apache.spark.ml.Pipeline", PIPELINE_CLASS),
            ("org.apache.spark.ml.PipelineModel", PIPELINE_MODEL_CLASS),
            (
                "org.apache.spark.ml.classification.OneVsRest",
                ONE_VS_REST_CLASS,
            ),
            (
                "org.apache.spark.ml.classification.OneVsRestModel",
                ONE_VS_REST_MODEL_CLASS,
            ),
            (
                "org.apache.spark.ml.tuning.CrossValidator",
                CROSS_VALIDATOR_CLASS,
            ),
            (
                "org.apache.spark.ml.tuning.CrossValidatorModel",
                CROSS_VALIDATOR_MODEL_CLASS,
            ),
            (
                "org.apache.spark.ml.tuning.TrainValidationSplit",
                TRAIN_VALIDATION_SPLIT_CLASS,
            ),
            (
                "org.apache.spark.ml.tuning.TrainValidationSplitModel",
                TRAIN_VALIDATION_SPLIT_MODEL_CLASS,
            ),
            (
                "org.apache.spark.ml.feature.QuantileDiscretizer",
                QUANTILE_DISCRETIZER_CLASS,
            ),
        ] {
            assert_eq!(python_class_of(jvm), python, "{jvm}");
            assert_eq!(python_class_of(python), python, "{python} is its own class");
        }
        // A stage of no meta-algorithm keeps its Python name, which no load reads.
        assert_eq!(
            python_class_of("org.apache.spark.ml.feature.VectorAssembler"),
            "pyspark.ml.feature.VectorAssembler"
        );
        assert_eq!(
            stage_kind("org.apache.spark.ml.Pipeline"),
            None,
            "the pipelines are read by their own readers"
        );
    }

    #[test]
    fn a_fitted_model_is_no_transformer_stage() {
        // A model given as a transformer has no server-side object to save by (see
        // `PipelineStage::model`), so a save of it fails, and so does a validator's of a pipeline
        // that holds it.
        let session = SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("session");
        let model = ObjectModel::new(
            MlRef::from_id(&session, "fitted"),
            MlOperator::new(
                "org.apache.spark.ml.classification.LinearSVCModel",
                OperatorType::Model,
            ),
            Params::new(),
        );
        let stage = PipelineStage::transformer(model.clone());
        let error = stage.write().expect_err("a model as a transformer");
        assert!(error.to_string().contains("is a fitted model"), "{error}");
        let as_model = PipelineStage::model(Box::new(model));
        assert!(as_model.write().is_ok());
        assert!(PipelineStage::transformer(VectorAssembler::new())
            .write()
            .is_ok());
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

    #[test]
    fn param_maps_set_only_the_params_that_the_stage_class_has() {
        // As PySpark's `_copyValues`: a param of another class is not set (not sent).
        let mut svc = LinearSVC::new().set_max_iter(5);
        let uid = svc.operator().uid.clone();
        let map = ParamMap::new().put(&uid, "rank", MlValue::Int(2)).put(
            &uid,
            "maxIter",
            MlValue::Int(7),
        );
        svc.apply_param_map(&map).unwrap();
        assert!(svc.params().get_param("rank").is_none());
        assert_eq!(svc.max_iter(), 7);
    }

    #[test]
    fn values_convert_as_pyspark_type_converters_convert_them() {
        let svc = "org.apache.spark.ml.classification.LinearSVC";
        assert_eq!(
            converted(svc, "regParam", &MlValue::Int(0)).unwrap(),
            MlValue::Double(0.0)
        );
        assert_eq!(
            converted(svc, "maxIter", &MlValue::Double(5.0)).unwrap(),
            MlValue::Int(5)
        );
        assert_eq!(
            converted(svc, "maxIter", &MlValue::Long(1 << 40)).unwrap(),
            MlValue::Long(1 << 40)
        );
        let error = converted(svc, "maxIter", &MlValue::Double(5.5)).unwrap_err();
        assert!(
            error.to_string().contains("Could not convert 5.5 to int"),
            "{error}"
        );
        let error = converted(svc, "featuresCol", &MlValue::Int(1)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Could not convert <class 'int'> to string type"),
            "{error}"
        );
        let error = converted(svc, "fitIntercept", &MlValue::Int(1)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Boolean Param requires value of type bool. Found <class 'int'>."),
            "{error}"
        );
        assert_eq!(
            converted(
                "org.apache.spark.ml.feature.QuantileDiscretizer",
                "inputCols",
                &MlValue::Array(vec![MlValue::String("a".into())])
            )
            .unwrap(),
            MlValue::Array(vec![MlValue::String("a".into())])
        );
    }

    #[test]
    fn grid_ints_that_pyspark_cannot_send_are_its_errors() {
        // PySpark's `toInt` and `toListInt` keep a Python int of any size, which its
        // `LiteralExpression` sends of `_infer_type`: a whole float beyond a long is the
        // `VALUE_NOT_BETWEEN` of the fit, not a saturated long, and a list is of the type of its
        // first element, whose `assert` fails of an element out of its range.
        use MlValue::{Array, Bool, Double, Int, Long};
        let svc = "org.apache.spark.ml.classification.LinearSVC";
        let not_between = "[VALUE_NOT_BETWEEN] Value for `value` must be between \
                           -9223372036854775808 and 32767.";
        for value in [1e20, -1e20, 9_223_372_036_854_775_808.0, f64::MAX] {
            let error = converted(svc, "maxIter", &Double(value)).unwrap_err();
            assert_eq!(error.to_string(), not_between, "{value}");
        }
        assert_eq!(
            converted(svc, "maxIter", &Double(-9_223_372_036_854_775_808.0)).unwrap(),
            Long(i64::MIN)
        );
        let mut copy = LinearSVC::new();
        let uid = copy.operator().uid.clone();
        let map = ParamMap::new().put(&uid, "maxIter", Double(1e20));
        let error = copy.apply_param_map(&map).unwrap_err();
        assert_eq!(error.to_string(), not_between);
        assert!(copy.params().get_param("maxIter").is_none());

        let mlp = "org.apache.spark.ml.classification.MultilayerPerceptronClassifier";
        let list = |values: Vec<MlValue>| converted(mlp, "layers", &Array(values));
        assert_eq!(
            list(vec![Int(2), Double(3.0)]).unwrap(),
            Array(vec![Int(2), Int(3)])
        );
        let longs = list(vec![Long(1 << 33), Int(2)]).unwrap();
        assert_eq!(longs, Array(vec![Long(1 << 33), Long(2)]));
        let literal = longs.to_literal().unwrap();
        let Some(proto::expression::literal::LiteralType::Array(array)) = literal.literal_type
        else {
            panic!("not an array: {literal:?}");
        };
        assert!(array.elements.iter().all(|element| matches!(
            element.literal_type,
            Some(proto::expression::literal::LiteralType::Long(_))
        )));
        assert_eq!(list(Vec::new()).unwrap(), Array(Vec::new()));
        for values in [
            vec![Int(2), Long(1 << 33)],
            vec![Int(2), Double(1e20)],
            vec![Long(1 << 33), Double(1e20)],
        ] {
            let error = list(values.clone()).unwrap_err();
            assert_eq!(error.kind, SparkErrorKind::AssertionError, "{values:?}");
        }
        let error = list(vec![Double(1e20), Int(2)]).unwrap_err();
        assert_eq!(error.to_string(), not_between);
        let error = list(vec![Int(2), Bool(true)]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Could not convert [2, True] to list of ints"),
            "{error}"
        );
    }

    #[test]
    fn a_fitted_model_is_of_the_jvm_class_that_pyspark_creates() {
        assert_eq!(
            model_class("org.apache.spark.ml.classification.DecisionTreeClassifier"),
            "org.apache.spark.ml.classification.DecisionTreeClassificationModel"
        );
        assert_eq!(
            model_class("org.apache.spark.ml.feature.QuantileDiscretizer"),
            "org.apache.spark.ml.feature.Bucketizer"
        );
        assert_eq!(
            model_class("org.apache.spark.ml.recommendation.ALS"),
            "org.apache.spark.ml.recommendation.ALSModel"
        );
    }
}
