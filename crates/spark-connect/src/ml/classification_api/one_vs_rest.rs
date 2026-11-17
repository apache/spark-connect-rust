// SPDX-License-Identifier: Apache-2.0
//! The typed accessors of OneVsRestModel. Its fit and transform run in the reduction of
//! [`crate::ml::OneVsRest`] and [`crate::ml::OneVsRestModel`], of the class count and the pool
//! of PySpark's `OneVsRest._fit` (see [`python_int`] and [`class_threads`]). Their persistence is
//! PySpark's Python format (`OneVsRestWriter`, `OneVsRestModelWriter` and their readers, see the
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
use crate::row::Value;
use crate::session::SparkSession;
use spark_connect_core::error::{Result, SparkError, SparkErrorKind};
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
    /// A model of the binary `models`, one per class in the order of the class index, as PySpark's
    /// `OneVsRestModel(models)` constructs one: of a new uid, no set params and no classifier. It
    /// transforms data of the session `session`, of which the models are.
    pub fn new(session: &SparkSession, models: Vec<Box<dyn Model>>) -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.classification.OneVsRestModel",
                OperatorType::Model,
            ),
            params: Params::new(),
            models,
            session_id: session.session_id().to_string(),
            classifier: None,
        }
    }

    /// Set the classifier of the model (its `classifier` param), the classifier that fitted its
    /// models, as PySpark's reader sets it (`ovaModel.set(ovaModel.classifier, classifier)`).
    pub fn set_classifier(mut self, classifier: impl Into<Classifier>) -> Self {
        self.classifier = Some(classifier.into());
        self
    }

    /// The binary submodels, one per class, in the order of the class index (`models`).
    pub fn models(&self) -> Vec<SubModel> {
        self.models
            .iter()
            .map(|model| SubModel::of(model.clone_box()))
            .collect()
    }
}

/// Python's `int(value)` of the max label `value` that PySpark's `OneVsRest._fit` reads, of which
/// it takes its class count (`int(...) + 1`), with the errors of Python's `int`, as the Python
/// values that PySpark collects of the label's type: a TypeError of a null (of no rows or only null
/// labels) and of a value of no number or string; of a float, the integer of it truncated toward
/// zero, a ValueError of NaN and an OverflowError of an infinity; of a string (and of bytes), its
/// decimal integer, of an optional sign, ASCII digits and single underscores between them, between
/// whitespace, else the ValueError of the literal (Python's `int` takes the decimal digits of other
/// scripts too); of a decimal, its integer truncated toward zero. An integer beyond the range of
/// i128 is its bound, as of any count beyond 2^63 a fit fails the same (see [`class_threads`]).
pub(crate) fn python_int(value: &Value) -> Result<i128> {
    let type_error = |name: &str| SparkError {
        kind: SparkErrorKind::BuiltinTypeError,
        ..SparkError::value_msg(format!(
            "int() argument must be a string, a bytes-like object or a real number, not '{name}'"
        ))
    };
    Ok(match value {
        Value::Null => return Err(type_error("NoneType")),
        Value::Bool(value) => i128::from(*value),
        Value::Byte(value) => i128::from(*value),
        Value::Short(value) => i128::from(*value),
        Value::Integer(value) => i128::from(*value),
        Value::Long(value) => i128::from(*value),
        Value::Float(value) => float_int(f64::from(*value))?,
        Value::Double(value) => float_int(*value)?,
        // Of a string, the whitespace of Unicode (C's `isspace` of ASCII, which Python's `int`
        // strips of the ASCII characters, and its `str.isspace` of the others).
        Value::String(text) => match decimal_digits(text.chars(), char::is_whitespace) {
            Some(value) => value,
            None => return Err(invalid_literal(&str_repr(text))),
        },
        // Of bytes, the whitespace of C's `isspace`.
        Value::Binary(bytes) => match decimal_digits(bytes.iter().map(|b| char::from(*b)), |c| {
            matches!(c, '\t'..='\r' | ' ')
        }) {
            Some(value) => value,
            None => return Err(invalid_literal(&bytes_repr(bytes))),
        },
        Value::Decimal { value, .. } => decimal_int(value)?,
        Value::Date(_) => return Err(type_error("datetime.date")),
        Value::Timestamp(_) | Value::TimestampNtz(_) => {
            return Err(type_error("datetime.datetime"))
        }
        Value::Time(_) => return Err(type_error("datetime.time")),
        Value::DayTimeInterval(_) => return Err(type_error("datetime.timedelta")),
        // PySpark's collect of a year-month interval raises this, before any `int`.
        Value::YearMonthInterval(_) => {
            return Err(SparkError::classed(
                SparkErrorKind::NotImplementedError,
                "NOT_IMPLEMENTED",
                &[(
                    "feature",
                    "Collecting a year-month interval value in Spark Connect",
                )],
            ))
        }
        Value::List(_) => return Err(type_error("list")),
        Value::Map(_) => return Err(type_error("dict")),
        Value::Struct(_) => return Err(type_error("Row")),
        Value::Variant { .. } => return Err(type_error("VariantVal")),
    })
}

/// `int(value)` of a Python float: truncated toward zero, and the errors of NaN and infinity.
fn float_int(value: f64) -> Result<i128> {
    if value.is_nan() {
        return Err(SparkError::builtin_value_msg(
            "cannot convert float NaN to integer",
        ));
    }
    if value.is_infinite() {
        return Err(SparkError {
            kind: SparkErrorKind::BuiltinOverflowError,
            ..SparkError::value_msg("cannot convert float infinity to integer")
        });
    }
    // `as` truncates toward zero, and saturates beyond the range.
    Ok(value as i128)
}

/// The integer of the decimal literal `chars` of Python's `int` of base 10, between the
/// whitespace of `is_space`: an optional sign, then ASCII digits with single underscores between
/// them; None of any other text. Saturated to the range of i128.
fn decimal_digits(
    chars: impl Iterator<Item = char>,
    is_space: impl Fn(char) -> bool,
) -> Option<i128> {
    let chars: Vec<char> = chars.collect();
    let start = chars.iter().position(|c| !is_space(*c))?;
    let end = chars.iter().rposition(|c| !is_space(*c))? + 1;
    let mut digits = &chars[start..end];
    let negative = digits.first() == Some(&'-');
    if matches!(digits.first(), Some('-' | '+')) {
        digits = &digits[1..];
    }
    let mut value: i128 = 0;
    let mut previous_digit = false;
    for c in digits {
        match c.to_digit(10) {
            Some(digit) if c.is_ascii_digit() => {
                value = value.saturating_mul(10).saturating_add(i128::from(digit));
                previous_digit = true;
            }
            // An underscore only between two digits.
            _ if *c == '_' && previous_digit => previous_digit = false,
            _ => return None,
        }
    }
    if !previous_digit {
        return None;
    }
    Some(if negative { -value } else { value })
}

/// `int(value)` of a Python `Decimal` of the text `text`: its integer, truncated toward zero.
fn decimal_int(text: &str) -> Result<i128> {
    let invalid = || SparkError::value_msg(format!("the decimal {text} is no number"));
    let (mantissa, exponent) = match text.find(['e', 'E']) {
        Some(at) => (
            &text[..at],
            text[at + 1..].parse::<i64>().map_err(|_| invalid())?,
        ),
        None => (text, 0),
    };
    let negative = mantissa.starts_with('-');
    let mantissa = mantissa.trim_start_matches(['-', '+']);
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits: Vec<u32> = whole
        .chars()
        .chain(fraction.chars())
        .map(|c| c.to_digit(10).ok_or_else(invalid))
        .collect::<Result<_>>()?;
    // The digits of the integer part, after the point moves by the exponent.
    let integer_digits = (whole.len() as i64).saturating_add(exponent);
    let mut value: i128 = 0;
    for index in 0..integer_digits.max(0) {
        let digit = digits.get(index as usize).copied().unwrap_or(0);
        value = value.saturating_mul(10).saturating_add(i128::from(digit));
    }
    Ok(if negative { -value } else { value })
}

/// Python's `repr` of the string `text`, of at most 200 characters, as the ValueError of `int`
/// shows it (`%.200R`): in single quotes, or in double quotes when it has a single quote and no
/// double quote, of the escapes of backslashes, quotes, tabs, newlines and returns, and of the
/// characters that do not print (of ASCII, and the controls and whitespace of Unicode).
fn str_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut repr = String::from(quote);
    for c in text.chars() {
        match c {
            '\\' => repr.push_str("\\\\"),
            '\t' => repr.push_str("\\t"),
            '\n' => repr.push_str("\\n"),
            '\r' => repr.push_str("\\r"),
            c if c == quote => {
                repr.push('\\');
                repr.push(c);
            }
            ' ' => repr.push(' '),
            c if c.is_control() || c.is_whitespace() => {
                let code = u32::from(c);
                repr.push_str(&match code {
                    0..=0xff => format!("\\x{code:02x}"),
                    0x100..=0xffff => format!("\\u{code:04x}"),
                    _ => format!("\\U{code:08x}"),
                });
            }
            c => repr.push(c),
        }
    }
    repr.push(quote);
    repr.chars().take(200).collect()
}

/// Python's `repr` of the bytes `bytes`, of at most 200 characters, as the ValueError of `int`
/// shows it: `b'...'`, of the quotes and escapes of [`str_repr`], and `\xhh` of a byte that is no
/// printable ASCII.
fn bytes_repr(bytes: &[u8]) -> String {
    let quote = if bytes.contains(&b'\'') && !bytes.contains(&b'"') {
        b'"'
    } else {
        b'\''
    };
    let mut repr = String::from("b");
    repr.push(char::from(quote));
    for &b in bytes {
        match b {
            b'\\' => repr.push_str("\\\\"),
            b'\t' => repr.push_str("\\t"),
            b'\n' => repr.push_str("\\n"),
            b'\r' => repr.push_str("\\r"),
            b if b == quote => {
                repr.push('\\');
                repr.push(char::from(b));
            }
            0x20..=0x7e => repr.push(char::from(b)),
            b => repr.push_str(&format!("\\x{b:02x}")),
        }
    }
    repr.push(char::from(quote));
    repr.chars().take(200).collect()
}

/// The ValueError of Python's `int` of a literal that is no integer, of its repr `repr`.
fn invalid_literal(repr: &str) -> SparkError {
    SparkError::builtin_value_msg(format!("invalid literal for int() with base 10: {repr}"))
}

/// The threads of the fits of the classes of a fit of `num_classes` classes of the parallelism
/// `parallelism` (None of a null), as PySpark's `OneVsRest._fit` makes its tasks and its pool,
/// with their errors. Its list of the tasks of the classes, `[trainSingleClass] * numClasses`,
/// has at most an index-sized number of items (an OverflowError beyond), and runs out of memory of
/// a count beyond `i32::MAX`, which this client takes no more classes than. Its pool is
/// `ThreadPool(processes=min(parallelism, numClasses))`: Python's `min`, of the TypeError of a
/// None, and a pool of one process at least ("Number of processes must be at least 1", of a count
/// of no class too).
pub(crate) fn class_threads(num_classes: i128, parallelism: Option<i64>) -> Result<usize> {
    if num_classes > i128::from(i64::MAX) {
        return Err(SparkError {
            kind: SparkErrorKind::BuiltinOverflowError,
            ..SparkError::value_msg("cannot fit 'int' into an index-sized integer")
        });
    }
    if num_classes > i128::from(i32::MAX) {
        return Err(SparkError::value_msg(format!(
            "OneVsRest cannot fit {num_classes} classes: the tasks of more than {} classes do \
             not fit in memory",
            i32::MAX
        )));
    }
    let Some(parallelism) = parallelism else {
        return Err(SparkError {
            kind: SparkErrorKind::BuiltinTypeError,
            ..SparkError::value_msg("'<' not supported between instances of 'int' and 'NoneType'")
        });
    };
    let threads = i128::from(parallelism).min(num_classes);
    if threads < 1 {
        return Err(SparkError::builtin_value_msg(
            "Number of processes must be at least 1",
        ));
    }
    Ok(threads as usize)
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
        // An unset classifier raises PySpark's KeyError of the param, of `getClassifier`.
        let classifier = self
            .classifier
            .as_ref()
            .ok_or_else(|| crate::ml::classifier_unset(&self.uid))?
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

    /// Load the estimator saved at `path`, as `OneVsRestReader.load`: its uid, its params and its
    /// classifier. As PySpark's reader, it loads metadata of any class (`loadMetadata` of no
    /// expected class): of PySpark's Python format, and of the JVM's class name, as a classic
    /// `JavaMLWriter` (and Scala's writer) saves a OneVsRest of a Java classifier, of the same
    /// layout.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let metadata = load_python_metadata(session, path)?;
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

    /// Load the model saved at `path`, as `OneVsRestModelReader.load`: its classifier, its
    /// `numClasses` models, and `OneVsRestModel(models)._resetUid(uid)` of the classifier and the
    /// params of its metadata. As PySpark's reader, it loads metadata of any class (`loadMetadata`
    /// of no expected class): of PySpark's Python format, and of the JVM's class name, as a
    /// classic `JavaMLWriter` (and Scala's writer) saves a OneVsRestModel of Java models, of the
    /// same layout.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let metadata = load_python_metadata(session, path)?;
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
        let mut model = Self::new(session, models).set_classifier(classifier);
        model.operator.uid = metadata_str(&metadata, "uid")?.to_string();
        model.params = read_params(&metadata)?;
        Ok(model)
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
    fn the_class_count_is_pythons_int_of_the_max_label() {
        // The values and the messages of Python 3.11's `int` of the same values, as PySpark's
        // `OneVsRest._fit` takes `int(max) + 1` of the max label that it collects.
        let int = |value: Value| python_int(&value).map_err(|e| (e.kind, e.to_string()));
        let decimal = |text: &str| Value::Decimal {
            value: text.into(),
            precision: None,
            scale: None,
        };
        for (value, expected) in [
            (Value::Bool(true), 1),
            (Value::Byte(-7), -7),
            (Value::Integer(3), 3),
            (Value::Long(i64::MAX), i128::from(i64::MAX)),
            (Value::Double(2.9), 2),
            (Value::Double(-2.9), -2),
            (Value::Double(-0.5), 0),
            (Value::Float(9.5), 9),
            (Value::String("10".into()), 10),
            (Value::String(" 9 ".into()), 9),
            (Value::String("+5".into()), 5),
            (Value::String("-0".into()), 0),
            (Value::String("1_000".into()), 1000),
            (Value::String("007".into()), 7),
            (Value::String("\u{a0}1\u{3000}".into()), 1),
            (Value::String("1\u{b}".into()), 1),
            (Value::Binary(b" 3 ".to_vec()), 3),
            (decimal("1.5"), 1),
            (decimal("-1.5"), -1),
            (decimal("-0.5"), 0),
            (decimal("1E+3"), 1000),
            (decimal("12.5E1"), 125),
        ] {
            assert_eq!(int(value.clone()), Ok(expected), "{value:?}");
        }
        // A float beyond i128 is its bound, of which a fit fails as of any count beyond 2^63.
        assert_eq!(int(Value::Double(1e300)), Ok(i128::MAX));
        let invalid = |repr: &str| {
            Err((
                SparkErrorKind::BuiltinValueError,
                format!("invalid literal for int() with base 10: {repr}"),
            ))
        };
        for (text, repr) in [
            ("b", "'b'"),
            ("1.5", "'1.5'"),
            ("1__0", "'1__0'"),
            ("_1", "'_1'"),
            ("1_", "'1_'"),
            ("", "''"),
            (" ", "' '"),
            ("it's", "\"it's\""),
            ("a'b\"c", "'a\\'b\"c'"),
            ("x\ny", "'x\\ny'"),
            ("tab\there", "'tab\\there'"),
            ("\\", "'\\\\'"),
            ("\u{2028}x", "'\\u2028x'"),
            ("\u{7f}", "'\\x7f'"),
            ("\u{1c}7\u{1f}", "'\\x1c7\\x1f'"),
            ("é", "'é'"),
        ] {
            assert_eq!(int(Value::String(text.into())), invalid(repr), "{text:?}");
        }
        // The repr of the message is of 200 characters at most, as `%.200R`.
        let long = int(Value::String("x".repeat(300))).unwrap_err().1;
        assert_eq!(
            long,
            format!(
                "invalid literal for int() with base 10: '{}",
                "x".repeat(199)
            )
        );
        for (bytes, repr) in [
            (&b"b"[..], "b'b'"),
            (b"it's", "b\"it's\""),
            (b"\xff1", "b'\\xff1'"),
            (b"1\x00", "b'1\\x00'"),
            (b"\x1c5", "b'\\x1c5'"),
        ] {
            assert_eq!(
                int(Value::Binary(bytes.to_vec())),
                invalid(repr),
                "{bytes:?}"
            );
        }
        assert_eq!(
            int(Value::Double(f64::NAN)),
            Err((
                SparkErrorKind::BuiltinValueError,
                "cannot convert float NaN to integer".into()
            ))
        );
        for infinity in [f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                int(Value::Double(infinity)),
                Err((
                    SparkErrorKind::BuiltinOverflowError,
                    "cannot convert float infinity to integer".into()
                ))
            );
        }
        let type_error = |name: &str| {
            Err((
                SparkErrorKind::BuiltinTypeError,
                format!(
                    "int() argument must be a string, a bytes-like object or a real number, not \
                     '{name}'"
                ),
            ))
        };
        for (value, name) in [
            (Value::Null, "NoneType"),
            (Value::Date(0), "datetime.date"),
            (Value::Timestamp(0), "datetime.datetime"),
            (Value::TimestampNtz(0), "datetime.datetime"),
            (Value::Time(0), "datetime.time"),
            (Value::DayTimeInterval(0), "datetime.timedelta"),
            (Value::List(vec![Value::Integer(1)]), "list"),
            (Value::Map(Vec::new()), "dict"),
        ] {
            assert_eq!(int(value.clone()), type_error(name), "{value:?}");
        }
    }

    #[test]
    fn the_class_threads_are_pysparks_pool_of_its_list_of_the_class_tasks() {
        let threads = |classes: i128, parallelism| {
            class_threads(classes, parallelism).map_err(|e| e.to_string())
        };
        // `ThreadPool(processes=min(parallelism, numClasses))`.
        assert_eq!(threads(3, Some(1)), Ok(1));
        assert_eq!(threads(3, Some(2)), Ok(2));
        assert_eq!(threads(3, Some(i64::MAX)), Ok(3));
        let none = Err("Number of processes must be at least 1".to_string());
        assert_eq!(threads(3, Some(0)), none);
        assert_eq!(threads(3, Some(i64::MIN)), none);
        // A max label below -1 is a count of no class, and of no process.
        assert_eq!(threads(0, Some(4)), none);
        assert_eq!(threads(-5, Some(4)), none);
        assert_eq!(
            threads(3, None),
            Err("'<' not supported between instances of 'int' and 'NoneType'".to_string())
        );
        // `[trainSingleClass] * numClasses` of more items than an index holds, before the pool.
        assert_eq!(
            threads(i128::from(i64::MAX) + 1, None),
            Err("cannot fit 'int' into an index-sized integer".to_string())
        );
        assert!(threads(i128::from(i32::MAX) + 1, Some(1))
            .unwrap_err()
            .contains("do not fit in memory"));
        assert_eq!(threads(i128::from(i32::MAX), Some(1)), Ok(1));
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
